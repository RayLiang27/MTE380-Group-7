import cv2
import numpy as np
import json
import math
from datetime import datetime

class SimpleCalibratorSP:
    def __init__(self):
        self.CAM_INDEX = 0
        self.FRAME_W, self.FRAME_H = 640, 480
        self.current_frame = None
        self.phase = "color"
        self.hsv_samples = []
        self.lower_hsv = None
        self.upper_hsv = None
        self.platform_points = []  # 3 servo corners
        self.pixel_to_meter_ratio = None
        self.origin_px = None

        # for persistent visualization
        self.ball_contour = None
        self.ball_center = None

        # known approx. edge distance (m)
        self.PLATFORM_EDGE_M = 0.10  

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.phase == "color":
                self.sample_color(x, y)
                self.detect_and_draw_ball()
            elif self.phase == "geometry" and len(self.platform_points) < 3:
                self.platform_points.append((x, y))
                print(f"[GEO] Point {len(self.platform_points)} selected at ({x}, {y})")
                if len(self.platform_points) == 3:
                    self.compute_geometry()

    def sample_color(self, x, y):
        """Collect HSV samples around click."""
        if self.current_frame is None:
            return
        hsv = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2HSV)
        region = hsv[max(y-2,0):y+3, max(x-2,0):x+3]
        pixels = region.reshape(-1,3)
        self.hsv_samples.extend(pixels)
        samples = np.array(self.hsv_samples)
        h_margin, s_margin, v_margin = 5, 20, 20
        self.lower_hsv = np.maximum([0,0,0], np.min(samples, axis=0) - [h_margin, s_margin, v_margin])
        self.upper_hsv = np.minimum([179,255,255], np.max(samples, axis=0) + [h_margin, s_margin, v_margin])
        print(f"[COLOR] Samples={len(self.hsv_samples)}  HSV range={self.lower_hsv}–{self.upper_hsv}")

    def detect_and_draw_ball(self):
        """Detect ball contour once color is sampled, hold the outline."""
        if self.lower_hsv is None or self.current_frame is None:
            return
        hsv = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_hsv, self.upper_hsv)
        mask = cv2.medianBlur(mask, 5)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            (x, y), radius = cv2.minEnclosingCircle(largest)
            if radius > 3:
                self.ball_contour = largest
                self.ball_center = (int(x), int(y))
                print(f"[BALL] Center=({int(x)}, {int(y)}), radius={radius:.2f}")

    def compute_geometry(self):
        """Estimate pixel-to-meter ratio and center origin."""
        pts = np.array(self.platform_points, dtype=np.float32)
        dists = [
            np.linalg.norm(pts[0]-pts[1]),
            np.linalg.norm(pts[1]-pts[2]),
            np.linalg.norm(pts[2]-pts[0])
        ]
        avg_pix_dist = np.mean(dists)
        self.pixel_to_meter_ratio = self.PLATFORM_EDGE_M / avg_pix_dist
        self.origin_px = np.mean(pts, axis=0)
        print(f"[GEO] pixel_to_meter_ratio={self.pixel_to_meter_ratio:.6f}, origin={self.origin_px}")
        self.phase = "complete"
        self.save_config()

    def save_config(self):
        data = {
            "timestamp": datetime.now().isoformat(),
            "camera": {
                "index": self.CAM_INDEX,
                "frame_width": self.FRAME_W,
                "frame_height": self.FRAME_H
            },
            "calibration": {
                "lower_hsv": self.lower_hsv.tolist(),
                "upper_hsv": self.upper_hsv.tolist(),
                "platform_points_px": self.platform_points,
                "origin_px": self.origin_px.tolist(),
                "pixel_to_meter_ratio": float(self.pixel_to_meter_ratio)
            }
        }
        with open("Stewart Platform/config_sp.json", "w") as f:
            json.dump(data, f, indent=4)
        print("[SAVE] config_sp.json created successfully.")

    def draw_overlays(self, frame):
        """Draw visual overlays depending on current phase."""
        # draw detected ball
        if self.ball_contour is not None:
            cv2.drawContours(frame, [self.ball_contour], -1, (0, 255, 255), 2)
            if self.ball_center is not None:
                cv2.circle(frame, self.ball_center, 5, (0, 0, 255), -1)

        # draw triangle
        if len(self.platform_points) > 0:
            for i, p in enumerate(self.platform_points):
                cv2.circle(frame, p, 5, (0, 0, 255), -1)
                cv2.putText(frame, f"P{i+1}", (p[0]+5, p[1]-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        if len(self.platform_points) == 3:
            pts = np.array(self.platform_points, np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], True, (255, 0, 0), 2)
            # draw centroid (red dot)
            cx = int(sum(p[0] for p in self.platform_points) / 3)
            cy = int(sum(p[1] for p in self.platform_points) / 3)
            cv2.circle(frame, (cx, cy), 6, (0, 0, 255), -1)
            cv2.putText(frame, "Center", (cx + 10, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)


    def run(self):
        cap = cv2.VideoCapture(self.CAM_INDEX)
        cv2.namedWindow("Calibration")
        cv2.setMouseCallback("Calibration", self.mouse_callback)
        print("[INFO] Phase 1: click on ball to sample color, press SPACE when done.")

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (self.FRAME_W, self.FRAME_H))
            self.current_frame = frame.copy()

            self.draw_overlays(frame)

            cv2.imshow("Calibration", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == 27:  # ESC
                break
            elif key == 32:  # SPACE
                if self.phase == "color":
                    self.phase = "geometry"
                    print("[INFO] Phase 2: click 3 servo pivots (A,B,C).")
                elif self.phase == "geometry" and len(self.platform_points) == 3:
                    self.compute_geometry()
                    print("[DONE] Calibration complete. ESC to exit.")

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    SimpleCalibratorSP().run()
