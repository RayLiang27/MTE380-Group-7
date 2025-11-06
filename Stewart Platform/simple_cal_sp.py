"""
Simple Calibration for 3-Servo Stewart Platform (Camera Only)
-------------------------------------------------------------
• Click on the ball in the live feed a few times to sample its color.
• Press SPACE when done sampling.
• Then click the 3 servo peg positions (A, B, C).
• Press SPACE again to finish.

Saves: config_sp.json with color thresholds, camera geometry, and scaling.
"""

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

        # assume platform diameter/edge distance known (approx.)
        self.PLATFORM_EDGE_M = 0.10  # ~10 cm between servo pivots

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.phase == "color":
                self.sample_color(x, y)
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

    def compute_geometry(self):
        """Estimate pixel-to-meter ratio and center origin."""
        pts = np.array(self.platform_points, dtype=np.float32)
        # average of edge lengths in pixels
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

            # draw current points
            if self.phase == "geometry":
                for i, p in enumerate(self.platform_points):
                    cv2.circle(frame, p, 5, (0, 0, 255), -1)
                    cv2.putText(frame, f"P{i+1}", (p[0]+5, p[1]-5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

            # preview mask for color phase
            if self.phase == "color" and self.lower_hsv is not None:
                hsv = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2HSV)
                mask = cv2.inRange(hsv, self.lower_hsv, self.upper_hsv)
                frame = cv2.bitwise_and(frame, frame, mask=mask)

            cv2.imshow("Calibration", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == 27:  # ESC
                break
            elif key == 32:  # SPACE to advance
                if self.phase == "color":
                    self.phase = "geometry"
                    print("[INFO] Phase 2: click the three servo pivot points (A, B, C), then press SPACE.")
                elif self.phase == "geometry" and len(self.platform_points) == 3:
                    self.compute_geometry()
                    print("[DONE] Calibration complete. ESC to exit.")

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    SimpleCalibratorSP().run()
