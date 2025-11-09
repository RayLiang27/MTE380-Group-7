import cv2
import numpy as np
import json
import serial
import time
from datetime import datetime

# Default neutral servo angles (degrees)
NEUTRAL_ANGLES = [50, 55, 45]

class SimpleCalibratorSP:
    def __init__(self):
        # Camera setup
        self.CAM_INDEX = 0
        self.FRAME_W, self.FRAME_H = 640, 480
        self.current_frame = None

        # Calibration states
        self.phase = "servo"
        self.hsv_samples = []
        self.lower_hsv = None
        self.upper_hsv = None
        self.platform_points = []
        self.pixel_to_meter_ratio = None
        self.origin_px = None

        # Visualization
        self.ball_contour = None
        self.ball_center = None

        # Known platform spacing (m)
        self.PLATFORM_EDGE_M = 0.10

        # Arduino serial setup
        self.arduino_port = "COM5" #"/dev/cu.usbmodem1301"  # Update as needed (COM3 on Windows)
        self.baud_rate = 115200
        self.arduino = None

    # ---------------- SERVO CALIBRATION FIRST ---------------- #

    def connect_arduino(self):
        try:
            self.arduino = serial.Serial(self.arduino_port, self.baud_rate, timeout=2)
            print(self.arduino)
            time.sleep(2)
            print(f"[SERVO] Connected to Arduino on {self.arduino_port}")
        except Exception as e:
            print(f"[ERROR] Could not connect to Arduino: {e}")
            self.arduino = None

    def send_servo_angles(self, angles):
        """Send [a1, a2, a3] in degrees to Arduino."""
        if not self.arduino:
            print("[WARN] Arduino not connected; skipping servo command.")
            return
        cmd = f"{angles[0]},{angles[1]},{angles[2]}\n"
        self.arduino.write(cmd.encode())
        print(f"[SERVO] Sent: {cmd.strip()}")

    def calibrate_servos(self):
        """Step 1: Level the platform using sliders for each motor."""
        self.connect_arduino()
        print("\n=== SERVO CALIBRATION ===")
        print("Use sliders to adjust each motor's neutral angle (0–70°).")
        print("Press SPACE when satisfied to continue calibration.")

        # Create window with sliders
        cv2.namedWindow("Servo Calibration")

        # Create trackbars for each motor
        cv2.createTrackbar("Motor 1", "Servo Calibration", NEUTRAL_ANGLES[0], 70, lambda x: None)
        cv2.createTrackbar("Motor 2", "Servo Calibration", NEUTRAL_ANGLES[1], 70, lambda x: None)
        cv2.createTrackbar("Motor 3", "Servo Calibration", NEUTRAL_ANGLES[2], 70, lambda x: None)

        last_angles = [None, None, None]

        while True:
            # Read slider values
            a1 = cv2.getTrackbarPos("Motor 1", "Servo Calibration")
            a2 = cv2.getTrackbarPos("Motor 2", "Servo Calibration")
            a3 = cv2.getTrackbarPos("Motor 3", "Servo Calibration")
            angles = [a1, a2, a3]

            # Only send command if angles changed
            if angles != last_angles:
                self.send_servo_angles(angles)
                last_angles = angles.copy()

            # Show live angle values on a black background
            frame = np.zeros((200, 400, 3), dtype=np.uint8)
            cv2.putText(frame, f"Motor 1: {a1}°", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            cv2.putText(frame, f"Motor 2: {a2}°", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            cv2.putText(frame, f"Motor 3: {a3}°", (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            cv2.imshow("Servo Calibration", frame)
            # self.send_servo_angles(angles)
            key = cv2.waitKey(50) & 0xFF
            if key == 27:  # ESC
                print("[INFO] Calibration aborted.")
                cv2.destroyWindow("Servo Calibration")
                return
            elif key == 32:  # SPACE
                print(f"[INFO] Final neutral angles: {angles}")
                # global NEUTRAL_ANGLES
                NEUTRAL_ANGLES[:] = angles
                break

        cv2.destroyWindow("Servo Calibration")
        self.save_servo_calibration()
        print("[INFO] Servo calibration complete.")
        self.phase = "color"


    def save_servo_calibration(self):
        """Save servo neutral angles (degrees only)."""
        data = {"servo_calibration": {"neutral_angles_deg": NEUTRAL_ANGLES}}
        try:
            with open("Stewart Platform/config_sp.json", "r") as f:
                old = json.load(f)
                old.update(data)
                data = old
        except FileNotFoundError:
            pass

        with open("Stewart Platform/config_sp.json", "w") as f:
            json.dump(data, f, indent=4)
        print("[SAVE] Servo calibration saved to config_sp.json.")

    # ---------------- CAMERA CALIBRATION ---------------- #

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
        """Collect HSV samples for ball color."""
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
        print(f"[COLOR] {len(self.hsv_samples)} samples → HSV {self.lower_hsv}-{self.upper_hsv}")

    def detect_and_draw_ball(self):
        """Detect and hold the ball contour."""
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
                print(f"[BALL] Center=({int(x)}, {int(y)}), r={radius:.2f}")

    def compute_geometry(self):
        """Compute platform geometry and save config."""
        pts = np.array(self.platform_points, dtype=np.float32)
        dists = [
            np.linalg.norm(pts[0]-pts[1]),
            np.linalg.norm(pts[1]-pts[2]),
            np.linalg.norm(pts[2]-pts[0])
        ]
        avg_pix_dist = np.mean(dists)
        self.pixel_to_meter_ratio = self.PLATFORM_EDGE_M / avg_pix_dist
        self.origin_px = np.mean(pts, axis=0)
        print(f"[GEO] Ratio={self.pixel_to_meter_ratio:.6f}, origin={self.origin_px}")

        self.save_camera_config()
        self.phase = "complete"
        print("[DONE] Calibration complete. ESC to exit.")

    def save_camera_config(self):
        data = {
            "timestamp": datetime.now().isoformat(),
            "camera": {
                "index": self.CAM_INDEX,
                "frame_width": self.FRAME_W,
                "frame_height": self.FRAME_H
            },
            "calibration": {
                "lower_hsv": self.lower_hsv.tolist() if self.lower_hsv is not None else None,
                "upper_hsv": self.upper_hsv.tolist() if self.upper_hsv is not None else None,
                "platform_points_px": self.platform_points,
                "origin_px": self.origin_px.tolist() if self.origin_px is not None else None,
                "pixel_to_meter_ratio": float(self.pixel_to_meter_ratio) if self.pixel_to_meter_ratio else None
            }
        }

        try:
            with open("Stewart Platform/config_sp.json", "r") as f:
                old = json.load(f)
                old.update(data)
                data = old
        except FileNotFoundError:
            pass

        with open("Stewart Platform/config_sp.json", "w") as f:
            json.dump(data, f, indent=4)
        print("[SAVE] Camera calibration saved to config_sp.json.")

    # ---------------- VISUALIZATION ---------------- #

    def draw_overlays(self, frame):
        if self.ball_contour is not None:
            cv2.drawContours(frame, [self.ball_contour], -1, (0, 255, 255), 2)
            if self.ball_center is not None:
                cv2.circle(frame, self.ball_center, 5, (0, 0, 255), -1)

        if len(self.platform_points) > 0:
            for i, p in enumerate(self.platform_points):
                cv2.circle(frame, p, 5, (0, 0, 255), -1)
                cv2.putText(frame, f"P{i+1}", (p[0]+5, p[1]-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        if len(self.platform_points) == 3:
            pts = np.array(self.platform_points, np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], True, (255, 0, 0), 2)
            cx = int(sum(p[0] for p in self.platform_points) / 3)
            cy = int(sum(p[1] for p in self.platform_points) / 3)
            cv2.circle(frame, (cx, cy), 6, (0, 0, 255), -1)
            cv2.putText(frame, "Center", (cx + 10, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # ---------------- MAIN LOOP ---------------- #

    def run(self):
        print("\n=== Stewart Platform Calibration ===")
        print("Step 1: Servo leveling\nStep 2: Ball color detection\nStep 3: Platform geometry\n")

        # Step 1 - Servo leveling
        self.calibrate_servos()

        # Step 2 - Camera setup
        cap = cv2.VideoCapture(self.CAM_INDEX)
        cv2.namedWindow("Calibration")
        cv2.setMouseCallback("Calibration", self.mouse_callback)
        print("[INFO] Phase 2: click ball to sample color, press SPACE when done.")

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
                    print("[INFO] Phase 3: click 3 servo pivots (A,B,C).")

        cap.release()
        cv2.destroyAllWindows()
        if self.arduino:
            self.arduino.close()


if __name__ == "__main__":
    SimpleCalibratorSP().run()
