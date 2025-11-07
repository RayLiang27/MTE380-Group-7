import cv2
import numpy as np
import json
import serial
import time
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from threading import Thread, Event
from collections import deque
import queue
import os
import math
import logging

# Make OpenCV behave on Windows (reduces contention with Tk)
try:
    cv2.setNumThreads(1)
except Exception:
    pass

# Optional kinematics helper
try:
    from spv4_kinematics import triangle_orientation_and_location, inverse_kinematics
except Exception:
    triangle_orientation_and_location = None
    inverse_kinematics = None

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s"
)

# -------------------- Non-blocking serial writer --------------------
class SerialWriter:
    """Moves serial writes off the control thread. Keeps only the latest command."""
    def __init__(self, ser: serial.Serial | None, write_timeout_s=0.01):
        self.ser = ser
        if self.ser:
            self.ser.timeout = 0.01
            self.ser.write_timeout = write_timeout_s
        self.q = queue.Queue(maxsize=1)  # latest only
        self.running = Event()
        self.drop_count = 0
        self._t = None

    def start(self):
        if not self.ser:
            return
        self.running.set()
        self._t = Thread(target=self._run, name="serial-writer", daemon=True)
        self._t.start()

    def stop(self):
        if not self.ser:
            return
        self.running.clear()
        if self._t:
            try:
                self._t.join(timeout=0.5)
            except Exception:
                pass

    def send(self, a0: int, a1: int, a2: int):
        """Queue a line 'a0,a1,a2\\n'. Drops the older line if queue is full."""
        if not self.ser:
            return
        cmd = f"{int(a0)},{int(a1)},{int(a2)}\n".encode('ascii', 'ignore')
        try:
            if self.q.full():
                _ = self.q.get_nowait()
                self.drop_count += 1
            self.q.put_nowait(cmd)
        except queue.Full:
            # extremely rare due to get above
            pass

    def _run(self):
        while self.running.is_set():
            try:
                cmd = self.q.get(timeout=0.05)
            except queue.Empty:
                continue
            try:
                self.ser.write(cmd)  # short write_timeout set
            except Exception as e:
                logging.warning(f"[SERVO] write failed: {e}")
                time.sleep(0.01)

# -------------------- Controller --------------------
class StewartPIDController:
    """Resource-optimized PID controller for 3-servos Stewart Platform."""
    def __init__(self, config_path="Stewart Platform/config_sp.json"):
        # Load config
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config not found: {config_path}")
        with open(config_path, "r") as f:
            self.config = json.load(f)

        # Calibration
        calib = self.config.get("calibration", {}) or {}
        self.lower_hsv = None
        self.upper_hsv = None
        lower = calib.get('lower_hsv')
        upper = calib.get('upper_hsv')
        if lower is not None and upper is not None:
            self.lower_hsv = np.array(lower, dtype=np.uint8)
            self.upper_hsv = np.array(upper, dtype=np.uint8)
        self.pixel_to_meter = float(calib.get('pixel_to_meter_ratio', 1.0))
        self.origin_px = np.array(calib.get('origin_px', [0, 0]), dtype=np.float32)

        # Servo neutral
        servos = (self.config.get('servo_calibration', {}) or {}).get('neutral_angles_deg')
        if servos and len(servos) >= 3:
            self.neutral_angles = [int(servos[0]), int(servos[1]), int(servos[2])]
        else:
            self.neutral_angles = [50, 50, 50]

        # Serial
        self.arduino_port = self.config.get('arduino_port', "COM4")
        self.baud_rate = int(self.config.get('baud_rate', 115200))
        self.arduino: serial.Serial | None = None
        self.serial_writer: SerialWriter | None = None

        # PID defaults
        self.Kp_x = 5.0; self.Ki_x = 0.0; self.Kd_x = 0.0
        self.Kp_y = 5.0; self.Ki_y = 0.0; self.Kd_y = 0.0

        # PID state
        self.integral_x = 0.0
        self.integral_y = 0.0
        self.prev_error_x = 0.0
        self.prev_error_y = 0.0

        # Mapping / geometry
        self.mapping_scale = float(self.config.get('mapping_scale_deg_per_m', 600.0))
        self.tilt_gain = float(self.config.get('tilt_gain_rad_per_m', 0.5))
        self.platform_h = float(self.config.get('platform_height_m', 12.0))

        # Camera / UI
        cam_cfg = self.config.get('camera', {}) or {}
        self.FRAME_W = int(cam_cfg.get('frame_width', 640))
        self.FRAME_H = int(cam_cfg.get('frame_height', 480))
        self.cam_index = int(cam_cfg.get('index', 0))
        self.preview = bool(cam_cfg.get('preview', True))

        # Logs (bounded)
        max_log = int(self.config.get('max_log_len', 2000))
        self.time_log = deque(maxlen=max_log)
        self.pos_x_log = deque(maxlen=max_log)
        self.pos_y_log = deque(maxlen=max_log)
        self.setpoint_x_log = deque(maxlen=max_log)
        self.setpoint_y_log = deque(maxlen=max_log)
        self.servo_log = deque(maxlen=max_log)

        # Runtime flags, queues
        self.running = Event()
        self.stop_request = Event()
        self.position_queue = queue.Queue(maxsize=1)  # latest center only

        # Setpoint
        self.setpoint_px = self.origin_px.copy()

        # Loop timing
        self.target_hz = float(self.config.get('control_rate_hz', 50.0))
        self.min_dt = 1.0 / max(1.0, self.target_hz)

        # Camera watchdog
        self.last_cam_time = time.monotonic()
        self.cam_stall_s = float(self.config.get('camera_stall_timeout_s', 3.0))

        # Tk root
        self.root = None

        # Send-on-change memory
        self._last_angles = (None, None, None)

    # ---------------- hardware -----------------
    def connect_arduino(self):
        try:
            self.arduino = serial.Serial(
                self.arduino_port,
                self.baud_rate,
                timeout=0.01,
                write_timeout=0.01
            )
            time.sleep(0.25)  # small settle
            logging.info(f"[SERVO] Connected {self.arduino_port} @ {self.baud_rate} bps")
            self.serial_writer = SerialWriter(self.arduino, write_timeout_s=0.01)
            self.serial_writer.start()
            return True
        except Exception as e:
            logging.warning(f"[SERVO] Could not open serial: {e} (simulation mode)")
            self.arduino = None
            self.serial_writer = None
            return False

    def _send_angles_nonblocking(self, angles):
        """Clamp + send via SerialWriter (if present)."""
        a0 = int(np.clip(angles[0], 0, 70))
        a1 = int(np.clip(angles[1], 0, 70))
        a2 = int(np.clip(angles[2], 0, 70))

        # send-on-change (>= 1 deg difference)
        la0, la1, la2 = self._last_angles
        if (la0 is None or
            abs(a0 - la0) >= 1 or
            abs(a1 - la1) >= 1 or
            abs(a2 - la2) >= 1):
            if self.serial_writer:
                self.serial_writer.send(a0, a1, a2)
            self._last_angles = (a0, a1, a2)

    # ---------------- vision -----------------
    def detect_ball_center(self, frame):
        """Returns (found: bool, center: np.array([x,y],float))"""
        if self.lower_hsv is None or self.upper_hsv is None:
            return False, None
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_hsv, self.upper_hsv)
        mask = cv2.medianBlur(mask, 5)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return False, None
        largest = max(contours, key=cv2.contourArea)
        (x, y), radius = cv2.minEnclosingCircle(largest)
        if radius < 3:
            return False, None
        return True, np.array([x, y], dtype=np.float32)

    # ---------------- PID math ----------------
    @staticmethod
    def update_pid(error, prev_error, integral, Kp, Ki, Kd, dt):
        if dt <= 0.0:
            return Kp*error + Ki*integral, integral, error
        integral_new = integral + error * dt
        derivative = (error - prev_error) / dt
        out = Kp * error + Ki * integral_new + Kd * derivative
        return out, integral_new, error

    # ---------------- camera & control threads ----------------
    def camera_thread(self):
        # Use DirectShow on Windows for stability
        cap = cv2.VideoCapture(self.cam_index, cv2.CAP_DSHOW) if os.name == 'nt' else cv2.VideoCapture(self.cam_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.FRAME_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.FRAME_H)
        cap.set(cv2.CAP_PROP_FPS, 60)  # hint
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # latest frame only (if backend supports)
        except Exception:
            pass

        if not cap.isOpened():
            logging.error("[CAM] Could not open camera")
            self.stop_request.set()
            self.running.clear()
            return

        logging.info("[CAM] Camera opened")
        preview_name = "SP Ball Tracking"

        while self.running.is_set() and not self.stop_request.is_set():
            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.005)
                continue

            if frame.shape[1] != self.FRAME_W or frame.shape[0] != self.FRAME_H:
                frame = cv2.resize(frame, (self.FRAME_W, self.FRAME_H), interpolation=cv2.INTER_AREA)

            found, center = self.detect_ball_center(frame)
            if found:
                try:
                    if self.position_queue.full():
                        _ = self.position_queue.get_nowait()
                    self.position_queue.put_nowait(center)
                except queue.Full:
                    pass

            if self.preview:
                if found:
                    x, y = int(center[0]), int(center[1])
                    cv2.circle(frame, (x, y), 6, (0, 0, 255), -1)
                    ox, oy = int(self.origin_px[0]), int(self.origin_px[1])
                    cv2.circle(frame, (ox, oy), 4, (255, 0, 0), -1)
                    cv2.line(frame, (ox, oy), (x, y), (255, 0, 0), 1)
                cv2.imshow(preview_name, frame)
                if (cv2.waitKey(1) & 0xFF) == 27:  # ESC to quit
                    self.stop_request.set()
                    break

            self.last_cam_time = time.monotonic()

        cap.release()
        if self.preview:
            try:
                cv2.destroyWindow(preview_name)
            except Exception:
                cv2.destroyAllWindows()

    def control_thread(self):
        arduino_ok = self.connect_arduino()
        clock = time.monotonic
        last_loop = clock()
        start_time = last_loop

        px2m = self.pixel_to_meter
        tilt_gain = self.tilt_gain
        neutral = self.neutral_angles
        platform_h = self.platform_h

        while self.running.is_set() and not self.stop_request.is_set():
            now = clock()
            dt = now - last_loop
            if dt < self.min_dt:
                time.sleep(self.min_dt - dt)
                now = clock()
                dt = now - last_loop
            last_loop = now

            # Camera watchdog
            if now - self.last_cam_time > self.cam_stall_s:
                logging.error("[CAM] Stall detected; stopping control")
                self.stop_request.set()
                break

            # Get latest center; if none, skip this cycle
            try:
                center = self.position_queue.get_nowait()
            except queue.Empty:
                continue

            # Error (pixels), convert to meters
            err_px = self.setpoint_px - center
            err_x_m = err_px[0] * px2m
            err_y_m = err_px[1] * px2m

            # PID per axis
            out_x, self.integral_x, self.prev_error_x = self.update_pid(
                err_x_m, self.prev_error_x, self.integral_x,
                self.Kp_x, self.Ki_x, self.Kd_x, dt
            )
            out_y, self.integral_y, self.prev_error_y = self.update_pid(
                err_y_m, self.prev_error_y, self.integral_y,
                self.Kp_y, self.Ki_y, self.Kd_y, dt
            )

            # Interpret PID outputs as small tilts
            pitch_rad = out_y * tilt_gain  # about Y
            roll_rad  = out_x * tilt_gain  # about X

            # Normal vector (small rotations applied to [0,0,1])
            cp = math.cos(pitch_rad); sp = math.sin(pitch_rad)
            cr = math.cos(roll_rad);  sr = math.sin(roll_rad)
            nrm = np.array([sp, -cp*sr, cp*cr], dtype=np.float64)

            S = np.array([0.0, 0.0, platform_h], dtype=np.float64)

            angles = None
            if triangle_orientation_and_location is not None:
                try:
                    res = triangle_orientation_and_location(nrm, S, initial_guess=5.0)
                    t11 = float(res.get('theta_11', 0.0))
                    t21 = float(res.get('theta_21', 0.0))
                    t31 = float(res.get('theta_31', 0.0))
                    # Use neutral +/- theta (your original intent)
                    angles = [neutral[0] - t11,
                              neutral[1] - t21,
                              neutral[2] - t31]
                except Exception as e:
                    logging.debug(f"[KIN] {e}")

            # Fallback linear mapping
            if angles is None:
                pitch_deg = math.degrees(pitch_rad) * self.mapping_scale / 100.0
                roll_deg  = math.degrees(roll_rad)  * self.mapping_scale / 100.0
                d1 =  pitch_deg
                d2 = -0.5 * pitch_deg + 0.86602540378 * roll_deg
                d3 = -0.5 * pitch_deg - 0.86602540378 * roll_deg
                angles = [neutral[0] + d1, neutral[1] + d2, neutral[2] + d3]

            # Non-blocking send
            self._send_angles_nonblocking(angles)

            # Thin logging (bounded deques)
            t = now - start_time
            self.time_log.append(t)
            self.pos_x_log.append(center[0])
            self.pos_y_log.append(center[1])
            self.setpoint_x_log.append(self.setpoint_px[0])
            self.setpoint_y_log.append(self.setpoint_px[1])
            self.servo_log.append([angles[0], angles[1], angles[2]])

            # Backoff if writer is dropping a lot (lets Arduino catch up)
            if self.serial_writer and self.serial_writer.drop_count > 20:
                time.sleep(0.01)
                self.serial_writer.drop_count = 0

        # Cleanup on exit: park to neutral & close serial
        if self.serial_writer:
            try:
                n0, n1, n2 = self.neutral_angles
                self.serial_writer.send(int(n0), int(n1), int(n2))
                time.sleep(0.02)
                self.serial_writer.stop()
            except Exception:
                pass
        if self.arduino:
            try:
                self.arduino.close()
            except Exception:
                pass

    # ---------------- GUI and plotting ----------------
    def create_gui(self):
        self.root = tk.Tk()
        self.root.title("Stewart Platform PID (Optimized)")
        self.root.geometry("640x620")

        def slider(label, var, frm, to):
            ttk.Label(self.root, text=label).pack()
            ttk.Scale(self.root, from_=frm, to=to, variable=var,
                      orient=tk.HORIZONTAL, length=550).pack()

        self.kp_x_var = tk.DoubleVar(value=self.Kp_x)
        self.ki_x_var = tk.DoubleVar(value=self.Ki_x)
        self.kd_x_var = tk.DoubleVar(value=self.Kd_x)
        self.kp_y_var = tk.DoubleVar(value=self.Kp_y)
        self.ki_y_var = tk.DoubleVar(value=self.Ki_y)
        self.kd_y_var = tk.DoubleVar(value=self.Kd_y)
        self.mapping_var = tk.DoubleVar(value=self.mapping_scale)

        slider("Kp X", self.kp_x_var, 0, 100)
        slider("Ki X", self.ki_x_var, 0, 50)
        slider("Kd X", self.kd_x_var, 0, 50)
        ttk.Separator(self.root, orient=tk.HORIZONTAL).pack(fill='x', pady=6)
        slider("Kp Y", self.kp_y_var, 0, 100)
        slider("Ki Y", self.ki_y_var, 0, 50)
        slider("Kd Y", self.kd_y_var, 0, 50)
        ttk.Separator(self.root, orient=tk.HORIZONTAL).pack(fill='x', pady=6)
        slider("Mapping scale (deg per meter)", self.mapping_var, 50, 2000)

        btns = ttk.Frame(self.root); btns.pack(pady=8)
        ttk.Button(btns, text="Reset Integrals", command=self.reset_integrals).pack(side=tk.LEFT, padx=6)
        ttk.Button(btns, text="Plot Results", command=self.plot_results).pack(side=tk.LEFT, padx=6)
        ttk.Button(btns, text="Stop", command=self.stop).pack(side=tk.LEFT, padx=6)

        # Periodic gain update
        def gui_update():
            if not self.running.is_set():
                return
            self.Kp_x = self.kp_x_var.get(); self.Ki_x = self.ki_x_var.get(); self.Kd_x = self.kd_x_var.get()
            self.Kp_y = self.kp_y_var.get(); self.Ki_y = self.ki_y_var.get(); self.Kd_y = self.kd_y_var.get()
            self.mapping_scale = self.mapping_var.get()
            self.root.after(150, gui_update)
        self.root.after(150, gui_update)

        self.root.protocol("WM_DELETE_WINDOW", self.stop)

    def reset_integrals(self):
        self.integral_x = 0.0
        self.integral_y = 0.0
        logging.info("[RESET] Integrals cleared")

    def plot_results(self):
        if not self.time_log:
            logging.info("[PLOT] no data to plot")
            return
        t = np.array(self.time_log)
        x = np.array(self.pos_x_log); xr = np.array(self.setpoint_x_log)
        y = np.array(self.pos_y_log); yr = np.array(self.setpoint_y_log)
        s = np.array(list(self.servo_log))
        fig, axs = plt.subplots(3, 1, figsize=(8, 10))
        axs[0].plot(t, x, label='ball_x_px'); axs[0].plot(t, xr, '--', label='setpoint_x')
        axs[0].legend(); axs[0].grid(True)
        axs[1].plot(t, y, label='ball_y_px'); axs[1].plot(t, yr, '--', label='setpoint_y')
        axs[1].legend(); axs[1].grid(True)
        if s.size:
            axs[2].plot(t, s[:, 0], label='s1')
            axs[2].plot(t, s[:, 1], label='s2')
            axs[2].plot(t, s[:, 2], label='s3')
            axs[2].legend(); axs[2].grid(True)
        plt.tight_layout()
        plt.show()

    def stop(self):
        self.stop_request.set()
        self.running.clear()
        try:
            if self.root:
                self.root.quit()
                self.root.destroy()
        except Exception:
            pass

    def run(self):
        logging.info("[INFO] Starting Stewart Platform PID controller")
        self.running.set()
        self.stop_request.clear()
        self.last_cam_time = time.monotonic()

        cam_thread = Thread(target=self.camera_thread, name="camera", daemon=True)
        ctrl_thread = Thread(target=self.control_thread, name="control", daemon=True)
        cam_thread.start()
        ctrl_thread.start()

        # GUI runs in main thread
        self.create_gui()
        self.root.mainloop()

        # Cleanup
        self.stop_request.set()
        self.running.clear()
        cam_thread.join(timeout=1.0)
        ctrl_thread.join(timeout=1.0)
        logging.info("[INFO] Controller stopped")


if __name__ == '__main__':
    try:
        controller = StewartPIDController()
        controller.run()
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
    except Exception as e:
        print(f"[ERROR] {e}")
