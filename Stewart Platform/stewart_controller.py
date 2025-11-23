import cv2
import numpy as np
import json
import serial
import time
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from threading import Thread
import queue
import os
from collections import deque

try:
    cv2.setNumThreads(1)
except Exception:
    pass

# Import the lightweight kinematics helper we added (safe — no plotting)
try:
    from spv4_kinematics import triangle_orientation_and_location, inverse_kinematics
    from SPV4_linkagesim import inverse_kinematics_from_orientation
except Exception:
    triangle_orientation_and_location = None
    inverse_kinematics = None


class StewartPIDController:
    """Basic PID controller for Stewart Platform (3 servos).

    Controls platform to move a tracked ball to the platform center using
    two PID loops (X & Y). Maps the two-axis control outputs to three
    servo angles using an equilateral 3-actuator mapping.

    Assumptions/Notes:
    - The calibration file `Stewart Platform/config_sp.json` contains
      `calibration.origin_px` and `calibration.pixel_to_meter_ratio` and
      `servo_calibration.neutral_angles_deg` saved by `simple_cal_sp.py`.
    - Arduino expects triple-angle commands in the form "a1,a2,a3\n"
      (this matches `simple_cal_sp.send_servo_angles`).
    - This is a basic geometric mapping (small-angle, linear). For
      accurate kinematics one should convert desired normal/orientation
      to leg lengths and then to servo angles using `SPV4.inverse_kinematics`.
    """

    def __init__(self, config_path="Stewart Platform/config_sp.json"):
        # Load config
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config not found: {config_path}")
        with open(config_path, "r") as f:
            self.config = json.load(f)

        self.lower_hsv = None
        self.upper_hsv = None
        if self.config.get("calibration"):
            lower = self.config['calibration'].get('lower_hsv')
            upper = self.config['calibration'].get('upper_hsv')
            if lower is not None and upper is not None:
                self.lower_hsv = np.array(lower, dtype=np.uint8)
                self.upper_hsv = np.array(upper, dtype=np.uint8)

        self.pixel_to_meter = self.config['calibration'].get('pixel_to_meter_ratio', 1.0)
        self.origin_px = np.array(self.config['calibration'].get('origin_px', [0, 0]), dtype=np.float32)

        # Servo information
        servos = self.config.get('servo_calibration', {}).get('neutral_angles_deg')
        if servos and len(servos) >= 3:
            self.neutral_angles = [int(s) for s in servos[:3]]
            self.neutral_angles = [39,50,50]
        else:
            # sensible reasonable default
            self.neutral_angles = [50, 50, 50]
            self.neutral_angles = [39,50,50]

        # self.arduino_port = self.config.get('arduino_port', "/dev/cu.usbmodem1301")
        self.arduino_port = self.config.get('arduino_port', "COM5")
        self.baud_rate = int(self.config.get('baud_rate', 115200))
        self.arduino = None

        # PID defaults (two independent controllers)
        self.Kp_x = 2.5
        self.Ki_x = 0.2
        self.Kd_x = 1.8
        self.Kp_y = 2.5
        self.Ki_y = 0.2
        self.Kd_y = 1.8

        # Internal PID state
        self.integral_x = 0.0
        self.integral_y = 0.0
        self.derivative_x = 0.0
        self.derivative_y = 0.0
        self.prev_error_x = 0.0
        self.prev_error_y = 0.0

        self.integral_deadband_m = float(self.config.get("integral_deadband_m", 0.007))
        self.integral_slowband_m = float(self.config.get("integral_deadband_m", 0.005))

        # Mapping scale: how many degrees of servo per meter of ball displacement
        self.mapping_scale = float(self.config.get('mapping_scale_deg_per_m', 600.0))

        # Logs
        self.experiment_tag = self.config.get('experiment_tag', 'servo')

        self.time_log = []
        self.pos_x_log = []
        self.pos_y_log = []
        self.setpoint_x_log = []
        self.setpoint_y_log = []
        self.servo_log = []

        self.max_points = 1000

        # PID diagnostic logs using rolling buffers
        self.time_log_window = deque(maxlen=self.max_points)
        self.err_x_log = deque(maxlen=self.max_points)
        self.err_y_log = deque(maxlen=self.max_points)
        self.int_x_log = deque(maxlen=self.max_points)
        self.int_y_log = deque(maxlen=self.max_points)
        self.der_x_log = deque(maxlen=self.max_points)
        self.der_y_log = deque(maxlen=self.max_points)

        # runtime flags and queue
        self.running = False
        self.frame = None
        self.FRAME_W = int(self.config.get('camera', {}).get('frame_width', 640))
        self.FRAME_H = int(self.config.get('camera', {}).get('frame_height', 480))
        self.cam_index = int(self.config.get('camera', {}).get('index', 0))
        self.position_queue = queue.Queue(maxsize=1)

        # setpoint: keep ball at the calibrated origin => error (0,0)
        self.setpoint_px = self.origin_px.copy()
        # Test mode parameters
        self.testing = False
        self.test_pre_delay_s = float(self.config.get('test_pre_delay_s', 1.0))
        self.test_duration_s = float(self.config.get('test_duration_s', 15.0))
        self.test_disp_m = float(self.config.get('test_disp_m', 0.01))
        self._test_thread = None
        
    def set_setpoint(self, x_px, y_px):
        """Set desired ball position in pixels."""
        if x_px < 0 or y_px < 0 or x_px >= self.FRAME_W or y_px >= self.FRAME_H:
            raise ValueError("Setpoint out of frame bounds")
        
        self.setpoint_px = np.array([x_px, y_px], dtype=np.float32)

    # ---------------- hardware -----------------
    def connect_arduino(self):
        try:
            self.arduino = serial.Serial(self.arduino_port, self.baud_rate, timeout=2)
            time.sleep(2)
            print(f"[SERVO] Connected to Arduino on {self.arduino_port}")
            return True
        except Exception as e:
            print(f"[SERVO] Could not open serial: {e}")
            self.arduino = None
            return False

    def send_servo_angles(self, angles):
        """Send angles as integers [a1,a2,a3] to Arduino.

        If Arduino not connected, prints message (simulation mode).
        """
        # clip and convert
        safe = [float(np.clip(a, 0, 70)) for a in angles] # TODO: find actual degree limits
        if self.arduino:
            cmd = f"{safe[0]},{safe[1]},{safe[2]}\n"
            try:
                self.arduino.write(cmd.encode())
            except Exception as e:
                print(f"[SERVO] Write error: {e}")
        else:
            print(f"[SERVO-SIM] -> {safe}")

    # ---------------- vision -----------------
    def detect_ball_center(self, frame):
        """Detect ball using HSV thresholds from calibration.

        Returns (found, center_px (x,y), vis_frame)
        center_px is returned as numpy array float (x,y) or None.
        """
        vis = frame.copy()
        if self.lower_hsv is None or self.upper_hsv is None:
            # can't detect without HSV; return center None
            return False, None, vis

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_hsv, self.upper_hsv)
        mask = cv2.medianBlur(mask, 5)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return False, None, vis

        largest = max(contours, key=cv2.contourArea)
        (x, y), radius = cv2.minEnclosingCircle(largest)
        if radius < 12:
            return False, None, vis

        center = np.array([x, y], dtype=np.float32)
        # draw
        cv2.circle(vis, (int(x), int(y)), int(radius), (0, 255, 255), 2)
        cv2.circle(vis, (int(x), int(y)), 3, (0, 0, 255), -1)
        # draw origin
        ox, oy = int(self.setpoint_px[0]), int(self.setpoint_px[1])
        cv2.circle(vis, (ox, oy), 4, (255, 0, 0), -1)
        cv2.line(vis, (ox, oy), (int(x), int(y)), (255, 0, 0), 1)
        return True, center, vis

    # ---------------- PID math ----------------

    def update_pid(self, error, prev_error, integral, Kp, Ki, Kd, dt):
        """Generic PID update returning (output, new_integral, new_prev_error, derivative).

        Integral logic:
        - If |error| > deadband: integrate error normally (with clamp).
        - If |error| <= deadband: bleed the integral toward zero (leaky integrator).
        """
        if dt <= 0:
            dt = 1e-6

        # --- Integral update ---
        if abs(error) > self.integral_deadband_m:
            # Normal integration when away from the setpoint
            integral_new = integral + error * dt
        else:
            # Inside deadband: exponential decay of existing integral
            tau = 1.67
            decay = np.exp(-dt / tau)
            integral_new = integral * decay

        # Clamp integral to avoid windup
        integral_new = float(np.clip(integral_new,
                                         -0.6,
                                         0.6))

        # --- Derivative & output ---
        derivative = (error - prev_error) / dt
        out = Kp * error + Ki * integral_new + Kd * derivative
        print(f"[PID] error: {error:.6f}, integral: {integral_new:.6f}, derivative: {derivative:.6f}, output: {out:.6f}")

        return out, integral_new, error, derivative
    
    # def update_pid(self, error, prev_error, integral, Kp, Ki, Kd, dt):
    #     """Generic PID update returning (output, new_integral, new_prev_error)."""
    #     # if abs(error) > self.integral_slowband_m:
    #     integral_new = integral + error * dt
    #     # if abs(error) < self.integral_deadband_m:
    #     #     # Inside the deadband: zero out the integral so it doesn't keep pushing
    #     #     integral_new = integral*0.97
    #     # else:
    #     #     integral_new = 0.0
    #     derivative = 0.0 if dt <= 0 else (error - prev_error) / dt

    #     if abs(derivative) < 1e-5 and abs(error) < self.integral_deadband_m:
    #         clipped_integral = 0.0
    #     # if error < self.integral_deadband_m and error > -self.integral_deadband_m:
    #     #     out = 0.0
    #     else:
    #         clipped_integral = np.clip(integral_new, -0.6, 0.6)
    #     out = Kp * error + Ki * clipped_integral + Kd * derivative
    #     print(f"[PID] error: {error:.6f}, integral: {clipped_integral:.6f}, derivative: {derivative:.6f}, output: {out:.6f}")
    #     return out, clipped_integral, error

    # ---------------- camera & control threads ----------------
    def camera_thread(self):
        # cap = cv2.VideoCapture(self.cam_index)
        # cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.FRAME_W)
        # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.FRAME_H)

        cap = cv2.VideoCapture(self.cam_index, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.FRAME_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.FRAME_H)
        cap.set(cv2.CAP_PROP_FPS, 60)  # hint only
        cv2.namedWindow("SP Ball Tracking", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("SP Ball Tracking", self._on_mouse_click)
        
        if not cap.isOpened():
            print("[CAM] Could not open camera")
            self.running = False
            return

        print("[CAM] Camera opened")

        while self.running:
            ret, frame = cap.read()
            if not ret:
                continue
            frame = cv2.resize(frame, (self.FRAME_W, self.FRAME_H))
            found, center, vis = self.detect_ball_center(frame)
            if found and center is not None:
                # put latest center into queue
                try:
                    if self.position_queue.full():
                        _ = self.position_queue.get_nowait()
                    self.position_queue.put_nowait(center)
                except Exception:
                    pass

            cv2.imshow("SP Ball Tracking", vis)
            if cv2.waitKey(1) & 0xFF == 27:
                self.running = False
                break

        cap.release()
        cv2.destroyAllWindows()

    def _on_mouse_click(self, event, x, y, flags, param):
        """Left-click on video to set setpoint."""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.setpoint_px = np.array([float(x), float(y)], dtype=np.float32)
            print(f"[CAM] Setpoint updated to: ({int(x)}, {int(y)})")

    def reset_setpoint_to_center(self):
        """Reset setpoint back to platform center."""
        self.setpoint_px = self.origin_px.copy()
        print(f"[CTRL] Setpoint reset to center: ({int(self.origin_px[0])}, {int(self.origin_px[1])})")

    def control_thread(self):
        if not self.connect_arduino():
            print("[INFO] Running in simulation mode (no Arduino)")
        last_time = time.time()
        self.start_time = last_time


        ## TODO Interchange
        while self.running:
        # while self.running and not self.position_queue.empty():
            try:
                center = self.position_queue.get(timeout=0.1)
                now = time.time()
                dt = max(1e-6, now - last_time)
                last_time = now

                # compute error (pixel): positive x right, positive y down
                error_px = self.setpoint_px - center  # want center -> setpoint
                # convert to meters
                error_m = error_px * self.pixel_to_meter
                error_x_m = error_m[0]
                error_y_m = error_m[1]

                # PID X (horizontal) and Y (vertical)
                out_x, self.integral_x, self.prev_error_x, self.derivative_x= self.update_pid(
                    error_x_m, self.prev_error_x, self.integral_x,
                    self.Kp_x, self.Ki_x, self.Kd_x, dt)

                out_y, self.integral_y, self.prev_error_y, self.derivative_y = self.update_pid(
                    error_y_m, self.prev_error_y, self.integral_y,
                    self.Kp_y, self.Ki_y, self.Kd_y, dt)

                # Use platform kinematics (if available) to compute servo angles.
                # Interpret PID outputs as desired small tilt angles (radians) by
                # applying a tilt_gain (radians per meter of ball displacement).
                # tilt_gain = float(self.config.get('tilt_gain_rad_per_m', 0.5))
                # desired pitch (rotation about Y) and roll (rotation about X)
                # pitch_rad = out_y * tilt_gain #TODO: check
                # roll_rad = out_x * tilt_gain
                z_scale_factor = np.float64(1.0)  
                out_z = 1.0
                # out_z = z_scale_factor / np.sqrt(out_x**2 + out_y**2)
                # normal_mag = np.sqrt(out_x**2 + out_y**2 + out_z**2)
                # out_x_norm = out_x / normal_mag
                # out_y_norm = -out_y / normal_mag
                # out_z_norm = out_z / normal_mag
                normal = np.array([out_x, -out_y, out_z])
                nrm = normal / np.linalg.norm(normal)
                print(f"[KIN] desired normal (pre-norm): {normal}")

                # Construct normal vector by applying R_x(roll) * R_y(pitch) to [0,0,1]
                # cp = np.cos(pitch_rad)
                # sp = np.sin(pitch_rad)
                # cr = np.cos(roll_rad)
                # sr = np.sin(roll_rad)
                # from derivation: n = [sin(pitch), -cos(pitch)*sin(roll), cos(pitch)*cos(roll)]
                # nrm = np.array([-sp * cr, sr, cp * cr])
                # nrm = normal
                # print(f"[KIN] nrm: {nrm}, pitch: {np.degrees(pitch_rad):.2f} deg, roll: {np.degrees(roll_rad):.2f} deg")

                # platform center height (meters) — default to 12 if not in config
                platform_h = float(self.config.get('platform_height_m', 10.0))
                S = np.array([0.0, 0.0, platform_h])
                res = inverse_kinematics_from_orientation(nrm, S, elbow_up=True, verbose=False)
                legs = res['legs']
                # print(legs)
                if None in legs:
                    legs = self.neutral_angles

                t11 = legs[0].get('theta2_deg', 0.0) if legs[0] else self.neutral_angles[0]
                t21 = legs[1].get('theta2_deg', 0.0) if legs[1] else self.neutral_angles[1]
                t31 = legs[2].get('theta2_deg', 0.0) if legs[2] else self.neutral_angles[2]
                angles = [self.neutral_angles[0]-t11, self.neutral_angles[1]-t21, self.neutral_angles[2]-t31]

                # angles = None
                # if triangle_orientation_and_location is not None:
                #     try:
                #         res = triangle_orientation_and_location(nrm, S, initial_guess=5.0)
                #         print(f"[KIN] Platform points: P1={res['P1']}, P2={res['P2']}, P3={res['P3']}")
                #         # triangle_orientation_and_location returns theta_11, theta_21, theta_31 (degrees)
                #         t11 = float(res.get('theta_11', 0.0))
                #         t21 = float(res.get('theta_21', 0.0))
                #         t31 = float(res.get('theta_31', 0.0))
                #         # print(res)
                #         # Compose servo commands as neutral + theta values (may need offset/tuning)
                #         angles = [t11 - 10,
                #                   t21 - 10,
                #                   t31 - 10]
                        
                #         """
                #         self.neutral_angles[0] - 
                #         self.neutral_angles[1] - 
                #         self.neutral_angles[2] - """

                #     except Exception as e:
                #         print(f"[KIN] kinematics error: {e}")

                # # Fallback to linear mapping if kinematics failed or missing
                # if angles is None:
                #     pitch_deg = np.degrees(pitch_rad) * self.mapping_scale / 100.0
                #     roll_deg = np.degrees(roll_rad) * self.mapping_scale / 100.0
                #     d1 = pitch_deg
                #     d2 = -0.5 * pitch_deg + 0.86602540378 * roll_deg
                #     d3 = -0.5 * pitch_deg - 0.86602540378 * roll_deg
                #     angles = [self.neutral_angles[0] + d1,
                #               self.neutral_angles[1] + d2,
                #               self.neutral_angles[2] + d3]

                t = time.time() - self.start_time

                print(f"t={t:.2f}s")

                # send to servos
                self.send_servo_angles(angles)

                # logging     bad - a human
                t = time.time() - self.start_time
                self.time_log.append(t)
                self.pos_x_log.append(float(center[0]))
                self.pos_y_log.append(float(center[1]))
                self.setpoint_x_log.append(float(self.setpoint_px[0]))
                self.setpoint_y_log.append(float(self.setpoint_px[1]))
                # store a copy so we don't accidentally mutate later
                self.servo_log.append([float(angles[0]),
                                       float(angles[1]),
                                       float(angles[2])])

                self.time_log_window.append(t)
                self.err_x_log.append(error_x_m)
                self.err_y_log.append(error_y_m)
                self.int_x_log.append(self.integral_x)
                self.int_y_log.append(self.integral_y)
                self.der_x_log.append(self.derivative_x)
                self.der_y_log.append(self.derivative_y)

                # self.time_log.append(t)
                # self.pos_x_log.append(center[0])
                # self.pos_y_log.append(center[1])
                # self.setpoint_x_log.append(self.setpoint_px[0])
                # self.setpoint_y_log.append(self.setpoint_px[1])
                # self.servo_log.append(angles)

                # print(f"qsize: {self.position_queue.qsize()}")
                print(f"t={t:.2f}s center={center.astype(int)} err_px={error_px.astype(int)} PID_out=[{out_x} {out_y}] servo={list(map(int,angles))}")

            except queue.Empty:
                continue
            # except Exception as e:
            #     print(f"[CONTROL] error: {e}")
            #     break

        # cleanup on exit
        if self.arduino:
            self.send_servo_angles(self.neutral_angles)
            try:
                self.arduino.close()
            except Exception:
                pass
    # ---------------- live PID plotting ----------------
    def init_live_plot(self):
        """Initialize live matplotlib figure for error / integral / derivative."""
        plt.ion()
        self.fig_pid, self.ax_pid = plt.subplots(3, 1, figsize=(7, 8), sharex=True)
        self.fig_pid.suptitle("PID Diagnostics (X & Y)")

        # Error subplot
        self.err_x_line, = self.ax_pid[0].plot([], [], label="err_x [m]")
        self.err_y_line, = self.ax_pid[0].plot([], [], label="err_y [m]")
        self.ax_pid[0].set_ylabel("Error")
        self.ax_pid[0].legend()
        self.ax_pid[0].grid(True)

        # Integral subplot
        self.int_x_line, = self.ax_pid[1].plot([], [], label="int_x")
        self.int_y_line, = self.ax_pid[1].plot([], [], label="int_y")
        self.ax_pid[1].set_ylabel("Integral")
        self.ax_pid[1].legend()
        self.ax_pid[1].grid(True)

        # Derivative subplot
        self.der_x_line, = self.ax_pid[2].plot([], [], label="der_x [m/s]")
        self.der_y_line, = self.ax_pid[2].plot([], [], label="der_y [m/s]")
        self.ax_pid[2].set_ylabel("Derivative")
        self.ax_pid[2].set_xlabel("Time [s]")
        self.ax_pid[2].legend()
        self.ax_pid[2].grid(True)

        self.fig_pid.tight_layout()
        self.fig_pid.canvas.draw()
        self.fig_pid.show()

    def update_live_plot(self):
        """Update live PID plots with latest logs."""
        if not self.time_log_window:
            return

        t  = list(self.time_log_window)

        err_x = list(self.err_x_log)
        err_y = list(self.err_y_log)

        int_x = list(self.int_x_log)
        int_y = list(self.int_y_log)

        der_x = list(self.der_x_log)
        der_y = list(self.der_y_log)

        # Update line data
        self.err_x_line.set_data(t, err_x)
        self.err_y_line.set_data(t, err_y)

        self.int_x_line.set_data(t, int_x)
        self.int_y_line.set_data(t, int_y)

        self.der_x_line.set_data(t, der_x)
        self.der_y_line.set_data(t, der_y)

        # Rescale axes
        for ax in self.ax_pid:
            ax.relim()
            ax.autoscale_view()

        self.fig_pid.canvas.draw_idle()
        self.fig_pid.canvas.flush_events()

    def save_csv(self):
        """Save full-run logs to a timestamped CSV for analysis."""
        if not self.time_log:
            print("[SAVE] no data to save")
            return

        import numpy as np

        # Make sure lengths stay consistent
        n = len(self.time_log)
        if not (len(self.pos_x_log) == len(self.pos_y_log) == len(self.setpoint_x_log) ==
                len(self.setpoint_y_log) == len(self.servo_log) == n):
            print("[SAVE] log length mismatch, not saving (something went off).")
            return

        os.makedirs("logs", exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        fname = f"logs/sp_{self.experiment_tag}_{ts}.csv"

        servo_arr = np.array(self.servo_log, dtype=float)
        data = np.column_stack([
            np.array(self.time_log, dtype=float),
            np.array(self.pos_x_log, dtype=float),
            np.array(self.pos_y_log, dtype=float),
            np.array(self.setpoint_x_log, dtype=float),
            np.array(self.setpoint_y_log, dtype=float),
            np.array(self.err_x_log, dtype=float) if len(self.err_x_log) == n else np.full(n, np.nan),
            np.array(self.err_y_log, dtype=float) if len(self.err_y_log) == n else np.full(n, np.nan),
            servo_arr[:, 0],
            servo_arr[:, 1],
            servo_arr[:, 2],
        ])

        header = (
            "t_s,ball_x_px,ball_y_px,setpoint_x_px,setpoint_y_px,"
            "err_x_m_window,err_y_m_window,servo1_deg,servo2_deg,servo3_deg"
        )

        np.savetxt(fname, data, delimiter=",", header=header, comments="")
        print(f"[SAVE] logs written to {fname}")


    def create_gui(self):
        self.root = tk.Tk()
        self.root.title("Stewart Platform PID")
        self.root.geometry("640x720")

        # Function to create a labeled scale with entry
        def create_param_control(parent, label, var, from_, to_, row):
            frame = ttk.Frame(parent)
            frame.grid(row=row, column=0, pady=5, sticky='ew')
            
            # Label
            ttk.Label(frame, text=label, width=15).grid(row=0, column=0)
            
            # Scale
            scale = ttk.Scale(frame, from_=from_, to=to_, variable=var, 
                            orient=tk.HORIZONTAL, length=400)
            scale.grid(row=0, column=1, padx=5)
            
            # Entry for direct value input
            entry = ttk.Entry(frame, width=8, justify=tk.RIGHT)
            entry.grid(row=0, column=2, padx=5)
            
            # Update functions
            def update_entry(event=None):
                entry.delete(0, tk.END)
                entry.insert(0, f"{var.get():.3f}")
                
            def update_scale(event=None):
                try:
                    value = float(entry.get())
                    var.set(value)
                except ValueError:
                    update_entry()
                    
            scale.configure(command=update_entry)
            entry.bind('<Return>', update_scale)
            entry.bind('<FocusOut>', update_scale)
            
            # Initial value
            update_entry()
            
            return frame

        main_frame = ttk.Frame(self.root)
        main_frame.pack(expand=True, fill='both', padx=10, pady=5)

        # X axis controls
        ttk.Label(main_frame, text="X Axis Control", font=('Arial', 10, 'bold')).grid(row=0, column=0, pady=10)
        self.kp_x_var = tk.DoubleVar(value=self.Kp_x)
        self.ki_x_var = tk.DoubleVar(value=self.Ki_x)
        self.kd_x_var = tk.DoubleVar(value=self.Kd_x)
        create_param_control(main_frame, "Kp X", self.kp_x_var, 0, 10, 1)
        create_param_control(main_frame, "Ki X", self.ki_x_var, 0, 10, 2)
        create_param_control(main_frame, "Kd X", self.kd_x_var, 0, 10, 3)

        ttk.Separator(main_frame, orient=tk.HORIZONTAL).grid(row=4, column=0, sticky='ew', pady=10)

        # Y axis controls
        ttk.Label(main_frame, text="Y Axis Control", font=('Arial', 10, 'bold')).grid(row=5, column=0, pady=10)
        self.kp_y_var = tk.DoubleVar(value=self.Kp_y)
        self.ki_y_var = tk.DoubleVar(value=self.Ki_y)
        self.kd_y_var = tk.DoubleVar(value=self.Kd_y)
        create_param_control(main_frame, "Kp Y", self.kp_y_var, 0, 10, 6)
        create_param_control(main_frame, "Ki Y", self.ki_y_var, 0, 10, 7)
        create_param_control(main_frame, "Kd Y", self.kd_y_var, 0, 10, 8)

        ttk.Separator(main_frame, orient=tk.HORIZONTAL).grid(row=9, column=0, sticky='ew', pady=10)

        # Mapping scale control
        ttk.Label(main_frame, text="Platform Control", font=('Arial', 10, 'bold')).grid(row=10, column=0, pady=10)
        self.mapping_var = tk.DoubleVar(value=self.mapping_scale)
        create_param_control(main_frame, "Map Scale", self.mapping_var, 50, 2000, 11)

        # --- Test setpoint controls (choose before pressing Test) ---
        test_frame = ttk.Frame(main_frame)
        test_frame.grid(row=11, column=0, pady=6, sticky='ew')
        ttk.Label(test_frame, text="Test Setpoint", font=('Arial', 10, 'bold')).grid(row=0, column=0, sticky='w')

        # Mode: relative (meters in +Y) or absolute pixels
        self.test_mode_var = tk.StringVar(value='relative')
        rb_rel = ttk.Radiobutton(test_frame, text="Relative +Y (m)", variable=self.test_mode_var, value='relative')
        rb_abs = ttk.Radiobutton(test_frame, text="Absolute (px)", variable=self.test_mode_var, value='absolute')
        rb_rel.grid(row=1, column=0, sticky='w', pady=2)
        rb_abs.grid(row=1, column=1, sticky='w', pady=2)

        # Relative displacement entry (meters)
        self.test_disp_var = tk.DoubleVar(value=self.test_disp_m)
        ttk.Label(test_frame, text="Disp (m):").grid(row=2, column=0, sticky='e')
        ttk.Entry(test_frame, textvariable=self.test_disp_var, width=10, justify='right').grid(row=2, column=1, sticky='w', padx=4)

        # Absolute pixel setpoint entries (x,y)
        self.test_set_x_var = tk.DoubleVar(value=float(self.origin_px[0]))
        self.test_set_y_var = tk.DoubleVar(value=float(self.origin_px[1]))
        ttk.Label(test_frame, text="X px:").grid(row=3, column=0, sticky='e')
        ttk.Entry(test_frame, textvariable=self.test_set_x_var, width=10, justify='right').grid(row=3, column=1, sticky='w', padx=4)
        ttk.Label(test_frame, text="Y px:").grid(row=4, column=0, sticky='e')
        ttk.Entry(test_frame, textvariable=self.test_set_y_var, width=10, justify='right').grid(row=4, column=1, sticky='w', padx=4)

        # Buttons
        btn_frame = ttk.Frame(main_frame)
        btn_frame.grid(row=12, column=0, pady=20)
        
        ttk.Button(btn_frame, text="Reset Integrals", command=self.reset_integrals).pack(side=tk.LEFT, padx=6)
        ttk.Button(btn_frame, text="Plot Results", command=self.plot_results).pack(side=tk.LEFT, padx=6)
        ttk.Button(btn_frame, text="Save CSV", command=self.save_csv).pack(side=tk.LEFT, padx=6)
        ttk.Button(btn_frame, text="Stop", command=self.stop).pack(side=tk.LEFT, padx=6)
        ttk.Button(btn_frame, text="Test", command=self.start_test).pack(side=tk.LEFT, padx=6)
        ttk.Button(btn_frame, text="Reset to Center", command=self.reset_setpoint_to_center).pack(side=tk.LEFT, padx=6)

        # Current Values Display
        self.values_label = ttk.Label(main_frame, text="", justify=tk.LEFT)
        self.values_label.grid(row=13, column=0, pady=10)

        self.root.after(100, self.gui_update)
    # # ---------------- GUI and plotting ----------------
    # def create_gui(self):
    #     self.root = tk.Tk()
    #     self.root.title("Stewart Platform PID")
    #     self.root.geometry("640x620")

    #     # X axis (horizontal) gains
    #     ttk.Label(self.root, text="Kp X").pack()
    #     self.kp_x_var = tk.DoubleVar(value=self.Kp_x)
    #     ttk.Scale(self.root, from_=0, to=100, variable=self.kp_x_var, orient=tk.HORIZONTAL, length=550).pack()
    #     ttk.Label(self.root, text="Ki X").pack()
    #     self.ki_x_var = tk.DoubleVar(value=self.Ki_x)
    #     ttk.Scale(self.root, from_=0, to=50, variable=self.ki_x_var, orient=tk.HORIZONTAL, length=550).pack()
    #     ttk.Label(self.root, text="Kd X").pack()
    #     self.kd_x_var = tk.DoubleVar(value=self.Kd_x)
    #     ttk.Scale(self.root, from_=0, to=50, variable=self.kd_x_var, orient=tk.HORIZONTAL, length=550).pack()

    #     ttk.Separator(self.root, orient=tk.HORIZONTAL).pack(fill='x', pady=6)

    #     # Y axis gains
    #     ttk.Label(self.root, text="Kp Y").pack()
    #     self.kp_y_var = tk.DoubleVar(value=self.Kp_y)
    #     ttk.Scale(self.root, from_=0, to=100, variable=self.kp_y_var, orient=tk.HORIZONTAL, length=550).pack()
    #     ttk.Label(self.root, text="Ki Y").pack()
    #     self.ki_y_var = tk.DoubleVar(value=self.Ki_y)
    #     ttk.Scale(self.root, from_=0, to=50, variable=self.ki_y_var, orient=tk.HORIZONTAL, length=550).pack()
    #     ttk.Label(self.root, text="Kd Y").pack()
    #     self.kd_y_var = tk.DoubleVar(value=self.Kd_y)
    #     ttk.Scale(self.root, from_=0, to=50, variable=self.kd_y_var, orient=tk.HORIZONTAL, length=550).pack()

    #     ttk.Separator(self.root, orient=tk.HORIZONTAL).pack(fill='x', pady=6)

    #     ttk.Label(self.root, text="Mapping scale (deg per meter)").pack()
    #     self.mapping_var = tk.DoubleVar(value=self.mapping_scale)
    #     ttk.Scale(self.root, from_=50, to=2000, variable=self.mapping_var, orient=tk.HORIZONTAL, length=550).pack()

    #     btn_frame = ttk.Frame(self.root)
    #     btn_frame.pack(pady=8)
    #     ttk.Button(btn_frame, text="Reset Integrals", command=self.reset_integrals).pack(side=tk.LEFT, padx=6)
    #     ttk.Button(btn_frame, text="Plot Results", command=self.plot_results).pack(side=tk.LEFT, padx=6)
    #     ttk.Button(btn_frame, text="Stop", command=self.stop).pack(side=tk.LEFT, padx=6)

    #     self.root.after(100, self.gui_update)

    def gui_update(self):
        if self.running:
            self.Kp_x = self.kp_x_var.get()
            self.Ki_x = self.ki_x_var.get()
            self.Kd_x = self.kd_x_var.get()
            self.Kp_y = self.kp_y_var.get()
            self.Ki_y = self.ki_y_var.get()
            self.Kd_y = self.kd_y_var.get()
            self.mapping_scale = self.mapping_var.get()
            self.update_live_plot()
            self.root.after(100, self.gui_update)

    # ---------------- test mode ----------------
    def start_test(self):
        """Start the automated test sequence (non-blocking).

        Behavior:
        - Record for `test_pre_delay_s` seconds at current setpoint (assume balanced center).
        - After pre-delay, command a new setpoint shifted by +Y (down) by `test_disp_m` meters.
        - Record for `test_duration_s` seconds, then save CSV and restore setpoint to origin.
        """
        if self.testing:
            print("[TEST] Test already running")
            return
        # ensure control is running
        if not self.running:
            print("[TEST] Controller not running; start controller before testing")
            return

        # Launch test thread
        self.testing = True
        self._test_thread = Thread(target=self._run_test, daemon=True)
        self._test_thread.start()

    def _run_test(self):
        try:
            print(f"[TEST] Starting test: pre-delay {self.test_pre_delay_s}s, run {self.test_duration_s}s, disp {self.test_disp_m} m")
            # Save original experiment tag and set a test tag
            old_tag = getattr(self, 'experiment_tag', 'servo')
            ts = time.strftime("%Y%m%d_%H%M%S")
            self.experiment_tag = f"test_{ts}"

            # Ensure logs are empty for a clean run
            self.time_log.clear()
            self.pos_x_log.clear()
            self.pos_y_log.clear()
            self.setpoint_x_log.clear()
            self.setpoint_y_log.clear()
            self.servo_log.clear()

            # Also clear rolling PID buffers
            try:
                self.time_log_window.clear()
                self.err_x_log.clear()
                self.err_y_log.clear()
                self.int_x_log.clear()
                self.int_y_log.clear()
                self.der_x_log.clear()
                self.der_y_log.clear()
            except Exception:
                pass

            # Start recording immediately for pre-delay
            test_start = time.time()
            pre_end = test_start + float(self.test_pre_delay_s)
            run_end = pre_end + float(self.test_duration_s)

            # Keep original setpoint
            orig_setpoint = self.setpoint_px.copy()

            # Wait for pre-delay while control loop logs data
            while time.time() < pre_end and self.running:
                time.sleep(0.05)

            # Determine new setpoint according to GUI selections (safe fallback to stored values)
            mode = 'relative'
            try:
                mode = self.test_mode_var.get()
            except Exception:
                mode = 'relative'

            if mode == 'absolute':
                try:
                    tx = float(self.test_set_x_var.get())
                    ty = float(self.test_set_y_var.get())
                except Exception:
                    tx, ty = float(self.origin_px[0]), float(self.origin_px[1])
                # Use set_setpoint to validate/clamp
                try:
                    self.set_setpoint(tx, ty)
                except Exception:
                    # fallback to direct assign
                    self.setpoint_px = np.array([float(np.clip(tx, 0, self.FRAME_W - 1)),
                                                  float(np.clip(ty, 0, self.FRAME_H - 1))], dtype=np.float32)
                print(f"[TEST] Commanded absolute new setpoint (px): {self.setpoint_px}")
            else:
                try:
                    disp_m = float(self.test_disp_var.get())
                except Exception:
                    disp_m = float(self.test_disp_m)

                # Convert meters -> pixels (pixel_to_meter is meters per pixel)
                if self.pixel_to_meter and self.pixel_to_meter != 0:
                    dy_px = disp_m / float(self.pixel_to_meter)
                else:
                    dy_px = 0.0

                new_setpoint = orig_setpoint.copy()
                new_setpoint[1] = float(orig_setpoint[1]) + dy_px
                # Use set_setpoint to validate/clamp
                try:
                    self.set_setpoint(new_setpoint[0], new_setpoint[1])
                except Exception:
                    self.setpoint_px = np.array([float(np.clip(new_setpoint[0], 0, self.FRAME_W - 1)),
                                                  float(np.clip(new_setpoint[1], 0, self.FRAME_H - 1))], dtype=np.float32)
                print(f"[TEST] Commanded relative new setpoint (px): {self.setpoint_px}")

            # Wait while recording run_duration
            while time.time() < run_end and self.running:
                time.sleep(0.05)

            # Assume balanced; save data to CSV
            print("[TEST] Test complete — saving CSV and restoring setpoint")
            self.save_csv()

            # Restore setpoint to platform center (origin_px) and experiment tag
            try:
                self.setpoint_px = self.origin_px.copy()
            except Exception:
                self.setpoint_px = orig_setpoint
            self.experiment_tag = old_tag
            print(f"[TEST] Setpoint restored to origin: {self.setpoint_px}")

        finally:
            self.testing = False
            self._test_thread = None

    def reset_integrals(self):
        self.integral_x = 0.0
        self.integral_y = 0.0
        print("[RESET] Integrals cleared")

    def plot_results(self):
        if not self.time_log:
            print("[PLOT] no data to plot")
            return
        fig, axs = plt.subplots(3, 1, figsize=(8, 10))
        axs[0].plot(self.time_log, self.pos_x_log, label='ball_x_px')
        axs[0].plot(self.time_log, self.setpoint_x_log, '--', label='setpoint_x')
        axs[0].legend(); axs[0].grid(True)
        axs[1].plot(self.time_log, self.pos_y_log, label='ball_y_px')
        axs[1].plot(self.time_log, self.setpoint_y_log, '--', label='setpoint_y')
        axs[1].legend(); axs[1].grid(True)
        # servo angles
        servo_arr = np.array(self.servo_log)
        if servo_arr.size:
            axs[2].plot(self.time_log, servo_arr[:, 0], label='s1')
            axs[2].plot(self.time_log, servo_arr[:, 1], label='s2')
            axs[2].plot(self.time_log, servo_arr[:, 2], label='s3')
            axs[2].legend(); axs[2].grid(True)
        plt.tight_layout()
        plt.show()

    def stop(self):
        self.running = False
        try:
            self.root.quit()
            self.root.destroy()
        except Exception:
            pass

    def run(self):
        print("[\033[92m INFO \033[0m] Starting Stewart Platform PID controller")
        self.running = True

        cam_thread = Thread(target=self.camera_thread, daemon=True)
        ctrl_thread = Thread(target=self.control_thread, daemon=True)
        cam_thread.start()
        ctrl_thread.start()

        self.init_live_plot()

        # GUI runs in main thread
        self.create_gui()
        self.root.mainloop()

        # cleanup
        self.running = False
        print("[INFO] Controller stopped")

    def TEST_input(self):
        le_in = ""

        while le_in != "EXIT":
            le_in = input("Enter as err_x,err_y or EXIT: ")

            if le_in == "EXIT":
                continue

            le_x = float(le_in.split(",")[0])
            le_y = float(le_in.split(",")[1])
            center = np.array([le_x, le_y], dtype=np.float32)

            try:
                if self.position_queue.full():
                    _ = self.position_queue.get_nowait()
                self.position_queue.put_nowait(center)
                self.control_thread()
            except Exception:
                pass

    def TEST_run(self):
        print("[\033[93m TEST \033[0m] Starting PID controller")
        # test_inp_thread = Thread(target=self.TEST_input, daemon=True)
        # ctrl_thread = Thread(target=self.control_thread, daemon=True)
        # test_inp_thread.start()
        # ctrl_thread.start()
        self.running = True
        self.TEST_input()


if __name__ == '__main__':
    try:
        controller = StewartPIDController()
        controller.run()
        # controller.TEST_run()
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
    except Exception as e:
        print(f"[ERROR] {e}")

#     def test_platform_orientation(self):
#         """Test platform by rotating normal vector like SPV4_linkagesim"""
#         if not self.connect_arduino():
#             print("[ERROR] Cannot test - Arduino not connected")
#             return

#         # Similar parameters to SPV4_linkagesim test
#         phi = np.deg2rad(70.0)  # Fixed tilt angle
#         platform_h = float(self.config.get('platform_height_m', 10.0))
#         S = np.array([0.0, 0.0, platform_h])
        
#         try:
#             while self.running:
#                 # Rotate alpha over time like SPV4_linkagesim
#                 t = time.time() - self.start_time
#                 alpha = (t) % (2 * np.pi)  # Complete rotation every 4π seconds
                
#                 # Compute normal vector
#                 nrm = np.array([
#                     np.cos(phi) * np.cos(alpha),
#                     np.cos(phi) * np.sin(alpha),
#                     np.sin(phi)
#                 ])
#                 print("[TEST] Normal vector:", nrm)
#                 # Get servo angles using the same solver as SPV4_linkagesim
#                 res = inverse_kinematics_from_orientation(nrm, S, elbow_up=True)
#                 legs = res['legs']

#                 # Extract angles and apply to servos
#                 if all(leg is not None for leg in legs.values()):
#                     t11 = legs[0].get('theta2_deg', 0.0)
#                     t21 = legs[1].get('theta2_deg', 0.0)
#                     t31 = legs[2].get('theta2_deg', 0.0)
                    
#                     # Convert to servo commands (adjust neutral angles as needed)
#                     angles = [
#                         self.neutral_angles[0] - t11,
#                         self.neutral_angles[1] - t21, 
#                         self.neutral_angles[2] - t31
#                     ]
                    
#                     # Send to servos
#                     self.send_servo_angles(angles)
                    
#                     # Print status
#                     print(f"t={t:.1f}s alpha={np.degrees(alpha):.1f}° angles={[int(a) for a in angles]}")
                
#                 # Sleep to control update rate
#                 time.sleep(0.05)

#         except KeyboardInterrupt:
#             print("\n[INFO] Test stopped by user")
#         finally:
#             # Return to neutral
#             self.send_servo_angles(self.neutral_angles)
#             if self.arduino:
#                 self.arduino.close()

# if __name__ == '__main__':
#     try:
#         controller = StewartPIDController()
        
#         # Add command line argument handling
#         import sys
#         # if len(sys.argv) > 1 and sys.argv[1] == 'test_orientation':
#         print("[INFO] Running orientation test mode")
#         controller.running = True
#         controller.start_time = time.time()
#         controller.test_platform_orientation()
#         # else:
#         #     # Normal PID control mode
#         #     controller.run()
            
#     except FileNotFoundError as e:
#         print(f"[ERROR] {e}")
#     except Exception as e:
#         print(f"[ERROR] {e}")
