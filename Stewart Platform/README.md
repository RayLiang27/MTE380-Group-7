# Stewart Platform Calibration Guide

## Overview

The calibration is split into three main steps:
1. **Servo Leveling** – Ensure the platform is physically level.
2. **Color Calibration** – Detect the ping-pong ball color for tracking.
3. **Geometry Calibration** – Mark the 3 servo pivot points for coordinate mapping.

All calibration data is saved to `Stewart Platform/config_sp.json`.

---

## Hardware Setup

- **Servos:** 3× MG995 connected to Adafruit 16-Channel PWM/Servo Shield (PCA9685)
- **Microcontroller:** Arduino (e.g., Uno or Mega)
- **Power:** Dedicated 5–6V supply for servos, with a common ground to Arduino.
- **Camera:** USB webcam for ball detection.

---

## Files

| File | Purpose |
|------|----------|
| `simple_cal_sp.py` | Python calibration tool for servo, color, and geometry setup |
| `servo_controller.ino` | Arduino sketch to control servos via PCA9685 |
| `config_sp.json` | Generated configuration file storing calibration data |

---

## Calibration Steps

### 1. Upload Arduino Sketch
1. Open `servo_controller.ino` in Arduino IDE.
2. Upload to the connected Arduino.
3. Power the servo supply (platform should move to 60° neutral position).

### 2. Run Python Calibration
```bash
python3 "Stewart Platform/simple_cal_sp.py"
```

Follow on-screen prompts:

#### Step 1 — Servo Leveling
- The servos move to 60°.
- Adjust linkage until the platform is level.
- Press **Space** to continue.

#### Step 2 — Color Calibration
- Click the ball several times to sample its color.
- Press **Space** when satisfied.

#### Step 3 — Geometry Calibration
- Click the three servo pivot points.
- A triangle and red center point will appear.
- Press **Esc** to exit.

---

## Output

A file `config_sp.json` is generated containing:
```json
{
  "servo_calibration": {
    "neutral_angles_deg": [60, 60, 60]
  },
  "camera": {...},
  "calibration": {...}
}
```

---

## Notes
- The Python script only works in **angles** (degrees). The Arduino converts to PWM.
- Ensure all servos and camera are powered before calibration.
- If communication fails, check the `self.arduino_port` in the Python script.

