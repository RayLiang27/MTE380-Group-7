import serial
import time

# -----------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------

PORT = "COM3"      # <- change to your ESP32 port
BAUD = 115200       # must match Serial.begin()
STARTUP_DELAY = 2   # seconds

# -----------------------------------------------------
# MAIN
# -----------------------------------------------------

def main():
    print(f"Connecting to {PORT} ...")

    try:
        ser = serial.Serial(PORT, BAUD, timeout=1)
    except:
        print("ERROR: Could not open serial port")
        return

    time.sleep(STARTUP_DELAY)
    print("Connected. Type an angle between 35 and 95. 'q' to quit.")

    while True:
        user_input = input("Angle > ")

        if user_input.lower() == "q":
            print("Exiting.")
            break

        # Validate integer
        try:
            angle = int(user_input)
        except:
            print("Please enter a number.")
            continue

        # Clamp for safety (matches your sketch)
        MIN_ANGLE = 0
        MAX_ANGLE = 360
        if angle < MIN_ANGLE or angle > MAX_ANGLE:
            print("Angle out of range (35–95).")
            continue

        # Send **one byte** because your ESP32 uses Serial.read()
        ser.write(bytes([angle]))

        print(f"Sent angle byte: {angle}")

    ser.close()

if __name__ == "__main__":
    main()
