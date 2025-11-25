import serial
import time

# -----------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------

PORT = "COM5"        # <-- change to your ESP32 port
BAUD = 115200          # must match your ESP32 Serial.begin()
DELAY_AFTER_OPEN = 2 # seconds

# -----------------------------------------------------
# MAIN
# -----------------------------------------------------

def main():
    print("Opening serial connection...")

    try:
        ser = serial.Serial(PORT, BAUD, timeout=1)
    except:
        print(f"Failed to open {PORT}")
        return

    time.sleep(DELAY_AFTER_OPEN)
    print("Connected!")
    print("Type an angle (35–95). Type 'q' to quit.")

    while True:
        cmd = input("Angle > ")

        if cmd.lower() == "q":
            print("Exiting.")
            break

        # Try to convert to a valid angle
        try:
            angle = int(cmd)
        except:
            print("Please enter a valid integer.")
            continue

        # Clamp for safety
        angle = max(0, min(360, angle))

        # Send as text + newline (modify if your ESP expects raw bytes)
        ser.write(f"{angle}\n".encode())

        print(f"Sent angle: {angle}")

    ser.close()


if __name__ == "__main__":
    main()
