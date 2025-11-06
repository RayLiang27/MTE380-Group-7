#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver();

#define SERVO_FREQ 50
#define SERVOMIN 250
#define SERVOMAX 450
const int servoChannels[3] = {2, 5, 7};

int angleToPulse(int angle) {
  return map(angle, 0, 180, SERVOMIN, SERVOMAX); //TODO: find the real angle range of motion.
}

void setup() {
  Serial.begin(9600);
  pwm.begin();
  pwm.setPWMFreq(SERVO_FREQ);
  delay(500);
  Serial.println("=== Stewart Platform Servo Control ===");
}

void loop() {
  if (Serial.available()) {
    String input = Serial.readStringUntil('\n');
    int a1, a2, a3;
    if (sscanf(input.c_str(), "%d,%d,%d", &a1, &a2, &a3) == 3) {
      pwm.setPWM(servoChannels[0], 0, angleToPulse(a1));
      pwm.setPWM(servoChannels[1], 0, angleToPulse(a2));
      pwm.setPWM(servoChannels[2], 0, angleToPulse(a3));
      Serial.print("Set angles: ");
      Serial.println(input);
    }
  }
}
