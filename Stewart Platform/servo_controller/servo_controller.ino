// #include <Wire.h>
// #include <Adafruit_PWMServoDriver.h>

// Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver();

// #define SERVO_FREQ 50
// #define SERVOMIN 310
// #define SERVOMAX 450
// const int servoChannels[3] = {2, 5, 7};

// float fmap(float x, float in_min, float in_max, float out_min, float out_max) {
//   return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min;
// }

// uint16_t angleToPulse(float angle) {
//   // Clamp to expected mechanical range
//   if (angle < 0.0f)  angle = 0.0f;
//   if (angle > 70.0f) angle = 70.0f;   // TODO: update with real range

//   float pulse = fmap(angle, 0.0f, 70.0f, SERVOMIN, SERVOMAX);
//   return (uint16_t)(pulse + 0.5f);    // round to nearest
// }

// void setup() {
//   Serial.begin(115200);
//   pwm.begin();
//   pwm.setPWMFreq(SERVO_FREQ);
//   delay(500);
//   Serial.println("=== Stewart Platform Servo Control ===");
// }

// void loop() {
//   if (Serial.available()) {
//     String input = Serial.readStringUntil('\n');

//     float a1, a2, a3;
//     if (sscanf(input.c_str(), "%f,%f,%f", &a1, &a2, &a3) == 3) {
//       pwm.setPWM(servoChannels[0], 0, angleToPulse(a1));
//       pwm.setPWM(servoChannels[1], 0, angleToPulse(a2));
//       pwm.setPWM(servoChannels[2], 0, angleToPulse(a3));
//       // Serial.print("Set angles: ");
//       // Serial.println(input);
//     }
//   }
// }

#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver();

#define SERVO_FREQ 50
#define SERVOMIN 310
#define SERVOMAX 450
const int servoChannels[3] = {2, 5, 7};

uint16_t intmap(uint16_t x, uint16_t in_min, uint16_t in_max, uint16_t out_min, uint16_t out_max) {
  return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min;
}

uint16_t angleToPulse(uint16_t angle) {
  return intmap(angle, 0, 70, SERVOMIN, SERVOMAX); //TODO: find the real angle range of motion.
}

void setup() {
  Serial.begin(115200);
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
      delay(5);
      pwm.setPWM(servoChannels[0], 0, angleToPulse(a1));
      pwm.setPWM(servoChannels[1], 0, angleToPulse(a2));
      pwm.setPWM(servoChannels[2], 0, angleToPulse(a3));
      // Serial.print("Set angles: ");
      // Serial.println(input);
    }
  }
}



