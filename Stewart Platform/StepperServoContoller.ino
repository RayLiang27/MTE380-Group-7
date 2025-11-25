#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

/* ============================================================
   ==========  SERVO CONFIG (Motors 2 & 3) =====================
   ============================================================ */

Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver();

#define SERVO_FREQ 50
#define SERVOMIN 310
#define SERVOMAX 450
const int servoChannels[2] = {5, 7};   // CH1 = motor2, CH2 = motor3

float fmapFloat(float x, float in_min, float in_max, float out_min, float out_max) {
  return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min;
}

uint16_t angleToPulse(float angle) {
  if (angle < 0.0f)  angle = 0.0f;
  if (angle > 70.0f) angle = 70.0f;
  float pulse = fmapFloat(angle, 0.0f, 70.0f, SERVOMIN, SERVOMAX);
  return (uint16_t)(pulse + 0.5f);
}

/* ============================================================
   ==========  STEPPER CONFIG (Motor 1) ========================
   ============================================================ */

// Your chosen pins
const int STEP_PIN = 14;  
const int DIR_PIN  = 12;  
const int EN_PIN   = 23;

// Config you provided
const int MIN_ANGLE = 0;
const int MAX_ANGLE = 360;
const int STEPS_PER_REV = 6400;
const float DEGREES_PER_REV = 360.0f;
const int STEP_DELAY_US = 10;

float STEPS_PER_DEG = (float)STEPS_PER_REV / DEGREES_PER_REV;

float currentAngle = 65.0f;   // default neutral angle

void stepOnce() {
  digitalWrite(STEP_PIN, HIGH);
  delayMicroseconds(STEP_DELAY_US);
  digitalWrite(STEP_PIN, LOW);
  delayMicroseconds(STEP_DELAY_US);
}

void moveSteps(long steps, bool cw) {
  digitalWrite(DIR_PIN, cw ? LOW : HIGH);
  for (long i = 0; i < steps; i++) stepOnce();
}

void moveStepperToAngle(float newAngle) {
  if (newAngle < MIN_ANGLE) newAngle = MIN_ANGLE;
  if (newAngle > MAX_ANGLE) newAngle = MAX_ANGLE;

  float delta = newAngle - currentAngle;
  if (fabs(delta) < 0.001f) return;

  long stepsToMove = lroundf(delta * STEPS_PER_DEG);
  bool dirCW = (stepsToMove > 0);

  moveSteps(labs(stepsToMove), dirCW);
  currentAngle = newAngle;
}

/* ============================================================
   ==========  SETUP ===========================================
   ============================================================ */

void setup() {
  Serial.begin(115200);

  // Servo driver
  pwm.begin();
  pwm.setPWMFreq(SERVO_FREQ);

  // Stepper pins
  pinMode(STEP_PIN, OUTPUT);
  pinMode(DIR_PIN, OUTPUT);
  pinMode(EN_PIN, OUTPUT);

  digitalWrite(EN_PIN, HIGH);  // enable driver
  digitalWrite(STEP_PIN, LOW);
  digitalWrite(DIR_PIN, LOW);

  Serial.println("=== Stewart Platform Mixed Control Ready ===");
}

/* ============================================================
   ==========  MAIN LOOP =======================================
   ============================================================ */

void loop() {
  if (Serial.available()) {

    String input = Serial.readStringUntil('\n');
    float a1, a2, a3;

    if (sscanf(input.c_str(), "%f,%f,%f", &a1, &a2, &a3) == 3) {

      // ---- Motor 1: STEPPER ----
      moveStepperToAngle(a1);

      // ---- Motor 2 & 3: SERVOS ----
      pwm.setPWM(servoChannels[0], 0, angleToPulse(a2));
      pwm.setPWM(servoChannels[1], 0, angleToPulse(a3));

      // Debug (optional)
      // Serial.println(input);
    }
  }
}
