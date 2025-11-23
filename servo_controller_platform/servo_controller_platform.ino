#include <Servo.h>
#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver();

#define SERVOMIN 250
#define SERVOMAX 450
#define SERVO_FREQ 50
const int MIN_ANGLE = 0;
const int MAX_ANGLE = 60;

const uint8_t motorChannel[3] = {2, 5, 7}; // Channels for motor 1, 2, and 3

void setup()
{
  // Initialize serial communication
  Serial.begin(9600);
  pwm.begin();
  pwm.setOscillatorFrequency(27000000);
  pwm.setPWMFreq(SERVO_FREQ); // Analog servos run at ~60 Hz updates
  delay(10);
  Serial.println("Arduino servo controller ready");
}

int angleToPulse(int angle)
{
  angle = constrain(angle, MIN_ANGLE, MAX_ANGLE);
  return map(angle, MIN_ANGLE, MAX_ANGLE, SERVOMIN, SERVOMAX);
}

void setMotor(int motorIndex, int angle)
{
  if (motorIndex < 1 || motorIndex > 3)
    return;
  int ch = motorChannel[motorIndex - 1];
  int pulse = angleToPulse(angle);
  pwm.setPWM(ch, 0, pulse);
}

void processLine(const char *line)
{
  // Parse the line to extract the angle
  // Formats accepted
  // "A x y z" where xyz is the angle in degrees
  // "S n angle" where n is servo number and angle in degrees
  if (line[0] == 'A')
  {
    int a = 0, b = 0, c = 0;
    if (sscanf(line, "A %d %d %d", &a, &b, &c) >= 3)
    {
      setMotor(1, a);
      setMotor(2, b);
      setMotor(3, c);
      Serial.println("OK");
    }
    else
    {
      Serial.println("ERR");
    }
  }
  else if (line[0] == 'S')
  {
    int n = 0, ang = 0;
    if (sscanf(line + 1, " %d %d", &n, &ang) >= 2)
    {
      setMotor(n, ang);
      Serial.println("OK");
    }
    else
    {
      Serial.println("ERR");
    }
  }
  else
  {
    Serial.println("UNK");
  }
}

void loop()
{
  static char buf[64];
  static uint8_t idx = 0;
  while (Serial.available())
  {
    char c = Serial.read();
    if (c == '\r')
      continue;
    if (c == '\n')
    {
      buf[idx] = 0;
      if (idx > 0)
        processLine(buf);
      idx = 0;
    }
    else if (idx < sizeof(buf) - 1)
    {
      buf[idx++] = c;
    }
    else
    {
      idx = 0; // overflow - reset
    }
  }
}