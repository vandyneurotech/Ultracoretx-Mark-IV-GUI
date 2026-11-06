/*
  obi_arm.ino - Arduino Uno firmware for the blink-controlled feeding arm.

  The PC app (Python GUI) detects a deliberate double blink and sends "S" over USB serial.
  This sketch then runs the scoop sequence: bowl -> dip -> scoop -> lift -> mouth -> hold -> home.

  Wiring (change the pin constants below to match your build):
    Base servo     -> D3     Shoulder servo -> D5
    Elbow servo    -> D6     Wrist servo    -> D9
    E-stop button  -> D2 to GND (normally open, uses INPUT_PULLUP)
    Manual scoop button (optional backup switch, like Obi's) -> D4 to GND
  Power the servos from a separate 5-6 V supply (>= 3 A for 4 servos) and tie its GND to the Uno GND.
  Do NOT power servos from the Uno's 5 V pin.

  Serial protocol @115200 - see obi_bci/arm.py for the full table.
  Keep SEQUENCE[] and HOME_POS in sync with SCOOP_SEQUENCE / HOME in obi_bci/arm.py
  so the GUI's simulated arm matches the real one.
*/

#include <Servo.h>

const uint8_t N_JOINTS = 4;
const uint8_t SERVO_PINS[N_JOINTS] = {3, 5, 6, 9};
const uint8_t ESTOP_PIN = 2;
const uint8_t SCOOP_BTN_PIN = 4;

// Safe angle limits per joint - tighten these once you know your arm's mechanical range.
const int MIN_ANGLE[N_JOINTS] = {20, 40, 30, 30};
const int MAX_ANGLE[N_JOINTS] = {160, 150, 150, 150};

const int HOME_POS[N_JOINTS] = {90, 100, 70, 90};

struct Waypoint {
  const char *name;
  int angles[N_JOINTS];  // base, shoulder, elbow, wrist
  uint16_t ms;           // time to reach this waypoint from the previous one
};

const Waypoint SEQUENCE[] = {
  {"to_bowl",  {60, 100, 70, 90},   900},
  {"dip",      {60, 70, 70, 60},    800},
  {"scoop",    {60, 75, 75, 115},   700},
  {"lift",     {60, 105, 80, 85},   800},
  {"to_mouth", {120, 110, 110, 50}, 1200},
  {"hold",     {120, 110, 110, 50}, 3000},
  {"home",     {90, 100, 70, 90},   1200},
};
const uint8_t SEQ_LEN = sizeof(SEQUENCE) / sizeof(SEQUENCE[0]);

Servo servos[N_JOINTS];
float current[N_JOINTS];
float startPos[N_JOINTS];
int target[N_JOINTS];
unsigned long moveStart = 0;
uint16_t moveMs = 0;
bool moving = false;

enum Mode { IDLE, SEQ, SINGLE, STOPPED };
Mode mode = IDLE;
int8_t seqIndex = -1;

unsigned long lastServoUpdate = 0;
unsigned long lastReport = 0;
char lineBuf[40];
uint8_t lineLen = 0;

int clampJoint(uint8_t j, int a) {
  return constrain(a, MIN_ANGLE[j], MAX_ANGLE[j]);
}

void attachAll() {
  for (uint8_t j = 0; j < N_JOINTS; j++) {
    if (!servos[j].attached()) servos[j].attach(SERVO_PINS[j]);
  }
}

void startMove(const int *angles, uint16_t ms) {
  for (uint8_t j = 0; j < N_JOINTS; j++) {
    startPos[j] = current[j];
    target[j] = clampJoint(j, angles[j]);
  }
  moveStart = millis();
  moveMs = ms ? ms : 1;
  moving = true;
}

void startWaypoint(uint8_t i) {
  seqIndex = i;
  Serial.print(F("STATE "));
  Serial.println(SEQUENCE[i].name);
  startMove(SEQUENCE[i].angles, SEQUENCE[i].ms);
}

void reportPos() {
  Serial.print(F("POS"));
  for (uint8_t j = 0; j < N_JOINTS; j++) {
    Serial.print(' ');
    Serial.print((int)current[j]);
  }
  Serial.println();
}

void emergencyStop() {
  moving = false;
  mode = STOPPED;
  seqIndex = -1;
  // Detach so the servos go limp instead of holding a possibly dangerous pose.
  for (uint8_t j = 0; j < N_JOINTS; j++) servos[j].detach();
  Serial.println(F("STOPPED"));
}

void beginScoop() {
  if (mode == STOPPED) { Serial.println(F("STOPPED")); return; }
  if (mode != IDLE)    { Serial.println(F("BUSY")); return; }
  mode = SEQ;
  startWaypoint(0);
}

void goHome() {
  if (mode == SEQ || mode == SINGLE) { Serial.println(F("BUSY")); return; }
  attachAll();
  mode = SINGLE;
  Serial.println(F("STATE home"));
  startMove(HOME_POS, 1200);
}

void handleLine(char *line) {
  switch (line[0]) {
    case 'S': beginScoop(); break;
    case 'H': goHome(); break;
    case 'X': emergencyStop(); break;
    case 'P': Serial.println(F("PONG")); break;
    case 'A': {
      if (mode != IDLE) { Serial.println(mode == STOPPED ? F("STOPPED") : F("BUSY")); break; }
      int a[N_JOINTS];
      if (sscanf(line + 1, "%d %d %d %d", &a[0], &a[1], &a[2], &a[3]) == 4) {
        mode = SINGLE;
        Serial.println(F("STATE manual"));
        startMove(a, 800);
      }
      break;
    }
    default: break;
  }
}

void readSerial() {
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n' || c == '\r') {
      if (lineLen) {
        lineBuf[lineLen] = '\0';
        handleLine(lineBuf);
        lineLen = 0;
      }
    } else if (lineLen < sizeof(lineBuf) - 1) {
      lineBuf[lineLen++] = c;
    }
  }
}

void updateMotion() {
  if (!moving) return;
  float f = (float)(millis() - moveStart) / moveMs;
  if (f >= 1.0) f = 1.0;
  float eased = 0.5 - 0.5 * cos(f * PI);  // smooth start and stop - gentler near the face
  for (uint8_t j = 0; j < N_JOINTS; j++) {
    current[j] = startPos[j] + (target[j] - startPos[j]) * eased;
    servos[j].write((int)(current[j] + 0.5));
  }
  if (f < 1.0) return;

  moving = false;
  if (mode == SEQ && seqIndex + 1 < SEQ_LEN) {
    startWaypoint(seqIndex + 1);
  } else {
    if (mode == SEQ) Serial.println(F("DONE"));
    else Serial.println(F("READY"));
    reportPos();
    mode = IDLE;
    seqIndex = -1;
  }
}

void setup() {
  Serial.begin(115200);
  pinMode(ESTOP_PIN, INPUT_PULLUP);
  pinMode(SCOOP_BTN_PIN, INPUT_PULLUP);
  for (uint8_t j = 0; j < N_JOINTS; j++) {
    current[j] = HOME_POS[j];
    servos[j].write(HOME_POS[j]);  // set position before attach to avoid a jump
  }
  attachAll();
  Serial.println(F("READY"));
  reportPos();
}

void loop() {
  readSerial();

  if (digitalRead(ESTOP_PIN) == LOW && mode != STOPPED) emergencyStop();

  static bool lastBtn = HIGH;
  bool btn = digitalRead(SCOOP_BTN_PIN);
  if (lastBtn == HIGH && btn == LOW) beginScoop();
  lastBtn = btn;

  unsigned long now = millis();
  if (now - lastServoUpdate >= 20) {
    lastServoUpdate = now;
    updateMotion();
  }
  if (moving && now - lastReport >= 50) {
    lastReport = now;
    reportPos();
  }
}
