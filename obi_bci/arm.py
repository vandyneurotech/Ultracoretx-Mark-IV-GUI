"""Feeding-arm control: serial link to the Arduino Uno, plus a simulated arm for testing.

Serial protocol (115200 baud, newline-terminated ASCII) - must match arduino/obi_arm/obi_arm.ino:

    PC -> Arduino            Arduino -> PC
    S   start scoop          READY            booted / idle at home
    H   go home              STATE <name>     entered a waypoint of the sequence
    X   emergency stop       POS b s e w      current servo angles (deg), ~20 Hz while moving
    P   ping                 DONE             scoop sequence finished
    A b s e w  move to       BUSY             command rejected, arm is moving
               angles        STOPPED          emergency stop latched (send H to recover)
                             PONG
"""

import math
import queue
import threading
import time

import serial
import serial.tools.list_ports

# (name, [base, shoulder, elbow, wrist] degrees, move time ms). Mirrors SEQUENCE[] in the sketch.
HOME = [90, 100, 70, 90]
SCOOP_SEQUENCE = [
    ("to_bowl",   [60, 100, 70, 90], 900),
    ("dip",       [60, 70, 70, 60], 800),
    ("scoop",     [60, 75, 75, 115], 700),
    ("lift",      [60, 105, 80, 85], 800),
    ("to_mouth",  [120, 110, 110, 50], 1200),
    ("hold",      [120, 110, 110, 50], 3000),
    ("home",      HOME, 1200),
]
JOINTS = ["Base", "Shoulder", "Elbow", "Wrist"]


def list_serial_ports():
    return [(p.device, p.description) for p in serial.tools.list_ports.comports()]


def _lerp(a, b, f):
    f = 0.5 - 0.5 * math.cos(min(max(f, 0.0), 1.0) * math.pi)  # ease in/out like the sketch
    return [x + (y - x) * f for x, y in zip(a, b)]


class SimulatedArm:
    """Runs the scoop sequence in software so the GUI and the trigger can be tested with no hardware."""

    name = "Simulated arm"

    def __init__(self):
        self.angles = list(HOME)
        self.state = "READY"
        self._plan = []      # list of (name, start_angles, target, t0, dur)
        self._events = queue.Queue()

    def connect(self):
        self._events.put("READY")

    def close(self):
        pass

    @property
    def busy(self):
        return bool(self._plan)

    def _run(self, steps):
        t0 = time.time()
        start = list(self.angles)
        plan = []
        for name, target, ms in steps:
            plan.append((name, start, list(target), t0, ms / 1000))
            start, t0 = list(target), t0 + ms / 1000
        self._plan = plan

    def scoop(self):
        if self.busy or self.state == "STOPPED":
            self._events.put("BUSY" if self.busy else "STOPPED")
            return
        self._run(SCOOP_SEQUENCE)

    def home(self):
        self.state = "READY"
        self._run([("home", HOME, 1200)])

    def stop(self):
        self._plan = []
        self.state = "STOPPED"
        self._events.put("STOPPED")

    def move_to(self, angles):
        if not self.busy and self.state != "STOPPED":
            self._run([("manual", angles, 800)])

    def poll(self):
        """Advance the simulation; return list of protocol lines (same as the real Arduino would send)."""
        lines = []
        while not self._events.empty():
            lines.append(self._events.get())
        if self._plan:
            now = time.time()
            while self._plan and now >= self._plan[0][3] + self._plan[0][4]:
                self.angles = self._plan.pop(0)[2]
            if self._plan:
                name, start, target, t0, dur = self._plan[0]
                self.angles = _lerp(start, target, (now - t0) / dur)
                if self.state != name:
                    self.state = name
                    lines.append(f"STATE {name}")
            else:
                self.state = "READY"
                lines.append("DONE")
            lines.append("POS " + " ".join(f"{a:.0f}" for a in self.angles))
        return lines


class SerialArm:
    """Talks to the Arduino Uno running obi_arm.ino."""

    def __init__(self, port, baud=115200):
        self.port = port
        self.name = port
        self.baud = baud
        self.ser = None
        self.angles = list(HOME)
        self.state = "CONNECTING"
        self._lines = queue.Queue()
        self._alive = False

    def connect(self):
        self.ser = serial.Serial(self.port, self.baud, timeout=0.1)
        self._alive = True
        threading.Thread(target=self._reader, daemon=True).start()
        # Opening the port resets the Uno; it prints READY after ~2 s.

    def _reader(self):
        buf = b""
        while self._alive:
            try:
                buf += self.ser.read(256)
            except serial.SerialException as exc:
                self._lines.put(f"ERROR {exc}")
                break
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                self._lines.put(line.decode(errors="replace").strip())

    def _send(self, cmd):
        if self.ser and self.ser.is_open:
            self.ser.write((cmd + "\n").encode())

    def close(self):
        self._alive = False
        if self.ser:
            try:
                self._send("X")
            finally:
                self.ser.close()

    @property
    def busy(self):
        return self.state not in ("READY", "STOPPED", "CONNECTING")

    def scoop(self):
        self._send("S")

    def home(self):
        self._send("H")

    def stop(self):
        self._send("X")

    def move_to(self, angles):
        self._send("A " + " ".join(str(int(a)) for a in angles))

    def poll(self):
        lines = []
        while not self._lines.empty():
            line = self._lines.get()
            if not line:
                continue
            lines.append(line)
            word, _, rest = line.partition(" ")
            if word == "POS":
                try:
                    self.angles = [float(v) for v in rest.split()][:4]
                except ValueError:
                    pass
            elif word == "STATE":
                self.state = rest
            elif word in ("READY", "DONE"):
                self.state = "READY"
            elif word == "STOPPED":
                self.state = "STOPPED"
        return lines
