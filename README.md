# Blink-to-Feed: Ultracortex EEG feeding-arm controller

An open, low-cost take on assistive feeding robots like **Obi** (WILA Innovations). Obi users press
body-operated switches to pick food and trigger the spoon. Here the "switch" is a **deliberate double
blink** detected from an OpenBCI Ultracortex headset.

```
Ultracortex (Cyton / Cyton+Daisy / Ganglion)
      │  BrainFlow (USB dongle)
      ▼
Python GUI ── filters ── time series / FFT / band power   (OpenBCI-GUI style)
      │
      └─ Fp1/Fp2 ── BlinkDetector ── BlinkTrigger (2 blinks in 1.2 s)
                                              │  "S\n" over USB serial
                                              ▼
                                     Arduino Uno (obi_arm.ino)
                                     bowl → dip → scoop → lift → mouth → hold → home
```

## Quick start

```bash
pip install -r requirements.txt
```

```bash
python main.py
```

1. **Board:** pick `Synthetic (no headset)` to try everything without hardware, or pick `Cyton`, `Cyton + Daisy` or `Ganglion` and choose the dongle's COM port. Then press **Start Stream**.
2. **Arm:** leave `Simulated arm` selected, or pick the Uno's COM port. Then press **Connect Arm** and tick **Blink control ON**.
3. With Synthetic, press **D** (simulated double blink) and watch the arm run the scoop. **B** sends a single blink, which should *not* trigger.
4. With a real headset, press **Calibrate** and follow the yellow banner: 5 s of rest, then 6 s of blinking.

**Esc** (or the red STOP button) sends an emergency stop: the servos go limp. Press **Home** to recover.

Run the tests with `python -m pytest tests`.

## Files

| Path | What it does |
|---|---|
| `obi_bci/board.py` | BrainFlow streaming, 10-20 channel labels, CSV recording, simulated blink injection |
| `obi_bci/processing.py` | Stateful band-pass and notch filters, ring buffer, FFT, band powers |
| `obi_bci/blink.py` | `BlinkDetector` (EOG or beta mode, adaptive or calibrated threshold) and `BlinkTrigger` |
| `obi_bci/arm.py` | Serial protocol to the Uno, `SimulatedArm`, scoop waypoints |
| `obi_bci/gui.py` | PyQt6 + pyqtgraph interface |
| `arduino/obi_arm/obi_arm.ino` | Uno firmware: smooth non-blocking servo sequence, E-stop, backup button |

## About "blinks spike beta"

A blink shows up on the frontal electrodes (Fp1/Fp2) in two ways, and the GUI can detect either one:

* **EOG amplitude (default, recommended):** the eyeball's electrical dipole moves when the lid closes. This produces a large (100–400 µV), slow (~300 ms) wave that is mostly under 8 Hz. It is far bigger than any brain rhythm, so it is the most reliable signal.
* **Beta-band burst:** the eyelid muscles fire during a blink. That muscle activity (EMG) adds broadband power, including 13–30 Hz, which is the "beta spike" you observed. This mode tracks the z-score of the beta envelope. It is noisier, because jaw clenching and frowning also raise frontal beta, and chewing will be common while eating.

Try both on your own recordings. **Record CSV** saves raw µV data with `# event blink` and `# event scoop` markers, so you can compare the two modes offline.

People also blink spontaneously about 15–20 times a minute. That is why the default trigger needs **2 blinks within 1.2 s**, with a 3 s cooldown after each scoop. All of these settings are in the GUI.

## Arduino wiring

| Part | Uno pin |
|---|---|
| Base servo | D3 |
| Shoulder servo | D5 |
| Elbow servo | D6 |
| Wrist (spoon) servo | D9 |
| E-stop button (to GND) | D2 |
| Backup scoop button (to GND) | D4 |

Power the servos from a separate 5–6 V supply that gives at least 3 A, and connect its GND to the Uno's GND. The Uno's 5 V pin can't supply that much current.

Compile and upload with the Arduino IDE (board: Arduino Uno; needs the Servo library).

**Tuning:** the waypoint angles in `SEQUENCE[]` (in the `.ino`) and `SCOOP_SEQUENCE` (in `arm.py`) are placeholders. Keep the two lists in sync. Find real angles with the GUI first: the `A b s e w` serial command moves the arm to a given pose. Also tighten `MIN_ANGLE` and `MAX_ANGLE` to your arm's safe mechanical range before anyone sits in front of it.

## Safety notes

This is a research prototype, not a medical device. Keep a caregiver present with the E-stop in reach. Use low-torque servos and a soft spoon. Test with the arm well away from the face until the waypoints are tuned.
