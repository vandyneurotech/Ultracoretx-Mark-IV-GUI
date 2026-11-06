"""Main window: OpenBCI-GUI-style EEG viewer + blink detection + feeding-arm control."""

import math
import os
import time

import numpy as np
import pyqtgraph as pg
from PyQt6 import QtCore, QtGui, QtWidgets

from .arm import HOME, JOINTS, SerialArm, SimulatedArm, list_serial_ports
from .blink import BlinkDetector, BlinkTrigger
from .board import BOARD_OPTIONS, EEGStreamer
from .processing import BANDS, RingBuffer, StreamFilter, band_powers, spectrum

WINDOW_S = 5
TICK_MS = 33
RAIL_UV = 187000  # Cyton input range is +/-187.5 mV; at the rail the electrode is not making contact
CHANNEL_COLORS = ["#a5a5a5", "#8a5cd1", "#4f6ee6", "#3fb27f", "#e8d33f", "#f29b38", "#ef5350", "#a1734c"]
BAND_COLORS = ["#6f6fd6", "#57b8d6", "#5fcf7e", "#e8c547", "#e8704a"]
ARM_LINKS_CM = (12.0, 10.0, 6.0)  # shoulder->elbow, elbow->wrist, wrist->spoon tip (for the 2D preview)

DARK_QSS = """
QWidget { background: #1f2229; color: #e6e6e6; font-size: 10pt; }
QGroupBox { border: 1px solid #3a3f4b; border-radius: 6px; margin-top: 14px; padding: 6px; }
QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 4px; color: #9ecbff; }
QPushButton { background: #2f3542; border: 1px solid #4a5162; border-radius: 4px; padding: 5px 10px; }
QPushButton:hover { background: #3a4252; }
QPushButton:checked { background: #2e6b3f; }
QPushButton#stop { background: #b3261e; font-weight: bold; color: white; }
QPushButton#stop:hover { background: #d32f2f; }
QComboBox, QSpinBox, QDoubleSpinBox, QPlainTextEdit { background: #282c35; border: 1px solid #4a5162; border-radius: 4px; padding: 2px 4px; }
QLabel#banner { background: #3d3a1e; color: #ffe08a; font-size: 13pt; padding: 8px; border-radius: 6px; }
"""


class ArmView(QtWidgets.QWidget):
    """2D stand-in for the CAD arm: side view (shoulder/elbow/wrist) and top view (base)."""

    def __init__(self):
        super().__init__()
        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.side = pg.PlotWidget(title="Side view")
        self.top = pg.PlotWidget(title="Top view (base)")
        for p, rng in ((self.side, (-6, 30, -12, 28)), (self.top, (-22, 22, -4, 24))):
            p.setAspectLocked(True)
            p.setXRange(rng[0], rng[1])
            p.setYRange(rng[2], rng[3])
            p.hideAxis("bottom")
            p.hideAxis("left")
            p.setMouseEnabled(False, False)
        self.side.plot([-5, 5], [0, 0], pen=pg.mkPen("#666", width=8))  # mount
        self.side_arm = self.side.plot(pen=pg.mkPen("#9ecbff", width=6), symbol="o", symbolSize=10,
                                       symbolBrush="#ffffff")
        self.spoon = self.side.plot(pen=pg.mkPen("#ffcc66", width=5))
        r = 18
        for ang, label, color in ((60, "bowl", "#7ccf7c"), (120, "mouth", "#ef8f8f")):
            x, y = r * math.cos(math.radians(ang)), r * math.sin(math.radians(ang))
            self.top.plot([x], [y], symbol="o", symbolSize=22, symbolBrush=color, pen=None)
            t = pg.TextItem(label, color=color, anchor=(0.5, -0.6))
            t.setPos(x, y)
            self.top.addItem(t)
        self.top_arm = self.top.plot(pen=pg.mkPen("#9ecbff", width=6), symbol="o", symbolBrush="#fff")
        lay.addWidget(self.side, 3)
        lay.addWidget(self.top, 2)
        self.set_angles(HOME)

    def set_angles(self, angles):
        base, sh, el, wr = angles
        l1, l2, l3 = ARM_LINKS_CM
        a1 = math.radians(sh)
        a2 = a1 - math.radians(180 - el)
        a3 = a2 + math.radians(wr - 90)
        p0 = (0.0, 0.0)
        p1 = (p0[0] + l1 * math.cos(a1), p0[1] + l1 * math.sin(a1))
        p2 = (p1[0] + l2 * math.cos(a2), p1[1] + l2 * math.sin(a2))
        p3 = (p2[0] + l3 * math.cos(a3), p2[1] + l3 * math.sin(a3))
        self.side_arm.setData([p0[0], p1[0], p2[0]], [p0[1], p1[1], p2[1]])
        self.spoon.setData([p2[0], p3[0]], [p2[1], p3[1]])
        reach = max(p3[0], 4.0)
        b = math.radians(base)
        self.top_arm.setData([0, reach * math.cos(b)], [0, reach * math.sin(b)])


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Blink-to-Feed | Ultracortex EEG feeding-arm controller")
        self.resize(1600, 1000)
        pg.setConfigOptions(antialias=True, background="#181a20", foreground="#cfd3dc")

        self.streamer = None
        self.arm = None
        self.detector = None
        self.trigger = BlinkTrigger()
        self.calib = None
        self.blink_count = 0
        self.scoop_count = 0
        self.last_scoop_t = 0.0
        self.blink_marks = []  # wall-clock times of detected blinks (for the score plot)
        self._tick_n = 0

        self._build_ui()
        self.refresh_ports()
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.tick)
        self.timer.start(TICK_MS)
        self.log("Ready. Choose a board and press Start Stream. Use 'Synthetic' to test without a headset.")

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        root = QtWidgets.QWidget()
        self.setCentralWidget(root)
        vbox = QtWidgets.QVBoxLayout(root)

        # --- top control bar
        bar = QtWidgets.QHBoxLayout()
        vbox.addLayout(bar)

        hs = QtWidgets.QGroupBox("Headset (Ultracortex)")
        h = QtWidgets.QHBoxLayout(hs)
        self.board_combo = QtWidgets.QComboBox()
        for opt in BOARD_OPTIONS:
            self.board_combo.addItem(opt.name)
        self.eeg_port = QtWidgets.QComboBox()
        self.eeg_port.setMinimumWidth(170)
        refresh = QtWidgets.QPushButton("Refresh ports")
        refresh.clicked.connect(self.refresh_ports)
        self.stream_btn = QtWidgets.QPushButton("Start Stream")
        self.stream_btn.setCheckable(True)
        self.stream_btn.clicked.connect(self.toggle_stream)
        self.rec_btn = QtWidgets.QPushButton("Record CSV")
        self.rec_btn.setCheckable(True)
        self.rec_btn.clicked.connect(self.toggle_record)
        for w in (QtWidgets.QLabel("Board"), self.board_combo, QtWidgets.QLabel("Port"), self.eeg_port,
                  refresh, self.stream_btn, self.rec_btn):
            h.addWidget(w)
        bar.addWidget(hs, 3)

        ag = QtWidgets.QGroupBox("Feeding arm (Arduino Uno)")
        a = QtWidgets.QHBoxLayout(ag)
        self.arm_port = QtWidgets.QComboBox()
        self.arm_port.setMinimumWidth(170)
        self.arm_btn = QtWidgets.QPushButton("Connect Arm")
        self.arm_btn.setCheckable(True)
        self.arm_btn.clicked.connect(self.toggle_arm)
        self.enable_chk = QtWidgets.QCheckBox("Blink control ON")
        self.enable_chk.setToolTip("When checked, a detected blink pattern sends the scoop command to the arm.")
        stop = QtWidgets.QPushButton("STOP (Esc)")
        stop.setObjectName("stop")
        stop.clicked.connect(self.arm_stop)
        QtGui.QShortcut(QtGui.QKeySequence("Esc"), self, activated=self.arm_stop)
        for w in (self.arm_port, self.arm_btn, self.enable_chk, stop):
            a.addWidget(w)
        bar.addWidget(ag, 2)

        self.banner = QtWidgets.QLabel()
        self.banner.setObjectName("banner")
        self.banner.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.banner.hide()
        vbox.addWidget(self.banner)

        # --- main area
        split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        vbox.addWidget(split, 1)

        left = QtWidgets.QWidget()
        lv = QtWidgets.QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        tsbar = QtWidgets.QHBoxLayout()
        tsbar.addWidget(QtWidgets.QLabel("Time series (1-45 Hz, 60 Hz notch)"))
        tsbar.addStretch()
        tsbar.addWidget(QtWidgets.QLabel("Vert scale"))
        self.scale_combo = QtWidgets.QComboBox()
        self.scale_combo.addItems(["50 uV", "100 uV", "200 uV", "500 uV", "1000 uV"])
        self.scale_combo.setCurrentIndex(2)
        tsbar.addWidget(self.scale_combo)
        self.notch_combo = QtWidgets.QComboBox()
        self.notch_combo.addItems(["60 Hz notch", "50 Hz notch", "No notch"])
        self.notch_combo.currentIndexChanged.connect(self._reset_display_filter)
        tsbar.addWidget(self.notch_combo)
        lv.addLayout(tsbar)
        self.ts_plot = pg.PlotWidget()
        self.ts_plot.setLabel("bottom", "Time", "s")
        self.ts_plot.setMouseEnabled(False, False)
        self.ts_plot.showGrid(x=True, y=False, alpha=0.2)
        lv.addWidget(self.ts_plot, 1)
        split.addWidget(left)

        right = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        self.fft_plot = pg.PlotWidget(title="FFT")
        self.fft_plot.setLogMode(y=True)
        self.fft_plot.setLabel("bottom", "Frequency", "Hz")
        self.fft_plot.setLabel("left", "uV/sqrt(Hz)")
        self.fft_plot.setXRange(0, 60)
        right.addWidget(self.fft_plot)
        self.band_plot = pg.PlotWidget(title="Band power (all channels)")
        self.band_plot.setLogMode(y=False)
        self.band_plot.getAxis("bottom").setTicks([list(enumerate(BANDS))])
        self.band_bars = pg.BarGraphItem(x=list(range(len(BANDS))), height=[0] * len(BANDS), width=0.6,
                                         brushes=BAND_COLORS)
        self.band_plot.addItem(self.band_bars)
        right.addWidget(self.band_plot)

        armbox = QtWidgets.QGroupBox("Arm")
        ab = QtWidgets.QVBoxLayout(armbox)
        self.arm_view = ArmView()
        ab.addWidget(self.arm_view, 1)
        row = QtWidgets.QHBoxLayout()
        self.arm_state = QtWidgets.QLabel("Arm: not connected")
        row.addWidget(self.arm_state, 1)
        for text, fn in (("Scoop now", lambda: self.request_scoop("manual button")), ("Home", self.arm_home)):
            b = QtWidgets.QPushButton(text)
            b.clicked.connect(fn)
            row.addWidget(b)
        ab.addLayout(row)
        self.angle_label = QtWidgets.QLabel()
        ab.addWidget(self.angle_label)
        right.addWidget(armbox)
        self.fft_plot.setMinimumHeight(140)
        self.band_plot.setMinimumHeight(120)
        armbox.setMinimumHeight(260)
        right.setSizes([220, 170, 330])
        split.addWidget(right)
        split.setSizes([900, 600])

        # --- blink panel + log
        bottom_w = QtWidgets.QWidget()
        bottom_w.setMaximumHeight(360)
        bottom = QtWidgets.QHBoxLayout(bottom_w)
        bottom.setContentsMargins(0, 0, 0, 0)
        vbox.addWidget(bottom_w)
        bb = QtWidgets.QGroupBox("Blink detection (Fp1/Fp2)")
        bl = QtWidgets.QHBoxLayout(bb)
        self.score_plot = pg.PlotWidget()
        self.score_plot.setMinimumHeight(150)
        self.score_plot.setMouseEnabled(False, False)
        self.score_curve = self.score_plot.plot(pen=pg.mkPen("#9ecbff", width=1.5))
        self.thr_line = pg.InfiniteLine(angle=0, pen=pg.mkPen("#ff6b6b", width=2, style=QtCore.Qt.PenStyle.DashLine))
        self.score_plot.addItem(self.thr_line)
        self.blink_scatter = pg.ScatterPlotItem(size=12, brush="#ffcc66", symbol="t")
        self.score_plot.addItem(self.blink_scatter)
        bl.addWidget(self.score_plot, 3)

        form = QtWidgets.QFormLayout()
        self.mode_combo = QtWidgets.QComboBox()
        self.mode_combo.addItems(["EOG amplitude (recommended)", "Beta-band burst (13-30 Hz)"])
        self.mode_combo.currentIndexChanged.connect(self._mode_changed)
        self.sens_spin = QtWidgets.QDoubleSpinBox()
        self.sens_spin.setRange(1.5, 20)
        self.sens_spin.setValue(5.0)
        self.sens_spin.setSingleStep(0.5)
        self.sens_spin.setToolTip("Threshold in multiples of background noise (lower = more sensitive)")
        self.sens_spin.valueChanged.connect(self._apply_detector_settings)
        self.req_spin = QtWidgets.QSpinBox()
        self.req_spin.setRange(1, 4)
        self.req_spin.setValue(2)
        self.req_spin.setToolTip("Blinks needed to trigger a scoop. 2+ avoids triggering on natural blinks.")
        self.req_spin.valueChanged.connect(self._apply_detector_settings)
        self.win_spin = QtWidgets.QDoubleSpinBox()
        self.win_spin.setRange(0.4, 4.0)
        self.win_spin.setValue(1.2)
        self.win_spin.setSuffix(" s")
        self.win_spin.valueChanged.connect(self._apply_detector_settings)
        self.cool_spin = QtWidgets.QDoubleSpinBox()
        self.cool_spin.setRange(0, 60)
        self.cool_spin.setValue(3.0)
        self.cool_spin.setSuffix(" s")
        form.addRow("Method", self.mode_combo)
        form.addRow("Threshold (x noise)", self.sens_spin)
        form.addRow("Blinks to trigger", self.req_spin)
        form.addRow("...within", self.win_spin)
        form.addRow("Cooldown after scoop", self.cool_spin)
        self.counts = QtWidgets.QLabel("Blinks: 0   Scoops: 0")
        form.addRow(self.counts)
        btns = QtWidgets.QGridLayout()
        cal = QtWidgets.QPushButton("Calibrate")
        cal.clicked.connect(self.start_calibration)
        rst = QtWidgets.QPushButton("Auto threshold")
        rst.clicked.connect(self.reset_calibration)
        sim1 = QtWidgets.QPushButton("Sim blink (B)")
        sim1.clicked.connect(lambda: self.simulate_blink(1))
        sim2 = QtWidgets.QPushButton("Sim double (D)")
        sim2.clicked.connect(lambda: self.simulate_blink(2))
        QtGui.QShortcut(QtGui.QKeySequence("B"), self, activated=lambda: self.simulate_blink(1))
        QtGui.QShortcut(QtGui.QKeySequence("D"), self, activated=lambda: self.simulate_blink(2))
        for i, w in enumerate((cal, rst, sim1, sim2)):
            btns.addWidget(w, i // 2, i % 2)
        form.addRow(btns)
        self.blink_led = QtWidgets.QLabel("BLINK")
        self.blink_led.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.blink_led.setFixedHeight(28)
        self._led(False)
        form.addRow(self.blink_led)
        fw = QtWidgets.QWidget()
        fw.setLayout(form)
        bl.addWidget(fw, 2)
        bottom.addWidget(bb, 3)

        lg = QtWidgets.QGroupBox("Log")
        ll = QtWidgets.QVBoxLayout(lg)
        self.log_box = QtWidgets.QPlainTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setMaximumBlockCount(500)
        ll.addWidget(self.log_box)
        bottom.addWidget(lg, 1)

    def _led(self, on):
        self.blink_led.setStyleSheet(
            "background:#ffcc66;color:#000;font-weight:bold;border-radius:4px;" if on
            else "background:#2a2e38;color:#666;border-radius:4px;")

    def log(self, msg):
        self.log_box.appendPlainText(time.strftime("%H:%M:%S  ") + msg)

    def refresh_ports(self):
        ports = list_serial_ports()
        cur_e, cur_a = self.eeg_port.currentText(), self.arm_port.currentText()
        self.eeg_port.clear()
        self.arm_port.clear()
        self.arm_port.addItem("Simulated arm")
        for dev, desc in ports:
            self.eeg_port.addItem(f"{dev} - {desc}", dev)
            self.arm_port.addItem(f"{dev} - {desc}", dev)
        for combo, cur in ((self.eeg_port, cur_e), (self.arm_port, cur_a)):
            i = combo.findText(cur)
            if i >= 0:
                combo.setCurrentIndex(i)
        if not ports:
            self.eeg_port.addItem("(no serial ports)", "")

    # ------------------------------------------------------------------ headset
    def toggle_stream(self, checked):
        if not checked:
            self.stop_stream()
            return
        opt = BOARD_OPTIONS[self.board_combo.currentIndex()]
        port = self.eeg_port.currentData() or ""
        if opt.needs_serial and not port:
            self.log("Select the serial port of the OpenBCI USB dongle first.")
            self.stream_btn.setChecked(False)
            return
        try:
            self.streamer = EEGStreamer(opt, port)
            self.streamer.start()
        except Exception as exc:  # BrainFlowError and serial errors
            self.log(f"Could not start {opt.name}: {exc}")
            self.streamer = None
            self.stream_btn.setChecked(False)
            return
        fs, n = self.streamer.fs, self.streamer.n_channels
        self.raw_buf = RingBuffer(n, fs * WINDOW_S)
        self.disp_buf = RingBuffer(n, fs * WINDOW_S)
        self.score_buf = RingBuffer(1, fs * WINDOW_S)
        self._reset_display_filter()
        self.frontal = self.streamer.frontal_indices()
        self.detector = BlinkDetector(fs, len(self.frontal))
        self._mode_changed()
        self._setup_ts_curves()
        self.stream_btn.setText("Stop Stream")
        self.board_combo.setEnabled(False)
        names = ", ".join(self.streamer.labels[i] for i in self.frontal)
        self.log(f"Streaming {opt.name} @ {fs} Hz, {n} ch. Blink detection on {names}.")

    def stop_stream(self):
        if self.streamer:
            try:
                self.streamer.stop()
            except Exception as exc:
                self.log(f"Error while stopping: {exc}")
            self.log("Stream stopped.")
        self.streamer = None
        self.calib = None
        self.banner.hide()
        self.stream_btn.setChecked(False)
        self.stream_btn.setText("Start Stream")
        self.rec_btn.setChecked(False)
        self.board_combo.setEnabled(True)

    def toggle_record(self, checked):
        if not self.streamer:
            self.rec_btn.setChecked(False)
            self.log("Start the stream before recording.")
            return
        if checked:
            os.makedirs("recordings", exist_ok=True)
            path = os.path.join("recordings", time.strftime("eeg_%Y%m%d_%H%M%S.csv"))
            self.streamer.start_recording(path)
            self.log(f"Recording to {path}")
        else:
            self.streamer.stop_recording()
            self.log("Recording stopped.")

    def _reset_display_filter(self):
        if not self.streamer:
            return
        notch = {0: 60.0, 1: 50.0}.get(self.notch_combo.currentIndex())
        self.display_filter = StreamFilter(self.streamer.fs, self.streamer.n_channels, low=1.0, high=45.0,
                                           notch=notch)

    def _setup_ts_curves(self):
        self.ts_plot.clear()
        self.ts_curves = []
        n = self.streamer.n_channels
        for i in range(n):
            pen = pg.mkPen(CHANNEL_COLORS[i % len(CHANNEL_COLORS)], width=1)
            self.ts_curves.append(self.ts_plot.plot(pen=pen))
        self.ts_plot.setYRange(-n + 0.5, 0.5, padding=0)
        self.ts_plot.setXRange(-WINDOW_S, 0, padding=0)
        self._update_channel_ticks()

    def _update_channel_ticks(self):
        raw = self.raw_buf.latest(self.streamer.fs)
        ticks = []
        for i, name in enumerate(self.streamer.labels):
            if raw.shape[1] and np.abs(raw[i]).max() >= RAIL_UV:
                status = "RAILED"
            elif self.disp_buf.filled:
                status = f"{np.std(self.disp_buf.latest(self.streamer.fs)[i]):.0f} uVrms"
            else:
                status = ""
            ticks.append((-i, f"{i + 1} {name}  {status}"))
        self.ts_plot.getAxis("left").setTicks([ticks])

    # ------------------------------------------------------------------ blink detection
    def _mode_changed(self):
        if not self.detector:
            return
        self.detector.set_mode("eog" if self.mode_combo.currentIndex() == 0 else "beta")
        self._apply_detector_settings()
        self.log(f"Blink method: {self.mode_combo.currentText()} (adaptive threshold)")

    def _apply_detector_settings(self):
        if self.detector:
            self.detector.sensitivity = self.sens_spin.value()
        self.trigger.required = self.req_spin.value()
        self.trigger.window_s = self.win_spin.value()

    def simulate_blink(self, count):
        if not self.streamer:
            self.log("Start a stream first (Synthetic works).")
            return
        self.streamer.inject_blink(count)

    def on_blink(self, ev):
        self.blink_count += 1
        self.blink_marks.append((ev.t, ev.peak))
        self._led(True)
        QtCore.QTimer.singleShot(250, lambda: self._led(False))
        self.counts.setText(f"Blinks: {self.blink_count}   Scoops: {self.scoop_count}")
        if self.streamer:
            self.streamer.mark_event("blink")
        if self.calib:
            return  # never move the arm while calibrating
        if self.trigger.feed(ev):
            self.request_scoop(f"{self.trigger.required}x blink")

    def request_scoop(self, source):
        if not self.arm:
            self.log(f"Scoop requested ({source}) but no arm is connected.")
            return
        if source != "manual button" and not self.enable_chk.isChecked():
            self.log(f"Scoop pattern detected ({source}) - blink control is OFF, ignoring.")
            return
        if self.arm.busy:
            self.log("Arm is busy - scoop ignored.")
            return
        if source != "manual button" and time.time() - self.last_scoop_t < self.cool_spin.value():
            self.log("In cooldown - scoop ignored.")
            return
        self.last_scoop_t = time.time()
        self.scoop_count += 1
        self.counts.setText(f"Blinks: {self.blink_count}   Scoops: {self.scoop_count}")
        if self.streamer:
            self.streamer.mark_event("scoop")
        self.log(f">>> SCOOP ({source})")
        self.arm.scoop()

    def start_calibration(self):
        if not self.streamer:
            self.log("Start the stream before calibrating.")
            return
        self.calib = {"phase": "rest", "rest": [], "blink": []}
        self.banner.setText("Calibration 1/2: relax, look straight ahead and try NOT to blink (5 s)")
        self.banner.show()
        QtCore.QTimer.singleShot(5000, self._calib_blink_phase)

    def _calib_blink_phase(self):
        if not self.calib:
            return
        self.calib["phase"] = "blink"
        self.banner.setText("Calibration 2/2: blink firmly 4-5 times, about once per second (6 s)")
        QtCore.QTimer.singleShot(6000, self._calib_finish)

    def _calib_finish(self):
        if not self.calib:
            return
        calib, self.calib = self.calib, None
        self.banner.hide()
        try:
            thr = self.detector.calibrate(np.array(calib["rest"]), np.array(calib["blink"]))
            self.log(f"Calibration done: threshold = {thr:.1f}" + (" (polarity flipped)" if self.detector.polarity < 0 else ""))
        except ValueError as exc:
            self.log(f"Calibration failed: {exc}")
        self.trigger.reset()

    def reset_calibration(self):
        if self.detector:
            self.detector.fixed_threshold = None
            self.log("Using adaptive threshold.")

    # ------------------------------------------------------------------ arm
    def toggle_arm(self, checked):
        if not checked:
            if self.arm:
                self.arm.close()
            self.arm = None
            self.arm_btn.setText("Connect Arm")
            self.arm_state.setText("Arm: not connected")
            self.log("Arm disconnected.")
            return
        port = self.arm_port.currentData()
        arm = SimulatedArm() if not port else SerialArm(port)
        try:
            arm.connect()
        except Exception as exc:
            self.log(f"Could not open arm on {port}: {exc}")
            self.arm_btn.setChecked(False)
            return
        self.arm = arm
        self.arm_btn.setText("Disconnect Arm")
        self.log(f"Arm connected: {arm.name}" + ("" if not port else " (Uno resets; waits ~2 s for READY)"))

    def arm_stop(self):
        if self.arm:
            self.arm.stop()
            self.log("EMERGENCY STOP sent. Press Home to recover.")

    def arm_home(self):
        if self.arm:
            self.arm.home()

    # ------------------------------------------------------------------ main loop
    def tick(self):
        self._tick_n += 1
        if self.streamer and self.streamer.running:
            try:
                eeg, _ = self.streamer.read()
            except Exception as exc:
                self.log(f"Stream error: {exc}")
                self.stop_stream()
                return
            if eeg.shape[1]:
                self.raw_buf.append(eeg)
                self.disp_buf.append(self.display_filter(eeg))
                scores, events = self.detector.process(eeg[self.frontal])
                self.score_buf.append(scores[None, :])
                if self.calib:
                    self.calib[self.calib["phase"]].extend(scores)
                for ev in events:
                    self.on_blink(ev)
            self._draw_eeg()

        if self.arm:
            for line in self.arm.poll():
                if not line.startswith("POS"):
                    self.log(f"arm: {line}")
            self.arm_view.set_angles(self.arm.angles)
            self.arm_state.setText(f"Arm ({self.arm.name}): {self.arm.state}")
            self.angle_label.setText("   ".join(f"{j} {a:.0f}deg" for j, a in zip(JOINTS, self.arm.angles)))

    def _draw_eeg(self):
        fs = self.streamer.fs
        disp = self.disp_buf.latest()
        n = disp.shape[1]
        if n < 2:
            return
        t = (np.arange(n) - n) / fs
        scale = float(self.scale_combo.currentText().split()[0])
        for i, curve in enumerate(self.ts_curves):
            curve.setData(t, np.clip(disp[i] / scale, -0.5, 0.5) - i)

        sc = self.score_buf.latest()[0]
        self.score_curve.setData(t[-len(sc):], sc)
        thr = self.detector.threshold
        if np.isfinite(thr):
            self.thr_line.setPos(thr)
        now = time.time()
        self.blink_marks = [m for m in self.blink_marks if now - m[0] < WINDOW_S]
        self.blink_scatter.setData([m[0] - now for m in self.blink_marks], [m[1] for m in self.blink_marks])
        self.score_plot.setXRange(-WINDOW_S, 0, padding=0)

        if self._tick_n % 6 == 0 and n > fs:
            freqs, amp = spectrum(disp, fs)
            if not hasattr(self, "fft_curves") or len(self.fft_curves) != disp.shape[0]:
                self.fft_plot.clear()
                self.fft_curves = [self.fft_plot.plot(pen=pg.mkPen(CHANNEL_COLORS[i % 8], width=1))
                                   for i in range(disp.shape[0])]
            for i, c in enumerate(self.fft_curves):
                c.setData(freqs[1:], np.maximum(amp[i, 1:], 1e-3))
            bp = band_powers(disp, fs)
            self.band_bars.setOpts(height=[bp[b] for b in BANDS])
        if self._tick_n % 15 == 0:
            self._update_channel_ticks()

    def closeEvent(self, event):
        self.timer.stop()
        if self.streamer:
            self.streamer.stop()
        if self.arm:
            self.arm.close()
        super().closeEvent(event)


def main():
    import sys
    app = QtWidgets.QApplication(sys.argv)
    app.setStyleSheet(DARK_QSS)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())
