"""EEG acquisition from OpenBCI boards (Ultracortex headset) via BrainFlow."""

import csv
import time
from dataclasses import dataclass

import numpy as np
from brainflow.board_shim import BoardIds, BoardShim, BrainFlowInputParams

# Default Ultracortex 10-20 electrode layout (OpenBCI Mark IV docs). The Daisy order
# for channels 9-16 follows the Daisy N1P..N8P wiring on the docs page; change it here
# if your headset is wired differently.
CYTON_LABELS = ["Fp1", "Fp2", "C3", "C4", "P7", "P8", "O1", "O2"]
DAISY_LABELS = CYTON_LABELS + ["F7", "F8", "F3", "F4", "T7", "T8", "P3", "P4"]
GANGLION_LABELS = ["Fp1", "Fp2", "O1", "O2"]


@dataclass(frozen=True)
class BoardOption:
    name: str
    board_id: int
    labels: tuple
    needs_serial: bool


BOARD_OPTIONS = [
    BoardOption("Synthetic (no headset)", BoardIds.SYNTHETIC_BOARD.value, tuple(CYTON_LABELS), False),
    BoardOption("Cyton (8 ch)", BoardIds.CYTON_BOARD.value, tuple(CYTON_LABELS), True),
    BoardOption("Cyton + Daisy (16 ch)", BoardIds.CYTON_DAISY_BOARD.value, tuple(DAISY_LABELS), True),
    BoardOption("Ganglion (4 ch, dongle)", BoardIds.GANGLION_BOARD.value, tuple(GANGLION_LABELS), True),
]


def blink_waveform(fs, amplitude_uv=250.0, duration_s=0.35):
    """A realistic-ish eye-blink artifact: a smooth ~300 ms monophasic bump."""
    n = int(fs * duration_s)
    t = np.linspace(-1, 1, n)
    return amplitude_uv * np.exp(-(t ** 2) / 0.18)


class EEGStreamer:
    """Wraps a BrainFlow BoardShim and hands out only the samples that arrived since the last read."""

    def __init__(self, option: BoardOption, serial_port: str = ""):
        self.option = option
        params = BrainFlowInputParams()
        if option.needs_serial:
            params.serial_port = serial_port
        self.board = BoardShim(option.board_id, params)
        self.fs = BoardShim.get_sampling_rate(option.board_id)
        all_eeg = BoardShim.get_eeg_channels(option.board_id)
        self.eeg_rows = all_eeg[: len(option.labels)]
        self.labels = list(option.labels)
        self.ts_row = BoardShim.get_timestamp_channel(option.board_id)
        self._pending_blinks = []  # waveforms queued for injection (testing)
        self._recorder = None
        self._csv = None
        self.running = False

    @property
    def n_channels(self):
        return len(self.eeg_rows)

    def frontal_indices(self):
        idx = [i for i, name in enumerate(self.labels) if name.lower().startswith("fp")]
        return idx or [0]

    def start(self):
        BoardShim.disable_board_logger()
        self.board.prepare_session()
        self.board.start_stream(45000)
        self.running = True

    def stop(self):
        self.stop_recording()
        if self.running:
            try:
                self.board.stop_stream()
            finally:
                self.running = False
        if self.board.is_prepared():
            self.board.release_session()

    def inject_blink(self, count=1, gap_s=0.45):
        """Queue simulated blink artifacts on the frontal channels (for testing without a headset)."""
        wave = blink_waveform(self.fs)
        gap = np.zeros(int(self.fs * gap_s))
        seq = wave
        for _ in range(count - 1):
            seq = np.concatenate([seq, gap, wave])
        self._pending_blinks.append([seq, 0])

    def read(self):
        """Return (eeg_uV[channels, n], timestamps[n]) for samples received since the last call."""
        data = self.board.get_board_data()
        eeg = data[self.eeg_rows].astype(float)
        ts = data[self.ts_row] if data.shape[1] else np.array([])
        n = eeg.shape[1]
        if n and self._pending_blinks:
            frontal = self.frontal_indices()
            for item in self._pending_blinks:
                seq, pos = item
                take = min(n, len(seq) - pos)
                eeg[frontal, :take] += seq[pos:pos + take]
                item[1] = pos + take
            self._pending_blinks = [it for it in self._pending_blinks if it[1] < len(it[0])]
        if n and self._recorder is not None:
            for i in range(n):
                self._recorder.writerow([f"{ts[i]:.4f}", *np.round(eeg[:, i], 3)])
        return eeg, ts

    # ---- recording -------------------------------------------------------
    def start_recording(self, path):
        self._csv = open(path, "w", newline="")
        self._recorder = csv.writer(self._csv)
        self._csv.write(f"# board={self.option.name} fs={self.fs} started={time.ctime()}\n")
        self._recorder.writerow(["timestamp", *self.labels])

    def mark_event(self, label):
        if self._recorder is not None:
            self._csv.write(f"# event {time.time():.4f} {label}\n")

    def stop_recording(self):
        if self._csv is not None:
            self._csv.close()
        self._csv = None
        self._recorder = None

    @property
    def recording(self):
        return self._recorder is not None
