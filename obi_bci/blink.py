"""Blink detection from frontal EEG (Fp1/Fp2) and the blink-pattern trigger for the arm.

Two detection modes are provided:

* ``eog``  - A blink is a large (100-400 uV), slow (~300 ms) deflection on Fp1/Fp2
             caused by the eye's electrical dipole moving. This is the classic, most
             reliable blink signature. The score is the 0.2-8 Hz signal in uV, signed
             by ``polarity`` (blinks are normally positive on Fp1/Fp2; calibration
             flips it automatically if your montage shows them negative).
* ``beta`` - Eyelid muscle (EMG) activity during a blink also adds a burst of
             13-30 Hz power on the frontal electrodes. The score is the z-score of the
             beta-band envelope against a rolling baseline.

Either way the detector emits a stream of scores plus a threshold, and a blink is a
threshold crossing that ends within a plausible blink duration.
"""

import time
from collections import deque
from dataclasses import dataclass

import numpy as np

from .processing import StreamFilter


@dataclass
class BlinkEvent:
    t: float          # wall-clock time (s)
    peak: float       # peak score during the blink
    duration: float   # seconds above threshold
    mode: str


class BlinkDetector:
    MODES = ("eog", "beta")

    def __init__(self, fs, n_frontal, mode="eog", sensitivity=5.0, min_uv=60.0,
                 min_dur=0.04, max_dur=0.8, refractory=0.25, baseline_s=10.0):
        self.fs = fs
        self.n_frontal = n_frontal
        self.sensitivity = sensitivity  # multiples of noise (eog) or z units (beta)
        self.min_uv = min_uv
        self.min_dur = min_dur
        self.max_dur = max_dur
        self.refractory = refractory
        self.baseline_len = int(baseline_s * fs)
        self.fixed_threshold = None     # set by calibration; overrides adaptive threshold
        self.polarity = 1.0
        self.set_mode(mode)

    def set_mode(self, mode):
        assert mode in self.MODES
        self.mode = mode
        if mode == "eog":
            self.filter = StreamFilter(self.fs, self.n_frontal, low=0.2, high=8.0, order=2)
        else:
            self.filter = StreamFilter(self.fs, self.n_frontal, low=13.0, high=30.0, order=4)
        self._env = 0.0
        self._history = deque(maxlen=self.baseline_len)   # scores used for the adaptive baseline
        self._above = False
        self._start = 0
        self._peak = 0.0
        self._last_event_sample = -10 ** 9
        self._n = 0
        self._threshold = float("inf")
        self.fixed_threshold = None

    @property
    def threshold(self):
        return self._threshold

    def _update_threshold(self):
        if self.fixed_threshold is not None:
            self._threshold = self.fixed_threshold
            return
        if len(self._history) < self.fs:  # need ~1 s of data before detecting
            self._threshold = float("inf")
            return
        h = np.fromiter(self._history, float)
        med = np.median(h)
        noise = 1.4826 * np.median(np.abs(h - med)) + 1e-9
        if self.mode == "eog":
            self._threshold = max(self.min_uv, med + self.sensitivity * noise)
        else:
            self._threshold = self.sensitivity  # beta scores are already z-scores

    def _scores(self, frontal):
        y = self.filter(frontal).mean(axis=0)
        if self.mode == "eog":
            return self.polarity * y
        # beta: exponential moving RMS envelope (~80 ms), then z-score vs. baseline
        alpha = 1.0 / (0.08 * self.fs)
        env = np.empty_like(y)
        e = self._env
        for i, v in enumerate(y * y):
            e += alpha * (v - e)
            env[i] = e
        self._env = e
        env = np.sqrt(env)
        if len(self._history) < self.fs:
            self._history.extend(env)
            return np.zeros_like(env)
        h = np.fromiter(self._history, float)
        med = np.median(h)
        mad = 1.4826 * np.median(np.abs(h - med)) + 1e-9
        self._history.extend(env)
        return (env - med) / mad

    def process(self, frontal):
        """Feed raw frontal EEG (n_frontal, n). Returns (scores, list[BlinkEvent])."""
        frontal = np.atleast_2d(frontal)
        if frontal.shape[1] == 0:
            return np.array([]), []
        scores = self._scores(frontal)
        if self.mode == "eog":
            self._history.extend(scores)
        self._update_threshold()
        thr = self._threshold
        events = []
        now = time.time()
        n = len(scores)
        for i, s in enumerate(scores):
            k = self._n + i
            if not self._above:
                if s > thr and (k - self._last_event_sample) / self.fs > self.refractory:
                    self._above, self._start, self._peak = True, k, s
            else:
                self._peak = max(self._peak, s)
                if s < 0.5 * thr:  # hysteresis: blink ends when score falls well below threshold
                    self._above = False
                    dur = (k - self._start) / self.fs
                    if self.min_dur <= dur <= self.max_dur:
                        t = now - (n - i) / self.fs
                        events.append(BlinkEvent(t, float(self._peak), dur, self.mode))
                        self._last_event_sample = k
                elif (k - self._start) / self.fs > self.max_dur:
                    # too long to be a blink (movement, electrode pop); wait for it to settle
                    self._above = s > thr
                    if not self._above:
                        self._last_event_sample = k
        self._n += n
        return scores, events

    def calibrate(self, rest_scores, blink_scores):
        """Pick a fixed threshold halfway between resting noise and the blink peaks."""
        rest_scores, blink_scores = np.asarray(rest_scores), np.asarray(blink_scores)
        if self.mode == "eog" and -np.percentile(blink_scores, 0.5) > np.percentile(blink_scores, 99.5):
            # blinks show up as negative deflections on this headset: flip the sign
            self.polarity = -self.polarity
            rest_scores, blink_scores = -rest_scores, -blink_scores
        rest = np.percentile(rest_scores, 99.5)
        peaks = np.percentile(blink_scores, 99.5)
        if peaks <= rest * 1.5:
            raise ValueError("Blinks were not clearly larger than rest - check Fp1/Fp2 contact and try again.")
        self.fixed_threshold = float(rest + 0.5 * (peaks - rest))
        self._threshold = self.fixed_threshold
        return self.fixed_threshold


class BlinkTrigger:
    """Fires when ``required`` blinks happen within ``window_s`` seconds.

    People blink spontaneously ~15-20 times a minute, so requiring a deliberate
    double (or triple) blink stops the arm from scooping on every natural blink.
    """

    def __init__(self, required=2, window_s=1.2):
        self.required = required
        self.window_s = window_s
        self._times = deque()

    def feed(self, event: BlinkEvent):
        self._times.append(event.t)
        while self._times and event.t - self._times[0] > self.window_s:
            self._times.popleft()
        if len(self._times) >= self.required:
            self._times.clear()
            return True
        return False

    def pending(self):
        return len(self._times)

    def reset(self):
        self._times.clear()
