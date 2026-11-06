"""Offline tests for the blink detector using simulated EEG (run: python -m pytest tests)."""

import numpy as np
import pytest

from obi_bci.blink import BlinkDetector, BlinkTrigger, BlinkEvent
from obi_bci.board import blink_waveform

FS = 250


def make_eeg(seconds, blink_times=(), seed=0, emg=False):
    rng = np.random.default_rng(seed)
    n = int(seconds * FS)
    t = np.arange(n) / FS
    base = 15 * rng.standard_normal((2, n)) + 10 * np.sin(2 * np.pi * 10 * t) + 5000  # DC offset like Cyton
    wave = blink_waveform(FS)
    for bt in blink_times:
        i = int(bt * FS)
        base[:, i:i + len(wave)] += wave
        if emg:
            base[:, i:i + len(wave)] += 60 * rng.standard_normal((2, len(wave)))
    return base


def run(det, eeg, chunk=8):
    events = []
    for i in range(0, eeg.shape[1], chunk):
        _, ev = det.process(eeg[:, i:i + chunk])
        events.extend(ev)
    return events


@pytest.mark.parametrize("mode,emg", [("eog", False), ("beta", True)])
def test_detects_each_blink(mode, emg):
    blinks = [4.0, 6.0, 6.6, 9.0, 12.0]
    det = BlinkDetector(FS, 2, mode=mode)
    events = run(det, make_eeg(14, blinks, emg=emg))
    assert len(events) == len(blinks)


@pytest.mark.parametrize("mode", ["eog", "beta"])
def test_no_false_positives_at_rest(mode):
    det = BlinkDetector(FS, 2, mode=mode)
    assert run(det, make_eeg(30, seed=3)) == []


def test_double_blink_trigger():
    trig = BlinkTrigger(required=2, window_s=1.2)
    ev = lambda t: BlinkEvent(t, 200, 0.3, "eog")
    assert not trig.feed(ev(0.0))
    assert not trig.feed(ev(3.0))   # too far apart: natural blinks
    assert trig.feed(ev(3.5))       # deliberate double blink
    assert not trig.feed(ev(3.9))   # trigger resets after firing


def test_calibration_sets_threshold():
    det = BlinkDetector(FS, 2)
    rest, _ = det.process(make_eeg(5, seed=1))
    blink, _ = det.process(make_eeg(5, [1, 2.5, 4], seed=2))
    thr = det.calibrate(rest, blink)
    assert rest.max() < thr < blink.max()


def test_calibration_learns_negative_polarity():
    det = BlinkDetector(FS, 2)
    rest, _ = det.process(-make_eeg(5, seed=1))
    blink, _ = det.process(-make_eeg(5, [1, 2.5, 4], seed=2))
    det.calibrate(rest, blink)
    assert det.polarity == -1.0
    assert len(run(det, -make_eeg(8, [3, 5], seed=4))) == 2
