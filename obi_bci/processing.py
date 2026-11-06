"""Signal-processing helpers: streaming filters, ring buffer, spectra and band powers."""

import numpy as np
from scipy import signal

BANDS = {
    "Delta": (0.5, 4.0),
    "Theta": (4.0, 8.0),
    "Alpha": (8.0, 13.0),
    "Beta": (13.0, 30.0),
    "Gamma": (30.0, 45.0),
}


class StreamFilter:
    """Stateful IIR filter applied chunk-by-chunk to (channels, samples) arrays.

    Keeping the filter state between chunks means every sample is filtered exactly
    once, with no edge artifacts at chunk boundaries.
    """

    def __init__(self, fs, n_channels, low=None, high=None, notch=None, order=4):
        nyq = fs / 2.0
        if high is not None:
            high = min(high, 0.95 * nyq)
        sections = []
        if low and high:
            sections.append(signal.butter(order, [low, high], btype="bandpass", fs=fs, output="sos"))
        elif low:
            sections.append(signal.butter(order, low, btype="highpass", fs=fs, output="sos"))
        elif high:
            sections.append(signal.butter(order, high, btype="lowpass", fs=fs, output="sos"))
        if notch and notch < nyq:
            b, a = signal.iirnotch(notch, 30.0, fs)
            sections.append(signal.tf2sos(b, a))
        self.sos = np.vstack(sections) if sections else None
        self.n_channels = n_channels
        self.zi = None

    def __call__(self, x):
        x = np.atleast_2d(np.asarray(x, dtype=float))
        if self.sos is None or x.shape[1] == 0:
            return x
        if self.zi is None:
            # Start the filter at steady state for the first sample to avoid a huge
            # step transient from the board's DC offset.
            base = signal.sosfilt_zi(self.sos)
            self.zi = base[:, None, :] * x[:, 0][None, :, None]
        y, self.zi = signal.sosfilt(self.sos, x, axis=1, zi=self.zi)
        return y

    def reset(self):
        self.zi = None


class RingBuffer:
    """Fixed-length (channels, samples) buffer; newest samples at the right."""

    def __init__(self, n_channels, length):
        self.data = np.zeros((n_channels, length))
        self.filled = 0

    def append(self, chunk):
        chunk = np.atleast_2d(chunk)
        n = chunk.shape[1]
        if n == 0:
            return
        length = self.data.shape[1]
        if n >= length:
            self.data[:] = chunk[:, -length:]
        else:
            self.data = np.roll(self.data, -n, axis=1)
            self.data[:, -n:] = chunk
        self.filled = min(length, self.filled + n)

    def latest(self, n=None):
        n = self.filled if n is None else min(n, self.filled)
        return self.data[:, self.data.shape[1] - n:]


def spectrum(x, fs, max_freq=60.0):
    """Welch amplitude spectrum (uV/sqrt(Hz)) for each channel of a (channels, samples) array."""
    nperseg = min(x.shape[1], int(fs * 2))
    if nperseg < 16:
        return np.array([]), np.zeros((x.shape[0], 0))
    freqs, psd = signal.welch(x, fs=fs, nperseg=nperseg, axis=1)
    keep = freqs <= max_freq
    return freqs[keep], np.sqrt(psd[:, keep])


def band_powers(x, fs):
    """Average absolute band power (uV^2) per band, averaged across channels."""
    nperseg = min(x.shape[1], int(fs * 2))
    if nperseg < 16:
        return {name: 0.0 for name in BANDS}
    freqs, psd = signal.welch(x, fs=fs, nperseg=nperseg, axis=1)
    out = {}
    for name, (lo, hi) in BANDS.items():
        mask = (freqs >= lo) & (freqs < hi)
        out[name] = float(np.trapezoid(psd[:, mask], freqs[mask], axis=1).mean()) if mask.any() else 0.0
    return out
