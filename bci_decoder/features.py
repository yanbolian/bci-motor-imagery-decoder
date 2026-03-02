"""
Frequency-domain feature extraction for BCI motor imagery decoding.

Feature: Log Band Power
-----------------------
For each trial and each EEG/ECoG channel, we compute the mean power
spectral density (PSD) within five canonical frequency bands, then
take the log to compress the dynamic range.

Why these bands?
    delta (1-4 Hz)  : artefact / slow drift indicator
    theta (4-8 Hz)  : cognitive/memory-related; less relevant for motor BCI
    mu    (8-12 Hz) : sensorimotor idle rhythm — primary ERD band for hand BCI
    beta  (13-30 Hz): sensorimotor beta rhythm — secondary ERD band
    gamma (30-45 Hz): high-frequency broadband activity; emerging BCI feature

The mu and beta bands are the most discriminative: during motor imagery,
ERD causes a clear power drop in contralateral channels, which classifiers
(LDA, SVM, EEGNet) learn to separate across the three classes.

Feature vector size: n_channels × n_bands = 8 × 5 = 40 per trial.

Reference: Pfurtscheller & Neuper (2001). Motor imagery and direct
brain-computer communication. Proceedings of the IEEE, 89(7).
"""

import numpy as np
from scipy import signal as sp_signal
from typing import Dict, List, Tuple

# Standard EEG/ECoG frequency bands
FREQ_BANDS: Dict[str, Tuple[float, float]] = {
    "delta": (1,  4),
    "theta": (4,  8),
    "mu":    (8,  12),   # key motor BCI band
    "beta":  (13, 30),   # key motor BCI band
    "gamma": (30, 45),
}


def log_band_power(
    data: np.ndarray,
    sfreq: int,
    bands: Dict[str, Tuple[float, float]] = None,
    nperseg: int = 128,
) -> np.ndarray:
    """
    Extract log band power features from multichannel neural data.

    Uses Welch's method (averaged periodogram) for robust PSD estimation.
    Welch averages over overlapping segments to reduce variance in the
    PSD estimate — important for short (1–4 s) neural epochs.

    Parameters
    ----------
    data : ndarray, shape (n_trials, n_channels, n_samples)
        Preprocessed neural data.
    sfreq : int
        Sampling frequency in Hz.
    bands : dict, optional
        {band_name: (low_hz, high_hz)}. Defaults to FREQ_BANDS.
    nperseg : int
        Welch segment length in samples. Controls frequency resolution:
        frequency resolution = sfreq / nperseg.
        At sfreq=250, nperseg=128 → resolution ≈ 1.95 Hz.

    Returns
    -------
    features : ndarray, shape (n_trials, n_channels * n_bands), dtype float32.
        Log band power, flattened across channels and bands.
        Order: [ch0_delta, ch0_theta, ch0_mu, ch0_beta, ch0_gamma,
                ch1_delta, ..., ch7_gamma]
    """
    if bands is None:
        bands = FREQ_BANDS

    n_trials, n_channels, n_samples = data.shape
    nperseg = min(nperseg, n_samples)

    freqs, psd = sp_signal.welch(data, fs=sfreq, nperseg=nperseg, axis=-1)
    # psd shape: (n_trials, n_channels, n_freqs)

    band_powers = []
    for band_name, (low, high) in bands.items():
        mask = (freqs >= low) & (freqs < high)
        mean_power = psd[:, :, mask].mean(axis=-1)     # (n_trials, n_channels)
        log_power = np.log(mean_power + 1e-10)          # log-transform
        band_powers.append(log_power)

    # Stack: (n_trials, n_channels, n_bands) → flatten → (n_trials, n_channels*n_bands)
    features = np.stack(band_powers, axis=-1)           # (n_trials, n_channels, n_bands)
    return features.reshape(n_trials, -1).astype(np.float32)


def windowed_log_band_power(
    X: np.ndarray,
    y: np.ndarray,
    sfreq: int,
    window_samples: int = 250,
    step_samples: int = 125,
    bands: Dict[str, Tuple[float, float]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract log band power from sliding windows over each trial.

    Use this to prepare training data for a real-time decoder whose
    window length is shorter than the full trial duration.  The classifier
    must be trained on windows of the same length it will receive at
    inference time — otherwise feature statistics (mean, variance) will
    differ and performance degrades.

    Parameters
    ----------
    X : ndarray, shape (n_trials, n_channels, n_trial_samples)
        Full-length preprocessed trials.
    y : ndarray, shape (n_trials,)
        Trial labels.
    sfreq : int
        Sampling frequency.
    window_samples : int
        Length of each window in samples.
    step_samples : int
        Step between windows in samples (50% overlap by default).
    bands : dict, optional
        Frequency bands. Defaults to FREQ_BANDS.

    Returns
    -------
    X_wins : ndarray, shape (n_windows, n_features)
    y_wins : ndarray, shape (n_windows,)
        Each window inherits its parent trial's label.
    """
    X_wins, y_wins = [], []
    for trial, label in zip(X, y):
        for start in range(0, trial.shape[1] - window_samples + 1, step_samples):
            win = trial[np.newaxis, :, start:start + window_samples]   # (1, ch, T)
            feats = log_band_power(win, sfreq=sfreq, bands=bands)
            X_wins.append(feats[0])
            y_wins.append(label)
    return np.stack(X_wins), np.array(y_wins, dtype=np.int64)


def feature_names(
    n_channels: int = 8,
    bands: Dict[str, Tuple[float, float]] = None,
) -> List[str]:
    """
    Return descriptive names for each feature in the log band power vector.

    Useful for interpreting model weights (e.g., which channel/band
    contributes most to left vs right decoding).

    Returns
    -------
    list of str, length = n_channels * len(bands).
    """
    if bands is None:
        bands = FREQ_BANDS
    names = []
    for ch in range(n_channels):
        for band in bands:
            names.append(f"ch{ch:02d}_{band}")
    return names
