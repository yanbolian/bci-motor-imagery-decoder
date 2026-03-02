"""
Neural signal preprocessing for BCI motor imagery decoding.

Pipeline
--------
1. Bandpass filter (1–40 Hz, zero-phase Butterworth)
   - Removes DC drift and slow artefacts (< 1 Hz)
   - Removes line noise and high-frequency muscle artefacts (> 40 Hz)

2. Common Average Reference (CAR)
   - Subtracts the spatial mean across channels at each time point
   - Attenuates spatially diffuse noise (amplifier drift, reference electrode)
   - Sharpens focal cortical signals — standard for ECoG analysis

Both operations are linear and invertible, preserving the frequency-domain
content we care about (mu/beta ERD in 8–30 Hz).
"""

import numpy as np
from scipy import signal as sp_signal
from typing import Tuple


def bandpass_filter(
    data: np.ndarray,
    low_hz: float,
    high_hz: float,
    sfreq: int,
    order: int = 4,
) -> np.ndarray:
    """
    Apply a zero-phase Butterworth bandpass filter.

    Zero-phase (forward-backward via sosfiltfilt) filtering avoids phase
    distortion, which would smear event timing. Effective filter order is
    2 × `order` due to the forward-backward pass.

    Parameters
    ----------
    data : ndarray, shape (..., n_samples)
        Input signal. Filtering is applied along the last axis, so this
        works for single trials (n_channels, n_samples) or batches
        (n_trials, n_channels, n_samples).
    low_hz : float
        Lower cutoff frequency in Hz.
    high_hz : float
        Upper cutoff frequency in Hz.
    sfreq : int
        Sampling frequency in Hz.
    order : int
        One-pass filter order (effective order = 2× due to sosfiltfilt).

    Returns
    -------
    ndarray, same shape as `data`, dtype float32.
    """
    nyq = sfreq / 2.0
    sos = sp_signal.butter(
        order,
        [low_hz / nyq, high_hz / nyq],
        btype="band",
        output="sos",
    )
    return sp_signal.sosfiltfilt(sos, data, axis=-1).astype(np.float32)


def common_average_reference(data: np.ndarray) -> np.ndarray:
    """
    Apply Common Average Reference (CAR) spatial filter.

    CAR subtracts the instantaneous mean across channels:
        x_car[ch, t] = x[ch, t] - mean_over_channels(x[:, t])

    This is equivalent to a spatial high-pass filter that attenuates
    signals common to all channels (reference drifts, widespread artefacts)
    while preserving channel-specific neural signals.

    Parameters
    ----------
    data : ndarray, shape (..., n_channels, n_samples)
        The channel axis is second-to-last; averaging is performed over it.

    Returns
    -------
    ndarray, same shape as `data`, dtype float32.
    """
    return (data - data.mean(axis=-2, keepdims=True)).astype(np.float32)


def preprocess_dataset(
    X: np.ndarray,
    sfreq: int = 250,
    bandpass: Tuple[float, float] = (1.0, 40.0),
    apply_car: bool = True,
) -> np.ndarray:
    """
    Preprocess a batch of neural trials.

    Applies: bandpass filter → (optional) common average reference.

    Parameters
    ----------
    X : ndarray, shape (n_trials, n_channels, n_samples)
        Raw neural data.
    sfreq : int
        Sampling frequency in Hz.
    bandpass : (low_hz, high_hz)
        Bandpass filter cutoff frequencies.
    apply_car : bool
        Whether to apply Common Average Reference.

    Returns
    -------
    X_proc : ndarray, shape (n_trials, n_channels, n_samples), dtype float32.
        Preprocessed data.
    """
    X_proc = bandpass_filter(X, bandpass[0], bandpass[1], sfreq)
    if apply_car:
        X_proc = common_average_reference(X_proc)
    return X_proc


def preprocess_window(
    window: np.ndarray,
    sfreq: int = 250,
    bandpass: Tuple[float, float] = (1.0, 40.0),
) -> np.ndarray:
    """
    Preprocess a single decoding window for real-time use.

    Parameters
    ----------
    window : ndarray, shape (1, n_channels, n_samples)
        Single trial / decode window with leading batch dimension.

    Returns
    -------
    ndarray, shape (1, n_channels, n_samples), dtype float32.
    """
    return preprocess_dataset(window, sfreq=sfreq, bandpass=bandpass)
