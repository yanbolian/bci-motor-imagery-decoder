"""
Neural signal simulation for BCI motor imagery decoding.

Generates synthetic ECoG/EEG-like multichannel time series with realistic
spectral properties for three motor states:
    0 = rest
    1 = left-hand motor imagery
    2 = right-hand motor imagery

Neuroscience rationale
----------------------
Motor imagery (the mental rehearsal of movement without actual movement)
suppresses cortical oscillations in the mu (8-12 Hz) and beta (13-30 Hz)
bands over the sensorimotor cortex contralateral to the imagined hand.
This phenomenon is called Event-Related Desynchronization (ERD) and is
the primary neural signature exploited by motor BCI systems.

Reference: Pfurtscheller & Lopes da Silva (1999). Event-related EEG/MEG
synchronization and desynchronization. Clinical Neurophysiology, 110(11).
"""

import numpy as np
from typing import Tuple

# ── Default signal parameters ────────────────────────────────────────────────
SFREQ: int = 250          # Sampling frequency (Hz) — typical EEG/ECoG
N_CHANNELS: int = 8       # Simulated cortical electrodes
TRIAL_DURATION: float = 4.0  # Seconds per trial

CLASSES = {0: "rest", 1: "left_hand", 2: "right_hand"}


def pink_noise(
    n_channels: int,
    n_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate multichannel pink (1/f) noise via the FFT method.

    Pink noise power falls as 1/f, closely matching the background
    power spectrum of cortical local-field potentials and ECoG signals.
    This is a better background model than white noise for neural data.

    Returns
    -------
    ndarray, shape (n_channels, n_samples), unit-variance per channel.
    """
    white = rng.standard_normal((n_channels, n_samples))
    freqs = np.fft.rfftfreq(n_samples, d=1.0 / SFREQ)
    freqs[0] = 1.0  # avoid division by zero at DC
    fft_vals = np.fft.rfft(white, axis=1)
    fft_vals /= np.sqrt(freqs)[np.newaxis, :]
    pink = np.fft.irfft(fft_vals, n=n_samples, axis=1)
    return (pink / pink.std(axis=1, keepdims=True)).astype(np.float32)


def simulate_trial(
    class_label: int,
    sfreq: int = SFREQ,
    n_channels: int = N_CHANNELS,
    trial_duration: float = TRIAL_DURATION,
    noise_std: float = 1.0,
    rng: np.random.Generator = None,
) -> np.ndarray:
    """
    Simulate a single ECoG/EEG-like trial for motor imagery decoding.

    Signal model
    ------------
    Background : pink (1/f) noise — realistic cortical background spectrum.
    Oscillations: mu (~10 Hz) and beta (~20 Hz) rhythms, added to all channels.
    ERD         : During motor imagery the oscillatory amplitude is suppressed
                  (event-related desynchronisation) in the hemisphere
                  contralateral to the imagined hand.

    Channel layout:
        Channels 0-3  → left sensorimotor cortex (governs right hand)
        Channels 4-7  → right sensorimotor cortex (governs left hand)

    ERD scaling factor:
        rest         → left=1.0, right=1.0   (both hemispheres active)
        left imagery → right hemisphere ERD  (left=1.0, right=0.2)
        right imagery→ left hemisphere ERD   (left=0.2, right=1.0)

    Parameters
    ----------
    class_label : int
        0 = rest, 1 = left-hand imagery, 2 = right-hand imagery.
    sfreq : int
        Sampling frequency in Hz.
    n_channels : int
        Total number of simulated channels.
    trial_duration : float
        Trial length in seconds.
    noise_std : float
        Standard deviation of additive white measurement noise (μV).
    rng : np.random.Generator
        Random number generator (for reproducibility).

    Returns
    -------
    signal : ndarray, shape (n_channels, n_samples), dtype float32.
        Simulated multichannel neural signal in μV-scale units.
    """
    if rng is None:
        rng = np.random.default_rng()

    n_samples = int(sfreq * trial_duration)
    t = np.arange(n_samples) / sfreq

    # Background: pink noise (~3 μV RMS, scaled for detectable ERD)
    signal = pink_noise(n_channels, n_samples, rng) * 3.0

    # Random phase offsets per trial — prevents EEGNet from memorising
    # a fixed waveform fingerprint instead of the ERD power pattern.
    # Real mu/beta oscillations have no fixed phase relationship to trial onset.
    phase_mu   = rng.uniform(0, 2 * np.pi)
    phase_beta = rng.uniform(0, 2 * np.pi)

    # Amplitude jitter: ±30% trial-to-trial variability in oscillation strength,
    # reflecting natural fluctuations in thalamocortical drive.
    amp_mu   = 5.0 * rng.uniform(0.7, 1.3)
    amp_beta = 2.5 * rng.uniform(0.7, 1.3)

    mu_osc   = amp_mu   * np.sin(2 * np.pi * 10 * t + phase_mu)
    beta_osc = amp_beta * np.sin(2 * np.pi * 20 * t + phase_beta)

    # ERD scaling: imagined movement → reduced amplitude in contralateral hemisphere.
    # Added ±0.05 variability in ERD depth across trials (real ERD depth fluctuates).
    if class_label == 0:    # rest: both hemispheres show full rhythms
        erd_left  = float(rng.uniform(0.9, 1.1))
        erd_right = float(rng.uniform(0.9, 1.1))
    elif class_label == 1:  # left hand → right motor cortex ERD
        erd_left  = float(rng.uniform(0.9, 1.1))
        erd_right = float(rng.uniform(0.15, 0.30))   # ERD depth varies ~15–30%
    else:                   # right hand → left motor cortex ERD
        erd_left  = float(rng.uniform(0.15, 0.30))
        erd_right = float(rng.uniform(0.9, 1.1))

    half = n_channels // 2
    osc = mu_osc + beta_osc
    signal[:half] += erd_left * osc   # left hemisphere channels
    signal[half:] += erd_right * osc  # right hemisphere channels

    # Additive measurement noise (electrode, amplifier)
    signal += rng.standard_normal((n_channels, n_samples)).astype(np.float32) * noise_std

    return signal.astype(np.float32)


def generate_dataset(
    n_trials_per_class: int = 120,
    sfreq: int = SFREQ,
    n_channels: int = N_CHANNELS,
    trial_duration: float = TRIAL_DURATION,
    noise_std: float = 1.0,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a balanced, shuffled dataset of simulated BCI motor imagery trials.

    Parameters
    ----------
    n_trials_per_class : int
        Trials per class. Total trials = n_trials_per_class × 3.
    sfreq : int
        Sampling frequency in Hz.
    n_channels : int
        Number of simulated ECoG channels.
    trial_duration : float
        Duration of each trial in seconds.
    noise_std : float
        Noise level. Higher → harder decoding problem. Try 0.5–3.0.
    seed : int
        Random seed for full reproducibility.

    Returns
    -------
    X : ndarray, shape (n_trials, n_channels, n_samples)
        Raw neural data.
    y : ndarray, shape (n_trials,), dtype int64
        Class labels (0 = rest, 1 = left, 2 = right).
    """
    rng = np.random.default_rng(seed)
    X, y = [], []

    for label in range(len(CLASSES)):
        for _ in range(n_trials_per_class):
            trial = simulate_trial(
                class_label=label,
                sfreq=sfreq,
                n_channels=n_channels,
                trial_duration=trial_duration,
                noise_std=noise_std,
                rng=rng,
            )
            X.append(trial)
            y.append(label)

    X = np.stack(X)                          # (n_trials, n_channels, n_samples)
    y = np.array(y, dtype=np.int64)

    # Shuffle so classes are interleaved (important for batch training)
    perm = rng.permutation(len(y))
    return X[perm], y[perm]
