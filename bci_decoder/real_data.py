"""
Real EEG data loaders for BCI motor imagery decoding.

Supported datasets
------------------
1. PhysioNet EEG Motor Movement/Imagery Dataset (EEGMMIDB)
   - Schalk et al. (2004). BCI2000: a general-purpose BCI system.
     IEEE Trans. Biomed. Eng., 51(6), 1034-1043.
   - 109 subjects, 64 EEG channels, 160 Hz.
   - 3-class: rest, left-hand imagery, right-hand imagery.
   - 4-class: rest, left-hand, right-hand, feet imagery.
   - Download: MNE downloads automatically on first use.
   - URL: https://physionet.org/content/eegmmidb/1.0.0/

2. BCI Competition IV Dataset 2a (instructions only — manual download)
   - Brunner et al. (2008). BCI Competition 2008 – Graz data set A.
   - 9 subjects, 22 EEG channels, 250 Hz, 4 classes.
   - Download: https://www.bbci.de/competition/iv/

Usage
-----
    from bci_decoder.real_data import load_physionet_subject
    X, y = load_physionet_subject(subject=1, sfreq_resample=250)

    # 4-class (adds feet imagery):
    from bci_decoder.real_data import load_physionet_subject_4class
    X, y = load_physionet_subject_4class(subject=1)

    # Then plug straight into the pipeline:
    from bci_decoder.preprocess import preprocess_dataset
    X_proc = preprocess_dataset(X, sfreq=250)
"""

import numpy as np
from typing import Tuple, List, Optional

# MNE is only imported inside functions so the rest of the package
# remains importable even if MNE is not installed.


# ── PhysioNet EEGMMIDB ────────────────────────────────────────────────────────

# PhysioNet run numbers for motor imagery tasks (see MNE docs):
#
#   Runs 4, 8, 12  →  left-hand vs right-hand imagery
#                       T0=rest, T1=left hand, T2=right hand
#
#   Runs 6, 10, 14 →  both fists vs both feet imagery
#                       T0=rest, T1=both fists, T2=both feet
#
# We use runs 4, 8, 12 for the 3-class problem.
# For 4-class we combine both sets: hand runs give rest/left/right,
# feet runs contribute the feet class (T2 only).
_PHYSIONET_IMAGERY_RUNS = [4, 8, 12]          # 3-class: left/right/rest
_PHYSIONET_HAND_RUNS    = [4, 8, 12]          # 4-class: hand subset
_PHYSIONET_FEET_RUNS    = [6, 10, 14]         # 4-class: feet subset

# Event IDs in the PhysioNet EEGMMIDB dataset
_PHYSIONET_EVENT_ID = {
    "rest":       1,
    "left_hand":  2,   # T1 = left-hand imagery
    "right_hand": 3,   # T2 = right-hand imagery
}

# Channels that broadly correspond to sensorimotor cortex (subset for speed)
# Full list of 64 channels available; these 8 cover C3/C4 region.
SENSORIMOTOR_CHANNELS = [
    "FC3", "FC4",  # pre-motor
    "C3",  "Cz",  "C4",   # primary motor
    "CP3", "CP4",  # post-motor / somatosensory
    "Pz",
]

# For 4-class decoding (adds feet imagery), include FCz at the vertex.
# The foot motor area sits at the top of the head along the midline, so
# Cz / FCz capture the feet ERD that lateral channels (C3, C4) miss.
SENSORIMOTOR_CHANNELS_4CLASS = [
    "FC3", "FCz", "FC4",   # pre-motor  (FCz = vertex, foot motor area)
    "C3",  "Cz",  "C4",   # primary motor (Cz = foot representation in homunculus)
    "CP3", "CP4",          # post-motor / somatosensory
]


def load_physionet_subject(
    subject: int = 1,
    sfreq_resample: int = 250,
    tmin: float = 0.0,
    tmax: float = 4.0,
    channels: Optional[List[str]] = None,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load one subject from the PhysioNet EEG Motor Movement/Imagery Dataset.

    MNE downloads the data automatically on first call (~7 MB per subject)
    and caches it locally. Subsequent calls are instant.

    The function returns epochs in exactly the same format as
    `generate_dataset()` so the rest of the pipeline is unchanged:
        X : (n_trials, n_channels, n_samples)
        y : (n_trials,)   0=rest, 1=left_hand, 2=right_hand

    Parameters
    ----------
    subject : int
        Subject number 1–109.
    sfreq_resample : int
        Target sampling frequency after resampling.
        Original PhysioNet data is at 160 Hz; 250 Hz is a common choice.
    tmin : float
        Epoch start (seconds relative to event onset).
    tmax : float
        Epoch end (seconds relative to event onset). tmax - tmin = trial length.
    channels : list of str, optional
        Channel subset to load. Defaults to SENSORIMOTOR_CHANNELS (8 channels).
        Pass None to keep all 64 channels.
    verbose : bool
        If True, show MNE loading messages.

    Returns
    -------
    X : ndarray, shape (n_trials, n_channels, n_samples), dtype float32
        Raw EEG in Volts (MNE convention). Multiply by 1e6 for μV.
    y : ndarray, shape (n_trials,), dtype int64
        Class labels: 0 = rest, 1 = left_hand, 2 = right_hand.

    Example
    -------
    >>> X, y = load_physionet_subject(subject=1)
    >>> print(X.shape, y.shape)
    (45, 8, 1000)  (45,)
    """
    try:
        import mne
    except ImportError:
        raise ImportError(
            "MNE-Python is required for real data loading.\n"
            "Install with:  pip install mne"
        )

    mne.set_log_level("WARNING" if not verbose else "INFO")

    if channels is None:
        channels = SENSORIMOTOR_CHANNELS

    # Download (if needed) and load raw EEG
    raw_files = mne.datasets.eegbci.load_data(
        subjects=subject,
        runs=_PHYSIONET_IMAGERY_RUNS,
        update_path=True,   # suppress interactive path prompt
        verbose=verbose,
    )
    raw = mne.io.concatenate_raws(
        [mne.io.read_raw_edf(f, preload=True, verbose=verbose) for f in raw_files]
    )

    # Standardise channel names (PhysioNet uses e.g. "Fc3." → "FC3")
    mne.datasets.eegbci.standardize(raw)

    # Pick the desired channels
    available = [ch for ch in channels if ch in raw.ch_names]
    if not available:
        raise ValueError(
            f"None of {channels} found in data. "
            f"Available channels: {raw.ch_names[:10]} ..."
        )
    raw.pick_channels(available)

    # Resample to target frequency
    if sfreq_resample != raw.info["sfreq"]:
        raw.resample(sfreq_resample, verbose=verbose)

    # Extract events and epoch
    events, _ = mne.events_from_annotations(raw, verbose=verbose)

    # Keep only the three motor imagery event types
    event_id = {k: v for k, v in _PHYSIONET_EVENT_ID.items()}
    epochs = mne.Epochs(
        raw,
        events,
        event_id=event_id,
        tmin=tmin,
        tmax=tmax,
        baseline=None,
        preload=True,
        verbose=verbose,
    )

    # Extract data as numpy array: (n_trials, n_channels, n_samples)
    X = epochs.get_data().astype(np.float32)   # Volts
    X *= 1e6                                    # convert to μV for consistency

    # Map event IDs to our label scheme: 0=rest, 1=left_hand, 2=right_hand
    label_map = {
        _PHYSIONET_EVENT_ID["rest"]:       0,
        _PHYSIONET_EVENT_ID["left_hand"]:  1,
        _PHYSIONET_EVENT_ID["right_hand"]: 2,
    }
    y = np.array([label_map[ev] for ev in epochs.events[:, 2]], dtype=np.int64)

    print(f"  Loaded subject {subject:03d}: "
          f"{X.shape[0]} trials x {X.shape[1]} channels x {X.shape[2]} samples "
          f"@ {sfreq_resample} Hz")
    print(f"  Classes: { {0: (y==0).sum(), 1: (y==1).sum(), 2: (y==2).sum()} }")

    return X, y


def load_physionet_multi_subject(
    subjects: List[int],
    sfreq_resample: int = 250,
    tmin: float = 0.0,
    tmax: float = 4.0,
    channels: Optional[List[str]] = None,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load multiple subjects and concatenate into a single dataset.

    Useful for cross-subject experiments or building larger training sets.

    Returns
    -------
    X : ndarray, shape (total_trials, n_channels, n_samples)
    y : ndarray, shape (total_trials,)
    subject_ids : ndarray, shape (total_trials,)
        Which subject each trial came from — needed for cross-subject
        leave-one-subject-out evaluation.
    """
    Xs, ys, sids = [], [], []
    for subj in subjects:
        try:
            X_s, y_s = load_physionet_subject(
                subject=subj,
                sfreq_resample=sfreq_resample,
                tmin=tmin,
                tmax=tmax,
                channels=channels,
                verbose=verbose,
            )
            Xs.append(X_s)
            ys.append(y_s)
            sids.append(np.full(len(y_s), subj, dtype=np.int64))
        except Exception as e:
            print(f"  Warning: subject {subj} failed ({e}), skipping.")

    X = np.concatenate(Xs, axis=0)
    y = np.concatenate(ys, axis=0)
    subject_ids = np.concatenate(sids, axis=0)
    print(f"\n  Total: {len(y)} trials from {len(Xs)} subjects")
    return X, y, subject_ids


def load_physionet_subject_4class(
    subject: int = 1,
    sfreq_resample: int = 250,
    tmin: float = 0.0,
    tmax: float = 4.0,
    channels: Optional[List[str]] = None,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load 4-class motor imagery data from the PhysioNet EEGMMIDB dataset.

    Classes
    -------
        0 = rest          (baseline, between imagery cues)
        1 = left-hand     (T1 in runs 4, 8, 12)
        2 = right-hand    (T2 in runs 4, 8, 12)
        3 = feet          (T2 in runs 6, 10, 14 — both feet imagery)

    Why feet as a 4th class?
    -------------------------
    The foot area of the motor homunculus sits at the vertex (midline, top
    of the head).  Feet motor imagery therefore produces an ERD centred at
    Cz / FCz, NOT at the lateral C3 / C4 channels activated by hand imagery.
    This distinct spatial pattern makes feet the most separable class in a
    4-class motor BCI — classification errors concentrate on left vs right,
    while feet is correctly identified more reliably.

    In Synchron's Stentrode system, a 4-class interface maps naturally to:
        left hand  → navigate previous / scroll up
        right hand → navigate next / scroll down
        feet       → select / confirm  (high accuracy due to vertex ERD)
        rest       → no command

    Loading strategy
    ----------------
    Two separate run groups are loaded and concatenated:
        Hand runs (4, 8, 12):  extract T0=rest, T1=left, T2=right → labels 0,1,2
        Feet runs (6, 10, 14): extract T2=feet only               → label 3
    This gives ~15 trials per class (balanced).

    Returns
    -------
    X : ndarray, shape (n_trials, n_channels, n_samples), dtype float32
        EEG in μV.
    y : ndarray, shape (n_trials,), dtype int64
        Labels: 0=rest, 1=left_hand, 2=right_hand, 3=feet.
    """
    try:
        import mne
    except ImportError:
        raise ImportError("MNE-Python is required.  pip install mne")

    mne.set_log_level("WARNING" if not verbose else "INFO")

    if channels is None:
        channels = SENSORIMOTOR_CHANNELS_4CLASS

    def _load_raw(runs):
        """Helper: download, concatenate, standardise, pick channels, resample."""
        files = mne.datasets.eegbci.load_data(
            subjects=subject, runs=runs, update_path=True, verbose=verbose,
        )
        raw = mne.io.concatenate_raws(
            [mne.io.read_raw_edf(f, preload=True, verbose=verbose) for f in files]
        )
        mne.datasets.eegbci.standardize(raw)
        available = [ch for ch in channels if ch in raw.ch_names]
        if not available:
            raise ValueError(
                f"None of {channels} found. Available: {raw.ch_names[:10]} ..."
            )
        raw.pick_channels(available)
        if sfreq_resample != raw.info["sfreq"]:
            raw.resample(sfreq_resample, verbose=verbose)
        return raw

    # ── Hand runs: rest + left + right ───────────────────────────────────────
    raw_hand = _load_raw(_PHYSIONET_HAND_RUNS)
    events_hand, _ = mne.events_from_annotations(raw_hand, verbose=verbose)
    epochs_hand = mne.Epochs(
        raw_hand, events_hand,
        event_id={"rest": 1, "left_hand": 2, "right_hand": 3},
        tmin=tmin, tmax=tmax, baseline=None, preload=True, verbose=verbose,
    )
    X_hand = epochs_hand.get_data().astype(np.float32) * 1e6
    label_map_hand = {1: 0, 2: 1, 3: 2}
    y_hand = np.array(
        [label_map_hand[ev] for ev in epochs_hand.events[:, 2]], dtype=np.int64
    )

    # ── Feet runs: feet only (T2 = both feet) ────────────────────────────────
    raw_feet = _load_raw(_PHYSIONET_FEET_RUNS)
    events_feet, _ = mne.events_from_annotations(raw_feet, verbose=verbose)
    epochs_feet = mne.Epochs(
        raw_feet, events_feet,
        event_id={"feet": 3},          # T2 in feet runs = both feet imagery
        tmin=tmin, tmax=tmax, baseline=None, preload=True, verbose=verbose,
    )
    X_feet = epochs_feet.get_data().astype(np.float32) * 1e6
    y_feet = np.full(len(X_feet), 3, dtype=np.int64)   # label 3 = feet

    # ── Concatenate and report ────────────────────────────────────────────────
    X = np.concatenate([X_hand, X_feet], axis=0)
    y = np.concatenate([y_hand, y_feet])

    counts = {name: int((y == i).sum())
              for i, name in enumerate(["rest", "left_hand", "right_hand", "feet"])}
    print(f"  Loaded subject {subject:03d} (4-class): "
          f"{X.shape[0]} trials × {X.shape[1]} ch × {X.shape[2]} samples "
          f"@ {sfreq_resample} Hz")
    print(f"  Classes: {counts}")

    return X, y


# ── BCI Competition IV 2a (manual download) ───────────────────────────────────

def load_bcic4_2a(
    gdf_path: str,
    sfreq_resample: int = 250,
    tmin: float = 0.5,
    tmax: float = 2.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a single GDF file from BCI Competition IV Dataset 2a.

    The dataset must be downloaded manually from:
        https://www.bbci.de/competition/iv/

    Parameters
    ----------
    gdf_path : str
        Path to a .gdf file (e.g., 'A01T.gdf' for subject 1 training set).
    sfreq_resample : int
        Target sampling frequency (original is 250 Hz).
    tmin, tmax : float
        Epoch window relative to cue onset (seconds).
        The cue appears at t=2 s; imagery runs from t=2 to t=6 s.
        tmin=0.5, tmax=2.5 gives the 2 s imagery window with 0.5 s offset.

    Returns
    -------
    X : ndarray, shape (n_trials, 22, n_samples), dtype float32
    y : ndarray, shape (n_trials,), dtype int64
        Labels: 0=left hand, 1=right hand, 2=feet, 3=tongue.

    Note
    ----
    This dataset has 4 classes; to use it with the 3-class pipeline,
    filter to classes 0 and 1 (left/right hand) after loading:
        mask = y <= 1
        X, y = X[mask], y[mask]
    """
    try:
        import mne
    except ImportError:
        raise ImportError("Install MNE: pip install mne")

    raw = mne.io.read_raw_gdf(gdf_path, preload=True, verbose=False)

    if sfreq_resample != raw.info["sfreq"]:
        raw.resample(sfreq_resample, verbose=False)

    events, event_id = mne.events_from_annotations(raw, verbose=False)

    # BCIC4 2a event IDs for motor imagery cues
    imagery_events = {
        "left_hand":  769,
        "right_hand": 770,
        "feet":       771,
        "tongue":     772,
    }

    epochs = mne.Epochs(
        raw, events, event_id=imagery_events,
        tmin=tmin, tmax=tmax,
        baseline=None, preload=True, verbose=False,
    )

    X = (epochs.get_data() * 1e6).astype(np.float32)
    label_map = {769: 0, 770: 1, 771: 2, 772: 3}
    y = np.array([label_map[e] for e in epochs.events[:, 2]], dtype=np.int64)

    return X, y
