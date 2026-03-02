#!/usr/bin/env python3
"""
BCI Motor Imagery Decoder — Real PhysioNet EEG Data
====================================================

Runs the same pipeline as run_pipeline.py but on real EEG from the
PhysioNet EEG Motor Movement/Imagery Dataset (EEGMMIDB).

MNE downloads the data automatically (~7 MB per subject).
Total download for subjects 1-5: ~35 MB.

Run:
    python run_real_data.py                  # subjects 1-5 (default)
    python run_real_data.py --subjects 1     # single subject
    python run_real_data.py --subjects 1 2 3 # multiple subjects

Key difference from simulated data
------------------------------------
Real EEG is much noisier and more variable than simulation:
- Accuracy typically drops to 55-75% for single subjects (LDA).
- Accuracy varies a lot across subjects (range ~45-85%).
- Cross-subject generalisation is harder (different electrode impedances,
  anatomy, and signal-to-noise across individuals).
- This is the real challenge in applied BCI research.
"""

import sys
import argparse
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, cross_val_score

from bci_decoder.real_data  import load_physionet_subject, SENSORIMOTOR_CHANNELS
from bci_decoder.preprocess import preprocess_dataset
from bci_decoder.features   import log_band_power, windowed_log_band_power
from bci_decoder.models     import build_lda, build_svm, train_eegnet
from bci_decoder.realtime   import RealTimeDecoder
from bci_decoder.evaluate   import (
    print_report, plot_confusion_matrix,
    plot_training_curves, plot_band_power_spectrum,
    plot_latency_distribution,
)

import torch
from sklearn.model_selection import train_test_split

SFREQ   = 250
SEED    = 42
OUT_DIR = Path("results_real")
OUT_DIR.mkdir(exist_ok=True)


def banner(text: str) -> None:
    print(f"\n{'-' * 60}")
    print(f"  {text}")
    print(f"{'-' * 60}")


def run_subject(subject: int) -> None:
    subj_dir = OUT_DIR / f"subject_{subject:03d}"
    subj_dir.mkdir(exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"  Subject {subject:03d} — PhysioNet EEGMMIDB")
    print(f"{'=' * 60}")

    # ── 1. Load real EEG ──────────────────────────────────────────
    banner("Step 1 / 5 — Load real EEG (PhysioNet auto-download)")

    X_raw, y = load_physionet_subject(
        subject=subject,
        sfreq_resample=SFREQ,
        tmin=0.0,
        tmax=4.0,
        channels=SENSORIMOTOR_CHANNELS,
    )
    n_trials, n_ch, n_samp = X_raw.shape
    print(f"  Shape: {X_raw.shape}  ({n_samp/SFREQ:.1f} s @ {SFREQ} Hz)")

    # ── 2. Preprocess ─────────────────────────────────────────────
    banner("Step 2 / 5 — Preprocess (bandpass 1-40 Hz + CAR)")

    X_proc = preprocess_dataset(X_raw, sfreq=SFREQ)
    plot_band_power_spectrum(
        X_proc, y, sfreq=SFREQ,
        save_path=str(subj_dir / "psd_by_class.png"),
    )

    # ── 3. Feature extraction ──────────────────────────────────────
    banner("Step 3 / 5 — Log band power features")

    X_feat = log_band_power(X_proc, sfreq=SFREQ)
    print(f"  Feature matrix: {X_feat.shape}")

    # ── 4. Cross-validated evaluation ─────────────────────────────
    # Use k-fold CV instead of a single train/test split because real
    # BCI datasets have fewer trials than simulation (typically 30-90
    # trials per class per subject).
    banner("Step 4 / 5 — Cross-validated classifiers (5-fold)")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    for name, clf in [("LDA", build_lda()), ("SVM", build_svm())]:
        scores = cross_val_score(clf, X_feat, y, cv=cv, scoring="accuracy")
        print(f"\n  [{name}]")
        print(f"    CV accuracy : {scores.mean():.3f} +/- {scores.std():.3f}")
        print(f"    Per fold    : {[f'{s:.2f}' for s in scores]}")

    # Also do a single split for confusion matrix + EEGNet
    idx_tr, idx_te = train_test_split(
        np.arange(n_trials), test_size=0.25, stratify=y, random_state=SEED,
    )
    X_tr_raw, X_te_raw = X_proc[idx_tr], X_proc[idx_te]
    X_tr_feat, X_te_feat = X_feat[idx_tr], X_feat[idx_te]
    y_tr, y_te = y[idx_tr], y[idx_te]

    lda = build_lda()
    lda.fit(X_tr_feat, y_tr)
    plot_confusion_matrix(
        y_te, lda.predict(X_te_feat), model_name="LDA",
        save_path=str(subj_dir / "cm_lda.png"),
    )

    # ── 5. EEGNet ─────────────────────────────────────────────────
    banner("Step 5 / 5 — EEGNet (PyTorch)")

    model, train_losses, val_accs = train_eegnet(
        X_tr_raw, y_tr, X_te_raw, y_te,
        n_classes=3, sfreq=SFREQ, n_epochs=80, batch_size=16, lr=5e-4,
    )
    plot_training_curves(
        train_losses, val_accs,
        save_path=str(subj_dir / "eegnet_training.png"),
    )

    model.eval()
    with torch.no_grad():
        logits  = model(torch.tensor(X_te_raw, dtype=torch.float32))
        y_pred  = logits.argmax(dim=1).numpy()
    print_report(y_te, y_pred, model_name="EEGNet")
    plot_confusion_matrix(
        y_te, y_pred, model_name="EEGNet",
        save_path=str(subj_dir / "cm_eegnet.png"),
    )

    print(f"\n  Results saved to: {subj_dir.resolve()}/")


def main() -> None:
    parser = argparse.ArgumentParser(description="BCI pipeline on real PhysioNet EEG")
    parser.add_argument(
        "--subjects", type=int, nargs="+", default=[1, 2, 3, 4, 5],
        help="Subject IDs to process (1-109). Default: 1 2 3 4 5",
    )
    args = parser.parse_args()

    print(f"Processing subjects: {args.subjects}")
    for subj in args.subjects:
        run_subject(subj)

    print(f"\n{'=' * 60}")
    print(f"  Done.  All results in: {OUT_DIR.resolve()}/")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
