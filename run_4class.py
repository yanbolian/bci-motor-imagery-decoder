#!/usr/bin/env python3
"""
4-Class BCI Motor Imagery — Left / Right / Feet / Rest
=======================================================

This script extends the 3-class pipeline by adding a 4th motor imagery
command: both-feet imagery.  It answers two concrete questions:

  1. Does adding a 4th class improve overall decoder throughput (ITR),
     despite the harder classification problem?

  2. Which class confusions dominate?  (The 4×4 confusion matrix reveals
     that left vs right is the hard pair, while feet is naturally separable
     due to its distinct spatial ERD pattern at the vertex.)

Run
---
    python run_4class.py                     # subjects 1-3 (default)
    python run_4class.py --subjects 1 2 3 4 5

Output (results_4class/<subject_NNN>/)
--------------------------------------
    psd_by_class.png          PSD for all 4 classes (feet shows vertex ERD)
    cm_lda_4class.png         4×4 LDA confusion matrix
    cm_eegnet_4class.png      4×4 EEGNet confusion matrix
    eegnet_training_4class.png Training curves (chance line at 25%)

Output (results_4class/)
    itr_comparison.png        3-class vs 4-class ITR per subject

Neuroscience background
-----------------------
The foot motor representation sits at the vertex (top of the head) along
the interhemispheric fissure.  During feet motor imagery:

  • ERD occurs at Cz / FCz (vertex channels) — midline suppression
  • C3 and C4 (lateral channels) show little change

This contrasts with hand imagery:
  • Left-hand imagery  → ERD at C4 (contralateral right hemisphere)
  • Right-hand imagery → ERD at C3 (contralateral left hemisphere)

The orthogonal spatial patterns make feet naturally the most separable
class.  Classification errors concentrate on left-hand vs right-hand
confusion, which is the inherently hard problem (both lateral, opposite
hemispheres but same frequency bands).

ITR scaling with number of classes
-----------------------------------
ITR = B × M  where B = information per decision, M = decisions per minute.

For a perfect decoder (100% accuracy):
    3 classes:  B = log₂(3) = 1.58 bits  →  ITR = 23.8 bits/min @ 15 dec/min
    4 classes:  B = log₂(4) = 2.00 bits  →  ITR = 30.0 bits/min @ 15 dec/min

So the 4-class interface has +26% higher theoretical throughput.
In practice, accuracy drops with more classes, so the actual ITR gain
is smaller — but the question is whether the gain outweighs the loss.
This script answers that question empirically for each subject.

Synchron relevance
------------------
A 4-class interface (rest / left / right / feet) maps directly to the
type of BCI control demonstrated in Synchron's published clinical trials
(Oxley et al., 2021, 2023):

    left hand  → navigate previous / scroll up
    right hand → navigate next / scroll down
    feet       → select / confirm  (high accuracy due to vertex ERD)
    rest       → idle (decoder outputs no command)

This gives a cursor-plus-click interface analogous to how Stentrode
patients interact with standard Windows and iOS applications.

Learning notes
--------------
  • The pipeline is IDENTICAL to run_real_data.py up to feature extraction.
    Only the loader (load_physionet_subject_4class) and n_classes=4 change.
    This illustrates a key design principle: keep the pipeline modular so
    new tasks/classes require minimal code changes.

  • The 4-class ITR comparison is the central plot.  Notice that ITR can
    increase OR decrease with more classes depending on the subject — some
    subjects' ERD is strong enough that the extra class costs nothing, while
    others see a net throughput loss.  This per-subject analysis is exactly
    the kind of result that guides clinical deployment decisions.
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

import torch

from bci_decoder.real_data  import (
    load_physionet_subject,
    load_physionet_subject_4class,
    SENSORIMOTOR_CHANNELS,
    SENSORIMOTOR_CHANNELS_4CLASS,
)
from bci_decoder.preprocess import preprocess_dataset
from bci_decoder.features   import log_band_power
from bci_decoder.models     import build_lda, build_svm, train_eegnet
from bci_decoder.evaluate   import (
    information_transfer_rate,
    print_report,
    plot_confusion_matrix,
    plot_training_curves,
    plot_band_power_spectrum,
    plot_itr_comparison,
    CLASS_NAMES,
    CLASS_NAMES_4,
)

# ── Constants ──────────────────────────────────────────────────────────────────

SFREQ       = 250           # Hz — target sampling rate after MNE resample
SEED        = 42            # fixed random seed for reproducibility
TRIAL_SEC   = 4.0           # seconds per trial → 15 decisions/min
DEC_PER_MIN = 60.0 / TRIAL_SEC
OUT_DIR     = Path("results_4class")
OUT_DIR.mkdir(exist_ok=True)


# ── Helpers ────────────────────────────────────────────────────────────────────

def banner(text: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {text}")
    print(f"{'=' * 60}")


# ── Per-subject pipeline ───────────────────────────────────────────────────────

def run_subject(subject: int) -> dict:
    """
    Run the full 4-class pipeline for one subject.

    Returns a dict of key metrics for the cross-subject summary table.
    """
    subj_dir = OUT_DIR / f"subject_{subject:03d}"
    subj_dir.mkdir(exist_ok=True)

    print(f"\n{'─' * 60}")
    print(f"  Subject {subject:03d} — 4-class motor imagery")
    print(f"{'─' * 60}")

    # ── Step 1: Load 4-class EEG ───────────────────────────────────────────
    #
    # load_physionet_subject_4class combines two PhysioNet run groups:
    #   Hand runs (4, 8, 12)  → rest (T0) + left (T1) + right (T2)
    #   Feet runs (6, 10, 14) → feet (T2 only)
    # Result: ~15 trials per class, ~60 trials total.

    banner("Step 1 / 5 — Load 4-class EEG (rest / left / right / feet)")

    X_raw, y = load_physionet_subject_4class(
        subject=subject,
        sfreq_resample=SFREQ,
        tmin=0.0,
        tmax=4.0,
        channels=SENSORIMOTOR_CHANNELS_4CLASS,
    )
    n_trials, n_ch, n_samp = X_raw.shape
    print(f"  Shape: {X_raw.shape}  ({n_samp / SFREQ:.1f} s @ {SFREQ} Hz)")

    # Also load 3-class data from the same subject so we can compare ITR
    # on the same individual.  We use the standard 3-class channels here
    # so the comparison is fair (both use 8 channels).
    print(f"\n  Also loading 3-class data for ITR comparison ...")
    X_raw_3, y_3 = load_physionet_subject(
        subject=subject,
        sfreq_resample=SFREQ,
        tmin=0.0,
        tmax=4.0,
        channels=SENSORIMOTOR_CHANNELS,
    )

    # ── Step 2: Preprocess ─────────────────────────────────────────────────
    #
    # Bandpass 1-40 Hz + Common Average Reference.
    # Same pipeline as 3-class — no changes needed.

    banner("Step 2 / 5 — Preprocess (bandpass 1-40 Hz + CAR)")

    X_proc   = preprocess_dataset(X_raw, sfreq=SFREQ)
    X_proc_3 = preprocess_dataset(X_raw_3, sfreq=SFREQ)

    # PSD plot — 4 classes now.
    # The key thing to look for: feet line should dip at Cz (right panel
    # will show midline channels), while left/right dip on opposite laterals.
    plot_band_power_spectrum(
        X_proc, y, sfreq=SFREQ,
        save_path=str(subj_dir / "psd_by_class.png"),
        class_names=CLASS_NAMES_4,
    )

    # ── Step 3: Feature extraction ─────────────────────────────────────────

    banner("Step 3 / 5 — Log band power features")

    X_feat   = log_band_power(X_proc,   sfreq=SFREQ)
    X_feat_3 = log_band_power(X_proc_3, sfreq=SFREQ)
    print(f"  4-class feature matrix: {X_feat.shape}")
    print(f"  3-class feature matrix: {X_feat_3.shape}")

    # ── Step 4: Cross-validated classifiers ────────────────────────────────
    #
    # Use 5-fold stratified CV so every trial contributes to the estimate
    # (important with only ~60 trials for 4-class).

    banner("Step 4 / 5 — 5-fold CV classifiers (LDA + SVM)")

    cv      = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    results = {"subject": subject}

    # 3-class LDA for comparison
    scores_3 = cross_val_score(build_lda(), X_feat_3, y_3, cv=cv,
                               scoring="accuracy")
    acc_3    = float(scores_3.mean())
    itr_3    = information_transfer_rate(acc_3, n_classes=3,
                                         decisions_per_min=DEC_PER_MIN)
    print(f"\n  [LDA — 3-class baseline]")
    print(f"    CV accuracy   : {acc_3:.3f} ± {scores_3.std():.3f}")
    print(f"    ITR           : {itr_3:.1f} bits/min")

    results["acc_3cls"] = round(acc_3, 4)
    results["itr_3cls"] = round(itr_3, 2)

    # 4-class LDA
    scores_4 = cross_val_score(build_lda(), X_feat, y, cv=cv,
                               scoring="accuracy")
    acc_4    = float(scores_4.mean())
    itr_4    = information_transfer_rate(acc_4, n_classes=4,
                                         decisions_per_min=DEC_PER_MIN)
    print(f"\n  [LDA — 4-class (+ feet)]")
    print(f"    CV accuracy   : {acc_4:.3f} ± {scores_4.std():.3f}")
    print(f"    ITR           : {itr_4:.1f} bits/min")

    delta_itr = itr_4 - itr_3
    sign      = "+" if delta_itr >= 0 else ""
    print(f"\n  ITR change from 3→4 class: {sign}{delta_itr:.1f} bits/min  "
          f"({'gain' if delta_itr >= 0 else 'loss'})")

    results["acc_4cls"] = round(acc_4, 4)
    results["itr_4cls"] = round(itr_4, 2)

    # Confusion matrix on a held-out split
    idx_tr, idx_te = train_test_split(
        np.arange(n_trials), test_size=0.25, stratify=y, random_state=SEED,
    )
    X_tr_feat, X_te_feat = X_feat[idx_tr], X_feat[idx_te]
    X_tr_raw,  X_te_raw  = X_proc[idx_tr], X_proc[idx_te]
    y_tr, y_te           = y[idx_tr], y[idx_te]

    lda = build_lda()
    lda.fit(X_tr_feat, y_tr)
    plot_confusion_matrix(
        y_te, lda.predict(X_te_feat), model_name="LDA (4-class)",
        save_path=str(subj_dir / "cm_lda_4class.png"),
        class_names=CLASS_NAMES_4,
    )

    # ── Step 5: EEGNet ─────────────────────────────────────────────────────
    #
    # EEGNet is told n_classes=4 so its final linear layer has 4 outputs.
    # Everything else (temporal conv, depthwise spatial conv) is unchanged.
    # Chance level is now 25% (shown as dashed line in training plot).

    banner("Step 5 / 5 — EEGNet (PyTorch, 4 output classes)")

    model, train_losses, val_accs = train_eegnet(
        X_tr_raw, y_tr, X_te_raw, y_te,
        n_classes=4, sfreq=SFREQ, n_epochs=80, batch_size=16, lr=5e-4,
    )
    plot_training_curves(
        train_losses, val_accs,
        save_path=str(subj_dir / "eegnet_training_4class.png"),
        n_classes=4,      # chance line at 25% instead of 33%
    )

    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X_te_raw, dtype=torch.float32))
        y_pred = logits.argmax(dim=1).numpy()

    print_report(y_te, y_pred, model_name="EEGNet (4-class)", n_classes=4)
    plot_confusion_matrix(
        y_te, y_pred, model_name="EEGNet (4-class)",
        save_path=str(subj_dir / "cm_eegnet_4class.png"),
        class_names=CLASS_NAMES_4,
    )

    eegnet_acc = float((y_pred == y_te).mean())
    results["eegnet_acc_4cls"] = round(eegnet_acc, 4)
    results["eegnet_itr_4cls"] = round(
        information_transfer_rate(eegnet_acc, n_classes=4,
                                  decisions_per_min=DEC_PER_MIN), 2
    )

    print(f"\n  Results saved to: {subj_dir.resolve()}/")
    return results


# ── Main ───────────────────────────────────────────────────────────────────────

def main(subjects: list) -> None:

    banner("4-Class BCI Motor Imagery — rest / left / right / feet")

    all_results = []
    for subj in subjects:
        try:
            r = run_subject(subj)
            all_results.append(r)
        except Exception as e:
            print(f"\n  Warning: subject {subj} failed — {e}.  Skipping.")

    if not all_results:
        print("No subjects completed successfully.")
        return

    # ── Cross-subject summary ──────────────────────────────────────────────

    banner("Summary — 3-class vs 4-class across subjects")

    df = pd.DataFrame(all_results)

    print(df.to_string(index=False))
    print(f"\n  Mean ITR (3-class LDA): {df['itr_3cls'].mean():.1f} bits/min")
    print(f"  Mean ITR (4-class LDA): {df['itr_4cls'].mean():.1f} bits/min")
    mean_delta = df["itr_4cls"].mean() - df["itr_3cls"].mean()
    sign = "+" if mean_delta >= 0 else ""
    print(f"  Mean ITR change:        {sign}{mean_delta:.1f} bits/min")

    print(
        f"\n  Interpretation:"
        f"\n    • 4-class ITR > 3-class ITR for a subject → feet imagery"
        f"\n      provides enough extra information to outweigh the harder"
        f"\n      classification problem."
        f"\n    • 4-class ITR < 3-class ITR → adding feet costs more in"
        f"\n      accuracy than it gains in vocabulary — that subject may"
        f"\n      produce weak foot ERD (a known individual difference)."
    )

    # ITR comparison plot
    plot_itr_comparison(
        df,
        save_path=str(OUT_DIR / "itr_comparison.png"),
    )

    # Save CSV
    df.to_csv(OUT_DIR / "summary_4class.csv", index=False)

    print(f"\n{'=' * 60}")
    print(f"  Files saved to: {OUT_DIR.resolve()}/")
    print(f"{'=' * 60}")
    print(f"\n  Files generated:")
    for f in sorted(OUT_DIR.iterdir()):
        if f.is_file():
            print(f"    {f.name}")
        else:
            print(f"    {f.name}/")


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="4-class BCI motor imagery: left / right / feet / rest."
    )
    parser.add_argument(
        "--subjects", type=int, nargs="+", default=[1, 2, 3],
        help="Subject IDs to process (1–109).  Default: 1 2 3",
    )
    args = parser.parse_args()
    print(f"Processing subjects: {args.subjects}")
    main(args.subjects)
