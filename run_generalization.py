#!/usr/bin/env python3
"""
BCI Generalisation Experiments — PhysioNet EEG (subjects 1–5)
==============================================================

This script extends run_real_data.py by asking three harder questions:

  1. Cross-subject generalisation (LOSO)
     Can the model decode a new participant with zero calibration data?

  2. Session drift + z-score correction
     Does signal drift across time degrade accuracy, and can per-session
     z-score normalisation fix it?

  3. Behavioral outcome correlation
     Does higher decoder ITR translate to better task performance?

These experiments directly address what Synchron (and BCI research more
broadly) actually needs to know before deploying a decoder in the real world.

Run
---
    python run_generalization.py                # subjects 1-5 (default)
    python run_generalization.py --subjects 1 2 3 4 5

Output (results_generalization/)
---------------------------------
  subject_summary.png          Per-subject within-session accuracy bar chart
  loso_results.png             Within-subject vs LOSO grouped bar chart
  session_drift.png            Drift condition comparison bar chart
  behavioral_correlation.png   ITR vs task performance scatter plot
  summary_table.csv            Full numerical results table

Learning notes
--------------
As you read through this script, notice:

  • The pipeline is the SAME as run_real_data.py up to feature extraction.
    Reusing bci_decoder.* modules means each experiment adds only the
    evaluation logic, not new preprocessing or feature code.

  • Features are extracted ONCE per subject and stored in a list.
    This avoids recomputing the Welch PSD (the slowest step) multiple
    times across experiments.

  • StratifiedKFold is used instead of a single train/test split because
    real EEG datasets have few trials per subject (~45 here).  With k-fold,
    every trial is used for both training and testing, giving a lower-
    variance accuracy estimate.

  • The LOSO loop is O(n_subjects²) in classifier fits but only O(n_subjects)
    in data loading — loading is the bottleneck (MNE + resampling).

  • Pandas DataFrames are used throughout for results because they make it
    trivial to merge, print, and save tabular results — a skill that comes
    up constantly in applied research.
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, cross_val_score

from bci_decoder.real_data      import load_physionet_subject, SENSORIMOTOR_CHANNELS
from bci_decoder.preprocess     import preprocess_dataset
from bci_decoder.features       import log_band_power
from bci_decoder.models         import build_lda
from bci_decoder.evaluate       import (
    information_transfer_rate,
    plot_subject_accuracy_summary,
    plot_loso_vs_within,
    plot_session_drift,
    plot_behavioral_correlation,
)
from bci_decoder.generalization import (
    leave_one_subject_out,
    session_drift_experiment,
    simulate_behavioral_scores,
)

# ── Constants ──────────────────────────────────────────────────────────────────

SFREQ   = 250           # target sampling frequency after MNE resample
SEED    = 42            # global random seed for reproducibility
OUT_DIR = Path("results_generalization")
OUT_DIR.mkdir(exist_ok=True)

# ITR is computed assuming 4 s trials → 15 decisions / minute
TRIAL_SEC     = 4.0
DEC_PER_MIN   = 60.0 / TRIAL_SEC


# ── Helpers ────────────────────────────────────────────────────────────────────

def banner(text: str) -> None:
    """Print a section header."""
    print(f"\n{'=' * 60}")
    print(f"  {text}")
    print(f"{'=' * 60}")


def sub_banner(text: str) -> None:
    """Print a sub-section header."""
    print(f"\n  {'─' * 56}")
    print(f"  {text}")
    print(f"  {'─' * 56}")


# ── Main pipeline ──────────────────────────────────────────────────────────────

def main(subjects: list) -> None:

    # ── Step 1: Load all subjects ──────────────────────────────────────────────
    #
    # MNE downloads PhysioNet data automatically on first run (~7 MB/subject).
    # Subsequent runs use the local cache (fast).
    #
    # We store:
    #   X_proc_list[i]  – preprocessed EEG trials (n_trials, n_ch, n_samp)
    #   X_feat_list[i]  – log band power features  (n_trials, n_features)
    #   y_list[i]       – class labels              (n_trials,)
    #   loaded_subjects – which subject IDs actually loaded (some may fail)

    banner("Step 1 / 5 — Load & preprocess all subjects")

    X_proc_list     = []
    X_feat_list     = []
    y_list          = []
    loaded_subjects = []

    for subj in subjects:
        print(f"\n  Subject {subj:03d}")
        try:
            # load_physionet_subject returns (n_trials, n_ch, n_samp) in μV
            X_raw, y = load_physionet_subject(
                subject=subj,
                sfreq_resample=SFREQ,
                tmin=0.0,
                tmax=4.0,
                channels=SENSORIMOTOR_CHANNELS,
            )

            # Bandpass 1–40 Hz + Common Average Reference
            X_proc = preprocess_dataset(X_raw, sfreq=SFREQ)

            # Log band power: (n_trials, n_channels * n_bands) = (n_trials, 40)
            # Extracting features here (once) avoids repeated Welch PSD calls
            # in the LOSO loop below.
            X_feat = log_band_power(X_proc, sfreq=SFREQ)

            X_proc_list.append(X_proc)
            X_feat_list.append(X_feat)
            y_list.append(y)
            loaded_subjects.append(subj)

        except Exception as e:
            print(f"  Warning: subject {subj} failed — {e}.  Skipping.")

    if len(loaded_subjects) < 2:
        print("Need at least 2 subjects for generalisation experiments.  Exiting.")
        return

    print(f"\n  Loaded {len(loaded_subjects)} subjects: {loaded_subjects}")

    # ── Step 2: Within-subject evaluation (5-fold CV) ─────────────────────────
    #
    # Why 5-fold CV instead of a single 75/25 split?
    # ─────────────────────────────────────────────────
    # With only ~45 trials per subject (PhysioNet 3 runs × ~15 trials each),
    # a single split gives a high-variance accuracy estimate — the result can
    # swing ±10 pp depending on which trials end up in the test set.
    # 5-fold CV uses every trial as a test trial exactly once, producing a
    # much more reliable estimate of true generalisation performance.
    # This is especially important when comparing across subjects.

    banner("Step 2 / 5 — Within-subject 5-fold CV evaluation (LDA)")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    within_rows = []

    for i, (X_feat, y, subj) in enumerate(zip(X_feat_list, y_list, loaded_subjects)):

        # cross_val_score handles the split, fit, and predict loop automatically
        scores = cross_val_score(build_lda(), X_feat, y,
                                 cv=cv, scoring="accuracy")
        acc = float(scores.mean())
        itr = information_transfer_rate(acc, n_classes=3,
                                        decisions_per_min=DEC_PER_MIN)

        print(f"  Subject {subj:03d}:  acc={acc:.3f} ± {scores.std():.3f}  "
              f"ITR={itr:.1f} bits/min  (N={len(y)} trials)")

        within_rows.append({
            "subject":         subj,
            "n_trials":        len(y),
            "within_accuracy": round(acc, 4),
            "within_itr":      round(itr, 2),
        })

    within_df = pd.DataFrame(within_rows)

    # Summary statistics — show inter-subject variability prominently
    print(f"\n  Across subjects:")
    print(f"    Mean accuracy : {within_df['within_accuracy'].mean():.3f}")
    print(f"    Std accuracy  : {within_df['within_accuracy'].std():.3f}")
    print(f"    Range         : {within_df['within_accuracy'].min():.3f} – "
          f"{within_df['within_accuracy'].max():.3f}")
    print(f"\n    (Range > 0.10 is normal for real EEG; in simulation it was ~0.02)")

    # Plot: per-subject accuracy summary
    plot_subject_accuracy_summary(
        within_df,
        save_path=str(OUT_DIR / "subject_summary.png"),
    )

    # ── Step 3: Cross-subject generalisation (LOSO) ───────────────────────────
    #
    # LOSO = Leave-One-Subject-Out.
    # For each subject s, train LDA on pooled data from all other subjects,
    # then evaluate on s without any fine-tuning.
    #
    # This answers: "What accuracy would a brand-new participant achieve on
    # day one, using only a pre-trained population model?"
    #
    # The gap between within-subject and LOSO accuracy is the "calibration
    # benefit" — the accuracy gained by collecting subject-specific data.

    banner("Step 3 / 5 — Cross-subject generalisation (LOSO)")

    print("\n  Training on N−1 subjects → evaluating on the held-out subject.")
    print("  No data from the test subject is seen during training.\n")

    loso_df = leave_one_subject_out(X_feat_list, y_list, loaded_subjects)

    mean_within = within_df["within_accuracy"].mean()
    mean_loso   = loso_df["loso_accuracy"].mean()

    print(f"\n  Mean within-subject accuracy : {mean_within:.3f}")
    print(f"  Mean LOSO accuracy           : {mean_loso:.3f}")
    print(f"  Calibration benefit (gap)    : {mean_within - mean_loso:+.3f} pp")
    print(
        f"\n  Interpretation: each new participant can expect ~{mean_loso:.0%} "
        f"accuracy with zero calibration."
    )

    plot_loso_vs_within(
        loso_df, within_df,
        save_path=str(OUT_DIR / "loso_results.png"),
    )

    # ── Step 4: Session drift + z-score correction ─────────────────────────────
    #
    # This experiment uses subject 1 (index 0) as a concrete example.
    #
    # Session drift is one of the most common failure modes in long-term BCI
    # deployment.  After calibration in the morning, the decoder may degrade
    # by afternoon due to electrode impedance changes, fatigue, or medication
    # timing.  The experiment simulates this and shows that per-session
    # z-score normalisation — which requires only the feature values, not
    # class labels — substantially recovers performance.

    banner("Step 4 / 5 — Session drift & z-score correction")

    print(f"\n  Using subject {loaded_subjects[0]:03d} as example.")
    print( "  Simulating realistic per-channel power offset + amplitude scaling.")
    print( "  Then comparing three decoding conditions.\n")

    drift_results = session_drift_experiment(
        X_feat_list[0],
        y_list[0],
        split_frac=0.60,    # 60% of trials = session 1 (training)
        drift_scale=0.50,   # drift magnitude = 50% of feature std
        seed=SEED,
    )

    plot_session_drift(
        drift_results,
        save_path=str(OUT_DIR / "session_drift.png"),
    )

    # ── Step 5: Behavioral outcome correlation ─────────────────────────────────
    #
    # "Task score" is simulated here, but the principle is real:
    # in Synchron's clinical trials, participants use the BCI to type,
    # control a cursor, or navigate apps.  The number of successful actions
    # per minute is a direct patient-facing outcome.
    #
    # The correlation between ITR and task score closes the loop from
    # signal processing → clinical benefit.
    #
    # In this simulation task_score ≈ 0.55 × ITR + 2.0 + noise.
    # The slope (0.55) is consistent with published P300 speller data
    # (Sellers et al., 2010: each additional bit/min → ~0.5 letters/min).

    banner("Step 5 / 5 — Behavioral outcome correlation")

    print("\n  Simulating downstream task performance (actions/min).")
    print("  task_score ≈ 0.55 × ITR + 2.0 + noise")
    print("  (linear in ITR, consistent with published BCI task data)\n")

    corr_df, r, p = simulate_behavioral_scores(within_df, seed=SEED)

    plot_behavioral_correlation(
        corr_df, r, p,
        save_path=str(OUT_DIR / "behavioral_correlation.png"),
    )

    # ── Summary table ──────────────────────────────────────────────────────────

    banner("Summary")

    # Merge all per-subject results into one table
    summary = (
        within_df
        .merge(loso_df[["subject", "loso_accuracy", "loso_itr"]], on="subject")
        .merge(corr_df[["subject", "task_score"]], on="subject")
    )

    # Rename for clarity
    summary = summary.rename(columns={
        "n_trials":        "N trials",
        "within_accuracy": "Within acc",
        "within_itr":      "Within ITR",
        "loso_accuracy":   "LOSO acc",
        "loso_itr":        "LOSO ITR",
        "task_score":      "Task score",
    })

    print(summary.to_string(index=False))

    # Also print session drift summary
    print(f"\n  Session drift results (subject {loaded_subjects[0]:03d}):")
    print(f"    Within-session acc : {drift_results['acc_within']:.3f}  "
          f"({drift_results['itr_within']:.1f} b/m)")
    print(f"    Drifted (no norm)  : {drift_results['acc_no_norm']:.3f}  "
          f"({drift_results['itr_no_norm']:.1f} b/m)")
    print(f"    Z-score corrected  : {drift_results['acc_z_score']:.3f}  "
          f"({drift_results['itr_z_score']:.1f} b/m)")

    # Save CSV
    summary.to_csv(OUT_DIR / "summary_table.csv", index=False)

    print(f"\n{'=' * 60}")
    print(f"  Plots and results saved to: {OUT_DIR.resolve()}/")
    print(f"{'=' * 60}")
    print(f"\n  Files generated:")
    for f in sorted(OUT_DIR.iterdir()):
        print(f"    {f.name}")


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="BCI generalisation experiments on PhysioNet EEG."
    )
    parser.add_argument(
        "--subjects", type=int, nargs="+", default=[1, 2, 3, 4, 5],
        help="Subject IDs to process (1–109).  Default: 1 2 3 4 5",
    )
    args = parser.parse_args()

    print(f"Processing subjects: {args.subjects}")
    main(args.subjects)
