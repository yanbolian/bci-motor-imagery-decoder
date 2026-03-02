"""
Generalization experiments for BCI motor imagery decoding.

Why generalization matters
--------------------------
The pipeline in run_real_data.py evaluates a model trained and tested on
the *same* subject within the *same* session.  That is the best-case
scenario.  In real deployed BCI the questions that matter are:

  1. Cross-subject:  Can the model work for a brand-new user who has never
     contributed training data?  (zero calibration)
  2. Cross-session:  Does the model degrade across time — different day,
     fatigue, impedance drift — and can we cheaply fix that?
  3. Outcome relevance: Does better ITR actually translate to better task
     performance for the user?

This module provides three experiments that directly address each question.

Experiment 1 — Leave-One-Subject-Out (LOSO)
--------------------------------------------
Train LDA on N-1 subjects → test on the held-out subject.

Why it matters for Synchron:
  After implant, participants cannot immediately generate large amounts of
  calibration data.  A pre-trained model that already achieves above-chance
  decoding on a new user drastically reduces the initial calibration burden.
  LOSO accuracy represents the "floor" performance before any fine-tuning.

Interpretation guide:
  within-subject accuracy >> LOSO accuracy
      → model is highly participant-specific.  Need per-user calibration.
  within-subject ≈ LOSO accuracy
      → strong cross-participant transfer.  Clinically valuable.
  Typical published gap (LDA, EEG motor imagery): 10–20 pp
  (Blankertz et al., 2011, NeuroImage 51(4)).

Experiment 2 — Session drift + z-score correction
---------------------------------------------------
ECoG / EEG amplitudes drift between recording sessions due to:
  • Electrode impedance increase (tissue encapsulation in implants)
  • Fatigue-related beta suppression (sustained mental effort)
  • Pharmacological shifts (morning vs afternoon medication)
  • Amplifier thermal drift and gain variation

We simulate drift by adding per-channel power offsets and scaling to a
held-out "session 2", then show that *per-session z-score normalisation*
(entirely unsupervised — no class labels needed) substantially recovers
accuracy.

Key insight for deployment:
  A naive approach — apply the session-1 classifier directly to session-2
  data — degrades sharply.  Standardising each new session's features to
  zero mean / unit variance before decoding is cheap and label-free, and
  is standard practice in clinical BCI pipelines.

  More advanced corrections (adaptive LDA, Riemannian alignment) exist but
  z-score is the simplest and should be tried first.
  References: Shenoy et al. (2006) J. Neural Eng.; Vidaurre et al. (2011).

Experiment 3 — Behavioral outcome correlation
---------------------------------------------
BCI papers report accuracy and ITR (bits/min), but users care about whether
they can type faster, browse the web, or call for help.  This experiment
simulates a downstream task performance score and shows its correlation
with ITR across subjects.

Theoretical grounding:
  If the decoder is the bottleneck, task throughput should scale linearly
  with ITR (Wolpaw et al., 2000).  The correlation quantifies how much of
  the variability in task performance is explained by decoder quality alone.
  In real BCI studies (e.g., Sellers et al., 2010, P300 speller),
  r > 0.7 between ITR and typing speed is common.

References
----------
Blankertz et al. (2011). Neurophysiological predictor of SMR-based BCI
    performance. NeuroImage, 51(4), 1303–1309.
Shenoy et al. (2006). Towards datasource-adaptive brain-machine interfaces.
    J. Neural Eng., 3(1), R13–R23.
Sellers et al. (2010). A comparison of two spelling interfaces for a
    brain-computer interface. J. Neural Eng., 7(1), 016026.
Vidaurre et al. (2011). Towards a cure for BCI illiteracy.
    Brain Topography, 23(2), 194–198.
Wolpaw et al. (2000). Brain-computer interface technology: a review.
    IEEE Trans. Rehab. Eng., 8(2), 164–173.
"""

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple

from .evaluate import information_transfer_rate
from .models import build_lda

# Assumed trial duration for ITR calculation (PhysioNet epochs are 4 s)
_TRIAL_SEC      = 4.0
_DEC_PER_MIN    = 60.0 / _TRIAL_SEC    # 15 decisions / minute


# ── Experiment 1: Leave-One-Subject-Out (LOSO) ────────────────────────────────

def leave_one_subject_out(
    X_feat_list: List[np.ndarray],
    y_list: List[np.ndarray],
    subject_ids: List[int],
) -> pd.DataFrame:
    """
    Leave-One-Subject-Out (LOSO) cross-subject generalisation.

    For each subject s in subject_ids:
        - Pool all trials from subjects ≠ s  →  training set
        - Evaluate on all trials from subject s  →  test set

    No data from the test subject is used during training.  This is the
    most realistic simulation of deploying a pre-trained model to a new
    participant on day one of BCI use.

    Parameters
    ----------
    X_feat_list : list of ndarray, each shape (n_trials_s, n_features)
        Pre-extracted log band power features, one array per subject.
        Must be in the same order as subject_ids.
    y_list : list of ndarray, each shape (n_trials_s,)
        Class labels (0=rest, 1=left, 2=right) for each subject.
    subject_ids : list of int
        Subject identifiers matching the two lists above.

    Returns
    -------
    pd.DataFrame with columns:
        subject         – subject ID used as test
        n_train         – total training trials (from all other subjects)
        n_test          – number of test trials for this subject
        loso_accuracy   – fraction correctly decoded
        loso_itr        – Information Transfer Rate in bits/min

    Notes
    -----
    LDA is the classifier of choice here because:
    - It has no hyperparameters that could be tuned towards specific
      subjects during model selection, keeping the comparison fair.
    - It is the most common BCI baseline classifier, making the results
      directly comparable to published literature.
    """
    rows = []
    n_subjects = len(subject_ids)

    for i, test_id in enumerate(subject_ids):
        # ── Build training set from all OTHER subjects ─────────────────────
        X_train = np.concatenate(
            [X_feat_list[j] for j in range(n_subjects) if j != i], axis=0
        )
        y_train = np.concatenate(
            [y_list[j]      for j in range(n_subjects) if j != i], axis=0
        )

        X_test = X_feat_list[i]
        y_test = y_list[i]

        # ── Train and evaluate ─────────────────────────────────────────────
        clf = build_lda()
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        acc = float((y_pred == y_test).mean())
        itr = information_transfer_rate(acc, n_classes=3,
                                        decisions_per_min=_DEC_PER_MIN)

        rows.append({
            "subject":        test_id,
            "n_train":        int(len(y_train)),
            "n_test":         int(len(y_test)),
            "loso_accuracy":  round(acc, 4),
            "loso_itr":       round(itr, 2),
        })

        print(f"  LOSO subject {test_id:03d} (test):  "
              f"acc={acc:.3f}  ITR={itr:.1f} bits/min  "
              f"[trained on {len(y_train)} trials from other subjects]")

    return pd.DataFrame(rows)


# ── Experiment 2: Session drift + z-score correction ──────────────────────────

def session_drift_experiment(
    X_feat: np.ndarray,
    y: np.ndarray,
    split_frac: float = 0.60,
    drift_scale: float = 0.5,
    seed: int = 42,
) -> Dict[str, float]:
    """
    Simulate between-session signal drift and demonstrate z-score mitigation.

    Pipeline
    --------
    1. Split trials into session 1 (first split_frac) and session 2 (rest).
    2. Add realistic per-channel drift to session 2 features:
           drifted = raw * scale + offset + noise
       where scale and offset are drawn randomly per feature but are
       FIXED across trials (systematic drift, not trial-level noise).
    3. Condition A — No normalisation:
           Train LDA on session 1, predict drifted session 2.
           The LDA pipeline's StandardScaler uses session-1 statistics, so
           the additive offset is NOT removed → accuracy degrades.
    4. Condition B — Per-session z-score:
           Fit a fresh StandardScaler on session 2 data (unsupervised,
           no labels used).  This subtracts the session-2 mean and divides
           by session-2 std, cancelling both the additive offset and the
           multiplicative scaling.  Then decode with a separately trained
           (z-scored session 1) LDA.
    5. Also compute a within-session CV baseline (no drift, no split penalty).

    Why z-score works for additive + multiplicative drift
    ─────────────────────────────────────────────────────
    If the true signal is X and the drifted signal is X' = a * X + b, then:
        z-score(X') = (X' - mean(X')) / std(X')
                    = (a*X + b - a*μ - b) / (a*σ)
                    = (X - μ) / σ  ≡  z-score(X)
    The drift parameters a and b vanish exactly.  In practice, class-specific
    drift (different drift per class) would resist this correction — that is
    when adaptive LDA or Riemannian alignment is needed.

    Parameters
    ----------
    X_feat : ndarray, shape (n_trials, n_features)
        Pre-extracted log band power features for ONE subject.
    y : ndarray, shape (n_trials,)
        Class labels.
    split_frac : float
        Fraction of trials treated as session 1 (training).
    drift_scale : float
        Controls drift magnitude relative to feature std.  drift_scale=0.5
        means the offset is drawn from [0.2, 0.8] × std of the features —
        a moderate but realistic drift level.
    seed : int
        Random seed for reproducible drift simulation.

    Returns
    -------
    dict with keys:
        acc_within   – within-session 5-fold CV accuracy (no drift, ideal case)
        itr_within   – corresponding ITR
        acc_no_norm  – accuracy after drift, no correction
        itr_no_norm  – corresponding ITR
        acc_z_score  – accuracy after drift + per-session z-score
        itr_z_score  – corresponding ITR
    """
    rng   = np.random.default_rng(seed)
    n     = len(y)
    split = int(n * split_frac)

    X_s1, y_s1 = X_feat[:split], y[:split]
    X_s2, y_s2 = X_feat[split:], y[split:]

    # ── Simulate drift on session 2 ────────────────────────────────────────
    # Per-feature drift parameters (same for every trial in the session)
    feat_std = X_s1.std(axis=0) + 1e-8

    offset = rng.uniform(0.2, 0.8, size=X_s1.shape[1]) * feat_std * drift_scale
    # scale: ±20% amplitude change per feature
    scale  = rng.uniform(0.8, 1.2, size=X_s1.shape[1])
    # Small residual trial-level noise
    noise  = rng.normal(0, 0.05, size=X_s2.shape) * feat_std[np.newaxis, :]

    X_s2_drifted = X_s2 * scale[np.newaxis, :] + offset[np.newaxis, :] + noise

    # ── Baseline: within-session 5-fold CV (no drift) ─────────────────────
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    scores_within = cross_val_score(build_lda(), X_s1, y_s1,
                                    cv=cv, scoring="accuracy")
    acc_within = float(scores_within.mean())
    itr_within = information_transfer_rate(acc_within, n_classes=3,
                                           decisions_per_min=_DEC_PER_MIN)

    # ── Condition A: No normalisation ─────────────────────────────────────
    # LDA pipeline's StandardScaler is fit on session 1 and applied to
    # session 2.  It subtracts the session-1 mean — but because session 2
    # has a DIFFERENT mean (due to drift), the residual offset remains.
    clf_no_norm = build_lda()
    clf_no_norm.fit(X_s1, y_s1)
    y_pred_nn  = clf_no_norm.predict(X_s2_drifted)
    acc_no_norm = float((y_pred_nn == y_s2).mean())
    itr_no_norm = information_transfer_rate(acc_no_norm, n_classes=3,
                                            decisions_per_min=_DEC_PER_MIN)

    # ── Condition B: Per-session z-score correction ────────────────────────
    # Fit an independent StandardScaler on session 2's drifted data.
    # This is purely unsupervised: we only need the raw feature values,
    # not the class labels — so it can be run in a deployed system before
    # the user starts a trial block.
    scaler_s1 = StandardScaler().fit(X_s1)
    scaler_s2 = StandardScaler().fit(X_s2_drifted)

    X_s1_z = scaler_s1.transform(X_s1)
    X_s2_z = scaler_s2.transform(X_s2_drifted)  # drift largely cancels

    # Use raw LDA (no pipeline scaler) since we pre-normalised manually
    lda_z = LinearDiscriminantAnalysis(solver="svd")
    lda_z.fit(X_s1_z, y_s1)
    y_pred_z = lda_z.predict(X_s2_z)
    acc_z   = float((y_pred_z == y_s2).mean())
    itr_z   = information_transfer_rate(acc_z, n_classes=3,
                                        decisions_per_min=_DEC_PER_MIN)

    results = {
        "acc_within":  round(acc_within,  4),
        "itr_within":  round(itr_within,  2),
        "acc_no_norm": round(acc_no_norm, 4),
        "itr_no_norm": round(itr_no_norm, 2),
        "acc_z_score": round(acc_z,       4),
        "itr_z_score": round(itr_z,       2),
    }

    print(f"  Session 1 (train): {len(y_s1)} trials")
    print(f"  Session 2 (test):  {len(y_s2)} trials  [drift applied]")
    print(f"  Within-session CV (no drift): acc={acc_within:.3f}  "
          f"ITR={itr_within:.1f} bits/min")
    print(f"  No normalisation (drifted):   acc={acc_no_norm:.3f}  "
          f"ITR={itr_no_norm:.1f} bits/min")
    print(f"  Z-score corrected:            acc={acc_z:.3f}  "
          f"ITR={itr_z:.1f} bits/min")

    return results


# ── Experiment 3: Behavioral outcome correlation ───────────────────────────────

def simulate_behavioral_scores(
    within_subject_df: pd.DataFrame,
    actions_per_bit: float = 0.55,
    noise_std: float = 0.8,
    seed: int = 42,
) -> Tuple[pd.DataFrame, float, float]:
    """
    Simulate downstream task performance and correlate with decoder ITR.

    In a real BCI research session, participants perform goal-directed tasks
    after the decoder is calibrated:  typing letters with a P300 speller,
    moving a cursor to targets, selecting icons in a communication app.
    "Task performance" is often reported as actions per minute or completion
    time per action.

    This function generates a simulated task score using the linear model:

        task_score = actions_per_bit × ITR + baseline + ε

    where ε ~ N(0, noise_std) captures real-world variability — fatigue,
    learning effects, individual differences in motor imagery skill, and
    interface usability factors that are independent of raw decoder accuracy.

    Why linear in ITR?
    ──────────────────
    Shannon's channel capacity theorem implies that if the BCI channel is
    the bottleneck, command throughput (commands/min) is bounded by ITR.
    Empirically, Sellers et al. (2010) found r > 0.7 between P300 speller
    ITR and offline typing speed across participants.  The linear model is
    a principled first approximation.

    Parameters
    ----------
    within_subject_df : pd.DataFrame
        Must contain columns 'subject' and 'within_itr'.
    actions_per_bit : float
        Slope of the task_score ~ ITR regression.
        0.55 means each additional bit/min of ITR yields ~0.55 more
        actions/min — consistent with published P300 speller data.
    noise_std : float
        Standard deviation of inter-subject variability beyond ITR.
        Set to 0.8 actions/min for moderate scatter.
    seed : int
        Random seed for reproducible noise.

    Returns
    -------
    df : pd.DataFrame
        Copy of within_subject_df with 'task_score' column added.
    r : float
        Pearson correlation coefficient (ITR vs task_score).
    p : float
        Two-tailed p-value of the Pearson correlation.
    """
    rng = np.random.default_rng(seed)
    itr_vals = within_subject_df["within_itr"].values

    noise      = rng.normal(0, noise_std, size=len(itr_vals))
    task_score = actions_per_bit * itr_vals + 2.0 + noise
    task_score = np.clip(task_score, 0, None)   # scores cannot be negative

    df = within_subject_df.copy()
    df["task_score"] = np.round(task_score, 2)

    r, p = pearsonr(itr_vals, task_score)
    print(f"  Pearson r={r:.3f}  p={p:.3f}  "
          f"(ITR explains {r**2*100:.0f}% of task-score variance)")

    return df, float(r), float(p)
