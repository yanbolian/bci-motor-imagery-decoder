"""
BCI decoder evaluation: metrics and visualisations.

Metrics
-------
Accuracy        : standard classification accuracy.
Balanced accuracy: mean recall per class — robust to class imbalance.
ITR             : Information Transfer Rate (bits/min) — the standard BCI
                  throughput metric combining accuracy and decision speed.

Plots (saved to results/ as PNG)
---------------------------------
psd_by_class.png     : Mean PSD per class, showing mu/beta ERD.
cm_{model}.png       : Normalised confusion matrix per model.
eegnet_training.png  : Training loss + validation accuracy curves.
latency_dist.png     : Real-time decode latency histogram.
"""

import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive: save to files
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, confusion_matrix,
)
from scipy import signal as sp_signal
from scipy.stats import linregress
from typing import Dict, Optional

CLASS_NAMES = ["Rest", "Left hand", "Right hand"]


# -- Metrics -------------------------------------------------------------------

def information_transfer_rate(
    accuracy: float,
    n_classes: int,
    decisions_per_min: float,
) -> float:
    """
    Information Transfer Rate (ITR) in bits/minute.

    Quantifies BCI communication throughput as the product of information
    per decision (B) and decision rate (M):
        ITR = B × M   (bits/min)

    where B (Nykopp's formula, based on Shannon entropy):
        B = log2(N) + P·log2(P) + (1-P)·log2((1-P)/(N-1))

    Parameters
    ----------
    accuracy : float         Classification accuracy ∈ [0, 1].
    n_classes : int          Number of target classes.
    decisions_per_min : float  Decode rate (trials/min).

    Reference: Wolpaw et al. (2000). Brain-computer interface technology.
    IEEE Trans. Rehabil. Eng., 8(2).
    """
    p = np.clip(accuracy, 1e-9, 1.0 - 1e-9)
    n = n_classes
    if p >= 1.0:
        b = math.log2(n)
    elif p <= 1.0 / n:
        b = 0.0
    else:
        b = (math.log2(n)
             + p * math.log2(p)
             + (1 - p) * math.log2((1 - p) / (n - 1)))
    return max(0.0, b * decisions_per_min)


def print_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "",
    trial_duration_sec: float = 4.0,
) -> None:
    """Print accuracy, balanced accuracy, and ITR."""
    acc     = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    dpm     = 60.0 / trial_duration_sec      # decisions per minute
    itr     = information_transfer_rate(acc, n_classes=3, decisions_per_min=dpm)

    header = f"  [{model_name}]" if model_name else "  [Results]"
    print(f"\n{header}")
    print(f"    Accuracy          : {acc:.3f}  ({acc*100:.1f} %)")
    print(f"    Balanced accuracy : {bal_acc:.3f}")
    print(f"    ITR               : {itr:.1f} bits/min")


# -- Plots ---------------------------------------------------------------------

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "",
    save_path: Optional[str] = None,
) -> None:
    """Normalised confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred, normalize="true")
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        vmin=0, vmax=1, ax=ax,
    )
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(f"Confusion matrix — {model_name}" if model_name else "Confusion matrix")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"    Saved -> {save_path}")
    plt.close(fig)


def plot_training_curves(
    train_losses: list,
    val_accuracies: list,
    save_path: Optional[str] = None,
) -> None:
    """EEGNet training loss + validation accuracy over epochs."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(train_losses, color="steelblue", lw=1.5)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-entropy loss")
    axes[0].set_title("Training loss")

    axes[1].plot(val_accuracies, color="darkorange", lw=1.5)
    axes[1].axhline(1 / 3, ls="--", color="grey", label="Chance (33.3 %)")
    axes[1].set_ylim(0, 1)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Validation accuracy")
    axes[1].legend(fontsize=9)

    plt.suptitle("EEGNet training", fontsize=13)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"    Saved -> {save_path}")
    plt.close(fig)


def plot_band_power_spectrum(
    X: np.ndarray,
    y: np.ndarray,
    sfreq: int = 250,
    save_path: Optional[str] = None,
) -> None:
    """
    Mean PSD per class and hemisphere, highlighting mu/beta ERD.

    This is the primary diagnostic plot: if ERD is present in the simulated
    data, the mu (8-12 Hz) and beta (13-30 Hz) bands should show clearly
    lower power for the motor imagery classes compared to rest —
    and the suppression should be lateralised (contralateral hemisphere).
    """
    n_channels = X.shape[1]
    half = n_channels // 2
    colors = {"Rest": "steelblue", "Left hand": "darkorange", "Right hand": "green"}

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax_idx, (ch_slice, hemisphere) in enumerate([
        (slice(0, half),    "Left hemisphere (ch 0–3)"),
        (slice(half, None), "Right hemisphere (ch 4–7)"),
    ]):
        ax = axes[ax_idx]
        for cls, name in enumerate(CLASS_NAMES):
            subset = X[y == cls][:, ch_slice, :]
            freqs, psd = sp_signal.welch(subset, fs=sfreq, nperseg=128, axis=-1)
            mean_psd = psd.mean(axis=(0, 1))
            ax.semilogy(freqs, mean_psd, color=list(colors.values())[cls],
                        label=name, lw=1.8)

        ax.axvspan(8,  12, alpha=0.12, color="purple", label="mu (8-12 Hz)")
        ax.axvspan(13, 30, alpha=0.08, color="red",    label="beta (13-30 Hz)")
        ax.set_xlim(0, 45)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("PSD (a.u.)")
        ax.set_title(hemisphere)
        ax.legend(fontsize=8)

    plt.suptitle(
        "Mean PSD by class — mu/beta suppression (ERD) visible during imagery",
        fontsize=12,
    )
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"    Saved -> {save_path}")
    plt.close(fig)


def plot_subject_accuracy_summary(
    df: pd.DataFrame,
    save_path: Optional[str] = None,
) -> None:
    """
    Horizontal bar chart of per-subject within-session LDA accuracy.

    This is the first figure to generate when working with a real EEG dataset.
    It shows *inter-subject variability* — a central fact of BCI research that
    is absent from simulated data.  In the PhysioNet EEGMMIDB dataset, accuracy
    routinely ranges from ~45% to ~85% across participants even with identical
    preprocessing and classifiers.

    Why variability is large in EEG motor imagery:
      - Individual differences in mu/beta ERD amplitude (SNR varies 3–10×)
      - Skull thickness and electrode placement affect spatial resolution
      - "BCI illiteracy": ~15–30% of participants produce no detectable ERD
        (Vidaurre & Blankertz, 2010, Frontiers in Neuroscience)
      - Cognitive strategy — some people visualise kinaesthetically, others
        visually, with different neural signatures

    The chance level (33.3% for 3 classes) is shown as a dashed line.
    Subjects below chance warrant investigation (label error, data quality).

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: 'subject', 'within_accuracy', 'within_itr'.
    """
    subjects   = [f"S{s:03d}" for s in df["subject"].values]
    accuracies = df["within_accuracy"].values
    itrs       = df["within_itr"].values
    chance     = 1 / 3

    y_pos = np.arange(len(subjects))

    fig, ax = plt.subplots(figsize=(7, max(3, len(subjects) * 0.7)))
    bars = ax.barh(y_pos, accuracies, color="steelblue", alpha=0.85,
                   edgecolor="white")
    ax.axvline(chance, ls="--", color="grey", lw=1.2,
               label=f"Chance ({chance:.1%})")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(subjects)
    ax.set_xlabel("Accuracy (5-fold CV)")
    ax.set_title(
        "Per-Subject Within-Session Accuracy (LDA)\n"
        "Inter-subject variability is the norm in real EEG — range can be 45–85%"
    )
    ax.set_xlim(0, 1.05)
    ax.legend(fontsize=9)

    # Annotate each bar with accuracy and ITR
    for i, (acc, itr) in enumerate(zip(accuracies, itrs)):
        ax.text(acc + 0.01, i, f"{acc:.2f}  ({itr:.0f} b/m)",
                va="center", fontsize=9)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"    Saved -> {save_path}")
    plt.close(fig)


def plot_loso_vs_within(
    loso_df: pd.DataFrame,
    within_df: pd.DataFrame,
    save_path: Optional[str] = None,
) -> None:
    """
    Grouped bar chart: within-subject vs LOSO accuracy per subject.

    The gap between the two bar groups is the *calibration benefit*:
    how much accuracy you gain by having subject-specific training data.
    This is a standard figure in BCI transfer-learning papers.

    Reading the chart:
      - Large gap → the model is highly participant-specific; subject-specific
        calibration is essential.
      - Small gap → the model generalises well across participants; a single
        pre-trained model may suffice for new users.
      - LOSO bar near chance → the model completely fails to transfer.

    In published motor-imagery EEG literature, the mean LOSO accuracy with
    LDA on log band power features is typically 10–20 pp below within-subject
    accuracy (Blankertz et al., 2011).

    Parameters
    ----------
    loso_df : pd.DataFrame
        Must contain 'subject' and 'loso_accuracy'.
    within_df : pd.DataFrame
        Must contain 'subject' and 'within_accuracy'.
    """
    merged   = within_df.merge(loso_df[["subject", "loso_accuracy"]], on="subject")
    subjects = merged["subject"].values
    x        = np.arange(len(subjects))
    width    = 0.35
    chance   = 1 / 3

    within_acc = merged["within_accuracy"].values
    loso_acc   = merged["loso_accuracy"].values

    fig, ax = plt.subplots(figsize=(max(6, len(subjects) * 1.5), 5))
    bars1 = ax.bar(x - width / 2, within_acc, width,
                   label="Within-subject", color="steelblue", alpha=0.85)
    bars2 = ax.bar(x + width / 2, loso_acc, width,
                   label="Cross-subject (LOSO)", color="darkorange", alpha=0.85)

    ax.axhline(chance, ls="--", color="grey", lw=1.2,
               label=f"Chance ({chance:.1%})")

    # Value annotations
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=8)

    # Annotate the calibration gap with a bracket on the first subject
    if len(subjects) > 0:
        gap = within_acc[0] - loso_acc[0]
        mid = (within_acc[0] + loso_acc[0]) / 2
        ax.annotate(
            f"gap\n{gap:+.2f}",
            xy=(x[0] + width / 2, loso_acc[0]),
            xytext=(x[0] + width + 0.1, mid),
            arrowprops=dict(arrowstyle="-", color="black", lw=0.8),
            fontsize=8, color="black",
        )

    ax.set_xlabel("Subject")
    ax.set_ylabel("Accuracy")
    ax.set_title(
        "Within-Subject vs Cross-Subject (LOSO) Accuracy\n"
        "Gap = calibration benefit: accuracy gained by having subject-specific data"
    )
    ax.set_xticks(x)
    ax.set_xticklabels([f"S{s}" for s in subjects])
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=9)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"    Saved -> {save_path}")
    plt.close(fig)


def plot_session_drift(
    drift_results: Dict[str, float],
    save_path: Optional[str] = None,
) -> None:
    """
    Bar chart comparing within-session, drifted, and z-score corrected accuracy.

    Three conditions are shown:

    1. Within-session (baseline)
       5-fold CV on session 1 only.  No drift.  Best-case accuracy.

    2. No normalisation (drifted)
       Train on session 1, test on session 2 with simulated drift.
       The classifier's internal StandardScaler uses session-1 statistics,
       so it cannot cancel the session-2-specific offset.  Accuracy drops.

    3. Z-score corrected
       Same drifted session 2, but independently z-scored before decoding
       (fit a new StandardScaler on session 2 data, no labels required).
       The drift-induced offset and scaling cancel algebraically, recovering
       most of the within-session accuracy.

    This plot makes the core argument for per-session normalisation in
    deployed BCI systems.  It also motivates more advanced approaches
    (adaptive LDA, Riemannian alignment) when z-score is insufficient.

    Parameters
    ----------
    drift_results : dict
        Output of generalization.session_drift_experiment().
        Expected keys: acc_within, acc_no_norm, acc_z_score
                       itr_within, itr_no_norm, itr_z_score.
    """
    labels     = ["Within-session\n(no drift)", "No normalisation\n(drifted)",
                  "Z-score corrected\n(per-session)"]
    accuracies = [drift_results["acc_within"],
                  drift_results["acc_no_norm"],
                  drift_results["acc_z_score"]]
    itrs       = [drift_results["itr_within"],
                  drift_results["itr_no_norm"],
                  drift_results["itr_z_score"]]
    colors     = ["steelblue", "firebrick", "seagreen"]
    chance     = 1 / 3

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(labels, accuracies, color=colors, alpha=0.85,
                  edgecolor="white", width=0.5)
    ax.axhline(chance, ls="--", color="grey", lw=1.2,
               label=f"Chance ({chance:.1%})")

    for bar, acc, itr in zip(bars, accuracies, itrs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{acc:.3f}\n({itr:.0f} b/m)",
                ha="center", va="bottom", fontsize=10, fontweight="bold")

    # Arrow showing the drop and recovery
    ax.annotate("", xy=(1, accuracies[1] + 0.02), xytext=(0, accuracies[0] - 0.02),
                arrowprops=dict(arrowstyle="->", color="firebrick", lw=1.5))
    ax.annotate("", xy=(2, accuracies[2] - 0.02), xytext=(1, accuracies[1] + 0.02),
                arrowprops=dict(arrowstyle="->", color="seagreen", lw=1.5))

    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.15)
    ax.set_title(
        "Session Drift & Z-Score Normalisation\n"
        "Drift degrades accuracy; per-session z-score recovers it"
    )
    ax.legend(fontsize=9)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"    Saved -> {save_path}")
    plt.close(fig)


def plot_behavioral_correlation(
    df: pd.DataFrame,
    r: float,
    p: float,
    save_path: Optional[str] = None,
) -> None:
    """
    Scatter plot: decoder ITR vs simulated downstream task performance.

    Each point is one subject.  The regression line and Pearson r are
    shown.  This figure is the bridge between signal-level metrics (ITR)
    and user-facing outcomes (task performance, quality of life).

    Why this matters:
      In clinical BCI trials, regulatory approval and reimbursement often
      depend on demonstrating functional benefit — not just high accuracy.
      Showing that better ITR translates to more commands per minute
      (and therefore faster communication, more independence) directly
      supports the case for the technology.

    In simulation here, the correlation is constructed by design.  In real
    data, ITR-vs-performance r values of 0.5–0.85 are routinely observed
    for P300 and motor-imagery BCIs.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'subject', 'within_itr', 'task_score'.
    r : float
        Pre-computed Pearson correlation coefficient.
    p : float
        p-value of the correlation.
    """
    itr  = df["within_itr"].values
    task = df["task_score"].values

    # Fit regression line
    slope, intercept, *_ = linregress(itr, task)
    x_line = np.linspace(itr.min() - 0.5, itr.max() + 0.5, 100)
    y_line = slope * x_line + intercept

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(itr, task, s=90, color="steelblue", zorder=3, label="Subjects")
    ax.plot(x_line, y_line, color="firebrick", lw=1.5, ls="--", label="Linear fit")

    # Subject labels
    for _, row in df.iterrows():
        ax.annotate(
            f"S{int(row['subject'])}",
            (row["within_itr"], row["task_score"]),
            textcoords="offset points", xytext=(7, 4), fontsize=9,
        )

    p_str = f"p={p:.3f}" if p >= 0.001 else "p<0.001"
    ax.set_xlabel("Decoder ITR (bits/min)")
    ax.set_ylabel("Task performance score (simulated actions/min)")
    ax.set_title(
        f"Decoder Throughput vs Task Performance\n"
        f"Pearson r={r:.3f}, {p_str}  —  higher ITR → more actions/min"
    )
    ax.legend(fontsize=9)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"    Saved -> {save_path}")
    plt.close(fig)


def plot_latency_distribution(
    latencies_ms: np.ndarray,
    save_path: Optional[str] = None,
) -> None:
    """Histogram of per-decode inference latencies."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(latencies_ms, bins=40, color="steelblue", edgecolor="white", linewidth=0.5)
    ax.axvline(
        np.mean(latencies_ms), color="red", ls="--", lw=1.5,
        label=f"Mean: {np.mean(latencies_ms):.2f} ms",
    )
    ax.axvline(
        np.percentile(latencies_ms, 95), color="orange", ls="--", lw=1.5,
        label=f"95th pct: {np.percentile(latencies_ms, 95):.2f} ms",
    )
    ax.set_xlabel("Decode latency (ms)")
    ax.set_ylabel("Count")
    ax.set_title("Real-time decode latency distribution")
    ax.legend(fontsize=9)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"    Saved -> {save_path}")
    plt.close(fig)
