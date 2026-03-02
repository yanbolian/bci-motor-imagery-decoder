#!/usr/bin/env python3
"""
BCI Motor Imagery Decoder — End-to-End Pipeline
================================================

Demonstrates a complete BCI signal processing and decoding workflow:

    Simulate → Preprocess → Feature extraction
    → Classical ML (LDA, SVM) → Deep learning (EEGNet)
    → Real-time streaming simulation → Evaluation plots

All results are saved to results/

Run:
    python run_pipeline.py
"""

import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

from bci_decoder.simulate   import generate_dataset, SFREQ, CLASSES
from bci_decoder.preprocess import preprocess_dataset
from bci_decoder.features   import log_band_power, windowed_log_band_power, feature_names
from bci_decoder.models     import build_lda, build_svm, train_eegnet
from bci_decoder.realtime   import RealTimeDecoder
from bci_decoder.evaluate   import (
    print_report,
    plot_confusion_matrix,
    plot_training_curves,
    plot_band_power_spectrum,
    plot_raw_signal_example,
    plot_latency_distribution,
)

import torch

SEED    = 42
OUT_DIR = Path("results")
OUT_DIR.mkdir(exist_ok=True)


def banner(text: str) -> None:
    print(f"\n{'-' * 60}")
    print(f"  {text}")
    print(f"{'-' * 60}")


def main() -> None:
    print("=" * 60)
    print("  BCI Motor Imagery Decoder — Full Pipeline")
    print("=" * 60)
    print(f"  Classes  : {CLASSES}")
    print(f"  Sfreq    : {SFREQ} Hz")
    print(f"  Seed     : {SEED}")

    # ── 1. Simulate neural data ───────────────────────────────────────────────
    banner("Step 1 / 6 — Simulate neural data")

    X_raw, y = generate_dataset(
        n_trials_per_class=500,
        noise_std=1.5,   # increase to make decoding harder; try 0.5–3.0
        seed=SEED,
    )
    n_trials, n_ch, n_samp = X_raw.shape
    print(f"  Trials   : {n_trials}  ({n_trials // 3} per class)")
    print(f"  Channels : {n_ch}")
    print(f"  Samples  : {n_samp}  ({n_samp / SFREQ:.1f} s @ {SFREQ} Hz)")

    # Plot example raw brain signals — show ERD as suppressed oscillations
    # in the contralateral hemisphere during motor imagery.
    plot_raw_signal_example(
        X_raw, y, sfreq=SFREQ,
        save_path=str(OUT_DIR / "raw_signal_example.png"),
    )

    # ── 2. Preprocess ─────────────────────────────────────────────────────────
    banner("Step 2 / 6 — Preprocess (bandpass 1–40 Hz + CAR)")

    X_proc = preprocess_dataset(X_raw, sfreq=SFREQ)
    print("  Preprocessing done.")

    plot_band_power_spectrum(
        X_proc, y, sfreq=SFREQ,
        save_path=str(OUT_DIR / "psd_by_class.png"),
    )

    # ── 3. Feature extraction ─────────────────────────────────────────────────
    banner("Step 3 / 6 — Extract log band power features")

    X_feat = log_band_power(X_proc, sfreq=SFREQ)
    feat_dim = X_feat.shape[1]
    print(f"  Feature matrix : {X_feat.shape}  "
          f"({n_ch} channels × {feat_dim // n_ch} bands)")
    print(f"  Feature names  : {feature_names(n_ch)[:5]} ...")

    # Single shared train/test split — both feature and raw views
    idx_tr, idx_te = train_test_split(
        np.arange(n_trials), test_size=0.2, stratify=y, random_state=SEED,
    )
    X_tr_feat, X_te_feat = X_feat[idx_tr], X_feat[idx_te]
    X_tr_raw,  X_te_raw  = X_proc[idx_tr], X_proc[idx_te]
    y_tr, y_te           = y[idx_tr],      y[idx_te]

    print(f"  Train : {len(y_tr)} trials  |  Test : {len(y_te)} trials")

    # ── 4. Classical models ───────────────────────────────────────────────────
    banner("Step 4 / 6 — Classical models (LDA, SVM)")

    trained_classifiers = {}
    for name, clf in [("LDA", build_lda()), ("SVM", build_svm(C=1.0))]:
        clf.fit(X_tr_feat, y_tr)
        y_pred = clf.predict(X_te_feat)
        print_report(y_te, y_pred, model_name=name)
        plot_confusion_matrix(
            y_te, y_pred, model_name=name,
            save_path=str(OUT_DIR / f"cm_{name.lower()}.png"),
        )
        trained_classifiers[name] = clf

    # ── 5. Deep learning — EEGNet ─────────────────────────────────────────────
    banner("Step 5 / 6 — EEGNet (PyTorch)")

    model, train_losses, val_accs = train_eegnet(
        X_tr_raw, y_tr,
        X_te_raw,  y_te,
        n_classes=3,
        sfreq=SFREQ,
        n_epochs=100,
        batch_size=64,
        lr=5e-4,
    )

    plot_training_curves(
        train_losses, val_accs,
        save_path=str(OUT_DIR / "eegnet_training.png"),
    )

    # Final EEGNet test evaluation
    model.eval()
    with torch.no_grad():
        logits  = model(torch.tensor(X_te_raw, dtype=torch.float32))
        y_pred_nn = logits.argmax(dim=1).numpy()

    print_report(y_te, y_pred_nn, model_name="EEGNet")
    plot_confusion_matrix(
        y_te, y_pred_nn, model_name="EEGNet",
        save_path=str(OUT_DIR / "cm_eegnet.png"),
    )

    # ── 6. Real-time streaming simulation ────────────────────────────────────
    banner("Step 6 / 6 — Real-time decoding simulation")

    # The batch LDA was trained on 4-second trial features.
    # A real-time decoder uses 1-second windows, so its feature statistics
    # differ — the StandardScaler has different mean/variance.
    # Solution: train a window-matched LDA on 1-second sliding windows
    # extracted from the training trials (50 % overlap).
    WINDOW_SAMPLES = 250   # 1 s @ 250 Hz
    STEP_SAMPLES   = 125   # 50 % overlap

    print(f"  Training window-matched LDA  "
          f"(window={WINDOW_SAMPLES/SFREQ:.1f} s, "
          f"step={STEP_SAMPLES/SFREQ:.2f} s) ...")
    X_tr_wins, y_tr_wins = windowed_log_band_power(
        X_tr_raw, y_tr, sfreq=SFREQ,
        window_samples=WINDOW_SAMPLES, step_samples=STEP_SAMPLES,
    )
    lda_rt = build_lda()
    lda_rt.fit(X_tr_wins, y_tr_wins)
    print(f"    Windowed training set: {X_tr_wins.shape[0]} windows")

    # Build a continuous stream from 40 held-out test trials.
    # Must transpose to (n_ch, n_trials, n_samples) before flatten so that
    # channel data is not interleaved across trials by the C-order reshape.
    n_stream_trials = min(40, len(y_te))
    stream_signal = (
        X_te_raw[:n_stream_trials]          # (n_trials, n_ch, n_samples)
        .transpose(1, 0, 2)                 # (n_ch, n_trials, n_samples)
        .reshape(n_ch, -1)                  # (n_ch, n_trials * n_samples)
    )
    # Per-sample ground-truth labels (repeated across each trial's samples)
    true_sample_labels = np.repeat(y_te[:n_stream_trials], n_samp)

    print(f"  Streaming {n_stream_trials} trials "
          f"({stream_signal.shape[1] / SFREQ:.1f} s of data)")
    print(f"  Window: 1.0 s   Step: 40 ms   Chunk: 10 samples (40 ms)")

    # The ring buffer is fed already-preprocessed data (X_te_raw),
    # matching the windowed training features which came from X_tr_raw.
    # No re-preprocessing needed inside the decode function.
    def lda_decode(window: np.ndarray) -> int:
        """window shape: (1, n_channels, window_samples) — already preprocessed."""
        feats = log_band_power(window, sfreq=SFREQ)
        return int(lda_rt.predict(feats)[0])

    rt_decoder = RealTimeDecoder(
        decode_fn=lda_decode,
        n_channels=n_ch,
        sfreq=SFREQ,
        window_sec=1.0,
        step_sec=0.04,   # 40 ms → 25 Hz output rate
        chunk_size=10,   # 40 ms packets
    )

    results = rt_decoder.run_simulation(
        stream_data=stream_signal,
        true_labels=true_sample_labels,
        verbose=True,
    )

    plot_latency_distribution(
        results["latencies_ms"],
        save_path=str(OUT_DIR / "latency_dist.png"),
    )

    # ── Summary ───────────────────────────────────────────────────────────────
    banner("Done")
    print(f"  All plots saved to: {OUT_DIR.resolve()}/")
    print()
    print("  Files produced:")
    for f in sorted(OUT_DIR.glob("*.png")):
        print(f"    {f.name}")


if __name__ == "__main__":
    main()
