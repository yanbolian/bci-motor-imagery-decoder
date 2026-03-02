# Pipeline Design: Neuroscience Rationale and Engineering Decisions

This document explains *why* each step of the pipeline was designed the
way it was — connecting the underlying neuroscience to the engineering
choices. This is the kind of reasoning expected in an applied BCI
research role.

---

## 1. Signal simulation

**Why simulate rather than use real data?**

Real ECoG/EEG BCI datasets require ethics approval and data-sharing
agreements. Simulation lets us validate the full pipeline end-to-end
before obtaining real participant data, and allows controlled experiments
(e.g., varying SNR to test decoder robustness under different conditions).

**Why pink (1/f) noise as the background?**

The power spectral density of local field potentials (LFPs) and ECoG
signals falls approximately as 1/f^α (α ≈ 1–3). Pink noise (α = 1) is
a reasonable first approximation. White noise would underestimate low-
frequency power and produce an unrealistically flat spectrum.

**Why ERD at 80% amplitude suppression?**

Published ERD values in motor imagery studies range from ~20–80%
power reduction (Pfurtscheller & Lopes da Silva, 1999). We use 80%
amplitude suppression (= 64% power reduction) — a clear effect that
produces a tractable decoding problem at moderate noise levels.
Increasing `noise_std` in `generate_dataset` makes the problem harder.

**Why 8 channels with left/right hemisphere grouping?**

Synchron's stentrode sits in the superior sagittal sinus overlying
sensorimotor cortex. A typical recording configuration captures bilateral
channels from both hemispheres, with contralateral channels most
informative for hand motor decoding.

---

## 2. Preprocessing

### Bandpass filter: 1–40 Hz

- **Low cut (1 Hz)**: removes DC offset and slow electrode drift.
  Below 1 Hz the signal is dominated by motion artefacts and baseline
  wander — not neural information.
- **High cut (40 Hz)**: removes power-line noise (50/60 Hz), high-
  frequency EMG contamination from facial muscles, and amplifier noise.
  The highest-frequency BCI-relevant band (gamma) peaks at ~35–40 Hz,
  so a 40 Hz cut preserves all informative content.
- **Zero-phase (sosfiltfilt)**: forward-backward filtering produces a
  zero-phase response, so no temporal smearing of event-locked activity.

### Common Average Reference (CAR)

CAR subtracts the instantaneous spatial mean. This is appropriate when:
- Signals of interest are focal (spatially localised ERD).
- Noise is spatially diffuse (reference drifts, EMG, amplifier offsets).

In ECoG, CAR consistently improves SNR for motor decoding (Crone et al.,
1998). In EEG, Laplacian referencing is sometimes preferred; CAR is a
simpler approximation suitable for dense electrode arrays.

---

## 3. Feature extraction: log band power

**Why log band power instead of raw waveform features?**

1. **Interpretability**: each feature has a direct neurophysiological
   meaning (power in a specific frequency band, channel, and class).
   This is critical for building trust with clinical teams and regulatory
   bodies.

2. **Robustness**: power features are more stable across sessions than
   raw amplitudes because they are invariant to phase, electrode
   impedance drift, and slow amplitude changes.

3. **Dimensionality**: 40 features (8 channels × 5 bands) is tractable
   for LDA without overfitting, even with < 200 training trials per class.

4. **Why log?**: Neural power spans several orders of magnitude across
   subjects and sessions. Log-transform compresses the range, making the
   feature distribution more Gaussian — a key assumption of LDA.

**Why Welch's method?**

Welch averages periodogram estimates over overlapping windows, reducing
variance by ~50–70% compared to a single FFT. For 4-second epochs at
250 Hz (1000 samples), Welch with 128-sample segments gives frequency
resolution ≈ 1.95 Hz — sufficient to resolve the mu (8–12 Hz) and beta
(13–30 Hz) bands.

**Alternative feature: Common Spatial Patterns (CSP)**

CSP finds spatial filters that simultaneously maximise variance for one
class while minimising it for another. It is the standard feature for
two-class motor imagery and often outperforms log band power, but requires
more trials to estimate reliably. A next step for this project would be
to add CSP features via `mne.decoding.CSP` or `pyriemann`.

---

## 4. Decoders

### Why LDA first?

LDA is the most widely benchmarked BCI classifier:
- It is the baseline in most motor BCI competitions (BCI Competition IV).
- Its weights directly indicate which channels/bands drive the decision —
  useful for debugging and explaining results to clinical staff.
- If LDA fails (low accuracy), it usually indicates a problem with the
  features or data, not the classifier.

### When to use SVM?

SVM with an RBF kernel is appropriate when:
- The class boundary in feature space is non-linear.
- The dataset is large enough to estimate the kernel well (> 100 trials/class).
- LDA achieves acceptable but not optimal accuracy.

### EEGNet design choices

**Why not a deeper network?**

Typical BCI datasets have 100–500 trials per session. Deep networks
(ResNet, transformer) overfit badly at this scale. EEGNet's constrained
architecture (depthwise + separable convolutions) acts as a strong
inductive prior: the spatial filter (block 2) must operate on raw
channel space, and the temporal filter (block 1) learns frequency
selectivity — both biologically plausible constraints.

**Why cosine annealing?**

Cosine annealing reduces the learning rate smoothly from `lr` to 0 over
training. This avoids the instability of step-decay schedules and tends
to find flatter minima that generalise better on small datasets.

---

## 5. Real-time decoding architecture

### Ring buffer

A ring buffer is the standard data structure for streaming neural signal
processing:
- **Fixed memory**: no allocation during runtime — critical for
  deterministic latency in safety-critical systems.
- **O(1) write**: each new sample overwrites the oldest entry.
- **Chronological read**: reconstructs the window without sorting.

### Sliding window

The 1-second decode window slides in 40 ms steps (25 Hz output rate).
This creates a trade-off:
- **Shorter windows**: lower latency, less signal averaging, noisier.
- **Longer windows**: better SNR, higher latency.

1 second is the standard in motor imagery BCIs; Synchron's click-detection
pipeline uses shorter windows (~0.3–0.5 s) because click events are brief.

### Latency budget (deployed system estimate)

| Component | Latency |
|---|---|
| Signal acquisition + USB | 20–40 ms |
| Preprocessing (bandpass + CAR) | 1–3 ms |
| Feature extraction (Welch PSD) | 0.5–2 ms |
| LDA inference | < 0.1 ms |
| **Total** | **~25–50 ms** |

Real-time responsive control typically requires < 100 ms total latency.
This pipeline's inference step (< 1 ms) is not the bottleneck.

---

## 6. Evaluation metrics

### Information Transfer Rate (ITR)

ITR quantifies how much information the BCI transmits per unit time
(bits/minute). It accounts for both classification accuracy and decision
speed, making it the standard metric for comparing BCI systems across
different paradigms and settings.

ITR alone is insufficient: a system with 60% accuracy but 120 decisions/min
may outperform one with 90% accuracy and 10 decisions/min by ITR, but
the low-accuracy system would be frustrating to use. Both metrics matter.

### Confusion matrix

The off-diagonal pattern reveals systematic confusions. In a 3-class
motor imagery BCI, the most common confusion is rest vs. one of the
imagery classes — rest is harder to maintain consistently than active
imagery.

---

## Next steps (to extend this project)

1. **Real EEG data**: plug in a public motor imagery dataset
   (e.g., BCI Competition IV Dataset 2a — 9 subjects, 4 classes,
   publicly available at www.bbci.de/competition/iv/).

2. **MNE-Python integration**: `mne` provides rich EEG/ECoG loading,
   preprocessing (ICA artefact removal), and event handling.

3. **Cross-session generalisation**: train on Session 1, test on Session 2.
   This is the hardest real-world challenge for BCI (electrode drift,
   fatigue, day-to-day signal variability).

4. **Common Spatial Patterns**: add CSP spatial filtering via
   `pyriemann.estimation.Covariances` + `pyriemann.tangentspace`.

5. **Online adaptation**: implement covariate-shift adaptation
   (e.g., Euclidean Alignment) to correct for session-to-session drift.
