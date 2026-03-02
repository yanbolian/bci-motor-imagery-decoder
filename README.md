# BCI Motor Imagery Decoder

A Python project demonstrating real-time neural signal decoding for
brain-computer interfaces (BCI), built to bridge computational
neuroscience fundamentals with applied BCI engineering.

---

## Motivation

This project targets the exact skill gap between a computational
neuroscience background and the requirements of an applied BCI
researcher role:

| Background strength | Applied in this project |
|---|---|
| Neural systems modelling | Physiologically accurate ERD/ERS simulation |
| Signal processing (MATLAB) | Bandpass filter + CAR in Python/SciPy |
| Statistical analysis | Welch PSD, log band power features |
| Reproducible pipelines | End-to-end script, fixed seeds, saved plots |
| **Gap: modern ML** | **scikit-learn LDA/SVM + PyTorch EEGNet** |
| **Gap: real-time decoding** | **Ring-buffer streaming decoder** |
| **Gap: BCI-specific metrics** | **ITR, confusion matrix, latency profiling** |

---

## What the pipeline demonstrates

```
Simulate neural data          bci_decoder/simulate.py
       ↓
Preprocess (bandpass + CAR)   bci_decoder/preprocess.py
       ↓
Extract log band power        bci_decoder/features.py
       ↓
Train LDA / SVM               bci_decoder/models.py   (scikit-learn)
Train EEGNet                  bci_decoder/models.py   (PyTorch)
       ↓
Evaluate (accuracy, ITR)      bci_decoder/evaluate.py
       ↓
Real-time stream simulation   bci_decoder/realtime.py
       ↓
Plots saved to results/
```

---

## Neuroscience background

### Signal: motor imagery ECoG/EEG

The pipeline decodes three states recorded from 8 simulated cortical
electrodes over sensorimotor cortex:

- **Class 0 — Rest**: both hemispheres show full mu (8–12 Hz) and
  beta (13–30 Hz) oscillations.
- **Class 1 — Left-hand imagery**: right sensorimotor cortex shows
  Event-Related Desynchronisation (ERD) — suppressed mu/beta amplitude.
- **Class 2 — Right-hand imagery**: left sensorimotor cortex shows ERD.

This contralateral ERD is the primary neural signature used in motor BCI
systems, including Synchron's stentrode work (Oxley et al., 2021).

### Why ERD? Why these frequency bands?

During motor imagery (mental rehearsal of movement without execution),
thalamocortical loops that generate the idle mu rhythm are interrupted.
This reduces oscillatory amplitude — a decrease in narrowband power
detectable by any electrode overlying sensorimotor cortex.

ERD is:
- **Contralateral**: imagining the left hand suppresses right hemisphere.
- **Band-specific**: strongest in mu (8–12 Hz) and beta (13–30 Hz).
- **Robust**: observable across EEG, ECoG, and implanted electrodes.

---

## Models

### LDA (Linear Discriminant Analysis)

The reference BCI classifier. Maximises the ratio of between-class
to within-class scatter in feature space. Advantages:
- Closed-form solution — no hyperparameter tuning.
- Works well with < 500 trials (common in BCI).
- Linear weights are directly interpretable (channel/band importance).

### SVM (RBF kernel)

Captures non-linear class boundaries. Useful when ERD patterns vary
non-linearly across sessions or participants.

### EEGNet (PyTorch CNN)

A compact convolutional architecture designed specifically for EEG/ECoG
(Lawhern et al., 2018). Three blocks:

1. **Temporal conv** — data-driven bandpass filter (learns mu/beta selectivity)
2. **Depthwise spatial conv** — learned spatial filter (equivalent to CSP)
3. **Separable conv** — temporal integration

Only ~3,000–10,000 parameters: prevents overfitting on small BCI datasets.

---

## Real-time simulation

`bci_decoder/realtime.py` implements the core real-time BCI loop:

```
Hardware packets (10 samples, 40 ms)
        ↓
Ring buffer (holds 1 s = 250 samples)
        ↓
Decode every 40 ms (25 Hz output)
        ↓
Preprocessing (bandpass + CAR) → features → LDA predict
        ↓
Measure decode latency (typically < 1 ms on CPU)
```

In Synchron's deployed system, additional latency comes from signal
acquisition hardware (~20–40 ms). Inference latency (this pipeline)
contributes < 5 ms — well within the 100 ms threshold for responsive
BCI control.

---

## Setup

```bash
# 1. Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate      # Linux/Mac
.venv\Scripts\activate         # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the full pipeline
python run_pipeline.py
```

Expected runtime: **~2–3 minutes** on CPU (dominated by EEGNet training).

---

## Expected output

```
Step 1 / 6 — Simulate neural data
  Trials   : 360  (120 per class)
  Channels : 8
  Samples  : 1000  (4.0 s @ 250 Hz)

Step 2 / 6 — Preprocess (bandpass 1–40 Hz + CAR)

Step 3 / 6 — Extract log band power features
  Feature matrix : (360, 40)  (8 channels × 5 bands)

Step 4 / 6 — Classical models (LDA, SVM)
  ── LDA ──
    Accuracy          : 0.875  (87.5 %)
    Balanced accuracy : 0.874
    ITR               : 59.1 bits/min

  ── SVM ──
    Accuracy          : 0.889  (88.9 %)
    ...

Step 5 / 6 — EEGNet (PyTorch)
  EEGNet parameters : 3,072
  ...

Step 6 / 6 — Real-time decoding simulation
  Streaming 40 trials (160.0 s of data)
    Mean latency : 0.31 ms
    Max  latency : 2.14 ms
```

### Plots saved to `results/`

| File | Description |
|---|---|
| `psd_by_class.png` | Mean PSD per class — ERD visible as mu/beta dip |
| `cm_lda.png` | LDA confusion matrix (normalised) |
| `cm_svm.png` | SVM confusion matrix |
| `cm_eegnet.png` | EEGNet confusion matrix |
| `eegnet_training.png` | Training loss + validation accuracy curves |
| `latency_dist.png` | Real-time decode latency distribution |

---

## Project structure

```
synchron_job/
├── README.md
├── requirements.txt
├── run_pipeline.py              ← main script (start here)
├── bci_decoder/
│   ├── __init__.py
│   ├── simulate.py              ← ERD/ERS neural signal generation
│   ├── preprocess.py            ← bandpass + CAR
│   ├── features.py              ← log band power (mu/beta ERD)
│   ├── models.py                ← LDA, SVM, EEGNet
│   ├── realtime.py              ← ring buffer + streaming decoder
│   └── evaluate.py              ← metrics + plots
├── docs/
│   └── pipeline_overview.md     ← neuroscience rationale, design decisions
└── results/                     ← generated plots (auto-created)
```

---

## Key references

- Pfurtscheller & Lopes da Silva (1999). Event-related EEG/MEG synchronization
  and desynchronization. *Clinical Neurophysiology*, 110(11), 1842–1857.
- Lawhern et al. (2018). EEGNet: a compact convolutional neural network for
  EEG-based BCIs. *Journal of Neural Engineering*, 15(5), 056013.
- Wolpaw et al. (2000). Brain-computer interface technology: a review of the
  first international meeting. *IEEE Trans. Rehab. Eng.*, 8(2), 164–173.
- Oxley et al. (2021). Motor neuroprosthesis implanted with neurointerventional
  surgery improves capacity for activities of daily living tasks in severe
  paralysis. *Journal of NeuroInterventional Surgery*, 13(2), 102–108.
