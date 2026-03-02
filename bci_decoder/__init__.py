"""
bci_decoder — Real-time motor imagery decoding for BCI applications.

A complete signal processing and machine learning pipeline for decoding
motor intentions from neural signals, targeting Synchron's stentrode BCI.

Modules
-------
simulate   : Synthetic ECoG/EEG-like motor imagery data generation.
preprocess : Bandpass filtering and Common Average Reference.
features   : Log band power feature extraction (mu/beta ERD).
models     : LDA, SVM (scikit-learn) and EEGNet (PyTorch) decoders.
realtime   : Ring-buffer streaming decoder with sliding-window inference.
evaluate   : Metrics (accuracy, ITR) and diagnostic visualisations.
"""
