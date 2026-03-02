"""
Motor imagery decoders: classical ML and deep learning.

Classical (scikit-learn)
------------------------
LDA  : Linear Discriminant Analysis — the reference BCI classifier.
       Maximises between-class variance relative to within-class variance.
       Fast, robust with small datasets, closed-form solution.
SVM  : Support Vector Machine with RBF kernel.
       Handles non-linear feature boundaries; useful when LDA underfits.

Deep Learning (PyTorch)
-----------------------
EEGNet: A compact CNN designed specifically for EEG/ECoG decoding
        (Lawhern et al., 2018, J. Neural Eng.).

    Block 1 — Temporal Conv:
        Learns frequency-selective filters (conceptually similar to
        bandpass filtering, but data-driven).

    Block 2 — Depthwise Spatial Conv:
        Learns one spatial filter per temporal feature map.
        Equivalent to a learned Common Spatial Pattern (CSP) filter —
        it will find the channel combination that best separates classes.

    Block 3 — Separable Conv:
        Efficient time integration across the spatially filtered signal.

    Key design property: only ~2,000–10,000 parameters, which prevents
    overfitting on small BCI datasets (typically < 500 trials per session).

Reference: Lawhern et al. (2018). EEGNet: a compact convolutional neural
network for EEG-based brain-computer interfaces. J. Neural Eng., 15(5).
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from typing import List, Tuple


# ── Classical Models ──────────────────────────────────────────────────────────

def build_lda() -> Pipeline:
    """
    LDA classifier with z-score normalisation.

    LDA is the most widely used BCI classifier because:
    - It is interpretable (linear weights = channel/band importance)
    - It needs very few training trials to generalise well
    - It is fast enough for real-time deployment
    - It has no hyperparameters to tune

    Returns
    -------
    sklearn Pipeline: StandardScaler → LinearDiscriminantAnalysis
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("lda",    LinearDiscriminantAnalysis(solver="svd")),
    ])


def build_svm(C: float = 1.0) -> Pipeline:
    """
    SVM classifier with RBF kernel.

    The RBF kernel can model non-linear class boundaries in feature space —
    useful when ERD patterns vary in a non-linear way across participants
    or sessions.

    Parameters
    ----------
    C : float
        Regularisation parameter. Larger C → less regularisation.

    Returns
    -------
    sklearn Pipeline: StandardScaler → SVC(RBF kernel)
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("svm",    SVC(kernel="rbf", C=C, gamma="scale", probability=True)),
    ])


# ── EEGNet (PyTorch) ──────────────────────────────────────────────────────────

class EEGNet(nn.Module):
    """
    Compact convolutional neural network for EEG/ECoG decoding.

    Architecture (3 blocks)
    -----------------------
    Block 1 — Temporal Conv + BN
        Input : (batch, 1, n_channels, n_samples)
        Conv  : kernel (1, sfreq//2) — learns frequency-selective filters
                (one filter per temporal_filters output maps)
        Output: (batch, F1, n_channels, n_samples)

    Block 2 — Depthwise Spatial Conv + BN + ELU + Pool + Dropout
        Conv  : kernel (n_channels, 1), groups=F1 — spatial filter per map
                collapses channel dimension → (batch, D*F1, 1, n_samples)
        Pool  : AvgPool(1, 4) — 4× temporal downsampling

    Block 3 — Separable Conv + BN + ELU + Pool + Dropout
        Conv  : kernel (1, 16) depthwise + kernel (1, 1) pointwise
        Pool  : AvgPool(1, 8) — 8× temporal downsampling

    Classifier: Flatten → Linear → logits
    """

    def __init__(
        self,
        n_channels: int = 8,
        n_samples: int = 1000,
        n_classes: int = 3,
        sfreq: int = 250,
        temporal_filters: int = 16,   # F1
        depth_multiplier: int = 2,    # D (total spatial maps = D * F1)
        dropout_rate: float = 0.5,
    ):
        super().__init__()
        F1 = temporal_filters
        F2 = F1 * depth_multiplier
        kern_t = sfreq // 2   # temporal kernel length ≈ 0.5 s

        # Block 1: temporal convolution — frequency selectivity
        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, kernel_size=(1, kern_t),
                      padding=(0, kern_t // 2), bias=False),
            nn.BatchNorm2d(F1),
        )

        # Block 2: depthwise spatial convolution — channel weighting
        self.block2 = nn.Sequential(
            nn.Conv2d(F1, F2, kernel_size=(n_channels, 1),
                      groups=F1, bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(dropout_rate),
        )

        # Block 3: separable convolution — temporal integration
        self.block3 = nn.Sequential(
            nn.Conv2d(F2, F2, kernel_size=(1, 16),
                      padding=(0, 8), groups=F2, bias=False),
            nn.Conv2d(F2, F2, kernel_size=1, bias=False),   # pointwise
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8)),
            nn.Dropout(dropout_rate),
        )

        # Dynamically compute flattened size (avoids hardcoding)
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_samples)
            dummy = self.block1(dummy)
            dummy = self.block2(dummy)
            dummy = self.block3(dummy)
            flat_size = dummy.numel()

        self.classifier = nn.Linear(flat_size, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor, shape (batch, n_channels, n_samples)

        Returns
        -------
        logits : Tensor, shape (batch, n_classes)
        """
        x = x.unsqueeze(1)    # → (batch, 1, n_channels, n_samples)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = x.flatten(1)
        return self.classifier(x)

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def train_eegnet(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_classes: int = 3,
    sfreq: int = 250,
    n_epochs: int = 60,
    batch_size: int = 32,
    lr: float = 5e-4,
) -> Tuple["EEGNet", List[float], List[float]]:
    """
    Train EEGNet on neural trial data.

    Uses Adam optimiser + cosine annealing learning-rate schedule.
    Cosine annealing smoothly reduces lr over training, helping EEGNet
    converge to a flatter, more generalisable minimum.

    Parameters
    ----------
    X_train, X_val : ndarray, shape (n_trials, n_channels, n_samples)
    y_train, y_val : ndarray, shape (n_trials,)
    n_classes : int
    sfreq : int
    n_epochs : int
    batch_size : int
    lr : float
        Initial learning rate for Adam.

    Returns
    -------
    model : EEGNet (trained, on CPU)
    train_losses : list of float (one per epoch)
    val_accuracies : list of float (one per epoch)
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    _, n_channels, n_samples = X_train.shape
    model = EEGNet(
        n_channels=n_channels,
        n_samples=n_samples,
        n_classes=n_classes,
        sfreq=sfreq,
    ).to(device)
    print(f"  EEGNet parameters : {model.count_parameters():,}")
    print(f"  Training device   : {device}")

    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=n_epochs)
    criterion = nn.CrossEntropyLoss()

    X_tr = torch.tensor(X_train, dtype=torch.float32)
    y_tr = torch.tensor(y_train, dtype=torch.long)
    X_v  = torch.tensor(X_val,   dtype=torch.float32).to(device)
    y_v  = torch.tensor(y_val,   dtype=torch.long).to(device)

    dataset = torch.utils.data.TensorDataset(X_tr, y_tr)
    loader  = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    train_losses, val_accuracies = [], []

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimiser.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimiser.step()
            epoch_loss += loss.item() * len(yb)
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_preds = model(X_v).argmax(dim=1)
            val_acc = (val_preds == y_v).float().mean().item()

        train_losses.append(epoch_loss / len(y_train))
        val_accuracies.append(val_acc)

        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1:3d}/{n_epochs}  "
                  f"loss={epoch_loss/len(y_train):.4f}  "
                  f"val_acc={val_acc:.3f}")

    return model.cpu(), train_losses, val_accuracies
