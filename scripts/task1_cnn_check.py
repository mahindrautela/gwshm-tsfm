"""Task 1 diagnostic: is the 1D-CNN's poor score a matter of leakage or of under-training?

Two checks quoted in the paper:
  1. The reference notebook's random split, with the early-stopping data also drawn at
     random (as in the notebook): repeated acquisitions leak and the AUC reaches 1.00.
  2. Grouped split (first fold, sign-symmetric) trained for 150 instead of 40 epochs:
     the AUC does not improve, so the gap is generalisation, not under-training.

    python scripts/task1_cnn_check.py        # about 10 minutes
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gwtsfm.baselines import train_cnn  # noqa: E402
from task1_gwtcw import first_repeat, grouped_folds, load_gwtcw  # noqa: E402

Xall, yall, cfgall = load_gwtcw()

# 1. leaky split: one group per signal, so validation signals are drawn at random too
tr, te = train_test_split(np.arange(len(Xall)), test_size=0.05, random_state=42)
s = train_cnn(Xall, yall, tr, te, np.arange(len(Xall)), n_classes=2, epochs=60)
print("random split, random validation, 60 epochs: AUC", round(roc_auc_score(yall[te], s), 3))

# 2. grouped split, first fold, 150 epochs
keep = first_repeat(len(Xall))
X, y, cfg = Xall[keep], yall[keep], cfgall[keep]
tr, te = grouped_folds(cfg)[0]
s = train_cnn(X, y, tr, te, cfg, n_classes=2, epochs=150, sign_symmetric=True)
print("grouped split, fold 0, sign-symmetric, 150 epochs: AUC", round(roc_auc_score(y[te], s), 3))
