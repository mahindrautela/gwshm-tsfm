"""Diagnostic: is the PyTorch port of the reference 1D-CNN under-trained?"""
import sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import gwtcw_classify as g

Xall, yall, cfgall = g.load("data/gwtcw/gwtcw.npz")
keep = np.arange(len(Xall)) % 1560 % 120 // 12 == 0
X, y, cfg = Xall[keep], yall[keep], cfgall[keep]
for ep in ():
    aucs = []
    for i, (tr, te) in enumerate(g.folds(cfg)):
        t0 = time.time()
        s = g.cnn_scores(X, y, tr, te, True, cfg, epochs=ep)
        aucs.append(roc_auc_score(y[te], s)); print(f"grouped sym fold {i} epochs {ep}: {aucs[-1]:.3f} ({time.time()-t0:.0f}s)", flush=True)
    print("grouped sym mean", np.mean(aucs).round(3), flush=True)
idx = np.arange(len(Xall)); tr, te = train_test_split(idx, test_size=0.05, random_state=42)
# leaky split, validation drawn at random too (as in the notebook)
s = g.cnn_scores(Xall, yall, tr, te, False, np.arange(len(Xall)), epochs=60)
print("leaky random split, random validation, 60 epochs:", round(roc_auc_score(yall[te], s), 3), flush=True)
