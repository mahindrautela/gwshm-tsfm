"""Task 3, path level: predict the damage zone from one path's residual signal (OGW-1).

This is the protocol of Rautela et al. (Ultrasonics 2021). One example is the residual of
one path at one frequency for one damage position: 28 positions x 36 paths x 3
frequencies = 3,024 examples. The target is the damage zone (7 zones of 4 positions).
Path and frequency are known in practice, so their one-hot codes are appended to every
feature set.

Splits (accuracy and macro-F1 per fold):
  random 90/10               10 random splits of the examples, as in the original paper
  grouped (position, path)   5 folds; the three frequencies of a residual stay together
  unseen positions           4 folds; fold j holds out the j-th position of every zone
"Pooled" accuracy (acc_structure) averages the class probabilities over the 36 paths of
each (position, frequency) case before taking the most likely zone.

Methods: frozen embeddings, the raw residual and its |FFT| spectrum, each with a
logistic-regression and a k-NN head; MiniRocket and the 1D-CNN trained on the residuals.
To save time the trained references run on the first random split and the four
unseen-position folds only (as in the paper).

Output: results/task3_ogw1/metrics.csv

    python scripts/task3_ogw1_paths.py       # about 45 minutes, mostly the 1D-CNN
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gwtsfm import baselines, heads  # noqa: E402
from gwtsfm.config import DATA, EMBEDDINGS, MODELS, RESULTS  # noqa: E402
from gwtsfm.splits import grouped_kfold  # noqa: E402

FROZEN, TRAINED = "foundation model (frozen)", "trained from scratch"


def make_splits(position, path, n_examples, random_reps, rng):
    """List of (split name, fold, train indices, test indices)."""
    splits = []
    for r in range(random_reps):
        p = rng.permutation(n_examples)
        splits.append(("random 90/10", r, p[n_examples // 10:], p[:n_examples // 10]))
    group = position * 36 + path                       # one group per (position, path)
    shuffled = rng.permutation(group.max() + 1)        # random fold assignment of the groups
    for k, (tr, te) in enumerate(grouped_kfold(shuffled[group], 5)):   # GroupKFold, any platform
        splits.append(("grouped (position, path)", k, tr, te))
    for k in range(4):
        te = np.flatnonzero(position % 4 == k)         # the k-th position of every zone
        splits.append(("unseen positions", k, np.setdiff1d(np.arange(n_examples), te), te))
    return splits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DATA / "ogw1" / "residuals.npz"))
    ap.add_argument("--out", default=str(RESULTS / "task3_ogw1"))
    ap.add_argument("--random-reps", type=int, default=10)
    ap.add_argument("--skip-trained", action="store_true", help="skip MiniRocket and the 1D-CNN")
    ap.add_argument("--cnn-epochs", type=int, default=60)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    R = np.load(args.data)["R"].astype(np.float32)       # (frequency, position, path, sample)
    nF, nD, nP, L = R.shape
    # index of the frequency, position and path of every example, in the order of R
    freq, position, path = (a.ravel() for a in np.meshgrid(np.arange(nF), np.arange(nD), np.arange(nP),
                                                           indexing="ij"))
    X = R.reshape(-1, L)
    X = X / X.std()
    zone = position // 4                                   # 0..6
    onehot = np.c_[np.eye(nP)[path], np.eye(nF)[freq]].astype(np.float32)

    feats = {"Raw residual": (TRAINED, X), "|FFT| spectrum": (TRAINED, np.abs(np.fft.rfft(X, axis=1)))}
    for key, name in MODELS.items():
        f = EMBEDDINGS / "ogw1_residuals" / f"{key}.npy"
        if f.exists():
            feats[name] = (FROZEN, np.load(f).astype(np.float32).reshape(len(X), -1))
    feats = {n: (fam, np.c_[F, onehot]) for n, (fam, F) in feats.items()}
    splits = make_splits(position, path, len(X), args.random_reps, np.random.default_rng(0))

    rows = []

    def record(method, family, head, split, k, te, P, seconds=np.nan):
        pred = P.argmax(1)
        case = freq[te] * nD + position[te]          # (frequency, position) of each test example
        cases = np.unique(case)
        P_case = np.stack([P[case == c].mean(0) for c in cases])
        y_case = np.array([zone[te][case == c][0] for c in cases])
        rows.append(dict(method=method, family=family, head=head, split=split, fold=k, seconds=seconds,
                         acc=accuracy_score(zone[te], pred), f1=f1_score(zone[te], pred, average="macro"),
                         acc_structure=accuracy_score(y_case, P_case.argmax(1)), n_test=len(te)))

    for name, (family, F) in feats.items():
        for split, k, tr, te in splits:
            record(name, family, "logistic regression", split, k, te,
                   heads.predict_proba(heads.logistic_regression, F, zone, tr, te))
            record(name, family, "kNN (k=5)", split, k, te,
                   heads.predict_proba(heads.knn_classifier, F, zone, tr, te))
        print(f"{name} done", flush=True)
        pd.DataFrame(rows).to_csv(out / "metrics.csv", index=False)

    if not args.skip_trained:
        for split, k, tr, te in splits:
            if split.startswith("grouped") or (split == "random 90/10" and k > 0):
                continue
            t0 = time.time()
            P = baselines.softmax(baselines.minirocket_decision(X, zone, tr, te, extra=onehot))
            record("MiniRocket", TRAINED, "ridge", split, k, te, P, time.time() - t0)
            t0 = time.time()
            P = baselines.train_cnn(X, zone, tr, te, position * nP + path, n_classes=7,
                                    epochs=args.cnn_epochs, rescale=True)
            record("1D-CNN", TRAINED, "end-to-end", split, k, te, P, time.time() - t0)
            print(f"trained references: {split} fold {k} done", flush=True)
            pd.DataFrame(rows).to_csv(out / "metrics.csv", index=False)

    M = pd.DataFrame(rows)
    M.to_csv(out / "metrics.csv", index=False)
    print(M.groupby(["split", "family", "method", "head"])[["acc", "f1", "acc_structure"]]
          .mean().round(3).to_string())


if __name__ == "__main__":
    main()
