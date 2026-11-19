"""Task 3: supervised damage localisation on OGW dataset 1 (after Rautela et al., Ultrasonics 2021).

One example = residual signal of one transmission path at one frequency for one
damage position (28 positions x 36 paths x 3 frequencies = 3,024).  Target = damage
zone DG1-DG7 (seven 2 x 2 clusters of positions).  Path and frequency are known in
practice, so their one-hot codes are appended to every feature set.

Splits (all report accuracy and macro-F1; structure-level accuracy averages the class
probabilities over the 36 paths of a (position, frequency) case):
  random        10 x random 90/10 split of examples (as in the paper; optimistic)
  grouped       5-fold, grouped by (position, path): the 3 frequencies stay together
  unseen pos.   4-fold leave-position-out: fold k holds out the k-th position of every
                zone, so every test position is unseen (interpolation within the zone)

Methods: frozen TSFM embeddings + logistic regression / k-NN; raw residual and |FFT|
+ logistic regression; MiniRocket + ridge; 1D-CNN trained from scratch.

python scripts/ogw1_localize.py --data data/ogw1/loc.npz --emb data/ogw1_emb_R --out results_ogw1
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gwtsfm import evalkit as ek  # noqa: E402

EMB = {"chronos2": "Chronos-2", "chronos_bolt": "Chronos-Bolt (base)", "tirex": "TiRex",
       "moment": "MOMENT-1 (base)", "mantis": "Mantis-8M", "ttm": "TTM-R2"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--emb", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip-trained", action="store_true")
    ap.add_argument("--random-reps", type=int, default=10)
    ap.add_argument("--only-trained", action="store_true",
                    help="reuse frozen-model rows from metrics_frozen.csv; run only the trained references")
    ap.add_argument("--trained-random-reps", type=int, default=3)
    ap.add_argument("--trained-skip-grouped", action="store_true")
    ap.add_argument("--cnn-epochs", type=int, default=60)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    z = np.load(args.data)
    R = z["R"].astype(np.float32)                      # (F, 28, P, L)
    nF, nD, nP, L = R.shape
    fi, di, pi = np.meshgrid(np.arange(nF), np.arange(nD), np.arange(nP), indexing="ij")
    fi, di, pi = fi.ravel(), di.ravel(), pi.ravel()
    X = R.reshape(-1, L)
    X = X / X.std()
    zone = (di // 4)                                   # 0..6
    onehot = np.c_[np.eye(nP)[pi], np.eye(nF)[fi]].astype(np.float32)
    feats = {"Raw residual": ("trained from scratch", X),
             "|FFT| spectrum": ("trained from scratch", np.abs(np.fft.rfft(X, axis=1)))}
    for k, n in EMB.items():
        f = Path(args.emb) / f"{k}.npy"
        if f.exists():
            feats[n] = ("foundation model (frozen)", np.load(f).astype(np.float32).reshape(len(X), -1))
    feats = {n: (fam, np.c_[F, onehot]) for n, (fam, F) in feats.items()}

    # splits
    rng = np.random.default_rng(0)
    splits = []
    for r in range(args.random_reps):
        p = rng.permutation(len(X))
        splits.append(("random 90/10", r, p[len(X) // 10:], p[:len(X) // 10]))
    g = di * nP + pi
    perm = rng.permutation(g.max() + 1)
    for k, (tr, te) in enumerate(GroupKFold(5).split(X, groups=perm[g])):
        splits.append(("grouped (position, path)", k, tr, te))
    for k in range(4):
        te = np.flatnonzero(di % 4 == k)
        splits.append(("unseen positions", k, np.setdiff1d(np.arange(len(X)), te), te))

    rows = pd.read_csv(out / "metrics_frozen.csv").to_dict("records") if args.only_trained else []

    def add(method, fam, head, split, k, te, P, secs=np.nan):
        pred = P.argmax(1)
        # structure level: average class probabilities over the paths of each (freq, position)
        case = fi[te] * nD + di[te]
        cs = np.unique(case)
        Pc = np.stack([P[case == c].mean(0) for c in cs])
        yc = np.array([zone[te][case == c][0] for c in cs])
        rows.append(dict(method=method, family=fam, head=head, split=split, fold=k, seconds=secs,
                         acc=accuracy_score(zone[te], pred),
                         f1=f1_score(zone[te], pred, average="macro"),
                         acc_structure=accuracy_score(yc, Pc.argmax(1)), n_test=len(te)))

    for name, (fam, F) in (feats.items() if not args.only_trained else []):
        for split, k, tr, te in splits:
            add(name, fam, "logistic regression", split, k, te, ek.fit_proba(ek.lr, F, zone, tr, te))
            add(name, fam, "kNN (k=5)", split, k, te, ek.fit_proba(ek.knn, F, zone, tr, te))
        print(f"{name} done", flush=True)
        pd.DataFrame(rows).to_csv(out / "metrics.csv", index=False)
    if not args.skip_trained:
        for split, k, tr, te in splits:
            if (split == "random 90/10" and k >= args.trained_random_reps) or \
                    (args.trained_skip_grouped and split.startswith("grouped")):
                continue  # trained references: 3 random repetitions are enough
            t0 = time.time()
            add("MiniRocket", "trained from scratch", "ridge", split, k, te,
                ek.minirocket_proba(X, zone, tr, te, extra=onehot), time.time() - t0)
            t0 = time.time()
            add("1D-CNN", "trained from scratch", "end-to-end", split, k, te,
                ek.cnn_proba(X, zone, tr, te, groups=g, n_classes=7, epochs=args.cnn_epochs),
                time.time() - t0)
            print(f"trained refs {split} fold {k} done", flush=True)
            pd.DataFrame(rows).to_csv(out / "metrics.csv", index=False)
    M = pd.DataFrame(rows)
    M.to_csv(out / "metrics.csv", index=False)
    print(M.groupby(["split", "family", "method", "head"])[["acc", "f1", "acc_structure"]]
          .mean().round(3).to_string())


if __name__ == "__main__":
    main()
