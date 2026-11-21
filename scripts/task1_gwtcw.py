"""Task 1: supervised damage classification on GW-TCW (healthy vs damaged signal).

Data: 1,560 healthy and 1,560 damaged signals, organised as 13 blocks x 12 sensor paths
x 10 repeated acquisitions. A (block, path) pair is a "configuration"; healthy row i and
damaged row i belong to the same configuration.

Protocol
  * The 10 repeats of a configuration are near copies (r > 0.9999), so we keep the first
    one (312 signals) and use 5-fold cross-validation grouped by configuration: no test
    configuration is seen in training.
  * In 71 of the 156 configurations the damaged signals are sign-inverted. Every model is
    therefore also run "sign-symmetric": trained on x and -x and scored on both, so that
    polarity cannot be used as a cue. A probe that only sees the sign of the largest peak
    checks that polarity alone does not separate the classes.
  * Frozen embeddings get a logistic-regression or k-NN head, plus a one-class score that
    uses no damage labels. Trained references: logistic regression on the raw signal and
    on the |FFT| spectrum, MiniRocket, and the 1D-CNN of the dataset's reference notebook.
  * Also reported: the leaky random split of the reference notebook, and a label-
    efficiency curve (training on 2 ... 64 labelled configurations).

Outputs in results/task1_gwtcw/:
    cv_metrics.csv     AUC and balanced accuracy per method, setting, head and fold
    fewshot.csv        AUC against the number of labelled training configurations
    data_summary.json  signal counts and the number of sign-inverted configurations

    python scripts/task1_gwtcw.py            # about 20 minutes on two CPU cores
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gwtsfm import baselines, heads  # noqa: E402
from gwtsfm.config import DATA, EMBEDDINGS, PAPER_MODELS, RESULTS, SEED  # noqa: E402
from gwtsfm.splits import grouped_kfold  # noqa: E402

FROZEN, TRAINED, PROBE = "foundation model (frozen)", "trained from scratch", "artefact probe"


# ------------------------------------------------------------------------------ data
def load_gwtcw():
    """All 3,120 signals (scaled to unit std), labels (1 = damaged) and configuration ids."""
    z = np.load(DATA / "gwtcw" / "gwtcw.npz")
    X = np.concatenate([z["B"], z["D"]]).astype(np.float32)
    X /= X.std()
    n = len(z["B"])
    y = np.r_[np.zeros(n, int), np.ones(n, int)]
    i = np.arange(n)
    cfg = np.tile((i // 120) * 12 + i % 12, 2)   # block * 12 + path, same for healthy and damaged
    return X, y, cfg


def first_repeat(n_rows: int) -> np.ndarray:
    """Mask keeping the first of the 10 repeated acquisitions of every configuration."""
    return np.arange(n_rows) % 1560 % 120 // 12 == 0


def grouped_folds(cfg, k=5):
    """5-fold split in which all signals of a configuration fall in the same fold."""
    shuffled = np.random.default_rng(SEED).permutation(cfg.max() + 1)  # random fold assignment
    return grouped_kfold(shuffled[cfg], k)          # GroupKFold, the same on every platform


def feature_sets(X, Xall, keep):
    """Every feature set: name -> (family, features of x, features of -x).

    Returns the sets for the 312 kept signals, and for all 3,120 signals where available
    (used only by the leaky random split).
    """
    feats, feats_all = {}, {}
    for key, name in PAPER_MODELS.items():
        f = EMBEDDINGS / "gwtcw" / f"{key}.npz"
        if not f.exists():
            continue
        z = np.load(f)
        P, N = z["pos"], z["neg"]          # embeddings of x and of -x
        if len(P) == len(Xall):            # embedded with --all-repeats
            feats_all[name] = (FROZEN, P, N)
            P, N = P[keep], N[keep]
        feats[name] = (FROZEN, P, N)
    feats["Raw signal"] = (TRAINED, X, -X)
    spectrum = np.abs(np.fft.rfft(X, axis=1))[:, :200]          # identical for x and -x
    feats["|FFT| spectrum"] = (TRAINED, spectrum, spectrum)
    peak = X[np.arange(len(X)), np.abs(X).argmax(1)]            # value of the largest peak
    feats["Peak polarity only"] = (PROBE, np.sign(peak)[:, None], -np.sign(peak)[:, None])
    feats_all["Raw signal"] = (TRAINED, Xall, -Xall)
    return feats, feats_all


# ----------------------------------------------------------------------------- heads
def damage_score(make_model, Fp, Fn, y, train, test, sign_symmetric):
    """Fit a head on the training rows and return P(damaged) for the test rows.

    Fp, Fn: features of x and of -x. Sign-symmetric: train on both and average the
    prediction over both.
    """
    m = make_model()
    if sign_symmetric:
        m.fit(np.r_[Fp[train], Fn[train]], np.r_[y[train], y[train]])
        return 0.5 * (m.predict_proba(Fp[test])[:, 1] + m.predict_proba(Fn[test])[:, 1])
    m.fit(Fp[train], y[train])
    return m.predict_proba(Fp[test])[:, 1]


def one_class_score(Fp, Fn, y, train, test, sign_symmetric, k=5):
    """No damage labels: mean distance to the k nearest healthy training signals.

    Sign-symmetric: the reference holds healthy x and -x, and a test signal gets the
    smaller of the distances of x and -x.
    """
    healthy = train[y[train] == 0]
    ref = np.r_[Fp[healthy], Fn[healthy]] if sign_symmetric else Fp[healthy]
    sc = StandardScaler().fit(ref)
    nn = NearestNeighbors(n_neighbors=k).fit(sc.transform(ref))
    d = nn.kneighbors(sc.transform(Fp[test]))[0].mean(1)
    if sign_symmetric:
        d = np.minimum(d, nn.kneighbors(sc.transform(Fn[test]))[0].mean(1))
    return d


def logistic():
    return heads.logistic_regression(C=1.0)


# ------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(RESULTS / "task1_gwtcw"))
    ap.add_argument("--skip-trained", action="store_true", help="skip MiniRocket and the 1D-CNN")
    ap.add_argument("--fewshot", type=int, nargs="+", default=[2, 4, 8, 16, 32, 64])
    ap.add_argument("--draws", type=int, default=5, help="random draws per few-shot size")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    Xall, yall, cfgall = load_gwtcw()
    keep = first_repeat(len(Xall))
    X, y, cfg = Xall[keep], yall[keep], cfgall[keep]
    folds = grouped_folds(cfg)
    feats, feats_all = feature_sets(X, Xall, keep)

    rows = []

    def record(method, family, mode, head, fold, y_true, s, seconds=np.nan, threshold=0.5):
        rows.append(dict(method=method, family=family, mode=mode, head=head, fold=fold,
                         seconds=seconds, auc=roc_auc_score(y_true, s),
                         bacc=balanced_accuracy_score(y_true, s > threshold)))

    # 1. grouped cross-validation, with and without sign symmetry
    for mode, sym in (("as recorded", False), ("sign-symmetric", True)):
        for name, (family, Fp, Fn) in feats.items():
            for i, (tr, te) in enumerate(folds):
                record(name, family, mode, "logistic regression", i, y[te],
                       damage_score(logistic, Fp, Fn, y, tr, te, sym))
                if family == PROBE:
                    continue
                record(name, family, mode, "kNN (k=5)", i, y[te],
                       damage_score(heads.knn_classifier, Fp, Fn, y, tr, te, sym))
                d = one_class_score(Fp, Fn, y, tr, te, sym)
                record(name, family, mode, "one-class (no damage labels)", i, y[te], d,
                       threshold=np.median(d))
            print(f"[{mode}] {name} done", flush=True)
        if not args.skip_trained:
            for i, (tr, te) in enumerate(folds):
                t0 = time.time()
                s = baselines.sigmoid(baselines.minirocket_decision(X, y, tr, te, sign_symmetric=sym))
                record("MiniRocket", TRAINED, mode, "ridge", i, y[te], s, time.time() - t0)
                t0 = time.time()
                s = baselines.train_cnn(X, y, tr, te, cfg, n_classes=2, epochs=40, sign_symmetric=sym)
                record("1D-CNN (reference notebook)", TRAINED, mode, "end-to-end", i, y[te], s,
                       time.time() - t0)
                print(f"[{mode}] trained references, fold {i} done", flush=True)
        pd.DataFrame(rows).to_csv(out / "cv_metrics.csv", index=False)

    # 2. the reference notebook's random 95/5 split over all 3,120 signals: repeats leak
    tr, te = train_test_split(np.arange(len(Xall)), test_size=0.05, random_state=42)
    for name, (family, Fp, Fn) in feats_all.items():
        for head_name, make in (("logistic regression", logistic), ("kNN (k=5)", heads.knn_classifier)):
            record(name, family, "random split (leaky)", head_name, 0, yall[te],
                   damage_score(make, Fp, Fn, yall, tr, te, False))
    if not args.skip_trained:
        record("1D-CNN (reference notebook)", TRAINED, "random split (leaky)", "end-to-end", 0,
               yall[te], baselines.train_cnn(Xall, yall, tr, te, cfgall, n_classes=2, epochs=40))
    pd.DataFrame(rows).to_csv(out / "cv_metrics.csv", index=False)

    # 3. label efficiency: train on m randomly chosen labelled configurations per fold
    rng = np.random.default_rng(SEED)
    fs_rows = []
    for name, (family, Fp, Fn) in feats.items():
        if family == PROBE:
            continue
        for i, (tr, te) in enumerate(folds):
            train_cfgs = np.unique(cfg[tr])
            for m in args.fewshot + [len(train_cfgs)]:          # the last point uses them all
                for d in range(args.draws if m < len(train_cfgs) else 1):
                    chosen = rng.choice(train_cfgs, min(m, len(train_cfgs)), replace=False)
                    sub = tr[np.isin(cfg[tr], chosen)]
                    s = damage_score(logistic, Fp, Fn, y, sub, te, True)
                    fs_rows.append(dict(method=name, family=family, fold=i, n_configs=m, draw=d,
                                        n_signals=len(sub), auc=roc_auc_score(y[te], s)))
        print(f"[label efficiency] {name} done", flush=True)
    pd.DataFrame(fs_rows).to_csv(out / "fewshot.csv", index=False)

    # sign-inverted configurations: first healthy and first damaged signal anti-correlated
    flipped = sum(np.corrcoef(X[(cfg == c) & (y == 0)][0], X[(cfg == c) & (y == 1)][0])[0, 1] < 0
                  for c in range(cfg.max() + 1))
    json.dump({"n_signals_used": len(X), "n_signals_all": len(Xall), "n_configs": int(cfg.max() + 1),
               "flipped_configs": int(flipped)}, open(out / "data_summary.json", "w"))


if __name__ == "__main__":
    main()
