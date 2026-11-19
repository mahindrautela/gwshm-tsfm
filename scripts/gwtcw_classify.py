"""GW-TCW damage classification with frozen TSFM embeddings vs trained baselines.

Data: Rautela et al. (2022) tapered composite wing, 1560 baseline + 1560 damaged
signals (1000 samples, 2 MHz).  Rows are 13 blocks x 12 sensor paths x 10
repeated acquisitions; baseline row i and damaged row i share a configuration.

Protocol
--------
* Splits are grouped by configuration (block, path): test paths are never seen in
  training, so near-identical repeats cannot leak across the split.
  A random (ungrouped) split is also reported to show how much it inflates scores.
* Two polarity settings, because 71/156 configurations have sign-inverted damaged
  signals:  "as recorded" and "sign-symmetric" (train on x and -x, score the
  average over x and -x), which removes polarity as a usable cue.
* Foundation models are frozen; only a logistic-regression or k-NN head is fit.
  A zero-label one-class score (distance to the nearest training baselines) is
  also reported.
* Trained-from-scratch references: raw-signal and |FFT| logistic regression,
  MiniRocket + ridge, and the 1D-CNN of the dataset's reference notebook.

python scripts/gwtcw_classify.py --data data/gwtcw/gwtcw.npz --emb data/gwtcw_emb --out results_gwtcw
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, RidgeClassifierCV
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

EMB_NAMES = {"chronos2": "Chronos-2", "chronos_bolt": "Chronos-Bolt (base)", "tirex": "TiRex",
             "moment": "MOMENT-1 (base)", "mantis": "Mantis-8M", "ttm": "TTM-R2"}
SEED = 0


def load(path):
    z = np.load(path)
    X = np.concatenate([z["B"], z["D"]]).astype(np.float32)
    X /= X.std()
    n = len(z["B"])
    y = np.r_[np.zeros(n, int), np.ones(n, int)]
    i = np.arange(n)
    cfg = np.tile((i // 120) * 12 + i % 12, 2)
    return X, y, cfg


def folds(cfg, k=5, seed=SEED):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(cfg.max() + 1)
    return list(GroupKFold(k).split(np.zeros(len(cfg)), groups=perm[cfg]))


# ---------------------------------------------------------------- heads ----------
def lr():
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=3000))


def knn():
    return make_pipeline(StandardScaler(), KNeighborsClassifier(5, metric="cosine"))


def fit_score(make, Fp, Fn, y, tr, te, sym):
    """Fit head on train rows, return P(damage) on test rows."""
    m = make()
    if sym:
        m.fit(np.r_[Fp[tr], Fn[tr]], np.r_[y[tr], y[tr]])
        return 0.5 * (m.predict_proba(Fp[te])[:, 1] + m.predict_proba(Fn[te])[:, 1])
    m.fit(Fp[tr], y[tr])
    return m.predict_proba(Fp[te])[:, 1]


def one_class(Fp, Fn, y, tr, te, sym, k=5):
    """Zero damage labels: distance to the k nearest *baseline* training signals."""
    ref = tr[y[tr] == 0]
    sc = StandardScaler().fit(np.r_[Fp[ref], Fn[ref]] if sym else Fp[ref])
    R = sc.transform(np.r_[Fp[ref], Fn[ref]] if sym else Fp[ref])
    nn = NearestNeighbors(n_neighbors=k).fit(R)
    d = nn.kneighbors(sc.transform(Fp[te]))[0].mean(1)
    if sym:
        d = np.minimum(d, nn.kneighbors(sc.transform(Fn[te]))[0].mean(1))
    return d


def metrics(y, s, thr=0.5):
    return dict(auc=roc_auc_score(y, s), bacc=balanced_accuracy_score(y, s > thr))


# --------------------------------------------------------- trained baselines ------
def minirocket_scores(X, y, tr, te, sym):
    from aeon.transformations.collection.convolution_based import MiniRocket
    Xtr = np.r_[X[tr], -X[tr]] if sym else X[tr]
    ytr = np.r_[y[tr], y[tr]] if sym else y[tr]
    mr = MiniRocket(random_state=SEED).fit(Xtr[:, None, :])
    clf = make_pipeline(StandardScaler(), RidgeClassifierCV(alphas=np.logspace(-3, 3, 10)))
    clf.fit(mr.transform(Xtr[:, None, :]), ytr)
    f = lambda A: clf.decision_function(mr.transform(A[:, None, :]))  # noqa: E731
    s = f(X[te]) if not sym else 0.5 * (f(X[te]) + f(-X[te]))
    return 1 / (1 + np.exp(-s))


def cnn_scores(X, y, tr, te, sym, cfg, epochs=40):
    """1D-CNN of the reference notebook (PyTorch port), early stopping on held-out configs."""
    import torch
    import torch.nn as nn
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    L = X.shape[1]

    def net():
        flat = 128 * ((((((L - 2) // 2 - 2) // 2) - 2) // 2 - 2) // 2)
        return nn.Sequential(
            nn.Conv1d(1, 16, 3), nn.MaxPool1d(2), nn.Conv1d(16, 32, 3), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 3), nn.ReLU(), nn.MaxPool1d(2), nn.Conv1d(64, 128, 3), nn.ReLU(),
            nn.MaxPool1d(2), nn.Flatten(), nn.Linear(flat, 128), nn.ReLU(), nn.Linear(128, 16),
            nn.ReLU(), nn.Linear(16, 1))

    g = np.unique(cfg[tr])
    rng = np.random.default_rng(SEED)
    val_g = rng.choice(g, max(1, int(0.15 * len(g))), replace=False)
    va = tr[np.isin(cfg[tr], val_g)]
    fit = tr[~np.isin(cfg[tr], val_g)]
    Xf, yf = (np.r_[X[fit], -X[fit]], np.r_[y[fit], y[fit]]) if sym else (X[fit], y[fit])
    T = lambda A: torch.from_numpy(np.ascontiguousarray(A, dtype=np.float32))[:, None, :]  # noqa: E731
    model = net()
    opt = torch.optim.Adam(model.parameters(), 1e-3)
    lossf = nn.BCEWithLogitsLoss()
    best, state = np.inf, None
    for _ in range(epochs):
        model.train()
        p = rng.permutation(len(Xf))
        for i in range(0, len(p), 32):
            b = p[i:i + 32]
            opt.zero_grad()
            loss = lossf(model(T(Xf[b]))[:, 0], torch.from_numpy(yf[b].astype(np.float32)))
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = lossf(model(T(X[va]))[:, 0], torch.from_numpy(y[va].astype(np.float32))).item()
        if vl < best:
            best, state = vl, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(state)
    with torch.no_grad():
        s = torch.sigmoid(model(T(X[te]))[:, 0]).numpy()
        if sym:
            s = 0.5 * (s + torch.sigmoid(model(T(-X[te]))[:, 0]).numpy())
    return s


# ------------------------------------------------------------------- main --------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--emb", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip-trained", action="store_true")
    ap.add_argument("--fewshot", type=int, nargs="+", default=[2, 4, 8, 16, 32, 64])
    ap.add_argument("--draws", type=int, default=5)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    Xall, yall, cfgall = load(args.data)
    # one acquisition per configuration (the 10 repeats correlate at r = 0.99998)
    keep = np.arange(len(Xall)) % 1560 % 120 // 12 == 0
    X, y, cfg = Xall[keep], yall[keep], cfgall[keep]
    F = folds(cfg)

    feats, feats_all = {}, {}
    for k, name in EMB_NAMES.items():
        f = Path(args.emb) / f"{k}.npz"
        if f.exists():
            z = np.load(f)
            P, N = z["pos"], z["neg"]
            if len(P) == len(Xall):
                feats_all[name] = ("foundation model (frozen)", P, N)
                P, N = P[keep], N[keep]
            feats[name] = ("foundation model (frozen)", P, N)
    feats["Raw signal"] = ("trained from scratch", X, -X)
    spec = np.abs(np.fft.rfft(X, axis=1))[:, :200]
    feats["|FFT| spectrum"] = ("trained from scratch", spec, spec)
    pk = X[np.arange(len(X)), np.abs(X).argmax(1)]
    feats["Peak polarity only"] = ("artefact probe", np.sign(pk)[:, None], -np.sign(pk)[:, None])

    rows = []

    def add(method, family, mode, head, fold, y_te, s, secs=np.nan, thr=0.5):
        rows.append(dict(method=method, family=family, mode=mode, head=head, fold=fold,
                         seconds=secs, **metrics(y_te, s, thr)))

    for mode, sym in (("as recorded", False), ("sign-symmetric", True)):
        for name, (fam, Fp, Fn) in feats.items():
            for i, (tr, te) in enumerate(F):
                add(name, fam, mode, "logistic regression", i, y[te], fit_score(lr, Fp, Fn, y, tr, te, sym))
                if fam != "artefact probe":
                    add(name, fam, mode, "kNN (k=5)", i, y[te], fit_score(knn, Fp, Fn, y, tr, te, sym))
                    d = one_class(Fp, Fn, y, tr, te, sym)
                    add(name, fam, mode, "one-class (no damage labels)", i, y[te], d,
                        thr=np.median(d))
            print(f"[{mode}] {name} done", flush=True)
        if not args.skip_trained:
            for i, (tr, te) in enumerate(F):
                t0 = time.time()
                add("MiniRocket", "trained from scratch", mode, "ridge", i, y[te],
                    minirocket_scores(X, y, tr, te, sym), time.time() - t0)
                t0 = time.time()
                add("1D-CNN (reference notebook)", "trained from scratch", mode, "end-to-end", i, y[te],
                    cnn_scores(X, y, tr, te, sym, cfg), time.time() - t0)
                print(f"[{mode}] trained baselines fold {i} done", flush=True)
        pd.DataFrame(rows).to_csv(out / "cv_metrics.csv", index=False)

    # random (ungrouped) split over ALL repeats, as in the reference notebook: shows leakage
    idx = np.arange(len(Xall))
    tr, te = train_test_split(idx, test_size=0.05, random_state=42)
    feats_all["Raw signal"] = ("trained from scratch", Xall, -Xall)
    for name, (fam, Fp, Fn) in feats_all.items():
        add(name, fam, "random split (leaky)", "logistic regression", 0, yall[te],
            fit_score(lr, Fp, Fn, yall, tr, te, False))
        add(name, fam, "random split (leaky)", "kNN (k=5)", 0, yall[te],
            fit_score(knn, Fp, Fn, yall, tr, te, False))
    if not args.skip_trained:
        add("1D-CNN (reference notebook)", "trained from scratch", "random split (leaky)", "end-to-end",
            0, yall[te], cnn_scores(Xall, yall, tr, te, False, cfgall))
    pd.DataFrame(rows).to_csv(out / "cv_metrics.csv", index=False)

    # few-shot learning curve: number of labelled training configurations
    rng = np.random.default_rng(SEED)
    fs_rows = []
    for name, (fam, Fp, Fn) in feats.items():
        if fam == "artefact probe":
            continue
        for i, (tr, te) in enumerate(F):
            g = np.unique(cfg[tr])
            for m in args.fewshot + [len(g)]:
                for d in range(args.draws if m < len(g) else 1):
                    sub = tr[np.isin(cfg[tr], rng.choice(g, min(m, len(g)), replace=False))]
                    s = fit_score(lr, Fp, Fn, y, sub, te, True)
                    fs_rows.append(dict(method=name, family=fam, fold=i, n_configs=m, draw=d,
                                        n_signals=len(sub), auc=roc_auc_score(y[te], s)))
        print(f"[few-shot] {name} done", flush=True)
    pd.DataFrame(fs_rows).to_csv(out / "fewshot.csv", index=False)
    json.dump({"n_signals_used": len(X), "n_signals_all": len(Xall), "n_configs": int(cfg.max() + 1),
               "flipped_configs": int(np.sum([np.corrcoef(X[(cfg == c) & (y == 0)][0],
                                                          X[(cfg == c) & (y == 1)][0])[0, 1] < 0
                                              for c in range(cfg.max() + 1)]))},
              open(out / "data_summary.json", "w"))


if __name__ == "__main__":
    main()
