"""Task 2: unsupervised damage (delamination) detection on NASA PCoE CFRP, layup 2.

Only healthy signals are ever used to fit anything.

Units: one signal = (measurement, frequency, path); 252 signals per measurement.
Healthy = early life (<= 3 fatigue cycles); damaged = 20,000 cycles (as in Rautela
et al., Composite Structures 2022); later cycles are used for the damage-growth curve.

Settings
--------
within-specimen  reference = 4 of the specimen's 8 healthy measurements (5 random
                 draws); test = the other 4 healthy + the 20k-cycle measurement.
cross-specimen   reference = the healthy measurements of the other 3 specimens;
                 test = the held-out specimen (8 healthy + 20k cycles).

Detectors on each feature set (frozen TSFM embedding, raw signal, |FFT|)
-----------------------------------------------------------------------
kNN distance     distance to the nearest reference signal (any path/frequency)
same-path        distance to the mean reference signal of the same path and frequency
PCA+ocSVM        PCA (10) + one-class SVM, pooled (the paper's ML approach)
Physics reference (raw signals): baseline subtraction (residual energy ratio and
1 - correlation against the mean same-path reference signal).

python scripts/nasa_anomaly.py --data data/nasa/l2.npz --emb data/nasa_emb --out results_nasa
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gwtsfm import evalkit as ek  # noqa: E402

EMB = {"chronos2": "Chronos-2", "chronos_bolt": "Chronos-Bolt (base)", "tirex": "TiRex",
       "moment": "MOMENT-1 (base)", "mantis": "Mantis-8M", "ttm": "TTM-R2"}
L = 1024


def same_path(Fr, Ft):
    """Fr (nr, F, P, d) reference, Ft (nt, F, P, d) -> distance to same-(f,p) reference mean."""
    mu = Fr.mean(0)
    sd = Fr.reshape(-1, Fr.shape[-1]).std(0) + 1e-6
    return np.sqrt((((Ft - mu[None]) / sd) ** 2).mean(-1))


def scores(name, kind, Fr, Ft):
    """All detectors for one feature set. Fr/Ft (n, F, P, d). Returns {detector: (nt, F, P)}."""
    nt, nf, npth = Ft.shape[:3]
    r, t = Fr.reshape(-1, Fr.shape[-1]), Ft.reshape(-1, Ft.shape[-1])
    out = {"kNN distance": ek.knn_distance(r, t).reshape(nt, nf, npth),
           "same-path distance": same_path(Fr, Ft),
           "PCA+ocSVM": ek.pca_ocsvm(r, t).reshape(nt, nf, npth)}
    if kind == "raw":
        mu = Fr.mean(0)[None]
        out["baseline subtraction (residual energy)"] = (((Ft - mu) ** 2).sum(-1) /
                                                          (mu ** 2).sum(-1))
        xc, mc = Ft - Ft.mean(-1, keepdims=True), mu - mu.mean(-1, keepdims=True)
        out["baseline subtraction (1 - correlation)"] = 1 - (xc * mc).sum(-1) / np.sqrt(
            (xc ** 2).sum(-1) * (mc ** 2).sum(-1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--emb", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--draws", type=int, default=5)
    ap.add_argument("--damage-cycles", type=int, default=20000)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    z = np.load(args.data)
    X = z["X"][..., :L].astype(np.float32)
    spec, cyc = z["specimen"], z["cycles"]
    feats = {"Raw signal": ("raw", X / X.std()),
             "|FFT| spectrum": ("fft", np.abs(np.fft.rfft(X, axis=-1)).astype(np.float32))}
    for k, n in EMB.items():
        f = Path(args.emb) / f"{k}.npy"
        if f.exists():
            feats[n] = ("tsfm", np.load(f).astype(np.float32))
    print("feature sets:", list(feats), flush=True)

    rows, curves = [], []
    rng = np.random.default_rng(0)
    healthy = cyc <= 3
    specimens = sorted(set(spec))
    plans = []  # (setting, draw, specimen, ref idx, test idx)
    for s in specimens:
        h = np.flatnonzero(healthy & (spec == s))
        later = np.flatnonzero(~healthy & (spec == s))
        for d in range(args.draws):
            ref = np.sort(rng.choice(h, len(h) // 2, replace=False))
            plans.append(("within-specimen", d, s, ref, np.r_[np.setdiff1d(h, ref), later]))
        ref = np.flatnonzero(healthy & (spec != s))
        plans.append(("cross-specimen", 0, s, ref, np.r_[h, later]))

    for name, (kind, F) in feats.items():
        for setting, d, s, ref, test in plans:
            sc = scores(name, kind, F[ref], F[test])
            lab = np.where(cyc[test] <= 3, 0, np.where(cyc[test] == args.damage_cycles, 1, -1))
            for det, S in sc.items():
                m = lab >= 0
                ys = np.repeat(lab[m], S.shape[1] * S.shape[2])
                rows.append(dict(method=name, kind=kind, detector=det, setting=setting, draw=d,
                                 specimen=s, auc_signal=roc_auc_score(ys, S[m].ravel()),
                                 n_pos=int(ys.sum()), n_neg=int((1 - ys).sum())))
                meas = S.reshape(len(test), -1).mean(1)
                base = meas[lab == 0]
                for i, t in enumerate(test):
                    curves.append(dict(method=name, detector=det, setting=setting, draw=d,
                                       specimen=s, cycles=int(cyc[t]), healthy_test=bool(lab[i] == 0),
                                       score=float(meas[i]),
                                       z=float((meas[i] - base.mean()) / (base.std() + 1e-12))))
        print(f"{name} done", flush=True)
    R = pd.DataFrame(rows)
    C = pd.DataFrame(curves)
    R.to_csv(out / "signal_auc.csv", index=False)
    C.to_csv(out / "measurement_scores.csv", index=False)
    # measurement-level: rank the 20k-cycle measurement among the specimen's healthy test
    # measurements (score = mean over its 252 signals); no normalisation, no label use
    mt = C[(C.healthy_test) | (C.cycles == args.damage_cycles)].copy()
    mt["y"] = (~mt.healthy_test).astype(int)
    M = (mt.groupby(["method", "detector", "setting", "draw", "specimen"])
           .apply(lambda g: pd.Series({
               "auc_measurement": roc_auc_score(g.y, g.score) if g.y.nunique() == 2 else np.nan,
               "detected_zero_fa": float(g.score[g.y == 1].min() > g.score[g.y == 0].max())}),
               include_groups=False).reset_index())
    M.to_csv(out / "measurement_auc.csv", index=False)
    summ = (R.groupby(["setting", "method", "detector"])["auc_signal"].agg(["mean", "std"]).round(3))
    print(summ.to_string())
    print(M.groupby(["setting", "method", "detector"])[["auc_measurement", "detected_zero_fa"]]
          .mean().round(3).to_string())
    json.dump({"n_measurements": int(len(X)), "healthy_per_specimen": {s: int((healthy & (spec == s)).sum())
               for s in specimens}, "signals_per_measurement": int(X.shape[1] * X.shape[2])},
              open(out / "summary.json", "w"))


if __name__ == "__main__":
    main()
