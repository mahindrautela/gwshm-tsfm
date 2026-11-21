"""Task 2: unsupervised delamination detection on NASA PCoE CFRP coupons (layup 2).

Only healthy measurements are ever used to fit anything.

Units: one signal = (measurement, frequency, path); 252 signals per measurement.
Healthy = early life (<= 3 fatigue cycles, 8 per specimen); damaged = 20,000 cycles (as in
Rautela et al., Composite Structures 2022). Later cycles give the damage-growth curve.

Two settings:
  within-specimen  reference = 4 of the specimen's 8 healthy measurements (5 random draws);
                   test = its other 4 healthy measurements and all later ones
  cross-specimen   reference = the healthy measurements of the other 3 specimens;
                   test = all measurements of the held-out specimen

Detectors, run on every feature set (frozen embeddings, raw signal, |FFT| spectrum):
  same-path distance  distance to the mean reference signal of the same path and frequency
  kNN distance        distance to the nearest reference signal of any path
  PCA+ocSVM           PCA (10) + one-class SVM pooled over all signals (Rautela et al. 2022)
and, on raw signals only, classical baseline subtraction against the same-path reference
mean: residual energy, and 1 - correlation.

Outputs in results/task2_nasa/:
    signal_auc.csv          signal-level AUC (healthy test vs 20k-cycle signals)
    measurement_scores.csv  score of every test measurement (mean over its 252 signals)
    measurement_auc.csv     per specimen: is the 20k measurement above every healthy one?
    growth_curve.csv        score of later measurements relative to the healthy level
    summary.json            counts

    python scripts/task2_nasa.py             # a few minutes
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
from gwtsfm import heads  # noqa: E402
from gwtsfm.config import DATA, EMBEDDINGS, PAPER_MODELS, RESULTS  # noqa: E402

L = 1024               # samples per signal (0.85 ms at 1.2 MHz)
DAMAGED_CYCLES = 20000
HEALTHY_MAX_CYCLES = 3


# ------------------------------------------------------------------------ detectors
def same_path_distance(Fr, Ft):
    """Distance of each test signal to the mean reference of its own path and frequency.

    Fr: reference features (n_ref, F, P, d); Ft: test features (n_test, F, P, d).
    Features are scaled by their standard deviation over all reference signals, and the
    distance is the root-mean-square difference over features (Eq. 1 of the paper).
    """
    mu = Fr.mean(0)                                         # (F, P, d)
    sd = Fr.reshape(-1, Fr.shape[-1]).std(0) + 1e-6         # (d,)
    return np.sqrt((((Ft - mu[None]) / sd) ** 2).mean(-1))  # (n_test, F, P)


def detector_scores(kind, Fr, Ft):
    """All detectors for one feature set. Returns {detector name: scores (n_test, F, P)}."""
    nt, nf, npth = Ft.shape[:3]
    r, t = Fr.reshape(-1, Fr.shape[-1]), Ft.reshape(-1, Ft.shape[-1])  # pool all signals
    out = {"kNN distance": heads.knn_distance(r, t).reshape(nt, nf, npth),
           "same-path distance": same_path_distance(Fr, Ft),
           "PCA+ocSVM": heads.pca_ocsvm(r, t).reshape(nt, nf, npth)}
    if kind == "raw":  # classical baseline subtraction against the same-path reference mean
        mu = Fr.mean(0)[None]
        out["baseline subtraction (residual energy)"] = ((Ft - mu) ** 2).sum(-1) / (mu ** 2).sum(-1)
        xc, mc = Ft - Ft.mean(-1, keepdims=True), mu - mu.mean(-1, keepdims=True)
        out["baseline subtraction (1 - correlation)"] = 1 - (xc * mc).sum(-1) / np.sqrt(
            (xc ** 2).sum(-1) * (mc ** 2).sum(-1))
    return out


# ------------------------------------------------------------------------------ data
def feature_sets(X):
    """Feature name -> (kind, array (n_meas, F, P, d)) for raw, |FFT| and every embedding."""
    feats = {"Raw signal": ("raw", X / X.std()),
             "|FFT| spectrum": ("fft", np.abs(np.fft.rfft(X, axis=-1)).astype(np.float32))}
    for key, name in PAPER_MODELS.items():
        f = EMBEDDINGS / "nasa" / f"{key}.npy"
        if f.exists():
            feats[name] = ("tsfm", np.load(f).astype(np.float32))
    return feats


def evaluation_plans(specimen, healthy, draws, rng):
    """(setting, draw, specimen, reference measurements, test measurements) for every run."""
    plans = []
    for s in sorted(set(specimen)):
        h = np.flatnonzero(healthy & (specimen == s))
        later = np.flatnonzero(~healthy & (specimen == s))
        for d in range(draws):
            ref = np.sort(rng.choice(h, len(h) // 2, replace=False))
            plans.append(("within-specimen", d, s, ref, np.r_[np.setdiff1d(h, ref), later]))
        plans.append(("cross-specimen", 0, s, np.flatnonzero(healthy & (specimen != s)), np.r_[h, later]))
    return plans


def growth_curve(C):
    """Within-specimen score of later measurements relative to the specimen's healthy level.

    For each draw, a measurement's score is divided by the mean score of the healthy test
    measurements; ratios are averaged over draws and combined over specimens by the
    geometric mean.
    """
    G = C[C.setting == "within-specimen"].copy()
    keys = ["method", "detector", "draw", "specimen"]
    healthy_mean = G[G.healthy_test].groupby(keys)["score"].mean().rename("healthy_mean")
    G = G.join(healthy_mean, on=keys)
    G["ratio"] = G["score"] / G["healthy_mean"]
    later = G[~G.healthy_test]
    curve = (later.groupby(["method", "detector", "specimen", "cycles"])["ratio"].mean().reset_index()
                  .groupby(["method", "detector", "cycles"])["ratio"]
                  .agg(lambda x: float(np.exp(np.mean(np.log(np.maximum(x, 1e-9)))))).reset_index())
    curve["n_specimens"] = later.groupby(["method", "detector", "cycles"])["specimen"].nunique().values
    return curve


# ------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DATA / "nasa" / "layup2.npz"))
    ap.add_argument("--out", default=str(RESULTS / "task2_nasa"))
    ap.add_argument("--draws", type=int, default=5, help="random reference draws within specimen")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    z = np.load(args.data)
    X = z["X"][..., :L].astype(np.float32)        # (measurements, frequencies, paths, samples)
    specimen, cycles = z["specimen"], z["cycles"]
    healthy = cycles <= HEALTHY_MAX_CYCLES
    feats = feature_sets(X)
    plans = evaluation_plans(specimen, healthy, args.draws, np.random.default_rng(0))
    print("feature sets:", list(feats), flush=True)

    rows, scores = [], []
    for name, (kind, F) in feats.items():
        for setting, draw, s, ref, test in plans:
            # label of each test measurement: 0 healthy, 1 damaged (20k cycles), -1 other cycles
            lab = np.where(cycles[test] <= HEALTHY_MAX_CYCLES, 0,
                           np.where(cycles[test] == DAMAGED_CYCLES, 1, -1))
            for det, S in detector_scores(kind, F[ref], F[test]).items():
                m = lab >= 0
                y_sig = np.repeat(lab[m], S.shape[1] * S.shape[2])      # one label per signal
                rows.append(dict(method=name, kind=kind, detector=det, setting=setting, draw=draw,
                                 specimen=s, auc_signal=roc_auc_score(y_sig, S[m].ravel()),
                                 n_pos=int(y_sig.sum()), n_neg=int((1 - y_sig).sum())))
                meas = S.reshape(len(test), -1).mean(1)                   # one score per measurement
                for i, t in enumerate(test):
                    scores.append(dict(method=name, detector=det, setting=setting, draw=draw,
                                       specimen=s, cycles=int(cycles[t]),
                                       healthy_test=bool(lab[i] == 0), score=float(meas[i])))
        print(f"{name} done", flush=True)

    R, C = pd.DataFrame(rows), pd.DataFrame(scores)
    R.to_csv(out / "signal_auc.csv", index=False)
    C.to_csv(out / "measurement_scores.csv", index=False)

    # measurement level: within each run, rank the 20k measurement against the healthy test
    # measurements of the same specimen (no normalisation, no use of labels beyond scoring)
    mt = C[C.healthy_test | (C.cycles == DAMAGED_CYCLES)].copy()
    mt["y"] = (~mt.healthy_test).astype(int)
    M = (mt.groupby(["method", "detector", "setting", "draw", "specimen"])
           .apply(lambda g: pd.Series({
               "auc_measurement": roc_auc_score(g.y, g.score) if g.y.nunique() == 2 else np.nan,
               "detected_zero_fa": float(g.score[g.y == 1].min() > g.score[g.y == 0].max())}),
               include_groups=False).reset_index())
    M.to_csv(out / "measurement_auc.csv", index=False)
    growth_curve(C).to_csv(out / "growth_curve.csv", index=False)

    summary = R.groupby(["setting", "method", "detector"])["auc_signal"].agg(["mean", "std"])
    print(summary.round(3).to_string())
    per_specimen = {s: int((healthy & (specimen == s)).sum()) for s in sorted(set(specimen))}
    json.dump({"n_measurements": int(len(X)), "healthy_per_specimen": per_specimen,
               "signals_per_measurement": int(X.shape[1] * X.shape[2])}, open(out / "summary.json", "w"))


if __name__ == "__main__":
    main()
