"""E1 step 2: unsupervised damage detection under temperature change (OGW temperature, 40 kHz).

Question: does the same-path embedding distance stay a useful damage index when the
temperature of a test differs from that of the healthy references, and how does it compare
with classical temperature compensation?

Protocol (run ext_temperature_embed.py first):
  references   healthy measurements of the first undamaged cycle (udam-1) on a temperature
               grid with spacing s in {0.5, 2, 5, 10, 20, 40} degC, starting at 20.5 degC: for
               each grid point, the measurement of the heating half and that of the cooling
               half of the cycle with the nearest recorded temperature
  healthy test the second undamaged cycle (udam-2): another session, all temperatures
  damaged test the four sessions with the defect at D04, D12, D16 or D24
  gap          |T_test - nearest reference temperature| (temperatures as recorded)

Detectors, per path, then averaged over the 36 paths of a measurement ("mean") or taken at
the most changed path ("max"):
  raw signal   residual energy |x - b|^2 / |b|^2 against
                 NT       the reference measurement at the nearest temperature
                 OBS      the best reference of that path (optimal baseline selection)
                 BSS      the NT reference stretched in time, best factor in 0.96-1.04
                          (baseline signal stretch)
                 OBS+BSS  stretch applied to the 3 nearest-temperature references
  embedding    distance of Eq. 1 (features scaled by the reference standard deviation) to
                 mean     the mean reference embedding of the path (as in Task 2)
                 NT       the embedding of the nearest-temperature reference
                 OBS      the closest reference embedding of the path

Thresholds for a fixed false-alarm rate (1% and 5%) come from healthy data only: every
udam-1 measurement is scored leave-one-out against the references without itself, and the
threshold is the (1 - rate) quantile. The realised false-alarm rate on udam-2 is reported
next to the detection rate.

Controls (damage is confounded with recording time: sessions run 13 Dec 2018 - 13 Jan 2019):
  drift     AUC of udam-2 against the leave-one-out udam-1 scores: both healthy, so 0.5
            unless the plate or the system drifted between sessions
  locality  Spearman correlation between a path's score change (damaged session over udam-2)
            and the path's proximity to the defect. Drift changes all paths; damage should
            change the paths through it.
  timing    lag (cross-correlation, sub-sample) of each session against udam-1 at the same
            temperature: a trigger offset between sessions would be a shortcut

Outputs in results/ext_temperature/: scores.csv, summary.csv, by_gap.csv, locality.csv,
timing.csv

    python scripts/ext_temperature.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gwtsfm.config import DATA, EMBEDDINGS, MODELS, RESULTS  # noqa: E402
from gwtsfm.ogw import DAMAGE_XY, TRANSDUCER_XY, path_distance  # noqa: E402

GRIDS = [0.5, 2, 5, 10, 20, 40]
ALPHAS = np.arange(0.96, 1.0401, 0.001)          # stretch factors for BSS
GAP_BINS = [0, 1, 2.5, 5, 10, 20, 40]
FARS = [0.01, 0.05]
DAMAGED = ["D04", "D12", "D16", "D24"]


# ------------------------------------------------------------------------------ references
def reference_set(temp, pool, s):
    """udam-1 measurements nearest each grid temperature, in the heating and in the cooling half."""
    grid = np.arange(20.5, 60.0 + 1e-9, s)
    peak = int(np.argmax(temp[pool]))                    # pool is in time order: up, then down
    refs = {int(half[np.argmin(np.abs(temp[half] - g))]) for g in grid
            for half in (pool[:peak + 1], pool[peak:])}
    return np.array(sorted(refs)), grid


# ------------------------------------------------------------------------------ raw-signal detectors
def residual_energy(x, b):
    return ((x - b) ** 2).sum(-1) / (b ** 2).sum(-1)


def stretched(b):
    """b (P, L) stretched by every factor in ALPHAS: (A, P, L), b_a(t) = b(t / a)."""
    L = b.shape[-1]
    pos = np.clip(np.arange(L)[None] / ALPHAS[:, None], 0, L - 1)
    i0 = np.floor(pos).astype(int)
    i1 = np.minimum(i0 + 1, L - 1)
    w = (pos - i0)[:, None, :]
    return b[:, i0].transpose(1, 0, 2) * (1 - w) + b[:, i1].transpose(1, 0, 2) * w


def raw_scores(X, temp, refs, t):
    """Per-path scores (P,) of measurement t for the four raw detectors."""
    order = refs[np.argsort(np.abs(temp[refs] - temp[t]), kind="stable")]
    nt = order[0]
    out = {"raw NT": residual_energy(X[t], X[nt]),
           "raw OBS": residual_energy(X[t][None], X[refs]).min(0)}
    out["raw BSS"] = residual_energy(X[t][None], stretched(X[nt])).min(0)
    out["raw OBS+BSS"] = np.min([residual_energy(X[t][None], stretched(X[r])).min(0) for r in order[:3]], 0)
    return out


# ------------------------------------------------------------------------------ embedding detectors
def embedding_scores(F, temp, refs, t, sd):
    nt = refs[np.argmin(np.abs(temp[refs] - temp[t]))]
    d = lambda a, b: np.sqrt((((a - b) / sd) ** 2).mean(-1))  # noqa: E731  Eq. 1
    return {"mean": d(F[t], F[refs].mean(0)), "NT": d(F[t], F[nt]),
            "OBS": d(F[t][None], F[refs]).min(0)}


# ------------------------------------------------------------------------------ main loop
def all_scores(X, temp, session, feats):
    """Long table: one row per (detector, grid, measurement) with mean- and max-path scores.

    Measurements of udam-1 are scored leave-one-out (role 'calibration'); the others are
    the tests. Also returns per-path scores for the locality control.
    """
    pool = np.flatnonzero(session == "udam-1")
    tests = np.flatnonzero(session != "udam-1")
    rows, per_path = [], {}
    for s in GRIDS:
        refs, _ = reference_set(temp, pool, s)
        sds = {name: F[refs].reshape(-1, F.shape[-1]).std(0) + 1e-6 for name, F in feats.items()}
        for t in np.r_[pool, tests]:
            r = refs[refs != t] if t in pool else refs
            gap = float(np.min(np.abs(temp[r] - temp[t])))
            res = raw_scores(X, temp, r, t)
            for name, F in feats.items():
                sd = sds[name] if t not in pool else F[r].reshape(-1, F.shape[-1]).std(0) + 1e-6
                for k, v in embedding_scores(F, temp, r, t, sd).items():
                    res[f"{name} | {k}"] = v
            for det, v in res.items():
                rows.append(dict(detector=det, grid=s, idx=int(t), session=session[t], temp=float(temp[t]),
                                 gap=gap, role="calibration" if t in pool else "test",
                                 score_mean=float(v.mean()), score_max=float(v.max())))
                per_path[(det, s, int(t))] = v.astype(np.float32)
        print(f"grid {s} degC: {len(refs)} references", flush=True)
    return pd.DataFrame(rows), per_path


def summarise(S):
    out, gaps = [], []
    for (det, s), g in S.groupby(["detector", "grid"]):
        cal = g[g.role == "calibration"]
        te = g[g.role == "test"]
        neg, pos = te[te.session == "udam-2"], te[te.session.isin(DAMAGED)]
        for agg in ["mean", "max"]:
            c = f"score_{agg}"
            row = dict(detector=det, grid=s, agg=agg,
                       auc=roc_auc_score(np.r_[np.zeros(len(neg)), np.ones(len(pos))], np.r_[neg[c], pos[c]]),
                       drift_auc=roc_auc_score(np.r_[np.zeros(len(cal)), np.ones(len(neg))], np.r_[cal[c], neg[c]]))
            for d in DAMAGED:
                p = te[te.session == d]
                row[f"auc_{d}"] = roc_auc_score(np.r_[np.zeros(len(neg)), np.ones(len(p))], np.r_[neg[c], p[c]])
            for a in FARS:
                thr = np.quantile(cal[c], 1 - a)
                row[f"tpr@{a:g}"] = float((pos[c] > thr).mean())
                row[f"far@{a:g}"] = float((neg[c] > thr).mean())
            out.append(row)
            for lo, hi in zip(GAP_BINS[:-1], GAP_BINS[1:]):
                n_, p_ = neg[(neg.gap >= lo) & (neg.gap < hi)], pos[(pos.gap >= lo) & (pos.gap < hi)]
                gr = dict(detector=det, grid=s, agg=agg, gap_lo=lo, gap_hi=hi, n_healthy=len(n_), n_damaged=len(p_))
                if len(n_) and len(p_):
                    gr["auc"] = roc_auc_score(np.r_[np.zeros(len(n_)), np.ones(len(p_))], np.r_[n_[c], p_[c]])
                for a in FARS:
                    thr = np.quantile(cal[c], 1 - a)
                    gr[f"tpr@{a:g}"] = float((p_[c] > thr).mean()) if len(p_) else np.nan
                    gr[f"far@{a:g}"] = float((n_[c] > thr).mean()) if len(n_) else np.nan
                gaps.append(gr)
    return pd.DataFrame(out), pd.DataFrame(gaps)


def locality(S, per_path, cross):
    """Spearman(path score change, proximity of the path to the defect), per damaged session."""
    tpos = np.array([TRANSDUCER_XY[k] for k in range(1, 13)], float)
    rows = []
    for (det, s), g in S[S.role == "test"].groupby(["detector", "grid"]):
        base = np.mean([per_path[(det, s, i)] for i in g[g.session == "udam-2"].idx], 0)
        for d in DAMAGED:
            dist = path_distance(cross, tpos, np.array([DAMAGE_XY[int(d[1:])]], float))[0]
            change = np.mean([per_path[(det, s, i)] for i in g[g.session == d].idx], 0) / base
            rows.append(dict(detector=det, grid=s, damage=d, spearman=spearmanr(change, -dist)[0]))
    return pd.DataFrame(rows)


def timing(X, temp, session):
    """Median lag (samples at 1 MHz) of each session against udam-1 at the nearest temperature."""
    pool = np.flatnonzero(session == "udam-1")
    rows = []
    for sname in dict.fromkeys(session):
        if sname == "udam-1":
            continue
        lags = []
        for t in np.flatnonzero(session == sname)[::8]:
            r = pool[np.argmin(np.abs(temp[pool] - temp[t]))]
            a, b = X[t], X[r]
            n = 2 * a.shape[-1]
            cc = np.fft.irfft(np.fft.rfft(a, n) * np.conj(np.fft.rfft(b, n)), n)
            cc = np.concatenate([cc[:, -50:], cc[:, :51]], 1)          # lags -50 .. 50
            k = cc.argmax(1)
            y0, y1, y2 = cc[np.arange(len(k)), np.clip(k - 1, 0, 100)], cc[np.arange(len(k)), k], \
                cc[np.arange(len(k)), np.clip(k + 1, 0, 100)]
            lags.append(k - 50 + 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2 + 1e-12))  # parabolic peak
        lags = np.concatenate(lags)
        rows.append(dict(session=sname, median_lag_samples=float(np.median(lags)),
                         iqr=float(np.subtract(*np.percentile(lags, [75, 25])))))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(RESULTS / "ext_temperature"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    z = np.load(DATA / "ogw_temp" / "signals_f40kHz.npz")
    X, session, temp, cross = z["X"].astype(np.float64), z["session"], z["temp"], z["cross"]
    feats = {name: np.load(EMBEDDINGS / "ogw_temp" / f"{key}.npy").astype(np.float32)
             for key, name in MODELS.items() if (EMBEDDINGS / "ogw_temp" / f"{key}.npy").exists()}
    print("feature sets:", list(feats), flush=True)
    timing(X, temp, session).to_csv(out / "timing.csv", index=False)
    S, per_path = all_scores(X, temp, session, feats)
    S.to_csv(out / "scores.csv", index=False)
    summary, by_gap = summarise(S)
    summary.to_csv(out / "summary.csv", index=False)
    by_gap.to_csv(out / "by_gap.csv", index=False)
    locality(S, per_path, cross).to_csv(out / "locality.csv", index=False)
    print(summary[summary["agg"] == "mean"].pivot_table(index="detector", columns="grid", values="auc").round(3).to_string())


if __name__ == "__main__":
    main()
