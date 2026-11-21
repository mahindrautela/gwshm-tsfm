"""E3 step 3: Task 2 on other boundary conditions and layups, and across layups.

Datasets: layups 1-3 x boundary conditions 1 (loaded), 2 (clamped), 3 (traction free),
from ext_nasa_prepare.py and ext_nasa_embed.py. Layup 2 under condition 1 is the paper's.
Healthy = <= 3 cycles; damaged = the 20,000-cycle measurement (as in the paper; no
independent damage measurement exists).

Settings, each with the same-path distance of Eq. 1 (task2_nasa.same_path_distance) on
every model's embeddings and, on raw signals, baseline subtraction (1 - correlation and
residual energy) and the raw same-path distance:
  within-specimen  half of a specimen's healthy measurements as reference, the other half
                   and its 20k measurement as test; every split if there are at most 5,
                   otherwise 5 random ones (seed 0). Needs >= 2 healthy and a 20k measurement.
  cross-specimen   reference = healthy measurements of the other specimens of the same layup
                   and condition; test = all healthy and the 20k measurement of the specimen
  cross-layup      condition 1 only: reference = healthy measurements of every specimen of
                   another layup; test = each specimen of the target layup

Per run: signal-level AUC (healthy test signals vs 20k signals) and detection at zero false
alarms (the 20k measurement scores above every healthy test measurement of the specimen).

Outputs in results/ext_nasa/: runs.csv (one row per run and detector), summary.csv

    python scripts/ext_nasa.py
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gwtsfm.config import DATA, EMBEDDINGS, MODELS, RESULTS  # noqa: E402
from task2_nasa import DAMAGED_CYCLES, HEALTHY_MAX_CYCLES, same_path_distance  # noqa: E402

L = 1024
BC = {1: "loaded", 2: "clamped", 3: "traction free"}


def load(name):
    """Measurements of one layup/condition: metadata and feature sets (n, F, P, d)."""
    z = np.load(DATA / "nasa" / "ext" / f"{name}.npz")
    X = z["X"][..., :L].astype(np.float32)
    healthy = z["cycles"] <= HEALTHY_MAX_CYCLES
    feats = {"Raw signal": X / X[healthy].std()}
    for key, m in MODELS.items():
        f = EMBEDDINGS / "nasa_ext" / name / f"{key}.npy"
        if f.exists():
            feats[m] = np.load(f).astype(np.float32)
    return dict(specimen=z["specimen"], cycles=z["cycles"], healthy=healthy, feats=feats)


def detectors(name, Fr, Ft):
    out = {"same-path distance": same_path_distance(Fr, Ft)}
    if name == "Raw signal":
        mu = Fr.mean(0)[None]
        out["baseline subtraction (residual energy)"] = ((Ft - mu) ** 2).sum(-1) / (mu ** 2).sum(-1)
        xc, mc = Ft - Ft.mean(-1, keepdims=True), mu - mu.mean(-1, keepdims=True)
        out["baseline subtraction (1 - correlation)"] = 1 - (xc * mc).sum(-1) / np.sqrt(
            (xc ** 2).sum(-1) * (mc ** 2).sum(-1))
    return out


def score_run(rows, info, ref_set, test_set, ref_feats, test_feats, test_cycles):
    """ref_feats/test_feats: {feature name: array}; appends one row per detector."""
    lab = np.where(test_cycles <= HEALTHY_MAX_CYCLES, 0, np.where(test_cycles == DAMAGED_CYCLES, 1, -1))
    m = lab >= 0
    for name in test_feats:
        if name not in ref_feats:
            continue
        for det, S in detectors(name, ref_feats[name][ref_set], test_feats[name][test_set]).items():
            y = np.repeat(lab[m], S.shape[1] * S.shape[2])
            meas = S.reshape(len(test_set), -1).mean(1)
            rows.append(dict(**info, method=name, detector=det,
                             auc_signal=roc_auc_score(y, S[m].ravel()),
                             detected_zero_fa=float(meas[lab == 1].min() > meas[lab == 0].max()),
                             n_ref=len(ref_set), n_healthy_test=int((lab == 0).sum())))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(RESULTS / "ext_nasa"))
    ap.add_argument("--max-splits", type=int, default=5)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    data = {f.stem: load(f.stem) for f in sorted((DATA / "nasa" / "ext").glob("layup*_bc*.npz"))}
    rows = []
    for name, d in data.items():
        layup, bc = int(name[5]), int(name[-1])
        sp, cy, h = d["specimen"], d["cycles"], d["healthy"]
        for s in sorted(set(sp)):
            hs = np.flatnonzero(h & (sp == s))
            dmg = np.flatnonzero((sp == s) & (cy == DAMAGED_CYCLES))
            if len(dmg) == 0:
                continue
            info = dict(layup=layup, condition=BC[bc], specimen=s)
            if len(hs) >= 2:   # within specimen: every split, or 5 random ones
                splits = list(itertools.combinations(hs, len(hs) // 2))
                if len(splits) > args.max_splits:
                    rng = np.random.default_rng(0)
                    splits = [splits[i] for i in sorted(rng.choice(len(splits), args.max_splits, replace=False))]
                for k, ref in enumerate(splits):
                    ref = np.array(ref)
                    test = np.r_[np.setdiff1d(hs, ref), dmg]
                    score_run(rows, {**info, "setting": "within-specimen", "draw": k, "source": name},
                              ref, test, d["feats"], d["feats"], cy[test])
            ref = np.flatnonzero(h & (sp != s))  # cross specimen, same layup and condition
            test = np.r_[hs, dmg]
            if len(ref) and len(hs):
                score_run(rows, {**info, "setting": "cross-specimen", "draw": 0, "source": name},
                          ref, test, d["feats"], d["feats"], cy[test])
            if bc == 1 and len(hs):            # cross layup, loaded condition
                for src_name, src in data.items():
                    if src_name.endswith("_bc1") and src_name != name:
                        score_run(rows, {**info, "setting": "cross-layup", "draw": 0, "source": src_name},
                                  np.flatnonzero(src["healthy"]), test, src["feats"], d["feats"], cy[test])
        print(f"{name} done", flush=True)
    R = pd.DataFrame(rows)
    R.to_csv(out / "runs.csv", index=False)
    # summary: per specimen first (mean over draws / sources), then over specimens
    per_spec = (R.groupby(["layup", "condition", "setting", "method", "detector", "specimen"])
                 [["auc_signal", "detected_zero_fa"]].mean().reset_index())
    summ = (per_spec.groupby(["layup", "condition", "setting", "method", "detector"])
            .agg(auc=("auc_signal", "mean"), auc_sd=("auc_signal", "std"),
                 detected=("detected_zero_fa", "sum"), n_specimens=("specimen", "nunique")).reset_index())
    summ.to_csv(out / "summary.csv", index=False)
    sp_ = summ[summ.detector == "same-path distance"]
    print(sp_.pivot_table(index="method", columns=["setting", "layup", "condition"], values="auc").round(2).to_string())


if __name__ == "__main__":
    main()
