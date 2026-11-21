"""E5: Task 3 beyond seven zones (OGW-1, network level).

Inputs: the 36-path damage-index maps of every method (results/ext_models/task3_ogw1/
di_maps.npz, from ext_models.py) and, for the control, the drift residuals and embeddings.
Features are the log-DI maps, per frequency or the three side by side, as in the paper.

1. Coordinates instead of zones. Ridge regression (features standardised, penalty chosen by
   generalised cross-validation on the training rows only) predicts the defect centre (x, y)
   in mm; error = Euclidean distance. Leave-position-out folds of the paper (fold j holds
   out the j-th position of every zone). References: RAPID (training-free, from the paper's
   script) and the centre of the zone predicted by the paper's classifier.
2. Leave one zone out. 7 folds; each holds out the 4 positions of one zone, so the test
   location was never seen in training. Classification cannot name an unseen zone, so only
   the regression is scored. Reference: predicting the mean training position. Control: the
   same with healthy maps of the same phase (10 draws), which carry no damage.
3. Missing sensors. At test time 1-4 transducers (all their paths) or 10-50% of the paths
   are missing (20 random draws each). Either the classifier is retrained on the remaining
   paths, or the missing log-DI values are replaced by their training mean without
   retraining. Zone accuracy, three frequencies combined, leave-position-out.

Outputs in results/ext_task3/: regression.csv, regression_per_position.csv,
leave_zone_out.csv, dropout.csv

    python scripts/ext_task3.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gwtsfm import heads  # noqa: E402
from gwtsfm.config import DATA, EMBEDDINGS, MODELS, RESULTS, SEED  # noqa: E402

import task3_ogw1_network as t3  # noqa: E402

ZONE = np.arange(28) // 4
LPO = [np.flatnonzero(np.arange(28) % 4 == j) for j in range(4)]     # the paper's folds
LZO = [np.flatnonzero(ZONE == g) for g in range(7)]                    # leave one zone out


def features(D, f):
    """log-DI features of a map (3, 28, 36): one frequency, or 'all' side by side."""
    return t3.log_di(D[f]) if f != "all" else np.concatenate([t3.log_di(D[i]) for i in range(3)], 1)


def regress(F, xy, folds):
    pred = np.zeros_like(xy, dtype=float)
    for te in folds:
        tr = np.setdiff1d(np.arange(28), te)
        m = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 4, 25))).fit(F[tr], xy[tr])
        pred[te] = m.predict(F[te])
    return pred


def classify(F, folds, keep=None, impute=False):
    """Zone predictions; keep = columns available at test time (all if None)."""
    pred = np.zeros(28, int)
    for te in folds:
        tr = np.setdiff1d(np.arange(28), te)
        if keep is None:
            Ftr, Fte = F[tr], F[te]
        elif impute:                       # trained on every path, missing ones set to the training mean
            Ftr, Fte = F[tr], F[te].copy()
            miss = np.setdiff1d(np.arange(F.shape[1]), keep)
            Fte[:, miss] = F[tr][:, miss].mean(0)
        else:                              # retrained on the available paths
            Ftr, Fte = F[tr][:, keep], F[te][:, keep]
        # the paper's head (logistic regression, C = 0.1); every zone has 3 training positions
        pred[te] = heads.logistic_regression(C=0.1).fit(Ftr, ZONE[tr]).predict(Fte)
    return pred


def control_maps(name, z, seeds=10):
    """Healthy maps of the same phase (as the paper's control), for every draw."""
    B = z["B"]
    if name == t3.PHYSICS:
        return [np.stack([t3.control_energy(B[f], np.random.default_rng(s)) for f in range(3)]) for s in range(seeds)]
    key = {v: k for k, v in MODELS.items()}[name]
    EB = np.load(EMBEDDINGS / "ogw1_baselines" / f"{key}.npy").astype(np.float32)
    return [np.stack([t3.control_embedding(EB[f], np.random.default_rng(s)) for f in range(3)]) for s in range(seeds)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(RESULTS / "ext_task3"))
    ap.add_argument("--draws", type=int, default=20)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    z = np.load(DATA / "ogw1" / "residuals.npz")
    dpos, cross, tpos = z["dpos"].astype(float), z["cross"], z["tpos"].astype(float)
    centres = np.array([dpos[4 * g:4 * g + 4].mean(0) for g in range(7)])
    maps = np.load(RESULTS / "ext_models" / "task3_ogw1" / "di_maps.npz")
    DI = {k.replace("_", " "): maps[k] for k in maps.files}
    gx, gy = np.meshgrid(np.arange(0, 501, 5), np.arange(0, 501, 5))
    grid = np.c_[gx.ravel(), gy.ravel()].astype(float)

    reg, per_pos, lzo, drop = [], [], [], []
    rng = np.random.default_rng(SEED)
    for name, D in DI.items():
        ctrl = control_maps(name, z)
        for f in [0, 1, 2, "all"]:
            label = {0: "40", 1: "60", 2: "80", "all": "all"}[f]
            F = features(D, f)
            # 1. coordinates, leave position out
            err = np.linalg.norm(regress(F, dpos, LPO) - dpos, axis=1)
            zc = centres[classify(F, LPO)]
            err_cls = np.linalg.norm(zc - dpos, axis=1)
            row = dict(method=name, freq=label, ridge_median_mm=np.median(err), ridge_mean_mm=err.mean(),
                       classifier_centre_median_mm=np.median(err_cls))
            if f != "all":
                xy = np.stack([t3.rapid(np.maximum(D[f, k] - 1, 0), cross, tpos, grid) for k in range(28)])
                row["rapid_median_mm"] = np.median(np.linalg.norm(xy - dpos, axis=1))
            reg.append(row)
            per_pos += [dict(method=name, freq=label, position=k + 1, ridge_error_mm=e) for k, e in enumerate(err)]
            # 2. leave one zone out (regression only), with the mean-position reference and the control
            err_z = np.linalg.norm(regress(F, dpos, LZO) - dpos, axis=1)
            err_mean = np.linalg.norm(np.stack([dpos[np.setdiff1d(np.arange(28), te)].mean(0)
                                                for te in LZO for _ in te]) - dpos[np.concatenate(LZO)], axis=1)
            err_ctrl = [np.median(np.linalg.norm(regress(features(M, f), dpos, LZO) - dpos, axis=1)) for M in ctrl]
            lzo.append(dict(method=name, freq=label, ridge_median_mm=np.median(err_z), ridge_mean_mm=err_z.mean(),
                            mean_position_median_mm=np.median(err_mean), control_median_mm=np.mean(err_ctrl),
                            **{f"zone{g + 1}_median_mm": np.median(err_z[LZO[g]]) for g in range(7)}))
        # 3. missing sensors (three frequencies combined)
        F = features(D, "all")
        path_cols = np.arange(36)
        for kind, sizes in [("transducers", [1, 2, 3, 4]), ("paths", [0.1, 0.25, 0.5])]:
            for size in sizes:
                for d in range(args.draws):
                    if kind == "transducers":
                        gone = rng.choice(np.arange(1, 13), size, replace=False)
                        avail = np.flatnonzero(~np.isin(cross, gone).any(1))
                    else:
                        avail = np.sort(rng.choice(path_cols, int(round(36 * (1 - size))), replace=False))
                    keep = np.concatenate([avail + 36 * i for i in range(3)])
                    for mode, imp in [("retrained", False), ("imputed", True)]:
                        acc = float((classify(F, LPO, keep, imp) == ZONE).mean())
                        drop.append(dict(method=name, missing=kind, amount=size, draw=d, mode=mode,
                                         paths_left=len(avail), acc=acc))
        print(f"{name} done", flush=True)
    pd.DataFrame(reg).to_csv(out / "regression.csv", index=False)
    pd.DataFrame(per_pos).to_csv(out / "regression_per_position.csv", index=False)
    pd.DataFrame(lzo).to_csv(out / "leave_zone_out.csv", index=False)
    D = pd.DataFrame(drop)
    D.to_csv(out / "dropout.csv", index=False)
    print(pd.DataFrame(reg).pivot_table(index="method", columns="freq", values="ridge_median_mm").round(1).to_string())
    print(pd.DataFrame(lzo).pivot_table(index="method", columns="freq",
                                        values=["ridge_median_mm", "control_median_mm"]).round(1).to_string())
    print(D.groupby(["method", "missing", "amount", "mode"]).acc.mean().unstack(["missing", "amount", "mode"]).round(2).to_string())


if __name__ == "__main__":
    main()
