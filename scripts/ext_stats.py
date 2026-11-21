"""E4: uncertainty and significance, resampling the physical units of each task.

Task 1 (unit: sensor configuration, 156)
  * out-of-fold scores of logistic regression (sign-symmetric) on the paper's folds; pooled
    AUC with a 95% bootstrap interval over configurations (2,000 resamples, both signals of
    a configuration together)
  * paired comparisons: bootstrap distribution of the AUC difference on the same resampled
    configurations; two-sided p = 2 * min(P(diff <= 0), P(diff >= 0)); Holm correction
  * split variability: the grouped 5-fold CV repeated with 20 different fold assignments
  * label efficiency with 20 instead of 5 random draws per size
Task 2 (unit: specimen, 4)
  * within specimen: every one of the C(8,4) = 70 reference splits instead of 5 draws
  * per specimen: mean, min and max AUC over splits, share of splits detected at zero
    false alarms
  * paired comparisons between detectors: exact sign-flip test over the 4 specimens. With 4
    units the smallest attainable two-sided p is 2/16 = 0.125, so no difference in Table 4
    can be significant at the specimen level; this is reported as such.
Task 3 (unit: damage position, 28)
  * per-position correctness of the network classifier (logistic regression on the
    36-path log-DI map, leave-position-out, as in the paper), Clopper-Pearson intervals
  * paired comparisons: exact McNemar test on per-position correctness; Holm correction

Outputs in results/ext_stats/.

    python scripts/ext_stats.py               # needs results/ext_models (ext_models.py)
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta, binomtest
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gwtsfm import baselines, heads  # noqa: E402
from gwtsfm.config import DATA, EMBEDDINGS, MODELS, RESULTS, SEED  # noqa: E402
from gwtsfm.splits import grouped_kfold  # noqa: E402

import task1_gwtcw as t1  # noqa: E402
import task2_nasa as t2  # noqa: E402
import task3_ogw1_network as t3  # noqa: E402

B = 2000


def holm(p):
    """Holm-Bonferroni adjusted p-values."""
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def boot_p(diff):
    return float(min(1.0, 2 * min((diff <= 0).mean(), (diff >= 0).mean())))


# ------------------------------------------------------------------------------ Task 1
def task1(out):
    t1.PAPER_MODELS = dict(MODELS)
    Xall, yall, cfgall = t1.load_gwtcw()
    keep = t1.first_repeat(len(Xall))
    X, y, cfg = Xall[keep], yall[keep], cfgall[keep]
    folds = t1.grouped_folds(cfg)
    feats, _ = t1.feature_sets(X, Xall, keep)
    feats = {k: v for k, v in feats.items() if v[0] != t1.PROBE and k != "Raw signal"}

    # out-of-fold scores on the paper's folds
    oof = {}
    for name, (_, Fp, Fn) in feats.items():
        s = np.zeros(len(y))
        for tr, te in folds:
            s[te] = t1.damage_score(t1.logistic, Fp, Fn, y, tr, te, True)
        oof[name] = s
    s = np.zeros(len(y))
    for tr, te in folds:
        s[te] = baselines.sigmoid(baselines.minirocket_decision(X, y, tr, te, sign_symmetric=True))
    oof["MiniRocket"] = s
    print("Task 1: out-of-fold scores done", flush=True)

    # bootstrap over configurations
    rng = np.random.default_rng(SEED)
    units = np.unique(cfg)
    rows_of = {c: np.flatnonzero(cfg == c) for c in units}
    boots = {m: np.zeros(B) for m in oof}
    for b in range(B):
        idx = np.concatenate([rows_of[c] for c in rng.choice(units, len(units))])
        for m, s in oof.items():
            boots[m][b] = roc_auc_score(y[idx], s[idx])
    ci = pd.DataFrame([dict(method=m, auc_pooled=roc_auc_score(y, s), lo=np.percentile(boots[m], 2.5),
                            hi=np.percentile(boots[m], 97.5)) for m, s in oof.items()])
    ci.sort_values("auc_pooled", ascending=False).to_csv(out / "task1_ci.csv", index=False)
    pairs = []
    for a, b_ in itertools.combinations(oof, 2):
        d = boots[a] - boots[b_]
        pairs.append(dict(a=a, b=b_, diff=roc_auc_score(y, oof[a]) - roc_auc_score(y, oof[b_]),
                          lo=np.percentile(d, 2.5), hi=np.percentile(d, 97.5), p=boot_p(d)))
    P = pd.DataFrame(pairs)
    P["p_holm"] = holm(P.p)
    P.to_csv(out / "task1_pairs.csv", index=False)

    # repeated CV with 20 fold assignments
    rep = []
    for seed in range(20):
        shuffled = np.random.default_rng(seed).permutation(cfg.max() + 1)
        f = grouped_kfold(shuffled[cfg], 5)
        for name, (_, Fp, Fn) in feats.items():
            aucs = [roc_auc_score(y[te], t1.damage_score(t1.logistic, Fp, Fn, y, tr, te, True)) for tr, te in f]
            rep.append(dict(method=name, seed=seed, auc=float(np.mean(aucs))))
    pd.DataFrame(rep).to_csv(out / "task1_repeated_cv.csv", index=False)
    print("Task 1: repeated CV done", flush=True)

    # label efficiency with 20 draws
    rng = np.random.default_rng(SEED)
    fs = []
    for name, (_, Fp, Fn) in feats.items():
        for i, (tr, te) in enumerate(folds):
            train_cfgs = np.unique(cfg[tr])
            for m in [2, 4, 8, 16, 32, 64]:
                for d in range(20):
                    chosen = rng.choice(train_cfgs, m, replace=False)
                    sub = tr[np.isin(cfg[tr], chosen)]
                    fs.append(dict(method=name, fold=i, n_configs=m, draw=d,
                                   auc=roc_auc_score(y[te], t1.damage_score(t1.logistic, Fp, Fn, y, sub, te, True))))
    pd.DataFrame(fs).to_csv(out / "task1_fewshot20.csv", index=False)
    print("Task 1: label efficiency done", flush=True)


# ------------------------------------------------------------------------------ Task 2
def task2(out):
    z = np.load(DATA / "nasa" / "layup2.npz")
    Xr = z["X"][..., :t2.L].astype(np.float32)
    specimen, cycles = z["specimen"], z["cycles"]
    healthy = cycles <= t2.HEALTHY_MAX_CYCLES
    feats = {"Raw signal": ("raw", Xr / Xr.std())}
    for key, name in MODELS.items():
        f = EMBEDDINGS / "nasa" / f"{key}.npy"
        if f.exists():
            feats[name] = ("tsfm", np.load(f).astype(np.float32))
    rows = []
    for s in sorted(set(specimen)):
        h = np.flatnonzero(healthy & (specimen == s))
        dmg = np.flatnonzero((specimen == s) & (cycles == t2.DAMAGED_CYCLES))
        for k, ref in enumerate(itertools.combinations(h, len(h) // 2)):
            ref = np.array(ref)
            test = np.r_[np.setdiff1d(h, ref), dmg]
            lab = np.r_[np.zeros(len(test) - len(dmg)), np.ones(len(dmg))]
            for name, (kind, F) in feats.items():
                dets = {"same-path distance": t2.same_path_distance(F[ref], F[test])}
                if kind == "raw":
                    full = t2.detector_scores("raw", F[ref], F[test])
                    dets["baseline subtraction (1 - correlation)"] = full["baseline subtraction (1 - correlation)"]
                for det, S in dets.items():
                    y = np.repeat(lab, S.shape[1] * S.shape[2])
                    meas = S.reshape(len(test), -1).mean(1)
                    rows.append(dict(method=name, detector=det, specimen=s, split=k,
                                     auc=roc_auc_score(y, S.ravel()),
                                     detected=float(meas[lab == 1].min() > meas[lab == 0].max())))
        print(f"Task 2: {s} done", flush=True)
    R = pd.DataFrame(rows)
    R["feature"] = np.where(R.method == "Raw signal", R.method + " | " + R.detector, R.method)
    R.to_csv(out / "task2_splits.csv", index=False)
    per = R.groupby(["feature", "specimen"]).agg(auc=("auc", "mean"), auc_min=("auc", "min"),
                                                 auc_max=("auc", "max"), detected=("detected", "mean")).reset_index()
    per.to_csv(out / "task2_per_specimen.csv", index=False)
    # exact sign-flip test over the 4 specimens on the per-specimen mean AUC
    wide = per.pivot(index="specimen", columns="feature", values="auc")
    signs = np.array(list(itertools.product([1, -1], repeat=len(wide))))
    pairs = []
    for a, b_ in itertools.combinations(wide.columns, 2):
        d = (wide[a] - wide[b_]).to_numpy()
        stat = (signs * d).mean(1)
        p = float((np.abs(stat) >= abs(d.mean()) - 1e-12).mean())
        pairs.append(dict(a=a, b=b_, diff=d.mean(), n_specimens=len(d), p_signflip=p))
    pd.DataFrame(pairs).to_csv(out / "task2_pairs.csv", index=False)


# ------------------------------------------------------------------------------ Task 3
def task3(out):
    z = np.load(RESULTS / "ext_models" / "task3_ogw1" / "di_maps.npz")
    zone = np.arange(28) // 4
    correct = {}
    for k in z.files:
        D = z[k]
        name = k.replace("_", " ")
        for f, label in [(0, "40"), (1, "60"), (2, "80"), (None, "all")]:
            F = t3.log_di(D[f]) if f is not None else np.concatenate([t3.log_di(D[i]) for i in range(3)], 1)
            pred = np.zeros(28, int)
            for j in range(4):
                te = np.flatnonzero(np.arange(28) % 4 == j)
                tr = np.setdiff1d(np.arange(28), te)
                pred[te] = heads.predict_proba(lambda: heads.logistic_regression(C=0.1), F, zone, tr, te).argmax(1)
            correct[(name, label)] = pred == zone
    rows = []
    for (name, label), c in correct.items():
        n = int(c.sum())
        rows.append(dict(method=name, freq=label, correct=n, n=28, acc=n / 28,
                         lo=beta.ppf(0.025, n, 29 - n) if n > 0 else 0.0,
                         hi=beta.ppf(0.975, n + 1, 28 - n) if n < 28 else 1.0))
    pd.DataFrame(rows).to_csv(out / "task3_ci.csv", index=False)
    pairs = []
    names = sorted({n for n, _ in correct})
    for label in ["40", "60", "80", "all"]:
        for a, b_ in itertools.combinations(names, 2):
            ca, cb = correct[(a, label)], correct[(b_, label)]
            n01, n10 = int((~ca & cb).sum()), int((ca & ~cb).sum())
            p = binomtest(n10, n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
            pairs.append(dict(freq=label, a=a, b=b_, acc_a=ca.mean(), acc_b=cb.mean(), a_only=n10, b_only=n01,
                              p_mcnemar=p))
    P = pd.DataFrame(pairs)
    P["p_holm"] = holm(P.p_mcnemar)
    P.to_csv(out / "task3_pairs.csv", index=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(RESULTS / "ext_stats"))
    ap.add_argument("--tasks", nargs="+", default=["task1", "task2", "task3"])
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for t in args.tasks:
        {"task1": task1, "task2": task2, "task3": task3}[t](out)


if __name__ == "__main__":
    main()
