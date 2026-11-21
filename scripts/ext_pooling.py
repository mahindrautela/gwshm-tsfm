"""E2 pooling ablation, step 2: the paper's protocols on every pooling variant.

Each (model, variant) from ext_pooling_embed.py is treated as its own feature set and run
through Task 1 (classification), Task 2 (detection) and Task 3 at network level, with the
task scripts unchanged (see ext_models.run_protocols). Task 3 at path level is left out: it
is near chance for every feature set in the paper.

Outputs: results/ext_pooling/{task1_gwtcw, task2_nasa, task3_ogw1}/ and summary.csv with
the headline metric of each task per model and variant:
    t1_auc         Task 1, logistic regression, sign-symmetric AUC
    t2_within      Task 2, same-path distance, signal AUC within specimen
    t2_cross       Task 2, same-path distance, signal AUC across specimens
    t3_acc         Task 3, zone accuracy from the 36-path map, mean over 40/60/80 kHz
    t3_spearman    Task 3, Spearman(damage index, path proximity), mean over frequencies

    python scripts/ext_pooling.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gwtsfm.config import DATA, MODELS, RESULTS  # noqa: E402
from ext_models import run_protocols  # noqa: E402

VARIANTS = ["model", "L25_mean", "L50_mean", "L75_mean", "L100_mean", "L100_max", "L100_last"]
POOL = DATA / "embeddings_pool"


def summary(out: Path) -> pd.DataFrame:
    cv = pd.read_csv(out / "task1_gwtcw" / "cv_metrics.csv")
    t1 = cv[(cv["mode"] == "sign-symmetric") & (cv["head"] == "logistic regression")].groupby("method").auc.mean()
    sa = pd.read_csv(out / "task2_nasa" / "signal_auc.csv")
    sp = sa[sa.detector == "same-path distance"].groupby(["method", "setting"]).auc_signal.mean().unstack()
    n = pd.read_csv(out / "task3_ogw1" / "network_localisation.csv")
    t3 = n[(n.approach == "logistic regression on DI map") & (n.freq > 0)].groupby("method").acc.mean()
    rho = pd.read_csv(out / "task3_ogw1" / "spearman.csv").groupby("method").spearman_mean.mean()
    S = pd.concat({"t1_auc": t1, "t2_within": sp["within-specimen"], "t2_cross": sp["cross-specimen"],
                   "t3_acc": t3, "t3_spearman": rho}, axis=1)
    S = S[S.index.str.contains(" \\| ")]
    S.index = pd.MultiIndex.from_tuples([tuple(i.split(" | ")) for i in S.index], names=["model", "variant"])
    return S.reset_index()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(RESULTS / "ext_pooling"))
    args = ap.parse_args()
    out = Path(args.out)
    feats = {}
    for key, name in MODELS.items():
        for v in VARIANTS:
            if (POOL / "nasa" / f"{key}__{v}.npy").exists():
                feats[f"{key}__{v}"] = f"{name} | {v}"
    print(f"{len(feats)} feature sets", flush=True)
    run_protocols(feats, out, embeddings=POOL, tasks=("task1", "task2", "task3_network"))
    S = summary(out)
    S.to_csv(out / "summary.csv", index=False)
    print(S.round(3).to_string())


if __name__ == "__main__":
    main()
