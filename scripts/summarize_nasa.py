"""Tables and damage-growth data for Task 2 (NASA PCoE).

python scripts/summarize_nasa.py --results results_nasa
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    args = ap.parse_args()
    r = Path(args.results)
    S = pd.read_csv(r / "signal_auc.csv")
    M = pd.read_csv(r / "measurement_auc.csv")
    C = pd.read_csv(r / "measurement_scores.csv")
    sig = S.groupby(["setting", "method", "detector"])["auc_signal"].agg(["mean", "std"])
    mea = M.groupby(["setting", "method", "detector"])[["auc_measurement", "detected_zero_fa"]].mean()
    T = sig.join(mea).round(3)
    T.to_csv(r / "summary_table.csv")
    with open(r / "tables.md", "w") as fh:
        for st in ("within-specimen", "cross-specimen"):
            fh.write(f"## {st}\n\n{T.loc[st].to_markdown()}\n\n")
    print((r / "tables.md").read_text())

    # damage growth: score relative to the mean of the specimen's healthy test measurements,
    # within-specimen setting, averaged over draws, geometric mean over specimens
    G = C[C.setting == "within-specimen"].copy()
    base = (G[G.healthy_test].groupby(["method", "detector", "draw", "specimen"])["score"].mean()
            .rename("healthy_mean"))
    G = G.join(base, on=["method", "detector", "draw", "specimen"])
    G["ratio"] = G["score"] / G["healthy_mean"]
    later = G[~G.healthy_test]
    curve = (later.groupby(["method", "detector", "specimen", "cycles"])["ratio"].mean().reset_index()
                  .groupby(["method", "detector", "cycles"])["ratio"]
                  .agg(lambda x: float(np.exp(np.mean(np.log(np.maximum(x, 1e-9)))))).reset_index())
    curve["n_specimens"] = (later.groupby(["method", "detector", "cycles"])["specimen"].nunique().values)
    curve.to_csv(r / "growth_curve.csv", index=False)
    hv = (G[G.healthy_test].groupby(["method", "detector"])["ratio"]
          .agg(lambda x: float(np.quantile(x, 0.95))).rename("healthy_q95").reset_index())
    hv.to_csv(r / "healthy_spread.csv", index=False)
    print(curve[curve.detector.isin(["same-path distance", "baseline subtraction (1 - correlation)"])]
          .pivot_table(index="cycles", columns="method", values="ratio").round(2).to_string())


if __name__ == "__main__":
    main()
