"""Tables and the label-efficiency figure for the GW-TCW study.

python scripts/summarize_gwtcw.py --results results_gwtcw --out report_gwtcw
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#a3a29c"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 150, "savefig.bbox": "tight"})


def mean_sd(x):
    return f"{x.mean():.3f} ± {x.std(ddof=1):.3f}" if len(x) > 1 else f"{x.mean():.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cv = pd.read_csv(Path(args.results) / "cv_metrics.csv")
    fs = pd.read_csv(Path(args.results) / "fewshot.csv")
    fs["n_configs"] = fs["n_configs"].where(fs["n_configs"] <= 64, 124)  # "all" (124-125 per fold)

    main_ = cv[cv["mode"] != "random split (leaky)"]
    piv = (main_.groupby(["family", "method", "mode", "head"])["auc"].mean()
           .unstack(["mode", "head"]).round(3))
    piv.to_csv(out / "auc_table.csv")
    sym = main_[main_["mode"] == "sign-symmetric"]
    best = (sym.groupby(["family", "method", "head"])["auc"].mean().unstack("head").round(3))
    bacc = (sym.groupby(["family", "method", "head"])["bacc"].mean().unstack("head").round(3))
    leaky = cv[cv["mode"] == "random split (leaky)"].pivot_table(index="method", columns="head",
                                                                 values="auc").round(3)
    sd = sym.groupby(["method", "head"])["auc"].apply(mean_sd).unstack("head")
    with open(out / "tables.md", "w") as fh:
        fh.write("## AUC, sign-symmetric, grouped CV (mean over 5 folds)\n\n" + best.to_markdown() + "\n\n")
        fh.write("## AUC mean ± sd over folds, sign-symmetric\n\n" + sd.to_markdown() + "\n\n")
        fh.write("## Balanced accuracy, sign-symmetric\n\n" + bacc.to_markdown() + "\n\n")
        fh.write("## AUC, all settings\n\n" + piv.to_markdown() + "\n\n")
        fh.write("## Random split over all repeats (leaky)\n\n" + leaky.to_markdown() + "\n\n")
        c = fs.groupby(["method", "n_configs"])["auc"].mean().unstack().round(3)
        fh.write("## Few-shot AUC (logistic regression, sign-symmetric)\n\n" + c.to_markdown() + "\n")
    print((out / "tables.md").read_text())

    # label-efficiency figure: grey context, accent on best frozen TSFM vs best trained reference
    c = fs.groupby(["method", "family", "n_configs"])["auc"].agg(["mean", "std"]).reset_index()
    final = c.sort_values("n_configs").groupby("method").tail(1).set_index("method")["mean"]
    fam = c.groupby("method")["family"].first()
    tsfm_best = final[fam[final.index] == "foundation model (frozen)"].idxmax()
    ref_best = final[fam[final.index] == "trained from scratch"].idxmax()
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for m, d in c.groupby("method"):
        d = d.sort_values("n_configs")
        x = d["n_configs"].where(d["n_configs"] <= 64, 124)
        if m in (tsfm_best, ref_best):
            col = BLUE if m == tsfm_best else ORANGE
            ax.plot(x, d["mean"], color=col, lw=2, marker="o", ms=4, zorder=3)
            ax.fill_between(x, d["mean"] - d["std"], d["mean"] + d["std"], color=col, alpha=0.12, lw=0)
            ax.text(x.iloc[-1] * 1.08, d["mean"].iloc[-1], m, color=INK, va="center", fontsize=8)
        else:
            ax.plot(x, d["mean"], color=GREY, lw=1, zorder=1)
    ax.set_xscale("log", base=2)
    ax.set_xticks([2, 4, 8, 16, 32, 64, 124], ["2", "4", "8", "16", "32", "64", "all"])
    ax.axhline(0.5, color=INK2, lw=0.8, ls=":")
    ax.set_ylim(0.4, 1.0)
    ax.set_xlabel("labelled training configurations (sensor paths)")
    ax.set_ylabel("AUC on unseen paths")
    ax.set_title(f"{tsfm_best} (frozen) vs {ref_best} (trained); grey: all other methods",
                 loc="left", fontsize=9, color=INK)
    ax.grid(color=GRID, lw=0.6)
    fig.savefig(out / "fig_fewshot.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
