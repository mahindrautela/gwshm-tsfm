"""Figures for the paper. Run from the repository root after the experiments:

    python paper/figures/make_figures.py

Reads results/ and data/ogw1/residuals.npz; writes paper/figures/*.pdf
(Figure 1 is drawn in LaTeX inside main.tex).
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("pdf")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from gwtsfm.config import DATA, MODELS, RESULTS  # noqa: E402
from gwtsfm.ogw import path_distance  # noqa: E402

OUT = Path(__file__).resolve().parent

# palette (validated on a white surface): categorical slots 1-4, one-hue blue ramp, neutrals
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
FREQ_COL = {40: "#86b6ef", 60: "#2a78d6", 80: "#104281"}
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e5e4e0"

plt.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{mathptmx}",
    "font.family": "serif", "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "axes.spines.top": False,
    "axes.spines.right": False, "legend.frameon": False, "lines.linewidth": 1.5,
    "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})

TSFM = list(MODELS.values())


def fig_task1():
    """Figure 2: AUC against the number of labelled configurations (Task 1)."""
    f = pd.read_csv(RESULTS / "task1_gwtcw/fewshot.csv")
    f.loc[f.n_configs > 64, "n_configs"] = 124  # "all": every training configuration of the fold
    m = f.groupby(["method", "n_configs"]).auc.mean().unstack()
    cv = pd.read_csv(RESULTS / "task1_gwtcw/cv_metrics.csv")
    ss = cv[cv["mode"] == "sign-symmetric"].groupby("method").auc.mean()
    x = m.columns.values.astype(float)
    tsfm = m.loc[TSFM]

    fig, ax = plt.subplots(figsize=(3.25, 1.95))
    ax.axhline(0.5, color=MUTED, lw=0.6, ls=(0, (3, 2)), zorder=1)
    ax.fill_between(x, tsfm.min(), tsfm.max(), color=BLUE, alpha=0.18, lw=0, zorder=2)
    ax.plot(x, tsfm.mean(), color=BLUE, marker="o", ms=3.2, zorder=4,
            label="Frozen TSFMs (mean and range of six)")
    ax.plot(x, m.loc["|FFT| spectrum"], color=ORANGE, marker="s", ms=3.2, zorder=3,
            label=r"$|$FFT$|$ spectrum")
    for name, lab, dy in [("MiniRocket", "MiniRocket", 0.0), ("1D-CNN (reference notebook)", "1D-CNN", 0.0)]:
        ax.plot([124], [ss[name]], marker="D", ms=3.6, color=INK2, zorder=5, ls="none")
        ax.annotate(lab, (124, ss[name] + dy), xytext=(-5, 0), textcoords="offset points",
                    ha="right", va="center", fontsize=7, color=INK2)
    ax.set_xscale("log", base=2)
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(v)) for v in x[:-1]] + ["all"])
    ax.minorticks_off()
    ax.set_xlabel("Labelled configurations in training")
    ax.set_ylabel("AUC, held-out configurations")
    ax.set_ylim(0.45, 0.95)
    ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9])
    ax.grid(axis="y", color=GRID, lw=0.5, zorder=0)
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2, handlelength=1.6,
              columnspacing=1.0, borderaxespad=0.1)
    fig.savefig(OUT / "task1_label_efficiency.pdf")
    plt.close(fig)


def fig_task2():
    """Figure 3: cross-specimen measurement scores over fatigue cycles (Task 2)."""
    s = pd.read_csv(RESULTS / "task2_nasa/measurement_scores.csv")
    s = s[s.setting == "cross-specimen"]
    panels = [("Mantis-8M", "same-path distance", "Mantis-8M embeddings, same-path distance"),
              ("Raw signal", "baseline subtraction (1 - correlation)",
               r"Raw signal, baseline subtraction (1 $-$ correlation)")]
    cols = {"L2S11": BLUE, "L2S17": ORANGE, "L2S18": AQUA, "L2S20": YELLOW}
    marks = {"L2S11": "o", "L2S17": "s", "L2S18": "^", "L2S20": "D"}
    fig, axes = plt.subplots(2, 1, figsize=(3.25, 2.75), sharex=True)
    for ax, (meth, det, title) in zip(axes, panels):
        d = s[(s.method == meth) & (s.detector == det)]
        ax.axvline(2e4, color=MUTED, lw=0.6, ls=(0, (3, 2)), zorder=1)
        for sp, g in d.groupby("specimen"):
            h = g[g.healthy_test]
            dmg = g[~g.healthy_test].sort_values("cycles")
            ref = h.score.mean()
            jit = np.linspace(-0.12, 0.12, len(h)) if len(h) > 1 else np.zeros(1)
            ax.plot(10 ** (0.0 + jit), h.score / ref, ls="none", marker=marks[sp], ms=2.8,
                    mfc="white", mec=cols[sp], mew=0.8, zorder=3)
            ax.plot(dmg.cycles, dmg.score / ref, color=cols[sp], marker=marks[sp], ms=2.8, lw=1.1,
                    zorder=4, label=sp.replace("L2", ""))
            hmax = (h.score / ref).max()
            ax.plot([0.7, 3e5], [hmax, hmax], color=cols[sp], lw=0.5, ls=(0, (1, 1.5)), zorder=2)
        ax.set_xscale("log")
        ax.set_title(title, loc="left", fontsize=7.5, pad=2)
        ax.grid(axis="y", color=GRID, lw=0.5, zorder=0)
        ax.set_ylabel("Score / healthy mean")
    axes[0].legend(loc="upper left", ncol=4, handlelength=1.4, columnspacing=0.9,
                   borderaxespad=0.1)
    ax = axes[1]
    ax.set_xlim(0.6, 4e5)
    ax.set_xticks([1, 1e1, 1e2, 1e3, 1e4, 1e5])
    ax.set_xticklabels(["healthy", "$10^1$", "$10^2$", "$10^3$", "$10^4$", "$10^5$"])
    ax.minorticks_off()
    ax.set_xlabel("Fatigue cycles")
    axes[0].annotate("20k cycles", (2e4, axes[0].get_ylim()[1]), xytext=(2, -2),
                     textcoords="offset points", ha="left", va="top", fontsize=6.5, color=INK2)
    fig.align_ylabels(axes)
    fig.savefig(OUT / "task2_cross_specimen.pdf")
    plt.close(fig)


def fig_task3(case=9, top=8):
    """Figure 4: damage-index maps of case D10 and the Spearman dot plot (Task 3)."""
    loc = np.load(DATA / "ogw1" / "residuals.npz", allow_pickle=True)
    di = np.load(RESULTS / "task3_ogw1/di_maps.npz")
    cross, tpos, dpos, freqs = loc["cross"], loc["tpos"], loc["dpos"], list(loc["freqs"])
    prox = -path_distance(cross, tpos, dpos)  # closeness of each path to the damage
    sp = pd.read_csv(RESULTS / "task3_ogw1/spearman.csv")

    fig = plt.figure(figsize=(6.75, 2.45))
    left, right = fig.subfigures(1, 2, width_ratios=[1, 1.45], wspace=0.04)
    axs = left.subplots(2, 2, gridspec_kw=dict(wspace=0.06, hspace=0.32))
    maps = [("Residual_energy_(physics)", "Residual energy"), ("MOMENT-1_(base)", "MOMENT-1 embedding")]
    shades = [RAMP[3], RAMP[4], RAMP[5], RAMP[6]]
    for r, (key, lab) in enumerate(maps):
        for c, f in enumerate([40, 80]):
            ax = axs[r, c]
            v = di[key][freqs.index(f), case]
            order = np.argsort(v)  # ascending: strongest paths drawn last
            rank_desc = np.empty(len(v), int)
            rank_desc[order[::-1]] = np.arange(len(v))
            for p in order:
                a, b = tpos[cross[p, 0] - 1], tpos[cross[p, 1] - 1]
                k = rank_desc[p]
                if k < top:
                    col, lw = shades[min(3, (top - 1 - k) * 4 // top)], 0.9 + 1.1 * (top - k) / top
                else:
                    col, lw = "#d9d8d3", 0.45
                ax.plot([a[0], b[0]], [a[1], b[1]], color=col, lw=lw, solid_capstyle="round",
                        zorder=2 if k >= top else 3 + (top - k))
            ax.plot(*tpos.T, ls="none", marker="s", ms=2.0, color=INK2, zorder=20)
            ax.plot(*dpos[case], marker="*", ms=8.5, color=ORANGE, mec="white", mew=0.6, zorder=21)
            rho = spearmanr(v, prox[case]).statistic
            ax.set_xlim(-12, 512)
            ax.set_ylim(-12, 512)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            for s_ in ax.spines.values():
                s_.set_visible(True)
                s_.set_color(GRID)
            ax.set_xlabel(rf"$\rho = {rho:.2f}$", fontsize=7, labelpad=1.5, color=INK2)
            if r == 0:
                ax.set_title(f"{f} kHz", fontsize=7.5, pad=2)
            if c == 0:
                ax.set_ylabel(lab, fontsize=7, labelpad=2)
    left.suptitle(r"(a) Damage-index maps, case D10", fontsize=7.5, x=0.5, y=1.02)

    ax = right.subplots()
    order = ["MOMENT-1 (base)", "TiRex", "Mantis-8M", "Chronos-Bolt (base)", "TTM-R2", "Chronos-2",
             "Residual energy (physics)"]
    names = {"MOMENT-1 (base)": "MOMENT-1 base", "Chronos-Bolt (base)": "Chronos-Bolt base",
             "Residual energy (physics)": "Residual energy"}
    for i, mth in enumerate(order):
        row = sp[sp.method == mth].set_index("freq_khz").spearman_mean
        y = len(order) - 1 - i
        ax.plot([row.min(), row.max()], [y, y], color=GRID, lw=2.4, zorder=1, solid_capstyle="round")
        for f in [40, 60, 80]:
            ax.plot(row[f], y, marker="o", ms=5, color=FREQ_COL[f], mec="white", mew=0.6, zorder=3,
                    ls="none", label=f"{f} kHz" if i == 0 else None)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([names.get(m, m) for m in order][::-1])
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.6, len(order) - 0.4)
    ax.set_xlabel(r"Spearman $\rho$ (mean over 28 cases)")
    ax.grid(axis="x", color=GRID, lw=0.5, zorder=0)
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=3, handletextpad=0.1,
              columnspacing=0.9, borderaxespad=0.1)
    right.suptitle("(b) Damage index vs. proximity of the path to the damage", fontsize=7.5,
                   x=0.55, y=1.02)
    fig.savefig(OUT / "task3_maps_spearman.pdf")
    plt.close(fig)


if __name__ == "__main__":
    fig_task1()
    fig_task2()
    fig_task3()
    print("figures written to", OUT)
