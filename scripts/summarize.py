"""Tables and figures from a results directory.

python scripts/summarize.py --results results --cache data/cache --out report
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gwtsfm import metrics, protocol  # noqa: E402
from gwtsfm.ogw import FreqCache, pair_index, pair_name  # noqa: E402

# reference palette (dataviz skill): categorical slots 1-3, sequential blue, ink
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

CLASSICAL = ["OBS", "OBS+BSS", "Stretch extrapolation", "Last value", "Linear trend", "AutoETS"]
GAP_BINS = (0, 2.5, 5, 10, 20, 40)

plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 150, "savefig.bbox": "tight"})


def load(results: str) -> pd.DataFrame:
    fs = [f for f in glob.glob(f"{results}/f*/Tc*/*.parquet")]
    return pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)


def order_methods(ms) -> list[str]:
    tsfm = sorted(m for m in ms if m not in CLASSICAL)
    return [m for m in CLASSICAL if m in ms] + tsfm


def best_di(row_di_z, row_di_res):
    return row_di_z if np.isfinite(row_di_z) else row_di_res


def heatmap(ax, M: pd.DataFrame, title: str, fmt="{:.2f}", vmin=0.5, vmax=1.0, cmap=None,
            log=False):
    from matplotlib.colors import LinearSegmentedColormap, LogNorm, Normalize
    cmap = cmap or LinearSegmentedColormap.from_list("seq", SEQ)
    norm = LogNorm(vmin, vmax) if log else Normalize(vmin, vmax)
    ax.imshow(M.to_numpy(float), aspect="auto", cmap=cmap, norm=norm)
    ax.set_xticks(range(M.shape[1]), [str(c) for c in M.columns], rotation=0)
    ax.set_yticks(range(M.shape[0]), M.index)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M.iat[i, j]
            if np.isfinite(v):
                dark = norm(v) > 0.55
                ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=7.5,
                        color="white" if dark else INK)
    ax.set_title(title, loc="left", color=INK, fontsize=10)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--example-tc", type=float, default=40.0)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df = load(args.results)
    methods = order_methods(df["method"].unique())
    tables = {}

    # ---- Table: pooled structure-level AUC (all paths fused), per DI ------------------
    rows = []
    for di in metrics.DI_COLS:
        a = metrics.fused_auc(df, di, by_state=True)
        a["di"] = di
        rows.append(a)
    A = pd.concat(rows)
    A.to_csv(out / "auc_fused.csv", index=False)
    piv = A[A.di == "di_res"].pivot_table(index="method", columns=["f_khz", "Tc"], values="auc")
    tables["auc_fused_res"] = piv.reindex(methods)
    pz = A[A.di == "di_z"].pivot_table(index="method", columns=["f_khz", "Tc"], values="auc")
    tables["auc_fused_z"] = pz.reindex([m for m in methods if m in pz.index])
    pt = A[A.di == "di_res"].pivot_table(index="method", columns=["f_khz", "Tc"], values="tpr")
    tables["tpr5_fused_res"] = pt.reindex(methods)

    # ---- AUC vs gap (fused, di_res), mean over Tc and f --------------------------------
    G = metrics.auc_vs_gap(df, "di_res", edges=GAP_BINS)
    G.to_csv(out / "auc_vs_gap.csv", index=False)
    g = G.groupby(["method", "bin"], observed=True)["auc"].mean().unstack().reindex(methods)
    g.columns = [f"{c.left:g}-{c.right:g}" for c in g.columns]
    tables["auc_vs_gap_res"] = g
    Gz = metrics.auc_vs_gap(df, "di_z", edges=GAP_BINS)
    gz = Gz.groupby(["method", "bin"], observed=True)["auc"].mean().unstack()
    gz = gz.reindex([m for m in methods if m in gz.index]).dropna(how="all")
    gz.columns = [f"{c.left:g}-{c.right:g}" for c in gz.columns]
    tables["auc_vs_gap_z"] = gz

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 0.32 * len(methods) + 1.2),
                             gridspec_kw={"width_ratios": [1, 1]})
    heatmap(axes[0], g, "Fused AUC vs extrapolation gap (residual DI)")
    axes[0].set_xlabel("gap T - Tc (°C)")
    E = metrics.forecast_error(df)
    e = E.groupby(["method", "bin"], observed=True)["rel_err"].median().unstack().reindex(methods)
    e.columns = [f"{c.left:g}-{c.right:g}" for c in e.columns]
    tables["pristine_rel_err"] = e
    heatmap(axes[1], e, "Prediction error on pristine signals", fmt="{:.2g}",
            vmin=max(1e-3, np.nanmin(e.to_numpy())), vmax=np.nanmax(e.to_numpy()), log=True)
    axes[1].set_yticks([])
    axes[1].set_xlabel("gap T - Tc (°C)")
    fig.savefig(out / "fig_gap_heatmaps.png")
    plt.close(fig)

    # ---- Ranked pooled AUC (mean over f, Tc): residual DI and, if available, z-DI -----
    r_res = A[A.di == "di_res"].groupby("method")["auc"].mean()
    r_z = A[A.di == "di_z"].groupby("method")["auc"].mean()
    rank = pd.DataFrame({"residual DI": r_res, "uncertainty-normalised DI": r_z}).reindex(methods)
    rank = rank.sort_values("residual DI")
    tables["auc_pooled_mean"] = rank
    fig, ax = plt.subplots(figsize=(6.2, 0.3 * len(rank) + 1))
    y = np.arange(len(rank))
    ax.barh(y + 0.18, rank["residual DI"], height=0.34, color=BLUE, label="residual DI")
    ax.barh(y - 0.18, rank["uncertainty-normalised DI"].fillna(0), height=0.34, color=ORANGE,
            label="uncertainty-normalised DI")
    ax.set_yticks(y, rank.index)
    ax.axvline(0.5, color=INK2, lw=0.8, ls=":")
    ax.set_xlim(0.4, 1.0)
    ax.set_xlabel("fused AUC, mean over frequencies and cut-offs")
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="lower right")
    fig.savefig(out / "fig_pooled_auc.png")
    plt.close(fig)

    # ---- Per-path AUC on the damage path T4-T9 ---------------------------------------
    pp = metrics.per_path_auc(df, "di_res")
    pp.to_csv(out / "auc_per_path.csv", index=False)
    p49 = pp[pp.path == pair_index(4, 9)].pivot_table(index="method", columns=["f_khz", "Tc"],
                                                      values="auc").reindex(methods)
    tables["auc_path_T4T9_res"] = p49

    # ---- Example forecast along temperature (path T4-T9) -----------------------------
    freqs = sorted(df.f_khz.unique())
    f0 = freqs[len(freqs) // 2]
    ex_files = sorted(glob.glob(f"{args.results}/f{f0}/Tc{args.example_tc:g}/*_examples.npz"))
    if ex_files:
        s = protocol.prepare(FreqCache.load(Path(args.cache) / f"f{f0}kHz.npz"))
        p = list(s.paths).index(pair_index(4, 9))
        pick = [f for f in ex_files if "Chronos-2" in f or "chronos2" in f.lower()] or ex_files[:1]
        ex = np.load(pick[0])
        k = ex["context"].shape[-1]
        n_samples = ex["context"].shape[1]
        energy = (s.ref_grid[p] ** 2).mean(0)
        cand = np.argsort(energy)[::-1]
        js = [cand[0], cand[min(len(cand) - 1, n_samples // 6)]]
        fig, axes = plt.subplots(1, 2, figsize=(9.5, 2.8), sharey=False)
        prist = s.state == "udam"
        for ax, j in zip(axes, js):
            Tg = s.grid[:k]
            Tf = s.grid[k - 1] + s.step * np.arange(1, ex["q50"].shape[-1] + 1)
            ax.plot(Tg, ex["context"][0, j], color=INK2, lw=1.5, label="pristine references (context)")
            ax.fill_between(Tf, ex["q10"][0, j], ex["q90"][0, j], color=BLUE, alpha=0.18, lw=0,
                            label="80 % forecast interval")
            ax.plot(Tf, ex["q50"][0, j], color=BLUE, lw=2, label="median forecast")
            sel = prist & (s.T > s.grid[k - 1])
            ax.scatter(s.T[sel], s.X[sel, p, j], s=9, color=ORANGE, zorder=3,
                       label="held-out pristine measurements")
            ax.axvline(s.grid[k - 1], color=INK2, lw=0.8, ls=":")
            ax.set_xlabel("plate temperature (°C)")
            ax.set_title(f"t = {(j - s.onset) / s.fs * 1e3:.3f} ms after excitation", loc="left",
                         fontsize=9, color=INK)
            ax.grid(color=GRID, lw=0.6)
        axes[0].set_ylabel("signal amplitude")
        axes[0].legend(frameon=False, fontsize=7.5, loc="best")
        name = Path(pick[0]).name.replace("_examples.npz", "")
        fig.suptitle(f"{name}, path {pair_name(pair_index(4, 9))}, {f0} kHz, Tc = {args.example_tc:g} °C",
                     x=0.01, y=1.06, ha="left", fontsize=10, color=INK)
        fig.savefig(out / "fig_example_forecast.png")
        plt.close(fig)

    # ---- Timing ------------------------------------------------------------------------
    tfile = Path(args.results) / "timing.jsonl"
    if tfile.exists():
        T = pd.DataFrame([json.loads(l) for l in tfile.read_text().splitlines() if l.strip()])
        T = T.drop_duplicates(["method", "f_khz", "Tc"], keep="last")
        T["ms_per_series"] = 1e3 * T["seconds"] / T["n_series"].replace(0, np.nan)
        tables["timing"] = T.groupby("method")[["seconds", "ms_per_series"]].mean()

    with open(out / "tables.md", "w") as fh:
        for k, v in tables.items():
            fh.write(f"## {k}\n\n{v.round(3).to_markdown()}\n\n")
    print((out / "tables.md").read_text())


if __name__ == "__main__":
    main()
