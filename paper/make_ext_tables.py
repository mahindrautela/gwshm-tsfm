"""Write the tables of the appendix "Extended experiments" (paper/tables/ext_*.tex) from results/.

Same pattern as make_tables.py: every number comes from the result files of the
scripts/ext_*.py runs; nothing is typed by hand. Values are rounded only at the end.

    python paper/make_ext_tables.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gwtsfm.config import MODELS, PAPER_MODELS, RESULTS  # noqa: E402

OUT = Path(__file__).resolve().parent / "tables"
NAME = {m: m.replace(" (base)", " base").replace(" (small)", " small") for m in MODELS.values()}
NEW = [m for m in MODELS.values() if m not in PAPER_MODELS.values()]


def f2(x):
    return "--" if pd.isna(x) else f"{x:.2f}"


def write(name, body):
    (OUT / name).write_text(body)
    print("wrote", name)


def rows(lines):
    return "\n".join(" & ".join(map(str, r)) + r" \\" for r in lines) + "\n"


# ------------------------------------------------------------------------------ E1
def e1():
    """Temperature: AUC vs reference grid spacing; detection at a nominal 5% false-alarm rate
    (realised rate in brackets) for the 2 and 5 degC grids."""
    f = RESULTS / "ext_temperature" / "summary.csv"
    if not f.exists():
        return
    S = pd.read_csv(f)
    S = S[S["agg"] == "mean"]
    grids = sorted(S.grid.unique())
    order = ["raw NT", "raw OBS", "raw BSS", "raw OBS+BSS"]
    emb = sorted({d.split(" | ")[0] for d in S.detector if " | " in d})
    best = S[S.detector.str.endswith("| OBS")].groupby("detector").auc.mean().sort_values(ascending=False)
    top = [d.split(" | ")[0] for d in best.index[:3]]
    lines = []
    for det in order + [f"{m} | {k}" for m in top for k in ("mean", "NT", "OBS")]:
        g = S[S.detector == det].set_index("grid")
        if g.empty:
            continue
        raw = {"raw NT": "Raw, nearest temperature", "raw OBS": "Raw, OBS", "raw BSS": "Raw, BSS",
               "raw OBS+BSS": "Raw, OBS + BSS"}
        if det in raw:
            label = raw[det]
        else:
            m, k = det.split(" | ")
            label = f"{NAME.get(m, m)}, {k}"
        lines.append([label] + [f2(g.auc.get(s)) for s in grids]
                     + [f"{f2(g['tpr@0.05'].get(s))} ({f2(g['far@0.05'].get(s))})" for s in (2, 5)])
        if det == "raw OBS+BSS":
            lines.append([r"\midrule"])
    body = "\n".join(r[0] if r[0] == r"\midrule" else " & ".join(r) + r" \\" for r in lines) + "\n"
    write("ext_temperature.tex", "% columns: grid spacing " + " ".join(map(str, grids)) + "\n" + body)
    print("   top embedding detectors:", top, "| embeddings evaluated:", emb)


# ------------------------------------------------------------------------------ E2
def e2_models():
    """All nine models on the headline metric of each task (paper protocols)."""
    R = RESULTS / "ext_models"
    cv = pd.read_csv(R / "task1_gwtcw" / "cv_metrics.csv")
    t1 = cv[(cv["mode"] == "sign-symmetric") & (cv["head"] == "logistic regression")].groupby("method").auc.mean()
    sa = pd.read_csv(R / "task2_nasa" / "signal_auc.csv")
    sp = sa[sa.detector == "same-path distance"].groupby(["method", "setting"]).auc_signal.mean().unstack()
    ma = pd.read_csv(R / "task2_nasa" / "measurement_auc.csv")
    dz = ma[(ma.detector == "same-path distance") & (ma.setting == "cross-specimen")].groupby("method").detected_zero_fa.sum()
    n = pd.read_csv(R / "task3_ogw1" / "network_localisation.csv")
    lr = n[n.approach == "logistic regression on DI map"].pivot_table(index="method", columns="freq", values="acc")
    rho = pd.read_csv(R / "task3_ogw1" / "spearman.csv").groupby("method").spearman_mean.mean()
    lines = []
    for m in sorted(MODELS.values(), key=lambda m: -t1[m]):
        lines.append([NAME[m] + (r"$^\star$" if m in NEW else ""), f2(t1[m]), f2(sp.loc[m, "within-specimen"]),
                      f2(sp.loc[m, "cross-specimen"]), f"{int(dz[m])}/4",
                      f2(lr.loc[m, [40, 60, 80]].mean()), f2(lr.loc[m, 0]), f2(rho[m])])
    write("ext_models.tex", rows(lines))


def e2_pooling():
    """Pooling ablation: each variant's metric, mean over the nine models, and how many
    models it improves over the model's own embedding."""
    S = pd.read_csv(RESULTS / "ext_pooling" / "summary.csv")
    metrics = ["t1_auc", "t2_within", "t2_cross", "t3_acc", "t3_spearman"]
    own = S[S.variant == "model"].set_index("model")[metrics]
    label = {"model": "Model's own embedding (paper)", "L25_mean": r"Mean, block at $\frac14$ depth",
             "L50_mean": r"Mean, block at $\frac12$ depth", "L75_mean": r"Mean, block at $\frac34$ depth",
             "L100_mean": "Mean, last block", "L100_max": "Max, last block", "L100_last": "Last token, last block"}
    lines = []
    for v in label:
        g = S[S.variant == v].set_index("model")[metrics]
        cells = [label[v]]
        for c in metrics:
            if v == "model":
                cells.append(f2(g[c].mean()))
            else:
                better = int((g[c] > own.loc[g.index, c] + 1e-9).sum())
                cells.append(f"{f2(g[c].mean())} ({better})")
        lines.append(cells)
    write("ext_pooling.tex", rows(lines))


# ------------------------------------------------------------------------------ E3
def e3():
    f = RESULTS / "ext_nasa" / "summary.csv"
    if not f.exists():
        return
    S = pd.read_csv(f)
    sp = S[S.detector == "same-path distance"]
    raw = S[(S.method == "Raw signal") & (S.detector == "baseline subtraction (1 - correlation)")]
    sets = [("within-specimen", c) for c in ["loaded", "clamped", "traction free"]] + \
        [("cross-specimen", "loaded"), ("cross-layup", "loaded")]
    feats = [m for m in MODELS.values() if m in set(sp.method)]
    lines = []
    for layup in (1, 2, 3):
        for m, lab, D in [(m, NAME[m], sp) for m in feats] + [("Raw signal", r"Raw, 1 $-$ corr.", raw)]:
            cells = [str(layup) if m == feats[0] else "", lab]
            for setting, cond in sets:
                g = D[(D.layup == layup) & (D.method == m) & (D.setting == setting) & (D.condition == cond)]
                cells.append("--" if g.empty else f"{f2(g.auc.iloc[0])} ({int(g.detected.iloc[0])}/{int(g.n_specimens.iloc[0])})")
            lines.append(cells)
        if layup < 3:
            lines.append([r"\midrule"])
    body = "\n".join(r[0] if r[0] == r"\midrule" else " & ".join(r) + r" \\" for r in lines) + "\n"
    write("ext_nasa.tex", body)


# ------------------------------------------------------------------------------ E4
def e4():
    R = RESULTS / "ext_stats"
    ci = pd.read_csv(R / "task1_ci.csv").set_index("method")
    rep = pd.read_csv(R / "task1_repeated_cv.csv").groupby("method").auc.agg(["mean", "std"])
    t2 = pd.read_csv(R / "task2_per_specimen.csv")
    t2 = t2.groupby("feature").agg(auc=("auc", "mean"), lo=("auc_min", "min"), det=("detected", "mean"))
    t3 = pd.read_csv(R / "task3_ci.csv")
    t3 = t3[t3.freq == "all"].set_index("method")
    lines = []
    for m in sorted(MODELS.values(), key=lambda m: -ci.loc[m, "auc_pooled"]):
        lines.append([NAME[m], f"{f2(ci.loc[m, 'auc_pooled'])} [{f2(ci.loc[m, 'lo'])}, {f2(ci.loc[m, 'hi'])}]",
                      f"{f2(rep.loc[m, 'mean'])} $\\pm$ {f2(rep.loc[m, 'std'])}",
                      f"{f2(t2.loc[m, 'auc'])} ({f2(t2.loc[m, 'lo'])})", f"{100 * t2.loc[m, 'det']:.0f}\\%",
                      f"{int(t3.loc[m.replace('_', ' '), 'correct'])}/28 [{f2(t3.loc[m, 'lo'])}, 1.00]"])
    lines.append([r"\midrule"])
    lines.append(["MiniRocket", f"{f2(ci.loc['MiniRocket', 'auc_pooled'])} [{f2(ci.loc['MiniRocket', 'lo'])}, "
                  f"{f2(ci.loc['MiniRocket', 'hi'])}]", "--", "--", "--", "--"])
    e = "Residual energy (physics)"
    lines.append([r"Raw, 1 $-$ corr. / residual energy", "--", "--",
                  f"{f2(t2.loc['Raw signal | baseline subtraction (1 - correlation)', 'auc'])} "
                  f"({f2(t2.loc['Raw signal | baseline subtraction (1 - correlation)', 'lo'])})",
                  f"{100 * t2.loc['Raw signal | baseline subtraction (1 - correlation)', 'det']:.0f}\\%",
                  f"{int(t3.loc[e, 'correct'])}/28 [{f2(t3.loc[e, 'lo'])}, {f2(t3.loc[e, 'hi'])}]"])
    body = "\n".join(r[0] if r[0] == r"\midrule" else " & ".join(r) + r" \\" for r in lines) + "\n"
    write("ext_stats.tex", body)


# ------------------------------------------------------------------------------ E5
def e5():
    R = RESULTS / "ext_task3"
    reg = pd.read_csv(R / "regression.csv")
    lzo = pd.read_csv(R / "leave_zone_out.csv")
    drop = pd.read_csv(R / "dropout.csv")
    d = drop[drop["mode"] == "retrained"].groupby(["method", "missing", "amount"]).acc.mean()
    names = ["Residual energy (physics)"] + list(MODELS.values())
    order = sorted(names, key=lambda m: reg[(reg.method == m) & (reg.freq == "all")].ridge_median_mm.iloc[0])
    lines = []
    for m in order:
        r = reg[reg.method == m].set_index("freq")
        z = lzo[lzo.method == m].set_index("freq")
        lines.append([NAME.get(m, "Residual energy"), f"{r.loc['all', 'ridge_median_mm']:.0f}",
                      f"{r.loc['40', 'rapid_median_mm']:.0f}",
                      f"{z.loc['all', 'ridge_median_mm']:.0f}", f"{z.loc['all', 'control_median_mm']:.0f}",
                      f2(d[(m, "transducers", 2.0)]), f2(d[(m, "transducers", 4.0)]), f2(d[(m, "paths", 0.5)])])
    ref = lzo[lzo.freq == "all"].mean_position_median_mm.iloc[0]
    write("ext_task3.tex", f"% mean-position reference, leave one zone out: {ref:.0f} mm\n" + rows(lines))


# ------------------------------------------------------------------------------ E6
def e6():
    f = RESULTS / "ext_adapt" / "folds.csv"
    if not f.exists():
        return
    F = pd.read_csv(f)
    g = F.groupby(["model", "variant"]).auc.agg(["mean", "std"])
    lines = []
    for key, label in [("moment", "MOMENT-1 base"), ("tirex", "TiRex")]:
        cells = [label]
        for v in ["LP (sklearn)", "LP (sklearn, 80%)", "LP (gradient)", "FT last 2", "LoRA r=8"]:
            cells.append(f"{f2(g.loc[(key, v), 'mean'])}\\,{{\\scriptsize$\\pm${f2(g.loc[(key, v), 'std'])}}}"
                         if (key, v) in g.index else "--")
        lines.append(cells)
    write("ext_adapt.tex", rows(lines))


if __name__ == "__main__":
    e1()
    e2_models()
    e2_pooling()
    e3()
    e4()
    e5()
    e6()
