"""Write the paper's result tables as LaTeX rows (paper/tables/*.tex), straight from results/.

Every number in a results table of the paper comes from here; nothing is typed by hand.
Values are rounded to two decimals only at the end, and ties at that precision are all
set in bold.

    python paper/make_tables.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gwtsfm.config import MODELS, RESULTS  # noqa: E402

OUT = Path(__file__).resolve().parent / "tables"
OUT.mkdir(exist_ok=True)

# model names as printed in the paper ("Chronos-Bolt (base)" -> "Chronos-Bolt base")
NAME = {m: m.replace(" (base)", " base") for m in MODELS.values()}
TSFM = list(NAME)


def pm(m, sd):
    """mean with a small '+- s.d.' after it"""
    return f"{m:.2f}\\,{{\\scriptsize$\\pm${sd:.2f}}}"


def f2(x):
    """two decimals, or a dash when missing"""
    return "--" if pd.isna(x) else f"{x:.2f}"


def bold(s, on):
    """wrap in \\textbf when on"""
    return rf"\textbf{{{s}}}" if on else s


def same2(a, b):
    """Equal at the two decimals shown in the table."""
    return (not pd.isna(a)) and (not pd.isna(b)) and f"{a:.2f}" == f"{b:.2f}"


def write(name, body):
    """save rows to paper/tables/<name>"""
    (OUT / name).write_text(body)
    print("wrote", name)


def task1():
    """Table 3: GW-TCW classification (rows sorted by sign-symmetric AUC)."""
    cv = pd.read_csv(RESULTS / "task1_gwtcw/cv_metrics.csv")
    g = cv.groupby(["method", "mode", "head"])
    auc, sd, bacc = g.auc.mean(), g.auc.std(), g.bacc.mean()

    def row(m, head_main, kind):
        ss = (m, "sign-symmetric", head_main)
        rec = (m, "as recorded", head_main)
        return dict(name=NAME.get(m, m), kind=kind, auc=auc.get(ss), sd=sd.get(ss), bacc=bacc.get(ss),
                    rec=auc.get(rec), knn=auc.get((m, "sign-symmetric", "kNN (k=5)")),
                    one=auc.get((m, "sign-symmetric", "one-class (no damage labels)")))
    rows = [row(m, "logistic regression", "frozen") for m in TSFM]
    rows.sort(key=lambda r: -r["auc"])
    refs = [row("MiniRocket", "ridge", "trained"), row("|FFT| spectrum", "logistic regression", "trained"),
            row("1D-CNN (reference notebook)", "end-to-end", "trained")]
    refs[1]["name"], refs[2]["name"] = r"$|$FFT$|$ + log.\ reg.", "1D-CNN"
    refs[0]["name"] = "MiniRocket + ridge"
    # bold marks the best frozen model in each column
    best_frozen = {k: max(r[k] for r in rows) for k in ["auc", "bacc", "rec", "knn", "one"]}
    lines = []
    for r in rows:
        c = [r["name"]]
        c.append(bold(pm(r["auc"], r["sd"]), same2(r["auc"], best_frozen["auc"])))
        for k in ["bacc", "rec", "knn", "one"]:
            c.append(bold(f2(r[k]), same2(r[k], best_frozen[k])))
        lines.append(" & ".join(c) + r" \\")
    lines.append(r"\midrule")
    lines.append(r"\multicolumn{6}{l}{\emph{Trained on the target data}} \\")
    for r in refs:
        c = [r["name"], pm(r["auc"], r["sd"])]
        for k in ["bacc", "rec", "knn", "one"]:
            c.append(f2(r[k]))
        lines.append(" & ".join(c) + r" \\")
    probe = auc.get(("Peak polarity only", "sign-symmetric", "logistic regression"))
    probe_rec = auc.get(("Peak polarity only", "as recorded", "logistic regression"))
    lines.append(r"\midrule")
    lines.append(f"Polarity probe & {probe:.2f} & -- & {probe_rec:.2f} & -- & -- \\\\")
    write("task1.tex", "\n".join(lines) + "\n")


def task2():
    """Table 4: NASA detection within and across specimens (rows sorted by cross-specimen AUC)."""
    sa = pd.read_csv(RESULTS / "task2_nasa/signal_auc.csv")
    ma = pd.read_csv(RESULTS / "task2_nasa/measurement_auc.csv")
    gs = sa.groupby(["setting", "method", "detector"]).auc_signal
    gm = ma.groupby(["setting", "method", "detector"]).detected_zero_fa.mean()
    t = {"mean": gs.mean(), "std": gs.std(), "detected_zero_fa": gm}

    def get(setting, method, det, col):
        return t[col].get((setting, method, det), np.nan)

    W, X = "within-specimen", "cross-specimen"
    rows = []
    for m in TSFM:
        rows.append(dict(name=NAME[m], kind="tsfm",
                         sp=get(W, m, "same-path distance", "mean"), sd=get(W, m, "same-path distance", "std"),
                         knn=get(W, m, "kNN distance", "mean"), pca=get(W, m, "PCA+ocSVM", "mean"),
                         zw=get(W, m, "same-path distance", "detected_zero_fa"),
                         xa=get(X, m, "same-path distance", "mean"),
                         zx=get(X, m, "same-path distance", "detected_zero_fa")))
    rows.sort(key=lambda r: (-r["xa"]))
    raw = []
    for det, lab in [("baseline subtraction (1 - correlation)", r"Baseline subtraction, 1 $-$ corr."),
                     ("baseline subtraction (residual energy)", "Baseline subtraction, residual energy"),
                     ("same-path distance", "Raw signal, same-path distance")]:
        raw.append(dict(name=lab, sp=get(W, "Raw signal", det, "mean"), sd=get(W, "Raw signal", det, "std"),
                        knn=get(W, "Raw signal", "kNN distance", "mean") if det == "same-path distance" else np.nan,
                        pca=get(W, "Raw signal", "PCA+ocSVM", "mean") if det == "same-path distance" else np.nan,
                        zw=get(W, "Raw signal", det, "detected_zero_fa"), xa=get(X, "Raw signal", det, "mean"),
                        zx=get(X, "Raw signal", det, "detected_zero_fa")))
    m = "|FFT| spectrum"
    raw.append(dict(name=r"$|$FFT$|$, same-path distance", sp=get(W, m, "same-path distance", "mean"),
                    sd=get(W, m, "same-path distance", "std"), knn=get(W, m, "kNN distance", "mean"),
                    pca=get(W, m, "PCA+ocSVM", "mean"), zw=get(W, m, "same-path distance", "detected_zero_fa"),
                    xa=get(X, m, "same-path distance", "mean"), zx=get(X, m, "same-path distance", "detected_zero_fa")))
    allr = rows + raw
    best = {k: max(r[k] for r in allr if not pd.isna(r[k])) for k in ["sp", "knn", "pca", "xa"]}
    lines = []

    def fmt(r):
        c = [r["name"], bold(pm(r["sp"], r["sd"]), same2(r["sp"], best["sp"])),
             bold(f2(r["knn"]), same2(r["knn"], best["knn"])),
             bold(f2(r["pca"]), same2(r["pca"], best["pca"])),
             f"{round(r['zw'] * 20)}/20", bold(f2(r["xa"]), same2(r["xa"], best["xa"])),
             bold(f"{round(r['zx'] * 4)}/4", r["zx"] == 1)]
        return " & ".join(c) + r" \\"
    for r in rows:
        lines.append(fmt(r))
    lines.append(r"\midrule")
    lines.append(r"\multicolumn{7}{l}{\emph{Raw-signal and spectrum references}} \\")
    for r in raw:
        lines.append(fmt(r))
    write("task2.tex", "\n".join(lines) + "\n")


def task3():
    """Tables 5, 6, 10 and 11: OGW-1 path-level and network-level localisation, RAPID."""
    n = pd.read_csv(RESULTS / "task3_ogw1/network_localisation.csv")
    lr = n[n.approach == "logistic regression on DI map"].pivot_table(index="method", columns="freq", values="acc")
    ctrl = n[n.approach.str.startswith("control")].set_index("method")
    rap = n[n.approach == "RAPID (training-free)"].pivot_table(index="method", columns="freq", values="acc")
    raperr = n[n.approach == "RAPID (training-free)"].pivot_table(index="method", columns="freq",
                                                                  values="median_error_mm")
    order = sorted(TSFM, key=lambda m: (-lr.loc[m, [40, 60, 80]].mean()))
    lines = []
    for m in order + ["Residual energy (physics)"]:
        if m == "Residual energy (physics)":
            lines.append(r"\midrule")
        nm = NAME.get(m, "Residual energy")
        c = [nm] + [bold(f2(lr.loc[m, f]), lr.loc[m, f] == 1) for f in [40, 60, 80, 0]]
        c += [f2(ctrl.loc[m, "acc"]), f2(ctrl.loc[m, "acc_max"])]
        lines.append(" & ".join(c) + r" \\")
    write("task3_network.tex", "\n".join(lines) + "\n")

    lines = []
    for m in order + ["Residual energy (physics)"]:
        if m == "Residual energy (physics)":
            lines.append(r"\midrule")
        nm = NAME.get(m, "Residual energy")
        c = [nm] + [f"{f2(rap.loc[m, f])} ({raperr.loc[m, f]:.0f})" for f in [40, 60, 80]]
        lines.append(" & ".join(c) + r" \\")
    write("task3_rapid.tex", "\n".join(lines) + "\n")

    p = pd.read_csv(RESULTS / "task3_ogw1/metrics.csv")
    g = p.groupby(["method", "head", "split"])[["acc", "acc_structure"]].mean()
    spl = ["random 90/10", "grouped (position, path)", "unseen positions"]

    def prow(m, head, label):
        c = [label] + [f2(g.acc.get((m, head, s))) for s in spl] + [f2(g.acc_structure.get((m, head, spl[2])))]
        return " & ".join(c) + r" \\"
    knn = "kNN (k=5)"
    ordp = sorted(TSFM, key=lambda m: -g.acc_structure.get((m, knn, spl[2])))
    lines = [prow(m, knn, NAME[m]) for m in ordp]
    lines.append(r"\midrule")
    lines.append(prow("|FFT| spectrum", knn, r"$|$FFT$|$ spectrum"))
    lines.append(prow("Raw residual", knn, "Raw residual"))
    lines.append(prow("MiniRocket", "ridge", "MiniRocket$^\\dagger$"))
    lines.append(prow("1D-CNN", "end-to-end", "1D-CNN$^\\dagger$"))
    write("task3_path.tex", "\n".join(lines) + "\n")

    # logistic-regression head, for the appendix
    lines = [prow(m, "logistic regression", NAME[m]) for m in ordp]
    lines.append(r"\midrule")
    lines.append(prow("|FFT| spectrum", "logistic regression", r"$|$FFT$|$ spectrum"))
    lines.append(prow("Raw residual", "logistic regression", "Raw residual"))
    write("task3_path_lr.tex", "\n".join(lines) + "\n")


def growth():
    """Table 9: damage growth, cycle counts with at least three specimens."""
    g = pd.read_csv(RESULTS / "task2_nasa/growth_curve.csv")
    g = g[g.detector == "same-path distance"]
    piv = g.pivot_table(index="method", columns="cycles", values="ratio")
    ns = g.groupby("cycles").n_specimens.min()
    cols = [c for c in piv.columns if ns[c] >= 3]
    order = TSFM + ["Raw signal", "|FFT| spectrum"]
    lines = []
    for m in order:
        nm = {**NAME, "Raw signal": "Raw signal", "|FFT| spectrum": r"$|$FFT$|$ spectrum"}[m]
        if m == "Raw signal":
            lines.append(r"\midrule")
        lines.append(" & ".join([nm] + [f"{piv.loc[m, c]:.1f}" for c in cols]) + r" \\")
    head = " & ".join([f"{int(c / 1000)}k" for c in cols])
    write("task2_growth.tex", "% columns: " + head + "\n" + "\n".join(lines) + "\n")


if __name__ == "__main__":
    task1()
    task2()
    task3()
    growth()
