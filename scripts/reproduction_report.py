"""Paper value next to reproduced value, for Tables 3-6 and every number quoted in Section 5.

Tables: each cell of the committed paper/tables/*.tex (git revision main by default) is set
next to the same cell of the regenerated file. Regenerate with paper/make_tables.py and copy
paper/tables/*.tex to results/paper_tables_reproduced/ (or pass --tables paper/tables). Rows
are matched by their label, since make_tables.py sorts rows by value.

Quoted numbers: each claim of Section 5 is recomputed from a folder of result tables, once
from the reference tables committed in paper_results/ (at the git revision) and once from
results/, and printed next to the value written in the paper.

    python scripts/reproduction_report.py > reproduction_tables.md
"""

from __future__ import annotations

import argparse
import io
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gwtsfm.config import PAPER_MODELS, RESULTS, ROOT  # noqa: E402

TSFM = list(PAPER_MODELS.values())
TABLES = {"task1.tex": "Table 3 (Task 1, GW-TCW)", "task2.tex": "Table 4 (Task 2, NASA)",
          "task3_path.tex": "Table 5 (Task 3, path level)", "task3_network.tex": "Table 6 (Task 3, network level)"}
HEADERS = {
    "task1.tex": ["Features", "Sign-sym. AUC", "Bal. acc.", "As rec.", "k-NN head", "One-class"],
    "task2.tex": ["Features and detector", "Same-path", "k-NN", "PCA + ocSVM", "Detected",
                  "Cross: same-path", "Cross: detected"],
    "task3_path.tex": ["Features", "Random", "Grouped", "Unseen", "36 paths, unseen"],
    "task3_network.tex": ["Damage index", "40 kHz", "60 kHz", "80 kHz", "All", "Control mean", "Control max"],
}


def git(rev: str, path: str) -> bytes:
    return subprocess.run(["git", "show", f"{rev}:{path}"], cwd=ROOT, check=True,
                          capture_output=True).stdout


# ---------------------------------------------------------------- tables

def plain(cell: str) -> str:
    """LaTeX cell -> plain text, e.g. '\\textbf{0.86\\,{\\scriptsize$\\pm$0.03}}' -> '0.86 ± 0.03'."""
    s = cell.strip()
    s = re.sub(r"\\textbf\{(.*)\}", r"\1", s)
    s = s.replace(r"\,{\scriptsize$\pm$", " ± ").replace("}", "").replace("{", "")
    s = s.replace("$|$", "|").replace(r"$-$", "−").replace("$^\\dagger$", "†").replace(r"\ ", " ")
    return s.replace("\\", "").strip().replace("|", "\\|")     # escape pipes for Markdown


def rows(tex: str) -> dict[str, list[str]]:
    out = {}
    for line in tex.splitlines():
        if "&" not in line or line.startswith("%") or "multicolumn" in line:
            continue
        cells = [plain(c) for c in line.rstrip().removesuffix(r"\\").split("&")]
        out[cells[0]] = cells[1:]
    return out


def table_markdown(name: str, rev: str, folder: Path) -> str:
    ref = rows(git(rev, f"paper/tables/{name}").decode())
    new = rows((folder / name).read_text())
    head = HEADERS[name]
    lines = [f"#### {TABLES[name]}", "", "Each cell: paper / reproduced. Differences in bold.", "",
             "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    n_cells = n_diff = 0
    for label, a in ref.items():
        b = new.get(label, ["(missing)"] * len(a))
        cells = []
        for x, y in zip(a, b):
            n_cells += 1
            if x == y:
                cells.append(f"{x} / {y}")
            else:
                n_diff += 1
                cells.append(f"**{x} / {y}**")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines += ["", f"{n_cells - n_diff} of {n_cells} cells identical.", ""]
    return "\n".join(lines)


# ---------------------------------------------------------------- quoted numbers

def rng(values) -> str:
    v = [f"{x:.2f}" for x in sorted(values)]
    return v[0] if v[0] == v[-1] else f"{v[0]}–{v[-1]}"


def quoted(load) -> list[tuple[str, str, str]]:
    """(section, claim, value) for every number in Section 5, from the tables that load() returns."""
    out = []
    add = lambda sec, claim, val: out.append((sec, claim, val))  # noqa: E731

    # Task 1 ----------------------------------------------------------------------------
    cv = load("task1_gwtcw/cv_metrics.csv")
    a = cv.groupby(["method", "mode", "head"]).auc.mean()
    ss, rec, lr = "sign-symmetric", "as recorded", "logistic regression"
    five = [m for m in TSFM if m != "TTM-R2"]
    add("5.1", "LR, sign-symmetric AUC, five of six models", rng(a[m, ss, lr] for m in five))
    add("5.1", "MiniRocket AUC", f"{a['MiniRocket', ss, 'ridge']:.2f}")
    add("5.1", "1D-CNN AUC", f"{a['1D-CNN (reference notebook)', ss, 'end-to-end']:.2f}")
    add("5.1", "|FFT| spectrum AUC (\"does clearly worse\")", f"{a['|FFT| spectrum', ss, lr]:.2f}")
    add("5.1", "TTM-R2 AUC", f"{a['TTM-R2', ss, lr]:.2f}")
    leak = cv[cv["mode"] == "random split (leaky)"].groupby(["method", "head"]).auc.mean()
    add("5.1", "Random split: k-NN AUC on MOMENT, Mantis, TTM",
        rng(leak[m, "kNN (k=5)"] for m in ["MOMENT-1 (base)", "Mantis-8M", "TTM-R2"]))
    add("5.1", "Polarity probe AUC (sign-symmetric / as recorded)",
        f"{a['Peak polarity only', ss, lr]:.2f} / {a['Peak polarity only', rec, lr]:.2f}")
    add("5.1", "AUC gain from sign symmetry, every frozen model",
        rng(a[m, ss, lr] - a[m, rec, lr] for m in TSFM))
    add("5.1", "Mantis one-class AUC (no damage labels)", f"{a['Mantis-8M', ss, 'one-class (no damage labels)']:.2f}")
    fs = load("task1_gwtcw/fewshot.csv").groupby(["method", "n_configs"]).auc.mean()
    add("5.1", "Best AUC of any method with up to 8 labelled configurations",
        f"{max(v for (m, n), v in fs.items() if n <= 8):.2f}")
    ns = sorted(n for n in fs.index.get_level_values(1).unique() if 32 <= n <= 64)
    ahead = all(fs[m, n] > fs["|FFT| spectrum", n] for m in TSFM for n in ns)
    add("5.1", "From 32 configurations on, every frozen model ahead of the spectrum", "yes" if ahead else "no")

    # Task 2 ----------------------------------------------------------------------------
    sa = load("task2_nasa/signal_auc.csv")
    ma = load("task2_nasa/measurement_auc.csv")
    s = sa.groupby(["setting", "method", "detector"]).auc_signal.mean()
    z = ma.groupby(["setting", "method", "detector"]).detected_zero_fa.mean()
    W, X, sp = "within-specimen", "cross-specimen", "same-path distance"
    fivec = [m for m in TSFM if m != "Chronos-2"]
    add("5.2", "Within: same-path AUC, five models", rng(s[W, m, sp] for m in fivec))
    add("5.2", "Within: same-path AUC, Chronos-2", f"{s[W, 'Chronos-2', sp]:.2f}")
    add("5.2", "Within: baseline subtraction, 1 − correlation", f"{s[W, 'Raw signal', 'baseline subtraction (1 - correlation)']:.2f}")
    add("5.2", "Within: damaged measurements detected at zero false alarms, worst model",
        f"{min(round(z[W, m, sp] * 20) for m in TSFM)}/20")
    feats = TSFM + ["Raw signal", "|FFT| spectrum"]
    add("5.2", "Same-path beats k-NN for every feature set",
        "yes" if all(s[W, m, sp] > s[W, m, "kNN distance"] for m in feats) else "no")
    add("5.2", "Within: PCA + ocSVM on embeddings", rng(s[W, m, "PCA+ocSVM"] for m in TSFM))
    add("5.2", "Within: PCA + ocSVM on raw signals", f"{s[W, 'Raw signal', 'PCA+ocSVM']:.2f}")
    raw_dets = [d for (st, m, d) in s.index if st == X and m == "Raw signal"]
    add("5.2", "Cross: raw-signal detectors (all five)", rng(s[X, "Raw signal", d] for d in raw_dets))
    raw_tab = ["baseline subtraction (1 - correlation)", "baseline subtraction (residual energy)", sp]
    add("5.2", "Cross: raw-signal detectors shown in Table 4", rng(s[X, "Raw signal", d] for d in raw_tab))
    add("5.2", "Cross: raw k-NN distance / PCA + ocSVM (not in Table 4)",
        f"{s[X, 'Raw signal', 'kNN distance']:.2f} / {s[X, 'Raw signal', 'PCA+ocSVM']:.2f}")
    for m, short in [("Mantis-8M", "Mantis"), ("Chronos-Bolt (base)", "Chronos-Bolt"), ("TiRex", "TiRex"),
                     ("MOMENT-1 (base)", "MOMENT")]:
        add("5.2", f"Cross: {short} same-path AUC, coupons detected",
            f"{s[X, m, sp]:.2f}, {round(z[X, m, sp] * 4)}/4")
    add("5.2", "Within: MOMENT same-path AUC", f"{s[W, 'MOMENT-1 (base)', sp]:.2f}")
    g = load("task2_nasa/growth_curve.csv")
    g = g[g.detector == sp].pivot_table(index="method", columns="cycles", values="ratio")
    r10 = [g.loc[m, 10000] for m in TSFM]
    add("5.2", "Growth: same-path ratio to healthy at 10k cycles, every model", f"{min(r10):.1f}–{max(r10):.1f}")
    peak = {m: int(g.loc[m, [c for c in g.columns if c <= 200000]].idxmax()) for m in TSFM}
    add("5.2", "Growth: cycle count where each model's ratio peaks",
        ", ".join(f"{k.split()[0]} {v // 1000}k" for k, v in peak.items()))

    # Task 3 ----------------------------------------------------------------------------
    p = load("task3_ogw1/metrics.csv")
    gp = p.groupby(["method", "head", "split"])[["acc", "acc_structure"]].mean()
    single = []
    for m, head in [(m, "kNN (k=5)") for m in TSFM + ["|FFT| spectrum", "Raw residual"]] + \
                   [("MiniRocket", "ridge"), ("1D-CNN", "end-to-end")]:
        for spl in ["random 90/10", "grouped (position, path)", "unseen positions"]:
            if (m, head, spl) in gp.index:          # MiniRocket and the CNN have no grouped split
                single.append(gp.acc[m, head, spl])
    add("5.3", "Path level: single-path accuracy, every feature set and split (Table 5)", rng(single))
    pooled = {m: gp.acc_structure[m, "kNN (k=5)", "unseen positions"] for m in TSFM}
    best = max(pooled, key=pooled.get)
    add("5.3", "Path level: best 36-path pooled accuracy", f"{pooled[best]:.2f} ({best})")
    n = load("task3_ogw1/network_localisation.csv")
    lrm = n[n.approach == "logistic regression on DI map"].pivot_table(index="method", columns="freq", values="acc")
    add("5.3", "Network: models with 28/28 (frequencies combined)",
        f"{sum(lrm.loc[m, 0] == 1 for m in TSFM)} of 6")
    lo = beta.ppf(0.025, 28, 1)
    add("5.3", "Clopper–Pearson 95% interval for 28/28", f"{lo:.2f}–1.00")
    every = [m.split()[0] for m in TSFM if all(lrm.loc[m, f] == 1 for f in [40, 60, 80])]
    add("5.3", "Network: models with 28/28 at every frequency alone", ", ".join(every))
    e = "Residual energy (physics)"
    add("5.3", "Network: residual energy at 40 / 60 / 80 kHz",
        " / ".join(f"{lrm.loc[e, f]:.2f}" for f in [40, 60, 80]))
    ctrl = n[n.approach.str.startswith("control")].set_index("method")
    add("5.3", "Control: mean accuracy over draws, frozen models", rng(ctrl.loc[m, "acc"] for m in TSFM))
    add("5.3", "Control: mean accuracy, residual energy", f"{ctrl.loc[e, 'acc']:.2f}")
    sp_ = load("task3_ogw1/spearman.csv").pivot_table(index="method", columns="freq_khz", values="spearman_mean")
    add("5.3", "Spearman, MOMENT and TiRex, all frequencies",
        rng(sp_.loc[m, f] for m in ["MOMENT-1 (base)", "TiRex"] for f in [40, 60, 80]))
    add("5.3", "Spearman, residual energy at 40 / 80 kHz", f"{sp_.loc[e, 40]:.2f} / {sp_.loc[e, 80]:.2f}")
    rap = n[n.approach == "RAPID (training-free)"].pivot_table(index="method", columns="freq", values="acc")
    for f in [40, 80]:
        add("5.3", f"RAPID at {f} kHz: embeddings vs residual energy",
            f"{rng(rap.loc[m, f] for m in TSFM)} vs {rap.loc[e, f]:.2f}")
    return out


# the values as written in the text of Section 5, in the order of quoted()
PAPER_TEXT = [
    "0.83–0.86", "0.90", "0.69", "(worse)", "0.74", "1.00", "chance (Table 3)", "0.02–0.07", "0.79",
    "≤ 0.64", "yes",
    "0.95–0.98", "0.88", "0.97", "20/20", "yes", "0.44–0.62", "0.23", "0.15–0.57", "0.15–0.57", "(in range)",
    "0.81, 4/4", "0.67, 4/4", "3/4", "0.47", "0.95", "1.9–3.2", "100k–150k",
    "0.17–0.29", "0.81 (MOMENT)", "6 of 6", "0.88–1.00", "MOMENT, TiRex", "0.96 / 0.61 / 0.25",
    "0.15–0.20", "0.15–0.20", "0.77–0.82", "0.49 / 0.16", "0.32–0.50 vs 0.75", "0.29–0.61 vs 0.07",
]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rev", default="main", help="git revision of the paper's tables and results")
    ap.add_argument("--tables", default=str(RESULTS / "paper_tables_reproduced"),
                    help="folder with the regenerated tables/*.tex")
    args = ap.parse_args()

    print("### Tables 3–6\n")
    for name in TABLES:
        print(table_markdown(name, args.rev, Path(args.tables)))

    ref = quoted(lambda f: pd.read_csv(io.BytesIO(git(args.rev, f"paper_results/{f}"))))
    new = quoted(lambda f: pd.read_csv(RESULTS / f))
    assert len(ref) == len(PAPER_TEXT), (len(ref), len(PAPER_TEXT))
    print("### Numbers quoted in Section 5\n")
    print("Claims marked † are printed by scripts/task1_cnn_check.py and are not kept in")
    print("paper_results/; the reproduced value is read from logs_task1_cnn.txt.\n")
    print("| § | Claim | Paper text | From paper_results/ | Reproduced | Same |")
    print("|---|---|---|---|---|---|")
    for (sec, claim, r), (_, _, v), t in zip(ref, new, PAPER_TEXT):
        claim = claim.replace("|", "\\|")
        print(f"| {sec} | {claim} | {t} | {r} | {v} | {'yes' if r == v else '**no**'} |")
    log = ROOT / "logs_task1_cnn.txt"
    text = log.read_text() if log.exists() else ""
    for claim, key, paper in [("Random split, random early-stopping data: 1D-CNN AUC †", "random split", "1.00"),
                              ("Grouped split, fold 0, 150 epochs: 1D-CNN AUC †", "150 epochs", "0.62")]:
        m = re.search(rf"{key}.*AUC ([0-9.]+)", text)
        v = f"{float(m.group(1)):.2f}" if m else "(not run)"
        print(f"| 5.1 | {claim} | {paper} | – | {v} | {'yes' if v == paper else '**no**'} |")


if __name__ == "__main__":
    main()
