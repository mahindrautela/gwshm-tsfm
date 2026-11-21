"""E2: the paper's three protocols, unchanged, on more frozen models.

The task scripts are imported and run as they are, with two module-level names replaced:
the dictionary of models (the paper's six plus TimesFM 2.5, Sundial and Moirai 2) and,
for the pooling ablation (ext_pooling.py), the embeddings folder. Splits, seeds, heads and
detectors are therefore exactly those of Tasks 1-3. The 1D-CNN and MiniRocket are skipped
(--skip-trained): they do not depend on the embeddings and are reported in the paper.

Outputs: results/ext_models/{task1_gwtcw, task2_nasa, task3_ogw1}/ with the same files as
results/, plus check_paper_rows.txt, which confirms that the paper's six models reproduce
results/ exactly in this run.

    python scripts/ext_models.py               # after embedding the new models:
    python scripts/embed.py gwtcw nasa ogw1_residuals ogw1_baselines --models timesfm sundial
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gwtsfm.config import EMBEDDINGS, MODELS, PAPER_MODELS, RESULTS  # noqa: E402

import task1_gwtcw  # noqa: E402
import task2_nasa  # noqa: E402
import task3_ogw1_network  # noqa: E402
import task3_ogw1_paths  # noqa: E402

TASKS = {  # module, results subfolder, extra arguments
    "task1": (task1_gwtcw, "task1_gwtcw", ["--skip-trained"]),
    "task2": (task2_nasa, "task2_nasa", []),
    "task3_paths": (task3_ogw1_paths, "task3_ogw1", ["--skip-trained"]),
    "task3_network": (task3_ogw1_network, "task3_ogw1", []),
}


def run_protocols(models: dict, out: Path, embeddings: Path = EMBEDDINGS, tasks=tuple(TASKS)):
    """Run the chosen task scripts with another model list and embeddings folder."""
    for t in tasks:
        mod, sub, extra = TASKS[t]
        mod.PAPER_MODELS, mod.EMBEDDINGS = models, embeddings
        sys.argv = [t, "--out", str(out / sub), *extra]
        print(f"=== {t} -> {out / sub}", flush=True)
        mod.main()


def check_paper_rows(out: Path) -> str:
    """The paper's models must give exactly the numbers in results/."""
    names = set(PAPER_MODELS.values())
    lines = []
    for f in ["task1_gwtcw/cv_metrics.csv", "task2_nasa/signal_auc.csv", "task3_ogw1/metrics.csv",
              "task3_ogw1/network_localisation.csv", "task3_ogw1/spearman.csv"]:
        a, b = pd.read_csv(RESULTS / f), pd.read_csv(out / f)
        a, b = a[a.method.isin(names)].drop(columns="seconds", errors="ignore"), \
            b[b.method.isin(names)].drop(columns="seconds", errors="ignore")
        keys = [c for c in a.columns if a[c].dtype.kind != "f"]
        m = a.merge(b, on=keys, suffixes=("", "_new"))
        vals = [c for c in a.columns if c not in keys]
        same = all(((m[c] == m[c + "_new"]) | (m[c].isna() & m[c + "_new"].isna())).all() for c in vals)
        lines.append(f"{f}: {len(a)} paper-model rows, {len(m)} matched, identical: {same}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(RESULTS / "ext_models"))
    ap.add_argument("--tasks", nargs="+", default=list(TASKS), choices=list(TASKS))
    args = ap.parse_args()
    out = Path(args.out)
    run_protocols(dict(MODELS), out, tasks=args.tasks)
    if set(args.tasks) == set(TASKS):
        report = check_paper_rows(out)
        (out / "check_paper_rows.txt").write_text(report + "\n")
        print(report)


if __name__ == "__main__":
    main()
