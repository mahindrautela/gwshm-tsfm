"""Copy the small result tables behind the paper into paper_results/ (tracked in git).

    python scripts/export_results.py
"""
import shutil
from pathlib import Path

FILES = {
    "task1_gwtcw": ["results_gwtcw/cv_metrics.csv", "results_gwtcw/fewshot.csv", "results_gwtcw/data_summary.json"],
    "task2_nasa": ["results_nasa/summary_table.csv", "results_nasa/tables.md", "results_nasa/signal_auc.csv",
                   "results_nasa/measurement_auc.csv", "results_nasa/growth_curve.csv",
                   "results_nasa/healthy_spread.csv"],
    "task3_ogw1": ["results_ogw1/metrics.csv", "results_ogw1_network/network_localisation.csv",
                   "results_ogw1_network/spearman.csv"],
}


def main(root=Path("."), out=Path("paper_results")):
    for task, files in FILES.items():
        (out / task).mkdir(parents=True, exist_ok=True)
        for f in files:
            src = root / f
            if src.exists():
                shutil.copy2(src, out / task / src.name)
                print(f"{task}/{src.name}")
            else:
                print(f"missing: {f}")


if __name__ == "__main__":
    main()
