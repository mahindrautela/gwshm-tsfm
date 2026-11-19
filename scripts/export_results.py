"""Copy the result tables behind the paper from results/ into paper_results/ (kept in git).

    python scripts/export_results.py
"""

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gwtsfm.config import RESULTS, ROOT  # noqa: E402

FILES = {
    "task1_gwtcw": ["cv_metrics.csv", "fewshot.csv", "data_summary.json"],
    "task2_nasa": ["signal_auc.csv", "measurement_auc.csv", "growth_curve.csv"],
    "task3_ogw1": ["metrics.csv", "network_localisation.csv", "spearman.csv"],
}

if __name__ == "__main__":
    for task, names in FILES.items():
        dst = ROOT / "paper_results" / task
        dst.mkdir(parents=True, exist_ok=True)
        for name in names:
            src = RESULTS / task / name
            if src.exists():
                shutil.copy2(src, dst / name)
                print(f"{task}/{name}")
            else:
                print(f"missing: {src}")
