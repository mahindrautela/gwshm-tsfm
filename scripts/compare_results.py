"""Compare every result table in results/ with the reference copy in paper_results/.

The reference is read from git (branch main by default), because audit_datasets.py and
export_results.py write into paper_results/ itself. The new audits.json is therefore the one
in the working tree's paper_results/; every other new table is the one in results/.

Rows are matched on their identifying columns (text and integer columns); every float column
except ``seconds`` is compared. For each file we print the number of matched rows, how many
values differ at all, how many differ at the paper's printed precision (|diff| >= 0.005), and
the largest difference per column with the row it occurs in.

    python scripts/compare_results.py               # all files
    python scripts/compare_results.py --rows 5      # also list the 5 largest differences
    python scripts/compare_results.py --rev HEAD    # reference from another commit
"""

from __future__ import annotations

import argparse
import io
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gwtsfm.config import RESULTS, ROOT  # noqa: E402

IGNORE = {"seconds"}


def git_file(rev: str, path: str) -> bytes:
    return subprocess.run(["git", "show", f"{rev}:{path}"], cwd=ROOT, check=True,
                          capture_output=True).stdout


def git_files(rev: str, folder: str) -> list[str]:
    out = subprocess.run(["git", "ls-tree", "-r", "--name-only", rev, folder], cwd=ROOT,
                         check=True, capture_output=True, text=True).stdout
    return sorted(out.split())


def compare_csv(ref: bytes, new: Path, n_rows: int) -> None:
    a, b = pd.read_csv(io.BytesIO(ref)), pd.read_csv(new)
    if list(a.columns) != list(b.columns):
        print(f"  columns differ: {list(a.columns)} vs {list(b.columns)}")
        return
    values = [c for c in a.columns if c not in IGNORE and a[c].dtype.kind == "f"]
    keys = [c for c in a.columns if c not in IGNORE and c not in values]
    # number repeated keys so that duplicate rows are matched in order
    for df in (a, b):
        df["_k"] = df.groupby(keys, dropna=False).cumcount() if keys else np.arange(len(df))
    m = a.merge(b, on=keys + ["_k"], how="outer", suffixes=("_ref", "_new"), indicator=True)
    only_ref, only_new = (m["_merge"] == "left_only").sum(), (m["_merge"] == "right_only").sum()
    m = m[m["_merge"] == "both"]
    print(f"  rows: {len(a)} reference, {len(b)} new, {len(m)} matched"
          + (f", {only_ref} only in reference, {only_new} only in new" if only_ref or only_new else ""))
    worst = []
    for c in values:
        x, y = m[f"{c}_ref"].to_numpy(float), m[f"{c}_new"].to_numpy(float)
        same = (x == y) | (np.isnan(x) & np.isnan(y))
        d = np.where(same, 0.0, np.abs(x - y))
        d = np.where(np.isnan(d), np.inf, d)            # NaN on one side only
        i = int(np.argmax(d)) if len(d) else 0
        print(f"  {c:20s} differ: {int((~same).sum()):5d}   >=0.005: {int((d >= 0.005).sum()):5d}   "
              f"max |diff| {d.max() if len(d) else 0:.3g}")
        for j in np.argsort(-d)[:n_rows]:
            if d[j] > 0:
                worst.append((d[j], c, m.iloc[j][keys].to_dict(), x[j], y[j]))
    for d, c, k, x, y in sorted(worst, key=lambda t: -t[0])[:n_rows]:
        print(f"    {c}: {x:.4f} -> {y:.4f}  {k}")


def compare_json(ref: bytes, new: Path) -> None:
    a, b = json.loads(ref), json.loads(new.read_text())
    print("  identical" if a == b else f"  differs:\n    reference {a}\n    new       {b}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows", type=int, default=0, help="list the largest differences")
    ap.add_argument("--rev", default="main", help="git revision holding the reference tables")
    args = ap.parse_args()
    for path in git_files(args.rev, "paper_results"):
        rel = Path(path).relative_to("paper_results")
        new = RESULTS / rel if len(rel.parts) > 1 else ROOT / path
        print(f"== {rel}")
        ref = git_file(args.rev, path)
        if not new.exists():
            print(f"  missing: {new}")
        elif rel.suffix == ".csv":
            compare_csv(ref, new, args.rows)
        else:
            compare_json(ref, new)


if __name__ == "__main__":
    main()
