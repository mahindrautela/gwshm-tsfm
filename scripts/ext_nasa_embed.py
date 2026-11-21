"""E3 step 2: frozen embeddings of the NASA layups and boundary conditions.

As scripts/embed.py does for Task 2 (first 1,024 samples; MOMENT and Mantis on native-rate
512-sample windows), with one change for the new data: each array is divided by the standard
deviation of its healthy (<= 3 cycles) signals only, so no damaged data enters the scaling.

Output: data/embeddings/nasa_ext/layup<l>_bc<b>/<model>.npy, shape (n, 7, 36, d), float16.

    python scripts/ext_nasa_embed.py                      # main environment models, on mps
    ~/miniconda3/envs/gwtsfm-moirai/bin/python scripts/ext_nasa_embed.py --models moirai2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gwtsfm import embed  # noqa: E402
from gwtsfm.config import DATA, EMBEDDINGS, MODELS  # noqa: E402

L = 1024
CPU_ONLY = {"mantis", "moirai2"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=[k for k in MODELS if k != "moirai2"], choices=list(MODELS))
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()
    for f in sorted((DATA / "nasa" / "ext").glob("layup*_bc*.npz")):
        z = np.load(f)
        X = z["X"][..., :L].astype(np.float32)
        X = X / X[z["cycles"] <= 3].std()
        folder = EMBEDDINGS / "nasa_ext" / f.stem
        folder.mkdir(parents=True, exist_ok=True)
        for key in args.models:
            out = folder / f"{key}.npy"
            if out.exists():
                continue
            m = embed.get(key, segment=True, device="cpu" if key in CPU_ONLY else args.device)
            Z, t = m(X.reshape(-1, L))
            np.save(out, Z.reshape(*X.shape[:3], -1).astype(np.float16))
            print(f"[done] {f.stem}/{key} {t / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
