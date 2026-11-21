"""E3 step 1: NASA measurements of every layup and boundary condition.

scripts/prepare_nasa.py is reused unchanged (same seed, same selection: up to 8 healthy
measurements at <= 3 cycles per specimen, then the first repetition at each later cycle
count), with its file-name pattern widened from layup 2 to layups 1-3. Layup 2 under
condition 1 is rebuilt too, so that every array is embedded the same way; it is checked to
equal the paper's data/nasa/layup2.npz.

Output: data/nasa/ext/layup<l>_bc<b>.npz, same arrays as data/nasa/layup2.npz.

    python scripts/ext_nasa_prepare.py
"""

from __future__ import annotations

import re
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402

from gwtsfm.config import DATA  # noqa: E402

import prepare_nasa  # noqa: E402

OUT = DATA / "nasa" / "ext"
COMPOSITES = DATA / "nasa" / "Composites.zip"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    prepare_nasa.NAME = re.compile(r"/(L\dS\d+)_(\d+)_(\d)_(\d+)\.mat$")
    for layup in (1, 2, 3):
        src = DATA / "nasa" / f"Layup{layup}.zip"
        if not src.exists():  # extract the layup archive from the 4.6 GB download
            with zipfile.ZipFile(COMPOSITES) as z, open(src, "wb") as fh:
                fh.write(z.read(f"2. Composites/Layup{layup}.zip"))
        for bc in (1, 2, 3):
            out = OUT / f"layup{layup}_bc{bc}.npz"
            if out.exists():
                continue
            sys.argv = ["prepare_nasa", "--zip", str(src), "--out", str(out), "--condition", str(bc)]
            print(f"=== layup {layup}, condition {bc}", flush=True)
            prepare_nasa.main()
    a, b = np.load(OUT / "layup2_bc1.npz"), np.load(DATA / "nasa" / "layup2.npz")
    print("layup 2, condition 1 identical to the paper's array:",
          all(np.array_equal(a[k], b[k]) for k in b.files))


if __name__ == "__main__":
    main()
