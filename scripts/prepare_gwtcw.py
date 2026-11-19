"""Download GW-TCW (Task 1) from Hugging Face and save it as one array file.

GW-TCW: guided waves on a tapered composite wing (Memmolo et al. 2016; Rautela et al.
2022). 1,560 healthy (baseline) and 1,560 damaged signals of 1,000 samples at 2 MHz.

Output data/gwtcw/gwtcw.npz:
    B  (1560, 1000)  healthy signals
    D  (1560, 1000)  damaged signals; row i of D has the same configuration as row i of B
    t  (1000,)       time vector

    python scripts/prepare_gwtcw.py
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gwtsfm.config import DATA  # noqa: E402

URL = "https://huggingface.co/datasets/mahindrautela/GW-TCW/resolve/main/"
FILES = {"B": "IISHM_BaselineExpData.txt", "D": "IISHM_DamageExpData.txt", "t": "IISHM_timevector.txt"}


def main():
    folder = DATA / "gwtcw"
    folder.mkdir(parents=True, exist_ok=True)
    for name in FILES.values():
        if not (folder / name).exists():
            print("downloading", name, flush=True)
            urllib.request.urlretrieve(URL + name, folder / name)
    np.savez(folder / "gwtcw.npz",
             B=np.loadtxt(folder / FILES["B"], delimiter=",").astype("f4"),
             D=np.loadtxt(folder / FILES["D"], delimiter=",").astype("f4"),
             t=np.loadtxt(folder / FILES["t"]))
    print("saved", folder / "gwtcw.npz")


if __name__ == "__main__":
    main()
