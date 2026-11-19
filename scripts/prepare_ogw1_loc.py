"""Residual-signal localisation dataset from OGW dataset 1 (after Rautela et al., Ultrasonics 2021).

* Residual = damaged signal minus the mean of the 20 baseline measurements of the
  preceding phase (D1-D11 <- baselines 1-20, D12-D20 <- 21-40, D21-D28 <- 41-60).
* Transmission paths only: the 36 pairs between the top row T1-T6 and the bottom
  row T7-T12.
* 40, 60 and 80 kHz; window of 512 samples at 1 MHz from the excitation onset
  (0.51 ms, the time-of-flight window of the paper).
* Labels: damage position D1-D28 and damage zone DG1-DG7 (the seven 2 x 2 clusters of
  positions in Table 3 of the OGW dataset description).

Output: R (3 freqs, 28 positions, 36 paths, 512), plus the baseline-vs-baseline
residuals of each phase (for reference) and the geometry.
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np

PAIRS = list(itertools.combinations(range(1, 13), 2))
CROSS = [PAIRS.index((i, j)) for i in range(1, 7) for j in range(7, 13)]
TPOS = {1: (450, 470), 2: (370, 470), 3: (290, 470), 4: (210, 470), 5: (130, 470), 6: (50, 470),
        7: (450, 30), 8: (370, 30), 9: (290, 30), 10: (210, 30), 11: (130, 30), 12: (50, 30)}
DPOS = {1: (50, 415), 2: (65, 415), 3: (50, 400), 4: (65, 400), 5: (250, 427), 6: (265, 427),
        7: (250, 412), 8: (265, 412), 9: (180, 345), 10: (195, 345), 11: (180, 330), 12: (195, 330),
        13: (320, 275), 14: (335, 275), 15: (320, 260), 16: (335, 260), 17: (130, 195),
        18: (145, 195), 19: (130, 180), 20: (145, 180), 21: (435, 205), 22: (450, 205),
        23: (435, 190), 24: (450, 190), 25: (250, 90), 26: (265, 90), 27: (250, 75), 28: (265, 75)}


def phase_of(k: int) -> int:
    return 0 if k <= 11 else (1 if k <= 20 else 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/ogw1")
    ap.add_argument("--out", default="data/ogw1/loc.npz")
    ap.add_argument("--freqs", type=int, nargs="+", default=[40, 60, 80])
    ap.add_argument("--length", type=int, default=512)
    args = ap.parse_args()
    R, BB = [], []
    for f in args.freqs:
        z = np.load(Path(args.cache) / f"ogw1_f{f}kHz.npz")
        X, names = z["X"].astype(np.float32), z["name"]
        on = int(round(float(z["onset"])))
        win = slice(on, on + args.length)
        bidx = [int(n.split("_")[1]) for n in names if n.startswith("baseline")]
        base = {p: np.mean([X[i] for i, n in enumerate(names) if n.startswith("baseline")
                            and (int(n.split("_")[1]) - 1) // 20 == p], 0) for p in range(3)}
        Rf = np.zeros((28, len(CROSS), args.length), np.float32)
        for i, n in enumerate(names):
            if n.startswith("D"):
                k = int(n[1:])
                Rf[k - 1] = (X[i] - base[phase_of(k)])[CROSS, win]
        # baseline-vs-baseline residuals (each baseline minus its phase mean)
        Bf = np.stack([(X[i] - base[(int(n.split('_')[1]) - 1) // 20])[CROSS, win]
                       for i, n in enumerate(names) if n.startswith("baseline")])
        R.append(Rf)
        BB.append(Bf)
        print(f"f={f} kHz: onset={on}, residuals {Rf.shape}, baseline residuals {Bf.shape}, "
              f"rms damage/baseline residual = {Rf.std() / Bf.std():.1f}")
    pos = np.arange(1, 29)
    np.savez(args.out, R=np.stack(R), B=np.stack(BB), freqs=np.array(args.freqs), position=pos,
             zone=(pos - 1) // 4 + 1, cross=np.array([PAIRS[c] for c in CROSS]),
             dpos=np.array([DPOS[k] for k in pos]), tpos=np.array([TPOS[k] for k in range(1, 13)]))
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
