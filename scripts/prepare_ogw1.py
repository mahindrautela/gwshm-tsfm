"""Build the Task 3 residual signals from OGW dataset 1 (after Rautela et al., Ultrasonics 2021).

A residual is what damage adds to a signal: the damaged signal minus the mean of the 20
baseline measurements of the same phase (D1-D11 use baselines 1-20, D12-D20 use 21-40,
D21-D28 use 41-60). We keep the 36 paths that cross the plate, three frequencies, and a
512-sample window (0.51 ms at 1 MHz) from the excitation onset.

The baselines also give "drift residuals": each baseline minus its phase mean. They show
how much a path changes without any damage and serve as the healthy reference.

Output data/ogw1/residuals.npz:
    R (3, 28, 36, 512)  damage residuals [frequency, position, path, sample]
    B (3, 60, 36, 512)  drift residuals  [frequency, baseline, path, sample]
    freqs, position, zone, cross (36, 2), dpos (28, 2), tpos (12, 2)

    python scripts/prepare_ogw1.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gwtsfm.config import DATA  # noqa: E402
from gwtsfm.ogw import CROSS_PATHS, DAMAGE_XY, PAIRS, TRANSDUCER_XY, phase_of, zone_of  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", default=str(DATA / "ogw1"), help="output of fetch_ogw1.py")
    ap.add_argument("--out", default=str(DATA / "ogw1" / "residuals.npz"))
    ap.add_argument("--freqs", type=int, nargs="+", default=[40, 60, 80])
    ap.add_argument("--length", type=int, default=512)
    args = ap.parse_args()

    R, B = [], []
    for f in args.freqs:
        z = np.load(Path(args.folder) / f"ogw1_f{f}kHz.npz")
        X, names = z["X"].astype(np.float32), z["name"]
        onset = int(round(float(z["onset"])))
        window = slice(onset, onset + args.length)

        def baseline_phase(name):  # 'baseline_1'..'baseline_60' -> phase 0, 1 or 2
            return (int(name.split("_")[1]) - 1) // 20

        is_baseline = np.array([n.startswith("baseline") for n in names])
        phase_mean = {p: np.mean([X[i] for i, n in enumerate(names)
                                  if is_baseline[i] and baseline_phase(n) == p], 0) for p in range(3)}

        Rf = np.zeros((28, len(CROSS_PATHS), args.length), np.float32)
        for i, n in enumerate(names):
            if n.startswith("D"):
                k = int(n[1:])  # damage position 1..28
                Rf[k - 1] = (X[i] - phase_mean[phase_of(k)])[CROSS_PATHS, window]
        Bf = np.stack([(X[i] - phase_mean[baseline_phase(n)])[CROSS_PATHS, window]
                       for i, n in enumerate(names) if is_baseline[i]])
        R.append(Rf)
        B.append(Bf)
        print(f"f={f} kHz: onset={onset}, residuals {Rf.shape}, drift residuals {Bf.shape}, "
              f"rms damage/drift = {Rf.std() / Bf.std():.1f}")

    pos = np.arange(1, 29)
    np.savez(args.out, R=np.stack(R), B=np.stack(B), freqs=np.array(args.freqs), position=pos,
             zone=np.array([zone_of(k) for k in pos]), cross=np.array([PAIRS[c] for c in CROSS_PATHS]),
             dpos=np.array([DAMAGE_XY[k] for k in pos]),
             tpos=np.array([TRANSDUCER_XY[k] for k in range(1, 13)]))
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
