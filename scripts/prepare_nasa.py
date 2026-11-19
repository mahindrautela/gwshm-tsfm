"""Extract the NASA PCoE CFRP measurements used in Task 2 into one array file.

Saxena, Goebel, Larrosa, Chang, "CFRP Composites Data Set", NASA Prognostics Data
Repository. Each .mat file in Layup2.zip is one measurement of one coupon (specimen) at
a number of fatigue cycles, under a boundary condition, with a repetition number. A
measurement holds 252 signals: 36 paths (actuators 1-6 to sensors 7-12) x 7 excitation
frequencies (150-450 kHz), 2,000 samples at 1.2 MHz.

We keep, for every layup-2 specimen, under the loaded boundary condition (condition 1):
  * healthy: 8 measurements chosen at random among those at <= 3 cycles
  * later:   the first repetition at each of 10, 110, 1k, 10k, 20k, 30k, 50k, 100k,
             150k and 200k cycles, where it exists (20k is the "damaged" state)

Output data/nasa/layup2.npz:
    X (62, 7, 36, 2000) signals [measurement, frequency, path, sample]
    specimen, cycles, bc, rep, condition: one entry per measurement
    freqs (7,), paths (36, 2), fs

    python scripts/prepare_nasa.py --zip data/nasa/Layup2.zip
"""

from __future__ import annotations

import argparse
import io
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import scipy.io as sio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gwtsfm.config import DATA, SEED  # noqa: E402

# file names look like .../L2S11_10000_1_2.mat = specimen, cycles, boundary condition, repetition
NAME = re.compile(r"/(L2S\d+)_(\d+)_(\d)_(\d+)\.mat$")
LATER_CYCLES = [10, 110, 1000, 10000, 20000, 30000, 50000, 100000, 150000, 200000]


def read_measurement(z: zipfile.ZipFile, name: str):
    """Read one .mat file into an array (frequency, path, sample) plus its metadata."""
    c = sio.loadmat(io.BytesIO(z.read(name)), squeeze_me=True, struct_as_record=False)["coupon"]
    p = c.path_data
    freqs = sorted({int(q.frequency) for q in p})
    paths = sorted({(int(q.actuator), int(q.sensor)) for q in p})
    X = np.zeros((len(freqs), len(paths), np.size(p[0].signal_sensor)), np.float32)
    for q in p:
        X[freqs.index(int(q.frequency)), paths.index((int(q.actuator), int(q.sensor)))] = q.signal_sensor
    return X, freqs, paths, str(c.condition), float(p[0].sampling_rate)


def select(files, condition, max_healthy_cycles, healthy_per_specimen):
    """Pick the healthy and later measurements of every specimen (see module docstring)."""
    rng = np.random.default_rng(SEED)
    pick = []
    for s in sorted({f[0] for f in files}):
        early = sorted(f for f in files
                       if f[0] == s and f[1] <= max_healthy_cycles and f[2] == condition)
        idx = rng.choice(len(early), min(healthy_per_specimen, len(early)), replace=False)
        pick += [early[i] for i in sorted(idx)]
        seen = set()
        for f in sorted(f for f in files if f[0] == s and f[1] in LATER_CYCLES and f[2] == condition):
            if f[1] not in seen:  # first repetition at each cycle count
                pick.append(f)
                seen.add(f[1])
    return pick


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", default=str(DATA / "nasa" / "Layup2.zip"))
    ap.add_argument("--out", default=str(DATA / "nasa" / "layup2.npz"))
    ap.add_argument("--condition", type=int, default=1, help="boundary condition (1 = loaded)")
    ap.add_argument("--max-healthy-cycles", type=int, default=3)
    ap.add_argument("--healthy-per-specimen", type=int, default=8)
    args = ap.parse_args()

    z = zipfile.ZipFile(args.zip)
    files = []  # (specimen, cycles, condition, repetition, file name)
    for n in z.namelist():
        m = NAME.search(n)
        if m and "/._" not in n:  # skip macOS resource-fork copies
            files.append((m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4)), n))

    Xs, meta = [], []
    for s, cycles, bc, rep, n in select(files, args.condition, args.max_healthy_cycles,
                                        args.healthy_per_specimen):
        X, freqs, paths, cond, fs = read_measurement(z, n)
        Xs.append(X)
        meta.append((s, cycles, bc, rep, cond))
        print(f"{s} cycles={cycles:>6} bc={bc} ({cond}) rep={rep}  {X.shape}", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, X=np.stack(Xs), specimen=np.array([m[0] for m in meta]),
             cycles=np.array([m[1] for m in meta]), bc=np.array([m[2] for m in meta]),
             rep=np.array([m[3] for m in meta]), condition=np.array([m[4] for m in meta]),
             freqs=np.array(freqs), paths=np.array(paths), fs=fs)
    print(f"saved {args.out}: {len(Xs)} measurements x {Xs[0].shape}")


if __name__ == "__main__":
    main()
