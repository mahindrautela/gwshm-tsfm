"""Extract NASA PCoE CFRP composites (layup 2) Lamb-wave measurements into one npz.

Saxena, Goebel, Larrosa, Chang, "CFRP Composites Data Set", NASA Prognostics Data
Repository.  Each .mat file is one measurement (specimen, fatigue cycles, boundary
condition, repetition) holding 252 signals: 36 paths (actuators 1-6 -> sensors
7-12) x 7 excitation frequencies (150-450 kHz), 2000 samples at 1.2 MHz.

python scripts/prepare_nasa.py --zip data_ext/nasa/Layup2.zip --out data/nasa/l2.npz \
    --condition 1 --baseline-max-cycles 3 --baseline-per-specimen 8
"""

from __future__ import annotations

import argparse
import io
import re
import zipfile
from pathlib import Path

import numpy as np
import scipy.io as sio

PAT = re.compile(r"/(L2S\d+)_(\d+)_(\d)_(\d+)\.mat$")


def read(z, name):
    c = sio.loadmat(io.BytesIO(z.read(name)), squeeze_me=True, struct_as_record=False)["coupon"]
    p = c.path_data
    freqs = sorted({int(q.frequency) for q in p})
    paths = sorted({(int(q.actuator), int(q.sensor)) for q in p})
    X = np.zeros((len(freqs), len(paths), np.size(p[0].signal_sensor)), np.float32)
    for q in p:
        X[freqs.index(int(q.frequency)), paths.index((int(q.actuator), int(q.sensor)))] = q.signal_sensor
    return X, freqs, paths, str(c.condition), float(p[0].sampling_rate)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--condition", type=int, default=1)
    ap.add_argument("--baseline-max-cycles", type=int, default=3)
    ap.add_argument("--baseline-per-specimen", type=int, default=8)
    ap.add_argument("--damage-cycles", type=int, nargs="+",
                    default=[10, 110, 1000, 10000, 20000, 30000, 50000, 100000, 150000, 200000])
    args = ap.parse_args()
    z = zipfile.ZipFile(args.zip)
    files = []
    for n in z.namelist():
        m = PAT.search(n)
        if m and "/._" not in n:
            files.append((m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4)), n))
    rng = np.random.default_rng(0)
    pick = []
    for s in sorted({f[0] for f in files}):
        early = [f for f in files if f[0] == s and f[1] <= args.baseline_max_cycles and f[2] == args.condition]
        early = sorted(early)
        idx = rng.choice(len(early), min(args.baseline_per_specimen, len(early)), replace=False)
        pick += [early[i] for i in sorted(idx)]
        later = [f for f in files if f[0] == s and f[1] in args.damage_cycles and f[2] == args.condition]
        # one repetition per cycle count (the first)
        seen = set()
        for f in sorted(later):
            if f[1] not in seen:
                pick.append(f)
                seen.add(f[1])
    Xs, meta = [], []
    for s, cy, bc, rep, n in pick:
        X, freqs, paths, cond, fs = read(z, n)
        Xs.append(X)
        meta.append((s, cy, bc, rep, cond))
        print(f"{s} cycles={cy:>6} bc={bc} ({cond}) rep={rep}  {X.shape}", flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, X=np.stack(Xs), specimen=np.array([m[0] for m in meta]),
             cycles=np.array([m[1] for m in meta]), bc=np.array([m[2] for m in meta]),
             rep=np.array([m[3] for m in meta]), condition=np.array([m[4] for m in meta]),
             freqs=np.array(freqs), paths=np.array(paths), fs=fs)
    print(f"saved {args.out}: {len(Xs)} measurements x {Xs[0].shape}")


if __name__ == "__main__":
    main()
