"""E1 step 1: signals and frozen embeddings of the OGW temperature dataset (40 kHz).

Input: data/ogw_temp/ogwT_f40kHz_<plate>.npz from ext_fetch_ogw_temperature.py.

Sessions, in time order of recording:
    udam-1   undamaged, first temperature cycle (161 measurements)   -> healthy references
    udam-2   undamaged, second cycle (161)                           -> healthy test
    D12, D16, D04, D24   one cycle each with the defect at that position (161 each) -> damaged test
Each cycle steps the chamber 20.5 -> 60 -> 20.5 degC in 0.5 degC steps.

Signals: the 36 paths that cross the plate (as in Task 3), the first 1,024 samples (1.02 ms
at 1 MHz), divided by the standard deviation of the udam-1 signals only (no test data).

Output:
    data/ogw_temp/signals_f40kHz.npz   X (966, 36, 1024), session, temp, stamp, cross
    data/embeddings/ogw_temp/<model>.npy   (966, 36, d) float16

    python scripts/ext_temperature_embed.py --models chronos2 tirex ...
    ~/miniconda3/envs/gwtsfm-moirai/bin/python scripts/ext_temperature_embed.py --models moirai2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gwtsfm import embed  # noqa: E402
from gwtsfm.config import DATA, EMBEDDINGS, MODELS  # noqa: E402
from gwtsfm.ogw import CROSS_PATHS, PAIRS  # noqa: E402

RAW = DATA / "ogw_temp"
SIGNALS = RAW / "signals_f40kHz.npz"
L = 1024
CPU_ONLY = {"mantis", "moirai2"}


def prepare():
    parts = []
    z = np.load(RAW / "ogwT_f40kHz_udam.npz")
    order = np.argsort(z["stamp"])
    half = len(order) // 2                                    # two temperature cycles
    for name, idx in [("udam-1", order[:half]), ("udam-2", order[half:])]:
        parts.append((name, z["X"][idx], z["temp"][idx], z["stamp"][idx]))
    for plate in ["D04", "D12", "D16", "D24"]:
        z = np.load(RAW / f"ogwT_f40kHz_{plate}.npz")
        order = np.argsort(z["stamp"])
        parts.append((plate, z["X"][order], z["temp"][order], z["stamp"][order]))
    parts.sort(key=lambda p: p[3][0])                         # sessions in recording order
    X = np.concatenate([p[1][:, CROSS_PATHS, :L] for p in parts]).astype(np.float32)
    session = np.concatenate([[p[0]] * len(p[1]) for p in parts])
    scale = X[session == "udam-1"].std()                     # reference data only
    np.savez(SIGNALS, X=X / scale, session=session, temp=np.concatenate([p[2] for p in parts]),
             stamp=np.concatenate([p[3] for p in parts]), scale=scale,
             cross=np.array([PAIRS[c] for c in CROSS_PATHS]))
    print(f"saved {SIGNALS}: X {X.shape}, sessions", {s: int((session == s).sum()) for s in dict.fromkeys(session)})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="*", default=[k for k in MODELS if k != "moirai2"], choices=list(MODELS))
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()
    if not SIGNALS.exists():
        prepare()
    z = np.load(SIGNALS)
    X = z["X"]
    folder = EMBEDDINGS / "ogw_temp"
    folder.mkdir(parents=True, exist_ok=True)
    for key in args.models:
        out = folder / f"{key}.npy"
        if out.exists():
            print(f"[skip] {out.name}", flush=True)
            continue
        dev = "cpu" if key in CPU_ONLY else args.device
        model = embed.get(key, segment=True, device=dev)           # native rate, 2 windows for MOMENT/Mantis
        Z, t = model(X.reshape(-1, L))
        np.save(out, Z.reshape(*X.shape[:2], -1).astype(np.float16))
        with open(folder / "timing.jsonl", "a") as fh:
            fh.write(json.dumps({"model": model.name, "seconds": t, "n": int(len(Z)), "dim": int(Z.shape[1]),
                                 "device": dev}) + "\n")
        print(f"[done] {out.name} ({t / 60:.1f} min on {dev})", flush=True)


if __name__ == "__main__":
    main()
