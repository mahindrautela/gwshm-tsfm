"""Compute frozen-model embeddings of every signal used in the three tasks.

    python scripts/embed.py                          # all datasets, all six models
    python scripts/embed.py nasa --models ttm mantis # one dataset, two models

Datasets (input -> output folder under data/embeddings/):
    gwtcw            GW-TCW signals x and their sign-flipped copies -x  -> gwtcw/<model>.npz
                     (arrays 'pos' and 'neg'; first of the 10 repeated acquisitions only,
                     unless --all-repeats)
    nasa             NASA signals, first 1,024 samples               -> nasa/<model>.npy
    ogw1_residuals   OGW-1 damage residuals R                        -> ogw1_residuals/<model>.npy
    ogw1_baselines   OGW-1 drift residuals B                         -> ogw1_baselines/<model>.npy

Each signal array is divided by its overall standard deviation first (one scale per
dataset, so relative amplitudes are kept). The .npy files keep the leading shape of the
input array, e.g. (62, 7, 36, d) for NASA, and are stored in half precision. Existing
files are skipped, so an interrupted run can simply be restarted. This is the slow step
of the project: hours on a CPU for all models and datasets.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gwtsfm import embed  # noqa: E402
from gwtsfm.config import DATA, EMBEDDINGS, MODELS  # noqa: E402

# dataset -> (file, array name, samples to keep, cut MOMENT/Mantis input into native-rate windows)
SOURCES = {
    "nasa": (DATA / "nasa" / "layup2.npz", "X", 1024, True),
    "ogw1_residuals": (DATA / "ogw1" / "residuals.npz", "R", None, True),
    "ogw1_baselines": (DATA / "ogw1" / "residuals.npz", "B", None, True),
}


def gwtcw_signals(all_repeats: bool = False) -> np.ndarray:
    """GW-TCW healthy then damaged signals (n, 1000), scaled to unit standard deviation.

    Rows come in blocks of 120 = 10 repeated acquisitions x 12 sensor paths. The repeats
    are near copies, so by default only the first acquisition of each configuration is
    kept (312 signals).
    """
    z = np.load(DATA / "gwtcw" / "gwtcw.npz")
    X = np.concatenate([z["B"], z["D"]]).astype(np.float32)
    X = X / X.std()
    if not all_repeats:
        X = X[np.arange(len(X)) % 1560 % 120 // 12 == 0]
    return X


def array_signals(dataset: str) -> tuple[np.ndarray, tuple]:
    """Signals of a NASA or OGW-1 array as rows (n, L), plus the leading shape to restore."""
    path, key, length, _ = SOURCES[dataset]
    X = np.load(path)[key]
    lead = X.shape[:-1]
    X = X.reshape(-1, X.shape[-1])[:, :length]
    return (X / X.std()).astype(np.float32), lead


def log_time(folder: Path, model, seconds, n, dim):
    with open(folder / "timing.jsonl", "a") as fh:
        fh.write(json.dumps({"model": model.name, "seconds": seconds, "n": n, "dim": dim}) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("datasets", nargs="*", default=["gwtcw", *SOURCES],
                    choices=["gwtcw", *SOURCES])
    ap.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    ap.add_argument("--all-repeats", action="store_true",
                    help="GW-TCW: embed all 3,120 signals (needed only for the leaky-split check)")
    args = ap.parse_args()

    for dataset in args.datasets:
        folder = EMBEDDINGS / dataset
        folder.mkdir(parents=True, exist_ok=True)
        if dataset == "gwtcw":
            X = gwtcw_signals(args.all_repeats)
        else:
            X, lead = array_signals(dataset)
        for key in args.models:
            out = folder / (f"{key}.npz" if dataset == "gwtcw" else f"{key}.npy")
            if out.exists():
                print(f"[skip] {dataset}/{out.name} exists", flush=True)
                continue
            try:
                if dataset == "gwtcw":
                    model = embed.get(key)                       # resample to 512 for MOMENT/Mantis
                    Zp, t1 = model(X)
                    Zn, t2 = model(-X)
                    np.savez(out, pos=Zp, neg=Zn)
                    log_time(folder, model, t1 + t2, 2 * len(X), Zp.shape[1])
                else:
                    model = embed.get(key, segment=SOURCES[dataset][3])
                    Z, t = model(X)
                    np.save(out, Z.reshape(*lead, -1).astype(np.float16))
                    log_time(folder, model, t, len(X), Z.shape[1])
                print(f"[done] {dataset}/{out.name}", flush=True)
            except Exception:
                print(f"[FAIL] {dataset}/{key}\n{traceback.format_exc()}", flush=True)


if __name__ == "__main__":
    main()
