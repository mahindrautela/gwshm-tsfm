"""E2 pooling ablation, step 1: embeddings of every signal under several pooling choices.

For each model, one forward pass over each dataset records all blocks (gwtsfm.pooling) and
keeps seven variants:

    model        the model's own embedding (as in gwtsfm.embed; the paper's choice)
    L25_mean     mean over tokens after the block at 1/4 of the depth
    L50_mean     ... at 1/2 of the depth
    L75_mean     ... at 3/4 of the depth
    L100_mean    ... after the last block (before any final normalisation)
    L100_max     elementwise max over tokens after the last block
    L100_last    last token after the last block

Inputs are exactly those of scripts/embed.py (same scaling, same GW-TCW selection, x and -x
for GW-TCW). Files go to data/embeddings_pool/<dataset>/<model>__<variant>.np[yz], in the
formats the task scripts read, so ext_pooling.py can run the paper's protocols on them.

Runs on mps by default (Mantis and Moirai 2 on cpu; see gwtsfm.embed); Moirai 2 needs the
gwtsfm-moirai environment.

    python scripts/ext_pooling_embed.py --models chronos2 tirex
    ~/miniconda3/envs/gwtsfm-moirai/bin/python scripts/ext_pooling_embed.py --models moirai2
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gwtsfm import embed  # noqa: E402
from gwtsfm.config import DATA, MODELS  # noqa: E402
from gwtsfm.pooling import LayerPool  # noqa: E402
from embed import SOURCES, array_signals, gwtcw_signals  # noqa: E402  (scripts/embed.py)

OUT = DATA / "embeddings_pool"
DEPTHS = {"L25": 0.25, "L50": 0.5, "L75": 0.75, "L100": 1.0}
CPU_ONLY = {"mantis", "moirai2"}


def variants(key, X, segment, device):
    """{variant name: (n, d)} for signals X (n, L)."""
    e = embed.get(key, segment=segment, device="cpu" if key in CPU_ONLY else device)
    with LayerPool(key, e, X.shape[1]) as lp:
        Z, _ = e(X)
    P = lp.result(len(X))
    out = {"model": Z}
    for name, frac in DEPTHS.items():
        out[f"{name}_mean"] = P[(max(1, round(frac * lp.n_layers)) - 1, "mean")]
    out["L100_max"] = P[(lp.n_layers - 1, "max")]
    out["L100_last"] = P[(lp.n_layers - 1, "last")]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=[k for k in MODELS if k != "moirai2"],
                    choices=list(MODELS))
    ap.add_argument("--datasets", nargs="+", default=["gwtcw", *SOURCES], choices=["gwtcw", *SOURCES])
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()
    for dataset in args.datasets:
        folder = OUT / dataset
        folder.mkdir(parents=True, exist_ok=True)
        if dataset == "gwtcw":
            X = gwtcw_signals()
        else:
            X, lead = array_signals(dataset)
        for key in args.models:
            done = folder / f"{key}__model.{'npz' if dataset == 'gwtcw' else 'npy'}"
            if done.exists():
                print(f"[skip] {dataset}/{key}", flush=True)
                continue
            t0 = time.time()
            if dataset == "gwtcw":
                pos, neg = variants(key, X, False, args.device), variants(key, -X, False, args.device)
                for v in sorted(pos, key=lambda v: v == "model"):
                    np.savez(folder / f"{key}__{v}.npz", pos=pos[v], neg=neg[v])
            else:
                V = variants(key, X, SOURCES[dataset][3], args.device)
                for v in sorted(V, key=lambda v: v == "model"):
                    Z = V[v]
                    np.save(folder / f"{key}__{v}.npy", Z.reshape(*lead, -1).astype(np.float16))
            # the "model" file is written last, so an interrupted run is redone
            print(f"[done] {dataset}/{key} in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
