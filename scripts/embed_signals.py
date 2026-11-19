"""Frozen TSFM embeddings for any array of signals (..., L) stored in an npz.

python scripts/embed_signals.py --npz data/nasa/l2.npz --key X --length 1024 --segment \
    --out data/nasa_emb --models ttm mantis chronos_bolt moment chronos2 tirex
"""
from __future__ import annotations

import argparse, json, sys, traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gwtsfm import embed  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--key", default="X")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--length", type=int, default=None)
    ap.add_argument("--segment", action="store_true", help="MOMENT/Mantis: native-rate 512 windows")
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", nargs="+", default=list(embed.REGISTRY))
    args = ap.parse_args()
    X = np.load(args.npz)[args.key]
    lead = X.shape[:-1]
    X = X.reshape(-1, X.shape[-1])[:, args.start:(args.start + args.length) if args.length else None]
    X = (X / X.std()).astype(np.float32)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for m in args.models:
        f = out / f"{m}.npy"
        if f.exists():
            print(f"[skip] {m}", flush=True)
            continue
        try:
            e = embed.get(m, segment=args.segment)
            Z, t = e(X)
            np.save(f, Z.reshape(*lead, -1).astype(np.float16))
            with open(out / "timing.jsonl", "a") as fh:
                fh.write(json.dumps({"model": m, "name": e.name, "seconds": t, "n": len(X),
                                     "dim": Z.shape[1]}) + "\n")
            print(f"[done] {m}: {Z.shape}  {t:.0f}s", flush=True)
        except Exception:
            print(f"[FAIL] {m}\n{traceback.format_exc()}", flush=True)


if __name__ == "__main__":
    main()
