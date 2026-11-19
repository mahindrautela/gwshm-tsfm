"""Extract frozen TSFM embeddings for the GW-TCW dataset (x and -x).

python scripts/gwtcw_embed.py --data data/gwtcw/gwtcw.npz --out data/gwtcw_emb \
    --models ttm mantis moment chronos_bolt chronos2 tirex
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", nargs="+", default=list(embed.REGISTRY))
    ap.add_argument("--all-repeats", action="store_true",
                    help="embed all 10 repeats (default: first repeat of each configuration)")
    args = ap.parse_args()
    z = np.load(args.data)
    X = np.concatenate([z["B"], z["D"]]).astype(np.float32)
    X = X / X.std()  # one global scale: relative amplitudes are preserved
    if not args.all_repeats:  # repeats correlate at r = 0.99998: keep the first of each
        X = X[np.arange(len(X)) % 1560 % 120 // 12 == 0]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for m in args.models:
        f = out / f"{m}.npz"
        if f.exists():
            print(f"[skip] {m}", flush=True)
            continue
        try:
            e = embed.get(m)
            Zp, tp = e(X)
            Zn, tn = e(-X)
            np.savez(f, pos=Zp, neg=Zn)
            with open(out / "timing.jsonl", "a") as fh:
                fh.write(json.dumps({"model": m, "name": e.name, "seconds": tp + tn,
                                     "n": 2 * len(X), "dim": Zp.shape[1]}) + "\n")
            print(f"[done] {m}: dim={Zp.shape[1]}  {tp + tn:.0f}s", flush=True)
        except Exception:
            print(f"[FAIL] {m}\n{traceback.format_exc()}", flush=True)


if __name__ == "__main__":
    main()
