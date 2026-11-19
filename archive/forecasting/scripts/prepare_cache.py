"""Build decimated per-frequency caches from the raw OGW folders.

python scripts/prepare_cache.py --root data/raw --out data/cache --freqs 40 100 160
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1])); sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from gwtsfm.ogw import FOLDERS, build_cache, describe_h5, list_measurements  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--freqs", type=int, nargs="+", required=True)
    ap.add_argument("--oversample", type=float, default=8.0)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    first = next((d for d in list_measurements(args.root, "udam")), None)
    if first is not None:
        f0 = next(first.glob("*.h5"))
        (out / "h5_structure.json").write_text(json.dumps(describe_h5(f0), indent=1))
        print(f"structure of {f0.name} -> {out / 'h5_structure.json'}")
    for st in FOLDERS:
        print(f"{st}: {len(list_measurements(args.root, st))} measurement folders")
    for f in args.freqs:
        c = build_cache(args.root, f, oversample=args.oversample)
        c.save(out / f"f{f}kHz.npz")
        print(f"f={f} kHz -> X{c.X.shape}, fs={c.fs/1e3:.1f} kHz, meta={c.meta}")


if __name__ == "__main__":
    main()
