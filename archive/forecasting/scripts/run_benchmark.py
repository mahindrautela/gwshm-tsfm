"""Run the zero-shot temperature-extrapolation benchmark (resumable).

Example
-------
python scripts/run_benchmark.py --cache data/cache --out results \
    --freqs 40 100 160 --tc 30 40 50 --methods baselines last linear chronos2 timesfm25

Results: <out>/f<freq>/Tc<tc>/<method>.parquet  (long format, one row per
test measurement x path) and <out>/timing.jsonl.  Forecast examples for a few
paths are stored as <out>/f<freq>/Tc<tc>/<method>_examples.npz.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1])); sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from forecasting import models, protocol  # noqa: E402
from gwtsfm.ogw import FreqCache, pair_index  # noqa: E402

EXAMPLE_PATHS = [pair_index(4, 9), pair_index(3, 10), pair_index(1, 7)]


def slug(name: str) -> str:
    return name.replace(" ", "_").replace("/", "-").replace("(", "").replace(")", "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--freqs", type=int, nargs="+", required=True)
    ap.add_argument("--tc", type=float, nargs="+", default=[30.0, 40.0, 50.0])
    ap.add_argument("--methods", nargs="+", default=["baselines", "last", "linear"])
    ap.add_argument("--t-end-ms", type=float, default=0.6)
    ap.add_argument("--chunk", type=int, default=8192)
    ap.add_argument("--warp", action="store_true",
                    help="forecast in the physics-warped (stretch-compensated) space")
    ap.add_argument("--paths", choices=["all", "cross"], default="all",
                    help="all 66 pairs, or the 36 cross-plate pairs T1-T6 -> T7-T12")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    setups = {}

    def setup(f):
        if f not in setups:
            c = FreqCache.load(Path(args.cache) / f"f{f}kHz.npz")
            paths = None if args.paths == "all" else np.array(
                [pair_index(i, j) for i in range(1, 7) for j in range(7, 13)])
            setups[f] = protocol.prepare(c, t_end_ms=args.t_end_ms, paths=paths)
            s = setups[f]
            print(f"[setup] f={f} kHz: refs={len(s.ref_T)} ({s.ref_T.min():.1f}-{s.ref_T.max():.1f} C), "
                  f"test={len(s.T)}, paths={len(s.paths)}, N={s.X.shape[-1]}, fs={s.fs/1e3:.0f} kHz",
                  flush=True)
        return setups[f]

    for method in args.methods:
        tag = "_warp" if args.warp else ""
        todo = [(f, tc) for f in args.freqs for tc in args.tc
                if not (out / f"f{f}" / f"Tc{tc:g}" / f"{slug(method)}{tag}.parquet").exists()]
        if not todo:
            print(f"[skip] {method}: all done", flush=True)
            continue
        model = None
        if method != "baselines":
            try:
                t0 = time.time()
                model = models.get(method)
                print(f"[load] {method} ({model.name}) in {time.time() - t0:.1f}s", flush=True)
            except Exception:
                print(f"[FAIL load] {method}\n{traceback.format_exc()}", flush=True)
                continue
        for f, tc in todo:
            s = setup(f)
            d = out / f"f{f}" / f"Tc{tc:g}"
            d.mkdir(parents=True, exist_ok=True)
            t0 = time.time()
            try:
                if method == "baselines":
                    df = protocol.run_baselines(s, tc)
                    n_series, H = 0, 0
                else:
                    if args.warp:
                        Y, H, rates = protocol.warped_context(s, tc)
                        fc = protocol.dewarp(model(Y, H, chunk=args.chunk), s, tc, rates)
                        name = f"Stretch warp + {model.name}"
                    else:
                        Y, sel, H = protocol.context(s, tc)
                        fc = model(Y, H, chunk=args.chunk)
                        name = model.name
                    t_fc = time.time() - t0
                    P, N = len(s.paths), s.X.shape[-1]
                    np.savez_compressed(d / f"{slug(method)}{tag}_fc.npz", paths=s.paths, grid=s.grid,
                                        **{q: getattr(fc, q).reshape(P, N, H).astype(np.float16)
                                           for q in ("q10", "q50", "q90")})
                    df_nom, ex = protocol.evaluate_forecast(s, tc, f"{name} [probe T]", fc,
                                                            keep_examples=True)
                    df = pd.concat([protocol.evaluate_library(s, tc, name, fc), df_nom],
                                   ignore_index=True)
                    ex_paths = [list(s.paths).index(p) for p in EXAMPLE_PATHS if p in s.paths]
                    np.savez_compressed(d / f"{slug(method)}{tag}_examples.npz",
                                        paths=s.paths[ex_paths], grid=s.grid,
                                        **{k: v[ex_paths].astype(np.float32) for k, v in ex.items()})
                    n_series = len(Y)
                df.to_parquet(d / f"{slug(method)}{tag if method != 'baselines' else ''}.parquet")
                dt = time.time() - t0
                with open(out / "timing.jsonl", "a") as fh:
                    fh.write(json.dumps({"method": method + tag, "f_khz": f, "Tc": tc, "seconds": dt,
                                         "forecast_seconds": t_fc if method != "baselines" else 0.0,
                                         "n_series": n_series, "horizon": H}) + "\n")
                print(f"[done] {method} f={f} Tc={tc:g}: {dt:.1f}s ({n_series} series, H={H})",
                      flush=True)
            except Exception:
                print(f"[FAIL] {method} f={f} Tc={tc:g}\n{traceback.format_exc()}", flush=True)
        del model
        gc.collect()


if __name__ == "__main__":
    main()
