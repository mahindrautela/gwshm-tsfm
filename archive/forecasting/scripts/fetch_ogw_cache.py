"""Stream selected frequencies of the OGW temperature dataset into compact caches.

Reads the per-measurement H5 files straight out of the figshare zip archives with
HTTP range requests, decimates them in memory and writes one cache per frequency
(``f<freq>kHz.npz``).  Raw files never touch the disk.  Progress is saved per
damage state, so an interrupted run resumes where it stopped.

python scripts/fetch_ogw_cache.py --out data/cache --freqs 40 100 160 --workers 4
"""

from __future__ import annotations

import argparse
import io
import json
import re
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from gwtsfm.ogw import FreqCache, decimate, decimation_factor, describe_h5, read_h5  # noqa: E402

API = "https://api.figshare.com/v2"
COLLECTION = 4488089
STATES = {"udam": "OGW_CFRP_Temperature_udam", "D04": "OGW_CFRP_Temperature_dam_D04",
          "D12": "OGW_CFRP_Temperature_dam_D12", "D16": "OGW_CFRP_Temperature_dam_D16",
          "D24": "OGW_CFRP_Temperature_dam_D24"}


def get_json(url):
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.load(r)


def archive_urls() -> dict[str, str]:
    urls = {}
    for art in get_json(f"{API}/collections/{COLLECTION}/articles?page_size=100"):
        for st, folder in STATES.items():
            if art["title"].strip() == folder:
                info = get_json(f"{API}/articles/{art['id']}")
                urls[st] = next(f["download_url"] for f in info["files"] if f["name"].endswith(".zip"))
    return urls


_local = threading.local()


def _zip(url):
    from remotezip import RemoteZip
    z = getattr(_local, "zips", {}).get(url)
    if z is None:
        z = RemoteZip(url)
        _local.zips = {**getattr(_local, "zips", {}), url: z}
    return z


def fetch_state(url: str, f_khz: int, workers: int, oversample: float, out_dir: Path):
    from remotezip import RemoteZip
    with RemoteZip(url) as z:
        members = sorted(n for n in z.namelist() if n.endswith(f"/pc_f{f_khz}kHz.h5"))
        first = z.read(members[0])
    struct = describe_h5(io.BytesIO(first))
    fs = float(struct["datasets"]["command/pitchcatch/sampling_frequency"]["value"][0])
    q = decimation_factor(fs, f_khz, oversample)

    def one(name):
        for attempt in range(4):
            try:
                raw = _zip(url).read(name)
                x, t, _, onset = read_h5(io.BytesIO(raw), with_onset=True)
                stamp = next(p for p in PurePosixPath(name).parts if re.fullmatch(r"\d{8}T\d{6}", p))
                return stamp, decimate(x, q), t, onset
            except Exception as e:  # transient network errors
                if attempt == 3:
                    raise
                _local.zips = {}
                time.sleep(2 + 3 * attempt)
                print(f"   retry {name}: {e!r}", flush=True)

    t0, done, res = time.time(), 0, []
    with ThreadPoolExecutor(workers) as ex:
        for r in ex.map(one, members):
            res.append(r)
            done += 1
            if done % 40 == 0 or done == len(members):
                el = time.time() - t0
                print(f"   {done}/{len(members)}  {done * len(first) / 1e6 / el:.1f} MB/s", flush=True)
    res.sort(key=lambda r: r[0])
    return dict(stamp=np.array([r[0] for r in res]), X=np.stack([r[1] for r in res]),
                temp=np.array([r[2] for r in res]), onset=np.array([r[3] for r in res]),
                fs=fs, q=q), struct


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--freqs", type=int, nargs="+", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--oversample", type=float, default=8.0)
    args = ap.parse_args()
    out = Path(args.out)
    part = out / "partial"
    part.mkdir(parents=True, exist_ok=True)
    urls = archive_urls()
    print("archives:", {k: v.rsplit("/", 1)[-1] for k, v in urls.items()}, flush=True)
    for f in args.freqs:
        final = out / f"f{f}kHz.npz"
        if final.exists():
            print(f"f={f}: done already", flush=True)
            continue
        for st in STATES:
            pf = part / f"f{f}_{st}.npz"
            if pf.exists():
                continue
            print(f"f={f} kHz {st}:", flush=True)
            d, struct = fetch_state(urls[st], f, args.workers, args.oversample, out)
            np.savez(pf, **d)
            sj = out / "h5_structure.json"
            if not sj.exists():
                sj.write_text(json.dumps(struct, indent=1))
        parts = {st: np.load(part / f"f{f}_{st}.npz") for st in STATES}
        q, fs = int(parts["udam"]["q"]), float(parts["udam"]["fs"])
        onset_raw = int(np.median(np.concatenate([p["onset"] for p in parts.values()])))
        n = min(p["X"].shape[-1] for p in parts.values())
        c = FreqCache(
            f_khz=f, fs=fs / q,
            X=np.concatenate([p["X"][..., :n] for p in parts.values()]),
            temp=np.concatenate([p["temp"] for p in parts.values()]),
            state=np.concatenate([[st] * len(p["temp"]) for st, p in parts.items()]),
            stamp=np.concatenate([p["stamp"] for p in parts.values()]),
            meta={"fs_raw": fs, "decimation": q, "fs_source": "file",
                  "onset_raw": onset_raw, "onset": onset_raw / q})
        c.save(final)
        print(f"f={f} kHz -> {final} X{c.X.shape} fs={c.fs / 1e3:.1f} kHz onset_raw={onset_raw}",
              flush=True)


if __name__ == "__main__":
    main()
