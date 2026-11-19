"""Fetch OGW dataset 1 (constant temperature, damage positions D1-D28) into compact caches.

Moll et al., Structural Health Monitoring 18(5-6), 2019.  Files are read from the
public Nextcloud share linked on openguidedwaves.de, decimated in memory and saved
as one ``ogw1_f<freq>kHz.npz`` per frequency: X (n_measurements, 66, N), names
('baseline_k' or 'Dk'), timestamps, fs.

python scripts/fetch_ogw1.py --out data/ogw1 --freqs 40 60 80 --fs-target 1e6
"""

from __future__ import annotations

import argparse
import io
import re
import sys
import time
import urllib.request
from base64 import b64encode
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gwtsfm.ogw import decimate, read_h5  # noqa: E402

SHARE = "UfcaKdsx1imUrTm"
DAV = "https://nextcloud.faserinstitut.de/public.php/webdav/"
AUTH = {"Authorization": "Basic " + b64encode(f"{SHARE}:".encode()).decode()}


def folders() -> list[str]:
    req = urllib.request.Request(DAV, method="PROPFIND", headers={**AUTH, "Depth": "1"})
    s = urllib.request.urlopen(req, timeout=120).read().decode()
    names = [h.rstrip("/").rsplit("/", 1)[-1] for h in re.findall(r"<d:href>([^<]+)</d:href>", s)]
    return sorted(n for n in names if re.fullmatch(r"\d{8}T\d{6}_(baseline_\d+|D\d+)", n))


def get(path: str) -> bytes:
    for attempt in range(5):
        try:
            req = urllib.request.Request(DAV + path, headers=AUTH)
            return urllib.request.urlopen(req, timeout=300).read()
        except Exception as e:  # transient network errors
            if attempt == 4:
                raise
            time.sleep(3 + 5 * attempt)
            print(f"   retry {path}: {e!r}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--freqs", type=int, nargs="+", default=[40, 60, 80])
    ap.add_argument("--fs-target", type=float, default=1e6)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fl = folders()
    print(f"{len(fl)} measurement folders", flush=True)
    for f in args.freqs:
        dst = out / f"ogw1_f{f}kHz.npz"
        if dst.exists():
            print(f"f={f}: done already", flush=True)
            continue

        def one(name):
            x, t, fs, onset = read_h5(io.BytesIO(get(f"{name}/pc_f{f}kHz.h5")), with_onset=True)
            q = max(1, int(round((fs or 10e6) / args.fs_target)))
            return name, decimate(x, q), t, fs, q, onset

        t0, res = time.time(), []
        with ThreadPoolExecutor(args.workers) as ex:
            for k, r in enumerate(ex.map(one, fl), 1):
                res.append(r)
                if k % 10 == 0 or k == len(fl):
                    print(f"  f={f} kHz {k}/{len(fl)}  {time.time() - t0:.0f}s", flush=True)
        names = np.array([r[0].split("_", 1)[1] for r in res])
        n = min(r[1].shape[-1] for r in res)
        np.savez(dst, X=np.stack([r[1][..., :n] for r in res]), name=names,
                 stamp=np.array([r[0].split("_", 1)[0] for r in res]),
                 temp=np.array([r[2] for r in res]), fs=res[0][3] / res[0][4],
                 onset=int(np.median([r[5] for r in res])) / res[0][4], f_khz=f)
        print(f"f={f} kHz -> {dst}  X{(len(res), 66, n)}", flush=True)


if __name__ == "__main__":
    main()
