"""Download OGW dataset 1 (Task 3) and keep compact copies of the frequencies we use.

Moll et al., Structural Health Monitoring 18(5-6), 2019. The 88 measurements (60
baselines and damage positions D1-D28) are read one file at a time from the public
Nextcloud share linked on openguidedwaves.de, reduced from 10 MHz to 1 MHz, and saved as
one file per excitation frequency.

Output data/ogw1/ogw1_f<freq>kHz.npz:
    X      (88, 66, N)  signals of the 66 transducer pairs at 1 MHz
    name   (88,)        'baseline_k' or 'Dk'
    stamp, temp         timestamp and mean surface temperature of each measurement
    fs, onset, f_khz    sampling rate after decimation, excitation onset (samples), frequency

    python scripts/fetch_ogw1.py --freqs 40 60 80
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
from gwtsfm.config import DATA  # noqa: E402
from gwtsfm.ogw import decimate, read_h5  # noqa: E402

# public share: WebDAV with the share token as user name and no password
SHARE = "UfcaKdsx1imUrTm"
DAV = "https://nextcloud.faserinstitut.de/public.php/webdav/"
AUTH = {"Authorization": "Basic " + b64encode(f"{SHARE}:".encode()).decode()}


def list_folders() -> list[str]:
    """Measurement folders on the share, e.g. '20190301T120000_baseline_1' or '..._D7'."""
    req = urllib.request.Request(DAV, method="PROPFIND", headers={**AUTH, "Depth": "1"})
    s = urllib.request.urlopen(req, timeout=120).read().decode()
    names = [h.rstrip("/").rsplit("/", 1)[-1] for h in re.findall(r"<d:href>([^<]+)</d:href>", s)]
    return sorted(n for n in names if re.fullmatch(r"\d{8}T\d{6}_(baseline_\d+|D\d+)", n))


def download(path: str) -> bytes:
    """Fetch one file, retrying a few times on network errors."""
    for attempt in range(5):
        try:
            return urllib.request.urlopen(urllib.request.Request(DAV + path, headers=AUTH),
                                          timeout=300).read()
        except Exception as e:
            if attempt == 4:
                raise
            print(f"   retry {path}: {e!r}", flush=True)
            time.sleep(3 + 5 * attempt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DATA / "ogw1"))
    ap.add_argument("--freqs", type=int, nargs="+", default=[40, 60, 80])
    ap.add_argument("--fs-target", type=float, default=1e6)
    ap.add_argument("--workers", type=int, default=4, help="parallel downloads")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    folders = list_folders()
    print(f"{len(folders)} measurement folders", flush=True)

    for f in args.freqs:
        dst = out / f"ogw1_f{f}kHz.npz"
        if dst.exists():
            print(f"f={f} kHz: already downloaded", flush=True)
            continue

        def one(folder):
            x, temp, fs, onset = read_h5(io.BytesIO(download(f"{folder}/pc_f{f}kHz.h5")),
                                         with_onset=True)
            q = max(1, int(round((fs or 10e6) / args.fs_target)))   # decimation factor
            return folder, decimate(x, q), temp, fs, q, onset

        t0, res = time.time(), []
        with ThreadPoolExecutor(args.workers) as ex:
            for k, r in enumerate(ex.map(one, folders), 1):
                res.append(r)
                if k % 10 == 0 or k == len(folders):
                    print(f"  f={f} kHz {k}/{len(folders)}  {time.time() - t0:.0f}s", flush=True)
        n = min(r[1].shape[-1] for r in res)
        np.savez(dst, X=np.stack([r[1][..., :n] for r in res]),
                 name=np.array([r[0].split("_", 1)[1] for r in res]),
                 stamp=np.array([r[0].split("_", 1)[0] for r in res]),
                 temp=np.array([r[2] for r in res]), fs=res[0][3] / res[0][4],
                 onset=int(np.median([r[5] for r in res])) / res[0][4], f_khz=f)
        print(f"f={f} kHz -> {dst}  X{(len(res), 66, n)}", flush=True)


if __name__ == "__main__":
    main()
