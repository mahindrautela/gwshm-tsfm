"""E1: download the OGW CFRP temperature dataset at one frequency (extended experiments).

Moll et al., "Temperature affected guided wave propagation in a composite plate
complementing the Open Guided Waves platform", Scientific Data 6, 191 (2019); figshare
collection 10.6084/m9.figshare.c.4488089. The same plate and 12 transducers as OGW-1,
measured in a climate chamber while the temperature was stepped between 20 and 60 degC:
two temperature cycles without damage ("udam", 322 measurements) and one cycle with a
reversible defect at each of D04, D12, D16 and D24 (161 measurements each).

Each archive is 27-54 GB; only the pc_f<freq>kHz.h5 members are read, with HTTP range
requests (gwtsfm.remote), and reduced from 10 MHz to 1 MHz as for OGW-1.

Output data/ogw_temp/ogwT_f<freq>kHz_<plate>.npz, one file per plate (udam, D04, ...):
    X         (n, 66, N)  signals of the 66 transducer pairs at 1 MHz (rows in ogw.PAIRS order)
    temp      (n,)        mean plate-surface temperature (degC), as read by ogw.read_h5
    temp_ctc, humid (n,)  climate-chamber temperature and humidity
    stamp     (n,)        measurement timestamp (folder name), in time order
    fs, onset, f_khz

    python scripts/ext_fetch_ogw_temperature.py --freq 40
"""

from __future__ import annotations

import argparse
import io
import json
import re
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gwtsfm.config import DATA  # noqa: E402
from gwtsfm.ogw import decimate, read_h5  # noqa: E402
from gwtsfm.remote import RemoteZip  # noqa: E402

API = "https://api.figshare.com/v2"
COLLECTION = 4488089
TITLE = re.compile(r"OGW_CFRP_Temperature_(udam|dam_D\d\d)$")


def articles() -> dict[str, str]:
    """plate label ('udam', 'D04', ...) -> download URL of its zip archive."""
    get = lambda u: json.load(urllib.request.urlopen(u, timeout=60))  # noqa: E731
    out, page = {}, 1
    while batch := get(f"{API}/collections/{COLLECTION}/articles?page={page}&page_size=100"):
        for a in batch:
            m = TITLE.search(a["title"])
            if m:
                files = get(f"{API}/articles/{a['id']}")["files"]
                out[m.group(1).replace("dam_", "")] = files[0]["download_url"]
        page += 1
    return out


def read_member(data: bytes, q: int):
    """Signals at the reduced rate and the climate-chamber readings of one measurement."""
    x, temp, fs, onset = read_h5(io.BytesIO(data), with_onset=True)
    with h5py.File(io.BytesIO(data), "r") as f:
        ctc = float(np.ravel(f["CTC/Temperature"][()])[0])
        hum = float(np.ravel(f["CTC/Humidity"][()])[0])
    return decimate(x, q), temp, ctc, hum, fs, onset


def fetch_plate(url: str, freq: int, fs_target: float, workers: int):
    with RemoteZip(url) as z:
        names = sorted(n for n in z.namelist() if n.endswith(f"pc_f{freq}kHz.h5"))
    print(f"   {len(names)} measurements", flush=True)
    chunks = [names[i::workers] for i in range(workers)]
    t0, done, results = time.time(), [0], {}

    def work(chunk):  # one connection per worker
        with RemoteZip(url) as zz:
            for n in chunk:
                q = 10  # 10 MHz -> 1 MHz; checked against the file's own rate below
                x, temp, ctc, hum, fs, onset = read_member(zz.read(n), q)
                assert fs is None or abs(fs / q - fs_target) < 1, fs
                results[n] = (x, temp, ctc, hum, fs, onset)
                done[0] += 1
                if done[0] % 20 == 0:
                    print(f"   {done[0]}/{len(names)}  {time.time() - t0:.0f}s", flush=True)

    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(work, chunks))
    r = [results[n] for n in names]
    stamp = [re.search(r"/(\d{8}T\d{6})/", n).group(1) for n in names]
    return dict(X=np.stack([v[0] for v in r]), temp=np.array([v[1] for v in r]),
                temp_ctc=np.array([v[2] for v in r]), humid=np.array([v[3] for v in r]),
                stamp=np.array(stamp), fs=r[0][4] / 10, onset=int(np.median([v[5] for v in r])) / 10)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--freq", type=int, default=40)
    ap.add_argument("--out", default=str(DATA / "ogw_temp"))
    ap.add_argument("--workers", type=int, default=6, help="parallel connections per archive")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for plate, url in sorted(articles().items()):
        dst = out / f"ogwT_f{args.freq}kHz_{plate}.npz"
        if dst.exists():
            print(f"{plate}: already downloaded", flush=True)
            continue
        print(f"{plate}: {url}", flush=True)
        d = fetch_plate(url, args.freq, 1e6, args.workers)
        np.savez(dst, **d, f_khz=args.freq, plate=plate)
        print(f"{plate} -> {dst}  X{d['X'].shape}  {d['temp'].min():.1f}-{d['temp'].max():.1f} degC",
              flush=True)


if __name__ == "__main__":
    main()
