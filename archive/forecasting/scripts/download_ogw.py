"""Download selected frequencies of the OGW CFRP temperature dataset from figshare.

The dataset is a figshare collection (doi:10.6084/m9.figshare.c.4488089) with one
article per folder (udam, dam_D04, dam_D12, dam_D16, dam_D24).  Archives are read
with HTTP range requests so that only the requested ``pc_f<freq>kHz.h5`` members
are transferred, not the whole multi-GB archive.

python scripts/download_ogw.py --root data/raw --freqs 40 100 160
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import urllib.request
from pathlib import Path, PurePosixPath

API = "https://api.figshare.com/v2"
COLLECTION = 4488089
WANTED = re.compile(r"OGW_CFRP_Temperature_(udam|dam_D\d\d)")


def get_json(url: str):
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.load(r)


def list_articles() -> list[dict]:
    arts, page = [], 1
    while True:
        batch = get_json(f"{API}/collections/{COLLECTION}/articles?page={page}&page_size=100")
        if not batch:
            break
        arts += batch
        page += 1
    return arts


def member_target(root: Path, member: str, folder: str) -> Path | None:
    """Map an archive member to <root>/<folder>/<timestamp>/<file>."""
    parts = PurePosixPath(member).parts
    ts = next((p for p in parts if re.fullmatch(r"\d{8}T\d{6}", p)), None)
    if ts is None:
        return None
    return root / folder / ts / parts[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--freqs", type=int, nargs="+", required=True)
    ap.add_argument("--only", nargs="*", default=None, help="subset, e.g. udam dam_D12")
    ap.add_argument("--list", action="store_true", help="only list articles and files")
    args = ap.parse_args()
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    pat = re.compile("|".join(rf"pc_f{f}kHz\.h5$" for f in args.freqs))

    for art in list_articles():
        m = WANTED.search(art["title"])
        if not m or (args.only and m.group(1) not in args.only):
            continue
        folder = f"OGW_CFRP_Temperature_{m.group(1)}"
        info = get_json(f"{API}/articles/{art['id']}")
        print(f"{art['title']} (id {art['id']}):")
        for fi in info["files"]:
            print(f"   {fi['name']}  {fi['size'] / 1e9:.2f} GB  {fi['download_url']}")
            if args.list:
                continue
            if fi["name"].lower().endswith(".zip"):
                from remotezip import RemoteZip
                with RemoteZip(fi["download_url"]) as z:
                    members = [n for n in z.namelist() if pat.search(n)]
                    print(f"   -> extracting {len(members)} members")
                    for k, n in enumerate(members):
                        tgt = member_target(root, n, folder)
                        if tgt is None or tgt.exists():
                            continue
                        tgt.parent.mkdir(parents=True, exist_ok=True)
                        with z.open(n) as src, open(tgt, "wb") as dst:
                            shutil.copyfileobj(src, dst)
                        if k % 50 == 0:
                            print(f"      {k}/{len(members)} {tgt.relative_to(root)}", flush=True)
            elif pat.search(fi["name"]):
                tgt = root / folder / fi["name"]
                urllib.request.urlretrieve(fi["download_url"], tgt)
    print("done", file=sys.stderr)


if __name__ == "__main__":
    main()
