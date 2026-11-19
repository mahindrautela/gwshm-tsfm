"""Dataset audits quoted in the paper.

GW-TCW: correlation between the 10 repeated acquisitions of a configuration, and the
configurations whose damaged signals are sign-inverted relative to the baseline.
OGW-1: residual energy of a path relative to its baseline-to-baseline drift, by the
distance between the path and the damage.

    python scripts/audit_datasets.py --gwtcw data/gwtcw/gwtcw.npz --ogw data/ogw1/loc.npz
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ogw1_spearman import proximity  # noqa: E402


def gwtcw(path):
    d = np.load(path)
    B, D = d["B"], d["D"]
    i = np.arange(len(B))
    cfg = (i // 120) * 12 + i % 12          # 13 blocks x 12 paths; 10 repeats per configuration
    rep_r, flips, abs_r = [], 0, []
    for c in np.unique(cfg):
        idx = np.flatnonzero(cfg == c)
        C = np.corrcoef(B[idx])
        rep_r.append(np.median(C[np.triu_indices(len(idx), 1)]))
        r = np.array([np.corrcoef(B[k], D[k])[0, 1] for k in idx])
        flips += int(np.all(r < 0))
        abs_r.append(np.median(np.abs(r)))
    return dict(n_configs=int(len(np.unique(cfg))), median_repeat_r=float(np.median(rep_r)),
                min_repeat_r=float(np.min(rep_r)), sign_inverted_configs=flips,
                median_abs_r_baseline_damaged=float(np.median(abs_r)))


def ogw(path, near=10.0, far=120.0):
    z = np.load(path, allow_pickle=True)
    R, B = z["R"], z["B"]
    dist = -proximity(z["cross"], z["tpos"], z["dpos"])     # mm from damage centre to each path
    ph = np.array([0] * 11 + [1] * 9 + [2] * 8)
    out = {}
    for fi, f in enumerate(z["freqs"]):
        eR, eB = (R[fi] ** 2).sum(-1), (B[fi] ** 2).sum(-1)
        drift = np.stack([eB[20 * q:20 * q + 20].mean(0) for q in range(3)])
        ratio = eR / drift[ph]
        out[int(f)] = dict(near_median=float(np.median(ratio[dist <= near])),
                           far_median=float(np.median(ratio[dist > far])),
                           n_near=int((dist <= near).sum()), n_far=int((dist > far).sum()))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gwtcw", default="data/gwtcw/gwtcw.npz")
    ap.add_argument("--ogw", default="data/ogw1/loc.npz")
    ap.add_argument("--out", default="paper_results/audits.json")
    a = ap.parse_args()
    res = {"gwtcw": gwtcw(a.gwtcw), "ogw1_energy_contrast": ogw(a.ogw)}
    Path(a.out).write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
