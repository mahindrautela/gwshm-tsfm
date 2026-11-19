"""Dataset audits quoted in the paper (Section 4.1 and Appendix A).

GW-TCW
  * how similar the 10 repeated acquisitions of a configuration are (median correlation)
  * how many configurations have sign-inverted damaged signals
OGW-1
  * residual energy of a path relative to its drift (baseline-to-baseline) energy, for
    paths within 10 mm of the damage and paths more than 120 mm away. This shows that only
    paths near the damage carry information about it, and less so at higher frequency.

Output: paper_results/audits.json

    python scripts/audit_datasets.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gwtsfm.config import DATA, ROOT  # noqa: E402
from gwtsfm.ogw import path_distance, phase_of  # noqa: E402


def audit_gwtcw(path):
    z = np.load(path)
    B, D = z["B"], z["D"]
    i = np.arange(len(B))
    cfg = (i // 120) * 12 + i % 12     # 13 blocks x 12 paths; 10 repeats per configuration
    repeat_r, flipped, abs_r = [], 0, []
    for c in np.unique(cfg):
        idx = np.flatnonzero(cfg == c)
        C = np.corrcoef(B[idx])                                   # repeats of one configuration
        repeat_r.append(np.median(C[np.triu_indices(len(idx), 1)]))
        r = np.array([np.corrcoef(B[k], D[k])[0, 1] for k in idx])   # healthy vs damaged
        flipped += int(np.all(r < 0))
        abs_r.append(np.median(np.abs(r)))
    return dict(n_configs=int(len(np.unique(cfg))), median_repeat_r=float(np.median(repeat_r)),
                min_repeat_r=float(np.min(repeat_r)), sign_inverted_configs=flipped,
                median_abs_r_baseline_damaged=float(np.median(abs_r)))


def audit_ogw1(path, near=10.0, far=120.0):
    z = np.load(path, allow_pickle=True)
    R, B = z["R"], z["B"]
    dist = path_distance(z["cross"], z["tpos"], z["dpos"])          # (28 cases, 36 paths), mm
    phase = np.array([phase_of(k) for k in range(1, 29)])
    out = {}
    for fi, f in enumerate(z["freqs"]):
        energy = (R[fi] ** 2).sum(-1)                                # (28, 36)
        drift = np.stack([(B[fi][20 * q:20 * q + 20] ** 2).sum(-1).mean(0) for q in range(3)])
        ratio = energy / drift[phase]
        out[int(f)] = dict(near_median=float(np.median(ratio[dist <= near])),
                           far_median=float(np.median(ratio[dist > far])),
                           n_near=int((dist <= near).sum()), n_far=int((dist > far).sum()))
    return out


if __name__ == "__main__":
    result = {"gwtcw": audit_gwtcw(DATA / "gwtcw" / "gwtcw.npz"),
              "ogw1_energy_contrast": audit_ogw1(DATA / "ogw1" / "residuals.npz")}
    (ROOT / "paper_results" / "audits.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
