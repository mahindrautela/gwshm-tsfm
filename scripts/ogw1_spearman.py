"""Task 3 mechanism check: does a path's damage index track how close the path passes to the damage?

For every damage case and frequency, Spearman correlation between the 36 per-path damage indices
(results_ogw1_network/di_maps.npz) and the path's proximity to the damage position. Proximity is the
negative perpendicular distance from the damage centre to the transmitter-receiver segment
(--kind ellipse uses the RAPID ellipse ratio instead). Reported as the mean over the 28 cases.

    python scripts/ogw1_spearman.py --loc data/ogw1/loc.npz --di results_ogw1_network/di_maps.npz \
        --out results_ogw1_network/spearman.csv
"""
import argparse

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def proximity(cross, tpos, dpos, kind="segment"):
    """(n_cases, n_paths) proximity of each path to each damage position (higher = closer)."""
    T = tpos[cross[:, 0] - 1]  # (P, 2); transducer ids in `cross` are 1-based (T1..T12)
    R = tpos[cross[:, 1] - 1]
    out = np.empty((len(dpos), len(cross)))
    for k, d in enumerate(dpos):
        if kind == "ellipse":
            ratio = (np.linalg.norm(T - d, axis=1) + np.linalg.norm(R - d, axis=1)) / np.linalg.norm(T - R, axis=1)
            out[k] = -ratio
        else:  # perpendicular distance to the segment
            v = R - T
            t = np.clip(((d - T) * v).sum(1) / (v * v).sum(1), 0, 1)
            out[k] = -np.linalg.norm(T + t[:, None] * v - d, axis=1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loc", default="data/ogw1/loc.npz")
    ap.add_argument("--di", default="results_ogw1_network/di_maps.npz")
    ap.add_argument("--out", default="results_ogw1_network/spearman.csv")
    ap.add_argument("--kind", default="segment", choices=["ellipse", "segment"])
    a = ap.parse_args()

    loc = np.load(a.loc, allow_pickle=True)
    cross, tpos, dpos, freqs = loc["cross"], loc["tpos"], loc["dpos"], loc["freqs"]
    prox = proximity(cross, tpos, dpos, a.kind)
    di = np.load(a.di)
    rows = []
    for name in di.files:
        D = di[name]  # (F, 28, 36)
        for fi, f in enumerate(freqs):
            rho = [spearmanr(D[fi, k], prox[k]).statistic for k in range(D.shape[1])]
            rows.append(dict(method=name.replace("_", " "), freq_khz=int(f), spearman_mean=np.mean(rho),
                             spearman_sd=np.std(rho)))
    df = pd.DataFrame(rows)
    df.to_csv(a.out, index=False)
    print(df.pivot(index="method", columns="freq_khz", values="spearman_mean").round(2).to_string())


if __name__ == "__main__":
    main()
