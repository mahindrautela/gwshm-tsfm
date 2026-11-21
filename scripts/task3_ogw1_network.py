"""Task 3, network level: locate the damage from the map of all 36 paths (OGW-1).

For every damage case (position k, frequency f) and path p, a damage index DI_p measures
how far the path's residual is from that path's drift residuals (the 20 baselines of the
same phase, each minus their mean):

  residual energy (physics)  DI_p = mean(r^2) / mean over baselines of mean(b^2)
  embedding distance         DI_p = |e(r) - mean e(b)| / mean |e(b) - mean e(b)|
                             with e the frozen embedding, standardised on healthy data (Eq. 2)

The 36 values form a map of the plate. We read the zone from it in two ways:
  logistic regression   on log DI, 4-fold leave-position-out (fold j holds out the j-th
                        position of every zone), per frequency and with all three combined
  RAPID                 training-free tomography (Hay et al. 2006, beta = 1.05); the zone
                        whose centre is nearest the image maximum
Control: the damage maps are replaced by maps of healthy baselines from the same phase
(10 random draws). If the classifier could recognise the phase instead of the damage,
it would still score well; it should fall to chance (1/7).
Mechanism check: Spearman correlation between a path's DI and how close it passes to the
damage, per case, averaged over the 28 cases.

Outputs in results/task3_ogw1/:
    network_localisation.csv  accuracy (and RAPID median error) per method and frequency
    di_maps.npz               the DI maps, one array (frequency, case, path) per method
    spearman.csv              mean and s.d. of the Spearman correlation per method and frequency

    python scripts/task3_ogw1_network.py     # about a minute
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gwtsfm import heads  # noqa: E402
from gwtsfm.config import DATA, EMBEDDINGS, PAPER_MODELS, RESULTS  # noqa: E402
from gwtsfm.ogw import path_distance, phase_of  # noqa: E402

PHYSICS = "Residual energy (physics)"
N_CONTROL = 10        # random draws of healthy maps for the control


def phase(k):
    """Phase of damage case k = 0..27 (position k + 1)."""
    return phase_of(k + 1)


def baselines_of(q):
    """Indices of the 20 drift residuals of phase q."""
    return np.arange(20 * q, 20 * q + 20)


# ---------------------------------------------------------------------- damage indices
def di_energy(R, B):
    """Residual-energy DI. R (28, P, L) damage residuals, B (60, P, L) drift -> (28, P)."""
    drift = np.stack([(B[baselines_of(q)] ** 2).mean(-1).mean(0) for q in range(3)])  # (3, P)
    return np.stack([(R[k] ** 2).mean(-1) / drift[phase(k)] for k in range(28)])


def standardise(EB, *arrays):
    """Scale embeddings by the mean and std of all drift embeddings (healthy data only)."""
    flat = EB.reshape(-1, EB.shape[-1])
    mu, sd = flat.mean(0), flat.std(0) + 1e-6
    return [(A - mu) / sd for A in arrays]


def di_embedding(ER, EB):
    """Embedding DI. ER (28, P, d) damage embeddings, EB (60, P, d) drift -> (28, P)."""
    ER, EB = standardise(EB, ER, EB)
    out = np.zeros(ER.shape[:2])
    for k in range(28):
        b = EB[baselines_of(phase(k))]                     # (20, P, d) same phase
        mu = b.mean(0)
        spread = np.linalg.norm(b - mu, axis=-1).mean(0)   # typical drift distance per path
        out[k] = np.linalg.norm(ER[k] - mu, axis=-1) / spread
    return out


def control_energy(B, rng):
    """Control map: for each case, the DI of one random healthy baseline of its phase,
    measured against the other 19 baselines of that phase."""
    out = np.zeros((28, B.shape[1]))
    for k in range(28):
        q = phase(k)
        b = 20 * q + rng.integers(20)
        others = np.setdiff1d(baselines_of(q), [b])
        out[k] = (B[b] ** 2).mean(-1) / (B[others] ** 2).mean(-1).mean(0)
    return out


def control_embedding(EB, rng):
    """As control_energy, with the embedding distance."""
    (EB,) = standardise(EB, EB)
    out = np.zeros((28, EB.shape[1]))
    for k in range(28):
        q = phase(k)
        b = 20 * q + rng.integers(20)
        others = EB[np.setdiff1d(baselines_of(q), [b])]
        mu = others.mean(0)
        out[k] = np.linalg.norm(EB[b] - mu, axis=-1) / np.linalg.norm(others - mu, axis=-1).mean(0)
    return out


# ------------------------------------------------------------------------ localisation
def rapid(di, cross, tpos, grid, beta=1.05):
    """RAPID tomography: each path adds its DI inside an ellipse with foci at its two
    transducers; returns the grid point where the image is largest."""
    img = np.zeros(len(grid))
    for d, (i, j) in zip(di, cross):
        a, b = tpos[i - 1], tpos[j - 1]
        r = (np.linalg.norm(grid - a, axis=1) + np.linalg.norm(grid - b, axis=1)) / np.linalg.norm(b - a)
        img += d * np.clip((beta - r) / (beta - 1), 0, None)
    return grid[img.argmax()]


def leave_position_out_accuracy(F, zone):
    """Zone accuracy of logistic regression on DI features F (28, n), 4 folds of 7 cases."""
    pred = np.zeros(28, int)
    for j in range(4):
        te = np.flatnonzero(np.arange(28) % 4 == j)
        tr = np.setdiff1d(np.arange(28), te)
        pred[te] = heads.predict_proba(lambda: heads.logistic_regression(C=0.1), F, zone, tr, te).argmax(1)
    return accuracy_score(zone, pred)


def log_di(D):
    return np.log(np.maximum(D, 1e-6))


# ------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DATA / "ogw1" / "residuals.npz"))
    ap.add_argument("--out", default=str(RESULTS / "task3_ogw1"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    z = np.load(args.data)
    R, B, cross, dpos, tpos, freqs = z["R"], z["B"], z["cross"], z["dpos"], z["tpos"], z["freqs"]
    nF = len(freqs)
    zone = np.arange(28) // 4
    gx, gy = np.meshgrid(np.arange(0, 501, 5), np.arange(0, 501, 5))     # 5 mm RAPID grid
    grid = np.c_[gx.ravel(), gy.ravel()].astype(float)
    centres = np.array([dpos[4 * g:4 * g + 4].mean(0) for g in range(7)])  # zone centres

    # DI maps (frequency, case, path) and control maps for every method
    DI = {PHYSICS: np.stack([di_energy(R[f], B[f]) for f in range(nF)])}
    CONTROL = {PHYSICS: [np.stack([control_energy(B[f], np.random.default_rng(s)) for f in range(nF)])
                         for s in range(N_CONTROL)]}
    for key, name in PAPER_MODELS.items():
        fr = EMBEDDINGS / "ogw1_residuals" / f"{key}.npy"
        fb = EMBEDDINGS / "ogw1_baselines" / f"{key}.npy"
        if fr.exists() and fb.exists():
            ER, EB = np.load(fr).astype(np.float32), np.load(fb).astype(np.float32)
            DI[name] = np.stack([di_embedding(ER[f], EB[f]) for f in range(nF)])
            CONTROL[name] = [np.stack([control_embedding(EB[f], np.random.default_rng(s)) for f in range(nF)])
                             for s in range(N_CONTROL)]

    rows = []
    for name, D in DI.items():
        for fi, f in enumerate(freqs):
            xy = np.stack([rapid(np.maximum(D[fi, k] - 1, 0), cross, tpos, grid) for k in range(28)])
            zone_pred = np.argmin(np.linalg.norm(xy[:, None] - centres[None], axis=-1), 1)
            rows.append(dict(method=name, freq=int(f), approach="RAPID (training-free)",
                             acc=accuracy_score(zone, zone_pred),
                             median_error_mm=float(np.median(np.linalg.norm(xy - dpos, axis=1)))))
            rows.append(dict(method=name, freq=int(f), approach="logistic regression on DI map",
                             acc=leave_position_out_accuracy(log_di(D[fi]), zone), median_error_mm=np.nan))

        def all_frequencies(M):  # the three maps side by side: (28, 3 * 36)
            return np.concatenate([log_di(M[fi]) for fi in range(nF)], 1)

        rows.append(dict(method=name, freq=0, approach="logistic regression on DI map",
                         acc=leave_position_out_accuracy(all_frequencies(D), zone), median_error_mm=np.nan))
        control = [leave_position_out_accuracy(all_frequencies(M), zone) for M in CONTROL[name]]
        rows.append(dict(method=name, freq=0, approach="control: healthy maps of the same phase",
                         acc=float(np.mean(control)), acc_max=float(np.max(control)), median_error_mm=np.nan))
    T = pd.DataFrame(rows)
    T.to_csv(out / "network_localisation.csv", index=False)
    np.savez(out / "di_maps.npz", **{k.replace(" ", "_"): v for k, v in DI.items()})

    # mechanism: does a path's DI rise the closer the path passes to the damage?
    closeness = -path_distance(cross, tpos, dpos)                       # (28, 36)
    sp = []
    for name, D in DI.items():
        for fi, f in enumerate(freqs):
            rho = [spearmanr(D[fi, k], closeness[k]).statistic for k in range(28)]
            sp.append(dict(method=name, freq_khz=int(f), spearman_mean=np.mean(rho), spearman_sd=np.std(rho)))
    pd.DataFrame(sp).to_csv(out / "spearman.csv", index=False)

    print(T.pivot_table(index="method", columns=["approach", "freq"], values="acc").round(3).to_string())


if __name__ == "__main__":
    main()
