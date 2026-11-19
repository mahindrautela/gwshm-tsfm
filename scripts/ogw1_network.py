"""Task 3, network level: localise from the 36-path damage-index map (OGW dataset 1).

Per path p and case (position k, frequency f) a damage index DI_p measures how far the
residual departs from that path's own baseline-to-baseline residuals (the 20 baselines
of the same phase):

  raw energy   DI = mean(r^2) / mean over baselines of mean(b^2)
  embedding    DI = ||e(r) - mean e(b)|| / mean ||e(b) - mean e(b)||   (features standardised)

Localisation from the DI map:
  RAPID        training-free tomography (Hay et al.; beta = 1.05) on max(DI - 1, 0); the
               zone is the one whose centre is nearest the image maximum.
  supervised   logistic regression on log DI (36 values), 4-fold leave-position-out
               (fold j holds out the j-th position of every zone).

python scripts/ogw1_network.py --data data/ogw1/loc.npz --embR data/ogw1_emb_R \
    --embB data/ogw1_emb_B --out results_ogw1_network
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gwtsfm import evalkit as ek  # noqa: E402

EMB = {"chronos2": "Chronos-2", "chronos_bolt": "Chronos-Bolt (base)", "tirex": "TiRex",
       "moment": "MOMENT-1 (base)", "mantis": "Mantis-8M", "ttm": "TTM-R2"}


def phase(k):  # k = 0..27
    return 0 if k < 11 else (1 if k < 20 else 2)


def di_raw(R, B):
    """R (28, P, L), B (60, P, L) -> (28, P)."""
    eB = np.stack([(B[20 * q:20 * q + 20] ** 2).mean(-1).mean(0) for q in range(3)])  # (3, P)
    return np.stack([(R[k] ** 2).mean(-1) / eB[phase(k)] for k in range(28)])


def di_emb(ER, EB):
    """ER (28, P, d), EB (60, P, d) -> (28, P)."""
    mu_all, sd_all = EB.reshape(-1, EB.shape[-1]).mean(0), EB.reshape(-1, EB.shape[-1]).std(0) + 1e-6
    ER, EB = (ER - mu_all) / sd_all, (EB - mu_all) / sd_all
    out = np.zeros(ER.shape[:2])
    for k in range(28):
        b = EB[20 * phase(k):20 * phase(k) + 20]           # (20, P, d)
        mu = b.mean(0)
        spread = np.linalg.norm(b - mu, axis=-1).mean(0)   # (P,)
        out[k] = np.linalg.norm(ER[k] - mu, axis=-1) / spread
    return out


def di_raw_baseline(B, rng):
    """Control: DI map of one healthy baseline per damage case, from that case's phase."""
    out = np.zeros((28, B.shape[1]))
    for k in range(28):
        q = phase(k)
        b = 20 * q + rng.integers(20)
        others = np.setdiff1d(np.arange(20 * q, 20 * q + 20), [b])
        out[k] = (B[b] ** 2).mean(-1) / (B[others] ** 2).mean(-1).mean(0)
    return out


def di_emb_baseline(EB, rng):
    mu_all, sd_all = EB.reshape(-1, EB.shape[-1]).mean(0), EB.reshape(-1, EB.shape[-1]).std(0) + 1e-6
    EB = (EB - mu_all) / sd_all
    out = np.zeros((28, EB.shape[1]))
    for k in range(28):
        q = phase(k)
        b = 20 * q + rng.integers(20)
        others = EB[np.setdiff1d(np.arange(20 * q, 20 * q + 20), [b])]
        mu = others.mean(0)
        out[k] = np.linalg.norm(EB[b] - mu, axis=-1) / np.linalg.norm(others - mu, axis=-1).mean(0)
    return out


def rapid(di, cross, tpos, grid, beta=1.05):
    img = np.zeros(len(grid))
    for d, (i, j) in zip(di, cross):
        a, b = tpos[i - 1], tpos[j - 1]
        r = (np.linalg.norm(grid - a, axis=1) + np.linalg.norm(grid - b, axis=1)) / np.linalg.norm(b - a)
        img += d * np.clip((beta - r) / (beta - 1), 0, None)
    return grid[img.argmax()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--embR", required=True)
    ap.add_argument("--embB", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    z = np.load(args.data)
    R, B, cross, dpos, tpos, freqs = z["R"], z["B"], z["cross"], z["dpos"], z["tpos"], z["freqs"]
    gx, gy = np.meshgrid(np.arange(0, 501, 5), np.arange(0, 501, 5))
    grid = np.c_[gx.ravel(), gy.ravel()].astype(float)
    centres = np.array([dpos[4 * g:4 * g + 4].mean(0) for g in range(7)])
    zone = np.arange(28) // 4

    rng = np.random.default_rng(0)
    DI = {"Residual energy (physics)": np.stack([di_raw(R[f], B[f]) for f in range(len(freqs))])}
    CTRL = {"Residual energy (physics)": np.stack([di_raw_baseline(B[f], rng) for f in range(len(freqs))])}
    for k, n in EMB.items():
        fr, fb = Path(args.embR) / f"{k}.npy", Path(args.embB) / f"{k}.npy"
        if fr.exists() and fb.exists():
            ER, EB = np.load(fr).astype(np.float32), np.load(fb).astype(np.float32)
            DI[n] = np.stack([di_emb(ER[f], EB[f]) for f in range(len(freqs))])
            CTRL[n] = np.stack([di_emb_baseline(EB[f], rng) for f in range(len(freqs))])
    rows = []
    for name, D in DI.items():
        for fi, f in enumerate(freqs):
            # training-free RAPID
            pred_xy = np.stack([rapid(np.maximum(D[fi, k] - 1, 0), cross, tpos, grid) for k in range(28)])
            zp = np.argmin(np.linalg.norm(pred_xy[:, None] - centres[None], axis=-1), 1)
            err = np.linalg.norm(pred_xy - dpos, axis=1)
            rows.append(dict(method=name, freq=int(f), approach="RAPID (training-free)",
                             acc=accuracy_score(zone, zp), median_error_mm=float(np.median(err))))
            # supervised on the DI map, leave-position-out
            F = np.log(np.maximum(D[fi], 1e-6))
            pred = np.zeros(28, int)
            for j in range(4):
                te = np.flatnonzero(np.arange(28) % 4 == j)
                tr = np.setdiff1d(np.arange(28), te)
                pred[te] = ek.fit_proba(lambda: ek.lr(C=0.1), F, zone, tr, te).argmax(1)
            rows.append(dict(method=name, freq=int(f), approach="logistic regression on DI map",
                             acc=accuracy_score(zone, pred), median_error_mm=np.nan))
        # all three frequencies together (supervised): concatenate the maps; plus the control
        for approach, M in (("logistic regression on DI map", D),
                            ("control: healthy maps of the same phase", CTRL[name])):
            F = np.concatenate([np.log(np.maximum(M[fi], 1e-6)) for fi in range(len(freqs))], 1)
            pred = np.zeros(28, int)
            for j in range(4):
                te = np.flatnonzero(np.arange(28) % 4 == j)
                tr = np.setdiff1d(np.arange(28), te)
                pred[te] = ek.fit_proba(lambda: ek.lr(C=0.1), F, zone, tr, te).argmax(1)
            rows.append(dict(method=name, freq=0, approach=approach,
                             acc=accuracy_score(zone, pred), median_error_mm=np.nan))
        # path-level separability of the DI: AUC of near (<= 10 mm) vs far (> 120 mm) paths
    T = pd.DataFrame(rows)
    T.to_csv(out / "network_localisation.csv", index=False)
    np.savez(out / "di_maps.npz", **{k.replace(" ", "_"): v for k, v in DI.items()})
    print(T.pivot_table(index="method", columns=["approach", "freq"], values="acc").round(3).to_string())
    print(T[T.approach.str.startswith("RAPID")].pivot_table(index="method", columns="freq",
                                                            values="median_error_mm").round(0).to_string())


if __name__ == "__main__":
    main()
