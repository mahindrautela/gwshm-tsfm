"""Classical temperature-compensation baselines.

* OBS       optimal baseline selection (Lu & Michaels 2005; Croxford et al. 2007)
* OBS+BSS   OBS combined with baseline signal stretch and amplitude scaling
            (Harley & Moura 2012; Clarke et al. 2010)
* stretch-extrapolation
            a physics-informed forecaster along the temperature axis: the
            stretch rate (velocity change per deg C) and amplitude rate are
            estimated from the reference ramp and used to extrapolate the
            hottest reference to unseen temperatures.

All functions work on signals already cut to the analysis window, shape
(..., N).  "Reference" signals are pristine signals at known temperatures.
"""

from __future__ import annotations

import numpy as np

from .dsp import stretch


def obs(x: np.ndarray, refs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Optimal baseline selection by minimum residual energy.

    x (M, N) test signals, refs (R, N).  Returns (index (M,), baseline (M, N)).
    """
    # ||x - r||^2 = ||x||^2 - 2 x.r + ||r||^2 ; the first term is constant per row
    score = -2.0 * x @ refs.T + np.sum(refs ** 2, -1)[None, :]
    idx = np.argmin(score, 1)
    return idx, refs[idx]


def obs_bss(x: np.ndarray, refs: np.ndarray, onset: int, s_grid: np.ndarray,
            chunk: int = 2048) -> tuple[np.ndarray, np.ndarray]:
    """OBS over all (reference, stretch) candidates with optimal amplitude.

    For each test signal the candidate c maximising (x.c)^2 / ||c||^2 is
    chosen; the returned baseline is a* c with a* = x.c / ||c||^2.
    Returns (best stretch (M,), baseline (M, N)).
    """
    R, N = refs.shape
    cands = stretch(np.repeat(refs[:, None, :], len(s_grid), 1), s_grid[None, :], onset)
    cands = cands.reshape(-1, N).astype(np.float32)
    cn = np.sum(cands.astype(np.float64) ** 2, -1) + 1e-30
    best = np.full(len(x), -np.inf)
    arg = np.zeros(len(x), dtype=np.int64)
    for k in range(0, len(cands), chunk):
        dots = (x.astype(np.float32) @ cands[k:k + chunk].T).astype(np.float64)
        sc = dots ** 2 / cn[None, k:k + chunk]
        j = np.argmax(sc, 1)
        v = sc[np.arange(len(x)), j]
        upd = v > best
        best[upd] = v[upd]
        arg[upd] = j[upd] + k
    c = cands[arg].astype(np.float64)
    a = np.sum(x * c, -1) / cn[arg]
    s_best = s_grid[arg % len(s_grid)]
    return s_best, (a[:, None] * c).astype(np.float32)


def estimate_rates(ref_grid: np.ndarray, temps: np.ndarray, onset: int,
                   span: float = 10.0,
                   s_grid: np.ndarray | None = None) -> tuple[float, float]:
    """Stretch rate kappa (1/degC) and amplitude rate beta (1/degC).

    ref_grid (G, N) pristine references on an increasing temperature grid.
    kappa: stretch that best maps the reference ``span`` deg C below the
    hottest one onto the hottest one, divided by the temperature difference.
    beta: slope of log-RMS vs temperature (least squares over all references).
    """
    if s_grid is None:
        s_grid = np.linspace(-0.05, 0.10, 301)
    hi = len(temps) - 1
    lo = int(np.argmin(np.abs(temps - (temps[hi] - span))))
    d = temps[hi] - temps[lo]
    if d <= 0:
        return 0.0, 0.0
    cands = stretch(np.repeat(ref_grid[lo][None], len(s_grid), 0), s_grid, onset)
    tgt = ref_grid[hi]
    sc = (cands @ tgt) ** 2 / (np.sum(cands ** 2, -1) + 1e-30)
    kappa = float(s_grid[np.argmax(sc)] / d)
    rms = np.sqrt(np.mean(ref_grid ** 2, -1)) + 1e-30
    beta = float(np.polyfit(temps, np.log(rms), 1)[0])
    return kappa, beta


def stretch_extrapolate(x_ref: np.ndarray, T_ref: float, T: np.ndarray, kappa: float,
                        beta: float, onset: int) -> np.ndarray:
    """Predict signals at temperatures T from the reference at T_ref."""
    dT = np.asarray(T, dtype=np.float64) - T_ref
    s = (1.0 + kappa) ** dT - 1.0
    y = stretch(np.broadcast_to(x_ref, (len(dT), x_ref.shape[-1])), s, onset)
    return (np.exp(beta * dT)[:, None] * y).astype(np.float32)
