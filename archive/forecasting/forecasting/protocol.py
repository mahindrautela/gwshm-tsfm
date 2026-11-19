"""Zero-shot damage-detection protocol: forecasting along the temperature axis.

Idea
----
For a fixed transducer pair and excitation frequency, the received signal is
a function x(t, T) of time-in-record t and plate temperature T.  At every
fixed sample t_n, the sequence of pristine reference signals ordered by
temperature, y_n[k] = x(t_n, T_k), is an ordinary time series on a uniform
0.5 deg C grid.  Temperature mainly changes wave speed, so y_n oscillates
in T with a frequency that grows with t_n -- a smooth, quasi-periodic series
of the kind that time series foundation models extrapolate well.

Given pristine references only up to a cut-off temperature T_c, every model
forecasts all y_n beyond T_c *zero-shot* (no training, no fine-tuning).  The
forecast evaluated at the measured temperature T of a new signal is the
model's prediction of the *pristine* signal there.  The prediction error is
the damage index (DI).  This is the realistic setting where the baseline
library does not cover the operating temperature, so classical optimal
baseline selection (OBS) must extrapolate.

Compared methods (same references, same cut-off)
------------------------------------------------
* TSFMs             Chronos-2, Chronos-Bolt, TimesFM-2.5, TimesFM-3, TiRex,
                    TiRex-2, Toto-2.0, Moirai-2.0, TTM-R2
* Naive forecasters last value, linear trend, AutoETS
* Physics forecaster stretch extrapolation (velocity / amplitude rates)
* Compensation      OBS, OBS + baseline signal stretch (BSS)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import baselines as bl
from .dsp import analysis_window, di_corr, di_residual, di_zscore, excitation_onset
from gwtsfm.ogw import PAIRS, FreqCache, split_reference


@dataclass
class Setup:
    f_khz: int
    fs: float
    onset: int                 # excitation onset inside the analysis window (samples, may be < 0)
    paths: np.ndarray          # (P,) pair indices
    grid: np.ndarray           # (G,) temperature grid of the references
    ref_grid: np.ndarray       # (P, G, N) pristine references interpolated onto the grid
    ref_T: np.ndarray          # (R,) measured temperatures of the raw references (sorted)
    ref_raw: np.ndarray        # (R, P, N) raw references (sorted by temperature)
    X: np.ndarray              # (M, P, N) test signals
    T: np.ndarray              # (M,) test temperatures
    state: np.ndarray          # (M,) 'udam' / 'D04' / ...
    meas: np.ndarray           # (M,) measurement index into the cache
    step: float


def prepare(cache: FreqCache, t_end_ms: float = 0.6, step: float = 0.5,
            paths: np.ndarray | None = None) -> Setup:
    ref, test = split_reference(cache)
    paths = np.arange(len(PAIRS)) if paths is None else np.asarray(paths)
    n = cache.X.shape[-1]
    if "onset" in cache.meta:  # from the recorded excitation signal
        onset = int(round(cache.meta["onset"]))
    else:
        onset = excitation_onset(cache.X[ref][:, paths])
    win = analysis_window(n, cache.fs, cache.f_khz, onset)
    stop = min(n, int(round(t_end_ms * 1e-3 * cache.fs)) + onset)
    win = slice(win.start, max(stop, win.start + 16))
    order = ref[np.argsort(cache.temp[ref])]
    Tr = cache.temp[order].astype(np.float64)
    Xr = np.asarray(cache.X[order][:, paths, win], dtype=np.float32)
    lo = np.ceil(Tr.min() / step) * step
    hi = np.floor(Tr.max() / step) * step
    grid = np.arange(lo, hi + 1e-9, step)
    # linear interpolation of the references onto the grid (per sample)
    j = np.clip(np.searchsorted(Tr, grid) - 1, 0, len(Tr) - 2)
    w = np.clip((grid - Tr[j]) / np.maximum(Tr[j + 1] - Tr[j], 1e-9), 0, 1)
    RG = (1 - w)[:, None, None] * Xr[j] + w[:, None, None] * Xr[j + 1]   # (G, P, N)
    Xt = np.asarray(cache.X[test][:, paths, win], dtype=np.float32)
    return Setup(f_khz=cache.f_khz, fs=cache.fs, onset=onset - win.start, paths=paths,
                 grid=grid, ref_grid=np.ascontiguousarray(RG.transpose(1, 0, 2)), ref_T=Tr,
                 ref_raw=Xr, X=Xt, T=cache.temp[test].astype(np.float64),
                 state=np.asarray(cache.state[test]), meas=test, step=step)


def _interp_forecast(V: np.ndarray, u: np.ndarray) -> np.ndarray:
    """V (P, N, H+1) knots at u = 0..H; u (M,) fractional positions -> (M, P, N)."""
    H = V.shape[-1] - 1
    u = np.clip(u, 0, H)
    lo = np.minimum(np.floor(u).astype(int), H - 1)
    w = (u - lo)[:, None, None]
    return ((1 - w) * np.moveaxis(V[..., lo], -1, 0) + w * np.moveaxis(V[..., lo + 1], -1, 0)
            ).astype(np.float32)


def _rows(s: Setup, sel: np.ndarray, Tc: float, method: str, d_res, d_cc, d_z=None) -> pd.DataFrame:
    M, P = d_res.shape
    df = pd.DataFrame({
        "f_khz": s.f_khz, "Tc": Tc, "method": method,
        "path": np.tile(s.paths, M),
        "meas": np.repeat(s.meas[sel], P),
        "T": np.repeat(s.T[sel], P),
        "state": np.repeat(s.state[sel], P),
        "di_res": d_res.ravel().astype(np.float32),
        "di_cc": d_cc.ravel().astype(np.float32),
    })
    df["di_z"] = np.nan if d_z is None else d_z.ravel().astype(np.float32)
    df["label"] = (df["state"] != "udam").astype(np.int8)
    return df


def context(s: Setup, Tc: float) -> tuple[np.ndarray, np.ndarray, int]:
    """Context series (P*N, k), test selector (T > Tc) and horizon H."""
    k = int(np.sum(s.grid <= Tc + 1e-9))
    Y = s.ref_grid[:, :k, :].transpose(0, 2, 1).reshape(-1, k)
    sel = np.flatnonzero(s.T > s.grid[k - 1] + 1e-9)
    H = int(np.ceil((s.T[sel].max() - s.grid[k - 1]) / s.step)) + 1
    return Y, sel, H


def evaluate_forecast(s: Setup, Tc: float, method: str, fc, keep_examples: bool = False):
    """Turn a forecast over the temperature grid into damage indices."""
    Y, sel, H = context(s, Tc)
    P, N = len(s.paths), s.ref_grid.shape[-1]
    last = Y[:, -1:]
    u = (s.T[sel] - s.grid[len(Y[0]) - 1]) / s.step
    knots = lambda q: np.concatenate([last, q], 1).reshape(P, N, H + 1)  # noqa: E731
    x = s.X[sel]
    q50 = _interp_forecast(knots(fc.q50), u)
    d_res, d_cc = di_residual(x, q50), di_corr(x, q50)
    d_z = None
    if getattr(fc, "probabilistic", True) and not np.allclose(fc.q10, fc.q90):
        q10 = _interp_forecast(knots(fc.q10), u)
        q90 = _interp_forecast(knots(fc.q90), u)
        d_z = di_zscore(x, q50, q10, q90)
    df = _rows(s, sel, Tc, method, d_res, d_cc, d_z)
    ex = None
    if keep_examples:
        ex = {"q10": fc.q10.reshape(P, N, H), "q50": fc.q50.reshape(P, N, H),
              "q90": fc.q90.reshape(P, N, H), "context": Y.reshape(P, N, -1)}
    return df, ex


def library(s: Setup, Tc: float, fc=None, sub: int = 5) -> tuple[np.ndarray, dict]:
    """Baseline library over temperature: references <= Tc, then forecasts (if any).

    Returns temperatures (K,) and arrays q10/q50/q90 of shape (K, P, N), linearly
    interpolated to ``sub`` sub-steps per grid step.  Without a forecast the
    library is the reference set only (classical OBS).
    """
    k = int(np.sum(s.grid <= Tc + 1e-9))
    R = s.ref_grid[:, :k, :].transpose(1, 0, 2)                     # (k, P, N)
    Tl = s.grid[:k]
    qs = {"q10": R, "q50": R, "q90": R}
    if fc is not None:
        P, N = R.shape[1:]
        H = fc.q50.shape[1]
        Tl = np.r_[Tl, s.grid[k - 1] + s.step * np.arange(1, H + 1)]
        qs = {q: np.concatenate([R, getattr(fc, q).reshape(P, N, H).transpose(2, 0, 1)])
              for q in ("q10", "q50", "q90")}
    # sub-step interpolation along temperature
    Tf = np.arange(Tl[0], Tl[-1] + 1e-9, s.step / sub)
    j = np.clip(np.searchsorted(Tl, Tf) - 1, 0, len(Tl) - 2)
    w = ((Tf - Tl[j]) / (Tl[j + 1] - Tl[j]))[:, None, None]
    return Tf, {q: ((1 - w) * v[j] + w * v[j + 1]).astype(np.float32) for q, v in qs.items()}


def select_from_library(x: np.ndarray, T: np.ndarray, Tl: np.ndarray, L: np.ndarray,
                        delta: float) -> np.ndarray:
    """Joint OBS over a library: one library temperature per measurement (all paths).

    x (M, P, N) test signals, T (M,) probe temperatures, L (K, P, N) library.
    Candidates are restricted to |Tl - T| <= delta; the criterion is the sum over
    paths of the normalised residual energy.  Returns the selected index (M,).
    """
    Ln = np.sum(L.astype(np.float64) ** 2, -1) + 1e-30                # (K, P)
    xx = np.sum(x.astype(np.float64) ** 2, -1)                         # (M, P)
    cross = np.einsum("mpn,kpn->mkp", x, L, optimize=True)             # (M, K, P)
    crit = ((xx[:, None, :] - 2 * cross + Ln[None]) / Ln[None]).sum(-1)  # (M, K)
    ok = np.abs(Tl[None, :] - T[:, None]) <= delta
    none = ~ok.any(1)
    ok[none, np.argmin(np.abs(Tl[None, :] - T[none, None]), 1)] = True  # fallback: nearest
    crit[~ok] = np.inf
    return np.argmin(crit, 1)


def evaluate_library(s: Setup, Tc: float, method: str, fc=None, delta: float = 3.0,
                     chunk: int = 128) -> pd.DataFrame:
    """Damage indices against the best library entry (references + forecasts)."""
    Tl, Q = library(s, Tc, fc)
    sel = np.flatnonzero(s.T > s.grid[int(np.sum(s.grid <= Tc + 1e-9)) - 1] + 1e-9)
    prob = fc is not None and not np.allclose(fc.q10, fc.q90)
    d_res, d_cc, d_z, Tsel = [], [], [], []
    for i in range(0, len(sel), chunk):
        idx = sel[i:i + chunk]
        x = s.X[idx]
        kk = select_from_library(x, s.T[idx], Tl, Q["q50"], delta)
        xh = Q["q50"][kk]
        d_res.append(di_residual(x, xh))
        d_cc.append(di_corr(x, xh))
        if prob:
            d_z.append(di_zscore(x, xh, Q["q10"][kk], Q["q90"][kk]))
        Tsel.append(Tl[kk])
    df = _rows(s, sel, Tc, method, np.concatenate(d_res), np.concatenate(d_cc),
               np.concatenate(d_z) if prob else None)
    df["T_lib"] = np.repeat(np.concatenate(Tsel), len(s.paths))
    return df


def warped_context(s: Setup, Tc: float):
    """Physics-warped context: references mapped onto the time axis of the Tc reference.

    Per path, a stretch rate kappa and amplitude rate beta are estimated from the
    references (as in stretch extrapolation).  Each reference x(t, T_k) is
    un-stretched and rescaled to T_c, u_k = e^{-beta dT} x(t (1+kappa)^{dT}, T_k),
    dT = T_k - T_c.  What remains in u along temperature is the part of the
    temperature effect that a uniform stretch does not explain; that is what the
    forecaster is asked to extrapolate.  Returns (Y (P*N, k), H, rates (P, 2)).
    """
    k = int(np.sum(s.grid <= Tc + 1e-9))
    Tk, T0 = s.grid[:k], s.grid[k - 1]
    P, N = len(s.paths), s.ref_grid.shape[-1]
    rates = np.zeros((P, 2))
    U = np.empty((P, k, N), np.float32)
    for p in range(P):
        kap, beta = bl.estimate_rates(s.ref_grid[p, :k], Tk, s.onset)
        rates[p] = kap, beta
        dT = Tk - T0
        U[p] = np.exp(-beta * dT)[:, None] * bl.stretch(s.ref_grid[p, :k], (1 + kap) ** (-dT) - 1, s.onset)
    _, sel, H = context(s, Tc)
    return U.transpose(0, 2, 1).reshape(P * N, k), H, rates


def dewarp(fc, s: Setup, Tc: float, rates: np.ndarray):
    """Map warped-space forecasts back to signals at the forecast temperatures."""
    k = int(np.sum(s.grid <= Tc + 1e-9))
    P, N = len(s.paths), s.ref_grid.shape[-1]
    H = fc.q50.shape[1]
    dT = s.step * np.arange(1, H + 1)

    class _F:
        pass

    out = _F()
    for q in ("q10", "q50", "q90"):
        V = getattr(fc, q).reshape(P, N, H)
        R = np.empty_like(V)
        for p in range(P):
            kap, beta = rates[p]
            R[p] = (np.exp(beta * dT)[:, None] *
                    bl.stretch(V[p].T, (1 + kap) ** dT - 1, s.onset)).T
        setattr(out, q, R.reshape(P * N, H))
    return out


def run_baselines(s: Setup, Tc: float, bss_span: float = 5.0,
                  s_grid: np.ndarray | None = None) -> pd.DataFrame:
    """OBS, OBS+BSS and stretch extrapolation with references T <= Tc."""
    if s_grid is None:
        s_grid = np.arange(-0.02, 0.15 + 1e-9, 0.001)
    k = int(np.sum(s.grid <= Tc + 1e-9))
    sel = np.flatnonzero(s.T > s.grid[k - 1] + 1e-9)
    x = s.X[sel]
    ok = s.ref_T <= Tc + 1e-9
    near = ok & (s.ref_T >= Tc - bss_span)
    xh_obs = np.empty_like(x)
    xh_bss = np.empty_like(x)
    P, N = len(s.paths), s.X.shape[-1]
    H = int(np.ceil((s.T[sel].max() - s.grid[k - 1]) / s.step)) + 1
    Tf = s.grid[k - 1] + s.step * np.arange(1, H + 1)
    ext = np.empty((P, N, H), np.float32)
    for p in range(P):
        _, xh_obs[:, p] = bl.obs(x[:, p], s.ref_raw[ok, p])
        _, xh_bss[:, p] = bl.obs_bss(x[:, p], s.ref_raw[near, p], s.onset, s_grid)
        kap, beta = bl.estimate_rates(s.ref_grid[p, :k], s.grid[:k], s.onset)
        ext[p] = bl.stretch_extrapolate(s.ref_grid[p, k - 1], s.grid[k - 1], Tf, kap, beta, s.onset).T
    ext = ext.reshape(P * N, H)

    class _F:  # forecast-like container for the physics extrapolation
        q10 = q50 = q90 = ext

    out = [
        # classical per-path OBS / OBS+BSS (no temperature restriction)
        _rows(s, sel, Tc, "OBS (per path)", di_residual(x, xh_obs), di_corr(x, xh_obs)),
        _rows(s, sel, Tc, "OBS+BSS", di_residual(x, xh_bss), di_corr(x, xh_bss)),
        # joint selection over the reference library within +-delta of the probe temperature
        evaluate_library(s, Tc, "OBS"),
        evaluate_library(s, Tc, "Stretch extrapolation", _F),
        evaluate_forecast(s, Tc, "Stretch extrapolation [probe T]", _F)[0],
    ]
    return pd.concat(out, ignore_index=True)
