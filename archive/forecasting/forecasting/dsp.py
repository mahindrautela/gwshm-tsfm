"""Signal-processing helpers: analysis window, time stretch, damage indices."""

from __future__ import annotations

import numpy as np


def excitation_onset(X: np.ndarray, frac: float = 0.2) -> int:
    """Sample index where the (crosstalk) excitation starts.

    ``X`` is any stack of signals (..., N).  Uses the mean absolute envelope
    and returns the first sample exceeding ``frac`` of its early maximum.
    """
    env = np.abs(X.reshape(-1, X.shape[-1])).mean(0)
    early = env[: max(8, len(env) // 4)]
    thr = frac * early.max()
    return int(np.argmax(early > thr))


def analysis_window(n: int, fs: float, f_khz: float, onset: int, cycles: int = 5,
                    guard: float = 1.5) -> slice:
    """Window that skips the excitation crosstalk and keeps the rest."""
    start = onset + int(np.ceil(guard * cycles / (f_khz * 1e3) * fs))
    return slice(min(start, n - 8), n)


def stretch(x: np.ndarray, s: float | np.ndarray, onset: int = 0) -> np.ndarray:
    """Time-stretch signals about the excitation onset: y(t) = x(t / (1 + s)).

    ``x`` (..., N); ``s`` scalar or broadcastable to x.shape[:-1].
    Linear interpolation; samples mapped outside the record are set to 0.
    """
    x = np.asarray(x)
    n = x.shape[-1]
    t = np.arange(n, dtype=np.float64) - onset
    s = np.asarray(s, dtype=np.float64)
    src = t / (1.0 + s[..., None]) + onset  # (..., N)
    i0 = np.floor(src).astype(np.int64)
    w = src - i0
    valid = (i0 >= 0) & (i0 < n - 1)
    i0c = np.clip(i0, 0, n - 2)
    xb = np.broadcast_to(x, np.broadcast_shapes(x.shape, src.shape))
    y = (1 - w) * np.take_along_axis(xb, i0c, -1) + w * np.take_along_axis(xb, i0c + 1, -1)
    return np.where(valid, y, 0.0).astype(np.float32)


# --------------------------------------------------------------------------
# Damage indices.  All take the measured signal x and a prediction/baseline
# xh over the analysis window (last axis) and return one number per signal.
# --------------------------------------------------------------------------

def di_residual(x: np.ndarray, xh: np.ndarray) -> np.ndarray:
    """Residual energy ratio ||x - xh||^2 / ||xh||^2 (baseline subtraction)."""
    num = np.sum((x - xh) ** 2, -1)
    den = np.sum(xh ** 2, -1) + 1e-30
    return num / den


def di_corr(x: np.ndarray, xh: np.ndarray) -> np.ndarray:
    """1 - Pearson correlation (insensitive to amplitude drift)."""
    xc = x - x.mean(-1, keepdims=True)
    hc = xh - xh.mean(-1, keepdims=True)
    r = np.sum(xc * hc, -1) / (np.sqrt(np.sum(xc ** 2, -1) * np.sum(hc ** 2, -1)) + 1e-30)
    return 1.0 - r


def di_zscore(x: np.ndarray, med: np.ndarray, lo: np.ndarray, hi: np.ndarray,
              floor_frac: float = 0.02) -> np.ndarray:
    """Uncertainty-normalised residual: mean(((x - q50) / sigma)^2).

    sigma is the Gaussian-equivalent scale from the 10-90 % predictive interval
    (width / 2.563), floored at ``floor_frac`` of the signal RMS so that
    over-confident samples do not dominate.
    """
    sigma = (hi - lo) / 2.563
    rms = np.sqrt(np.mean(med ** 2, -1, keepdims=True)) + 1e-30
    sigma = np.maximum(sigma, floor_frac * rms)
    return np.mean(((x - med) / sigma) ** 2, -1)
