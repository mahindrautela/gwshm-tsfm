"""Synthetic OGW-like guided-wave data for pipeline testing.

Writes files with exactly the on-disk layout of the OGW temperature dataset
(see ``ogw.py``) so that the full pipeline -- loader, caching, baselines,
TSFM protocol, metrics -- can be exercised before the real data are
available.  The physics is deliberately simple but captures the effects that
matter for temperature-compensated damage detection:

* two Lamb modes (fast S0, slower A0) with temperature-dependent velocity
  (time stretch) and amplitude,
* edge reflections via image sources on a 500 x 500 mm plate,
* a scatterer (the reversible "damage") adding small extra wave packets,
* excitation crosstalk, measurement noise and temperature-probe noise.

Nothing in this module is used to produce reported results on real data.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import h5py
import numpy as np

from gwtsfm.ogw import FOLDERS, PAIRS

PLATE = 0.5  # m
# 12 transducers: T1-T6 on the left, T7-T12 on the right (synthetic layout)
_ys = np.linspace(0.07, 0.43, 6)
TPOS = np.array([[0.06, y] for y in _ys] + [[0.44, y] for y in _ys[::-1]])
# damage positions roughly along the T4-T9 line
DPOS = {"D04": (0.16, 0.34), "D12": (0.22, 0.31), "D16": (0.28, 0.28), "D24": (0.34, 0.25)}


def toneburst(t: np.ndarray, f: float, cycles: int = 5) -> np.ndarray:
    dur = cycles / f
    w = np.where((t >= 0) & (t <= dur), 0.5 * (1 - np.cos(2 * np.pi * t / dur)), 0.0)
    return w * np.sin(2 * np.pi * f * t)


def _images(p: np.ndarray) -> list[np.ndarray]:
    """Source + first-order mirror images across the four plate edges."""
    x, y = p
    return [p, np.array([-x, y]), np.array([2 * PLATE - x, y]),
            np.array([x, -y]), np.array([x, 2 * PLATE - y])]


def simulate(i: int, j: int, T: float, f: float, t: np.ndarray, damage: str | None,
             rng: np.random.Generator) -> np.ndarray:
    dT = T - 20.0
    c_s0 = 5600.0 * (1 - 0.0012 * dT)
    c_a0 = min(1000.0 + 6.0 * f / 1e3, 2000.0) * (1 - 0.0020 * dT)
    amp_t = 1 - 0.004 * dT
    a_ratio = np.clip(150e3 / f, 0.3, 3.0)  # A0 dominates at low frequency
    src, rcv = TPOS[i - 1], TPOS[j - 1]
    y = np.zeros_like(t)
    for k, img in enumerate(_images(src)):
        L = np.linalg.norm(rcv - img)
        refl = 1.0 if k == 0 else 0.6
        for c, a in ((c_s0, 1.0), (c_a0, a_ratio)):
            y += refl * a / np.sqrt(L) * toneburst(t - L / c, f)
    if damage is not None:
        d = np.asarray(DPOS[damage])
        l1, l2 = np.linalg.norm(d - src), np.linalg.norm(rcv - d)
        # distance of scatterer from the direct line -> forward-scatter shadow
        u = rcv - src
        v = d - src
        dist = abs(u[0] * v[1] - u[1] * v[0]) / np.linalg.norm(u)
        shadow = 0.06 * np.exp(-(dist / 0.02) ** 2)
        y *= 1 - shadow
        for c, a in ((c_s0, 0.10), (c_a0, 0.25 * a_ratio)):
            y += a / np.sqrt(l1 * l2) * 0.2 * toneburst(t - (l1 + l2) / c, f)
    y *= amp_t * (1 + 0.002 * rng.standard_normal())
    y += 0.5 * toneburst(t, f)  # electrical crosstalk
    y += 0.004 * np.abs(y).max() * rng.standard_normal(t.shape)
    return y.astype(np.float32)


def temperature_cycle(step: float = 0.5) -> np.ndarray:
    up = np.arange(20.0, 60.0 + 1e-9, step)
    return np.concatenate([up, up[-2::-1]])


def write_dataset(root: str | Path, freqs_khz=(100,), fs: float = 2.5e6, n: int = 3277,
                  step: float = 0.5, n_udam_cycles: int = 2, seed: int = 0,
                  states=("udam", "D04", "D12", "D16", "D24")) -> Path:
    """Write a synthetic dataset with the OGW folder/file layout."""
    root = Path(root)
    rng = np.random.default_rng(seed)
    t = np.arange(n) / fs
    clock = datetime(2018, 12, 13, 9, 0, 0)
    for st in states:
        cycles = n_udam_cycles if st == "udam" else 1
        temps = np.concatenate([temperature_cycle(step)] * cycles)
        for T_nom in temps:
            clock += timedelta(minutes=7)
            d = root / FOLDERS[st] / clock.strftime("%Y%m%dT%H%M%S")
            d.mkdir(parents=True, exist_ok=True)
            T_meas = T_nom + 0.1 * rng.standard_normal(2)
            for fk in freqs_khz:
                X = np.stack([simulate(i, j, T_nom, fk * 1e3, t,
                                       None if st == "udam" else st, rng) for i, j in PAIRS])
                with h5py.File(d / f"pc_f{fk}kHz.h5", "w") as h:
                    g = h.create_group("pitchcatch")
                    g.create_dataset("catch", data=X)
                    g.attrs["fs"] = fs
                    h.create_group("Temperature").create_dataset("values", data=T_meas)
    return root
