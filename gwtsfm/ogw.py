"""Open Guided Waves (OGW) dataset 1: file reader and plate geometry (Task 3).

Moll et al., "Open Guided Waves: online platform for ultrasonic guided wave
measurements", Structural Health Monitoring 18(5-6), 2019.

The plate is 500 x 500 mm with 12 transducers: T1-T6 along the top edge (y = 470 mm)
and T7-T12 along the bottom edge (y = 30 mm). A reversible defect was placed at 28
positions D1-D28, which form seven 2 x 2 clusters (zones). Each OGW file holds one
measurement: the signal received on each of the 66 transducer pairs.
"""

from __future__ import annotations

import itertools

import h5py
import numpy as np
from scipy import signal as sps

# The 66 transducer pairs (i, j) with i < j, in the order used throughout this project.
PAIRS = list(itertools.combinations(range(1, 13), 2))

# The 36 paths that cross the plate, from a top transducer (T1-T6) to a bottom one (T7-T12).
CROSS_PATHS = [PAIRS.index((i, j)) for i in range(1, 7) for j in range(7, 13)]

# Transducer and defect-centre coordinates in mm (from the OGW dataset description).
TRANSDUCER_XY = {1: (450, 470), 2: (370, 470), 3: (290, 470), 4: (210, 470), 5: (130, 470),
                 6: (50, 470), 7: (450, 30), 8: (370, 30), 9: (290, 30), 10: (210, 30),
                 11: (130, 30), 12: (50, 30)}
DAMAGE_XY = {1: (50, 415), 2: (65, 415), 3: (50, 400), 4: (65, 400),
             5: (250, 427), 6: (265, 427), 7: (250, 412), 8: (265, 412),
             9: (180, 345), 10: (195, 345), 11: (180, 330), 12: (195, 330),
             13: (320, 275), 14: (335, 275), 15: (320, 260), 16: (335, 260),
             17: (130, 195), 18: (145, 195), 19: (130, 180), 20: (145, 180),
             21: (435, 205), 22: (450, 205), 23: (435, 190), 24: (450, 190),
             25: (250, 90), 26: (265, 90), 27: (250, 75), 28: (265, 75)}


def zone_of(position: int) -> int:
    """Damage zone 1-7 of position 1-28: four neighbouring positions per zone."""
    return (position - 1) // 4 + 1


def phase_of(position: int) -> int:
    """Measurement phase 0-2 of a damage position.

    The dataset was recorded as 20 baselines, D1-D11, 20 baselines, D12-D20, 20 baselines,
    D21-D28. Each damage case is compared with the baselines of its own phase.
    """
    return 0 if position <= 11 else (1 if position <= 20 else 2)


def path_distance(cross: np.ndarray, tpos: np.ndarray, dpos: np.ndarray) -> np.ndarray:
    """Distance (mm) from each damage centre to each transmitter-receiver segment.

    cross: (P, 2) 1-based transducer ids of each path; tpos: (12, 2) transducer xy;
    dpos: (K, 2) damage xy. Returns (K, P).
    """
    T = tpos[cross[:, 0] - 1]
    R = tpos[cross[:, 1] - 1]
    v = R - T
    out = np.empty((len(dpos), len(cross)))
    for k, d in enumerate(dpos):
        # closest point on the segment T -> R, then its distance to the damage centre
        t = np.clip(((d - T) * v).sum(1) / (v * v).sum(1), 0, 1)
        out[k] = np.linalg.norm(T + t[:, None] * v - d, axis=1)
    return out


def read_h5(path, with_onset: bool = False):
    """Read one OGW measurement file (a path or a file-like object).

    Returns (signals, mean temperature, sampling rate) and, with ``with_onset``, the
    sample at which the excitation starts. ``signals`` is (66, N) float32 with rows in
    the order of PAIRS. The sampling rate is None if the file does not state it.
    """
    with h5py.File(path, "r") as f:
        x = np.asarray(f["pitchcatch"]["catch"], dtype=np.float32)
        temps = np.asarray(f["Temperature"]["values"], dtype=np.float64).ravel()
        fs = None
        if "command/pitchcatch/sampling_frequency" in f:
            fs = float(np.asarray(f["command/pitchcatch/sampling_frequency"][()]).ravel()[0])
        onset = None
        if with_onset and "pitchcatch/pitch" in f:
            # first sample where the recorded excitation exceeds 1% of its peak
            p = np.abs(np.asarray(f["pitchcatch/pitch"][0], dtype=np.float64))
            onset = int(np.argmax(p > 0.01 * p.max()))
        if "command/pitchcatch/channels" in f:
            # the file lists the transducer pair (0-based) of each row: reorder to PAIRS
            ch = np.asarray(f["command/pitchcatch/channels"][()]).astype(int) + 1
            order = [PAIRS.index(tuple(sorted(r))) for r in ch]
            if order != list(range(len(PAIRS))):
                y = np.empty_like(x)
                y[order] = x
                x = y
    if with_onset:
        return x, float(np.nanmean(temps)), fs, onset
    return x, float(np.nanmean(temps)), fs


def decimate(x: np.ndarray, q: int) -> np.ndarray:
    """Reduce the sampling rate by an integer factor q (zero-phase anti-aliasing filter)."""
    if q <= 1:
        return x.astype(np.float32)
    return sps.decimate(x, q, ftype="fir", zero_phase=True, axis=-1).astype(np.float32)
