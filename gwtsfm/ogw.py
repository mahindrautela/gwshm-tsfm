"""I/O for the Open Guided Waves (OGW) CFRP temperature dataset.

Moll et al., "Temperature affected guided wave propagation in a composite plate
complementing the Open Guided Waves Platform", Scientific Data 6:191 (2019).

On-disk layout (as distributed):

    <root>/OGW_CFRP_Temperature_udam/<YYYYMMDDTHHMMSS>/pc_f<freq>kHz.h5
    <root>/OGW_CFRP_Temperature_dam_D04/<timestamp>/pc_f<freq>kHz.h5
    ... (dam_D12, dam_D16, dam_D24)

Each timestamped folder is one temperature step.  Each H5 file holds

    pitchcatch/catch   : (66, n_samples) received signals, one row per
                         transducer pair (i, j), i < j, lexicographic order
    Temperature/values : surface temperatures from the two probes (deg C)

The loader below reads these, decimates them, and caches one compact
``.npz`` per excitation frequency so that experiments never touch the raw
(~tens of GB) files again.
"""

from __future__ import annotations

import itertools
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import h5py
import numpy as np
from scipy import signal as sps

FREQS_KHZ = list(range(40, 261, 20))
N_TRANSDUCERS = 12
PAIRS = list(itertools.combinations(range(1, N_TRANSDUCERS + 1), 2))  # 66 pairs
FOLDERS = {
    "udam": "OGW_CFRP_Temperature_udam",
    "D04": "OGW_CFRP_Temperature_dam_D04",
    "D12": "OGW_CFRP_Temperature_dam_D12",
    "D16": "OGW_CFRP_Temperature_dam_D16",
    "D24": "OGW_CFRP_Temperature_dam_D24",
}
# Sampling rate used when the files carry no explicit attribute.  Verified
# against the files at load time where possible (see ``_infer_fs``).
DEFAULT_FS = 10e6

_TS_RE = re.compile(r"^\d{8}T\d{6}$")


def pair_index(i: int, j: int) -> int:
    """Row index of the pair (i, j) in ``pitchcatch/catch`` (1-based ids)."""
    a, b = (i, j) if i < j else (j, i)
    return PAIRS.index((a, b))


def pair_name(k: int) -> str:
    i, j = PAIRS[k]
    return f"T{i}-T{j}"


def _first_existing(group: h5py.Group, names) -> object | None:
    for n in names:
        if n in group:
            return group[n]
        if n in group.attrs:
            return group.attrs[n]
    return None


def describe_h5(path: str | os.PathLike) -> dict:
    """Return a JSON-able description of an OGW H5 file (structure + attrs)."""
    out: dict = {"datasets": {}, "attrs": {}}

    def visit(name, obj):
        if isinstance(obj, h5py.Dataset):
            out["datasets"][name] = {"shape": list(obj.shape), "dtype": str(obj.dtype)}
            if obj.size <= 16:
                out["datasets"][name]["value"] = np.asarray(obj[()]).tolist()
        for k, v in obj.attrs.items():
            try:
                v = np.asarray(v).tolist()
            except Exception:  # pragma: no cover - exotic attr types
                v = str(v)
            out["attrs"][f"{name}@{k}"] = v

    with h5py.File(path, "r") as f:
        for k, v in f.attrs.items():
            out["attrs"][f"/@{k}"] = np.asarray(v).tolist()
        f.visititems(visit)
    return out


def _infer_fs(f: h5py.File) -> float | None:
    """Try to read the sampling rate from common attribute/dataset names."""
    for grp in (f, f.get("pitchcatch")):
        if grp is None:
            continue
        for name in ("fs", "Fs", "samplingrate", "sampling_rate", "SampleFrequency",
                     "sample_frequency", "samplefrequency"):
            v = _first_existing(grp, [name])
            if v is not None:
                return float(np.asarray(v[()] if hasattr(v, "shape") else v).ravel()[0])
        for name in ("dt", "Ts", "sampletime", "time_step"):
            v = _first_existing(grp, [name])
            if v is not None:
                return 1.0 / float(np.asarray(v[()] if hasattr(v, "shape") else v).ravel()[0])
        for name in ("time", "t", "Time"):
            v = _first_existing(grp, [name])
            if v is not None and hasattr(v, "shape") and v.size > 2:
                t = np.asarray(v[()]).ravel()
                return 1.0 / float(np.median(np.diff(t)))
    return None


def read_h5(path: str | os.PathLike) -> tuple[np.ndarray, float, float | None]:
    """Read one OGW file -> (catch [66, N] float32, mean temperature, fs or None)."""
    with h5py.File(path, "r") as f:
        x = np.asarray(f["pitchcatch"]["catch"], dtype=np.float32)
        temps = np.asarray(f["Temperature"]["values"], dtype=np.float64).ravel()
        fs = _infer_fs(f)
    if x.shape[0] != len(PAIRS) and x.shape[1] == len(PAIRS):
        x = x.T
    return x, float(np.nanmean(temps)), fs


def list_measurements(root: str | os.PathLike, state: str) -> list[Path]:
    """Timestamped measurement folders for a state ('udam', 'D04', ...), in time order."""
    base = Path(root) / FOLDERS[state]
    if not base.exists():
        return []
    dirs = [d for d in base.iterdir() if d.is_dir() and _TS_RE.match(d.name)]
    return sorted(dirs, key=lambda d: d.name)


def decimation_factor(fs: float, f_khz: float, oversample: float = 8.0) -> int:
    """Integer decimation keeping >= ``oversample`` samples per carrier period."""
    return max(1, int(np.floor(fs / (oversample * f_khz * 1e3))))


def decimate(x: np.ndarray, q: int) -> np.ndarray:
    """Zero-phase anti-aliased decimation along the last axis."""
    if q <= 1:
        return x.astype(np.float32)
    return sps.decimate(x, q, ftype="fir", zero_phase=True, axis=-1).astype(np.float32)


@dataclass
class FreqCache:
    """All measurements of one excitation frequency, decimated."""

    f_khz: int
    fs: float                      # sampling rate after decimation
    X: np.ndarray                  # (n_meas, 66, n_samples) float32
    temp: np.ndarray               # (n_meas,) mean surface temperature
    state: np.ndarray              # (n_meas,) str: 'udam', 'D04', ...
    stamp: np.ndarray              # (n_meas,) str timestamp
    meta: dict = field(default_factory=dict)

    @property
    def label(self) -> np.ndarray:
        return (self.state != "udam").astype(int)

    def save(self, path: str | os.PathLike) -> None:
        np.savez(path, f_khz=self.f_khz, fs=self.fs, X=self.X, temp=self.temp,
                 state=self.state, stamp=self.stamp, meta=json.dumps(self.meta))

    @classmethod
    def load(cls, path: str | os.PathLike, mmap: bool = False) -> "FreqCache":
        d = np.load(path, mmap_mode="r" if mmap else None, allow_pickle=False)
        return cls(f_khz=int(d["f_khz"]), fs=float(d["fs"]), X=d["X"], temp=d["temp"],
                   state=d["state"], stamp=d["stamp"], meta=json.loads(str(d["meta"])))


def build_cache(root: str | os.PathLike, f_khz: int, states=tuple(FOLDERS),
                oversample: float = 8.0, fs_default: float = DEFAULT_FS,
                verbose: bool = True) -> FreqCache:
    """Read every measurement of one frequency and return a decimated cache."""
    Xs, temps, sts, stamps = [], [], [], []
    fs_seen = None
    q = None
    for st in states:
        dirs = list_measurements(root, st)
        for d in dirs:
            fp = d / f"pc_f{f_khz}kHz.h5"
            if not fp.exists():
                continue
            x, t, fs = read_h5(fp)
            fs = fs or fs_default
            if fs_seen is None:
                fs_seen = fs
                q = decimation_factor(fs, f_khz, oversample)
            Xs.append(decimate(x, q))
            temps.append(t)
            sts.append(st)
            stamps.append(d.name)
        if verbose:
            print(f"  f={f_khz} kHz  {st}: {sum(s == st for s in sts)} measurements")
    if not Xs:
        raise FileNotFoundError(f"no files for f={f_khz} kHz under {root}")
    n = min(x.shape[-1] for x in Xs)
    X = np.stack([x[..., :n] for x in Xs])
    return FreqCache(f_khz=f_khz, fs=fs_seen / q, X=X, temp=np.asarray(temps),
                     state=np.asarray(sts), stamp=np.asarray(stamps),
                     meta={"fs_raw": fs_seen, "decimation": q,
                           "fs_source": "file" if fs_seen != fs_default else "default"})


def split_reference(cache: FreqCache) -> tuple[np.ndarray, np.ndarray]:
    """Indices of the reference set and of the test set.

    Following Moll et al. (2019), the first heating ramp of the intact plate
    (the first ~81 'udam' measurements, 20 -> 60 deg C) is the reference /
    baseline set.  Everything else (later intact measurements and all damaged
    measurements) is the test set.  The ramp end is detected as the first
    temperature maximum, so the split does not depend on an exact count.
    """
    ud = np.flatnonzero(cache.state == "udam")
    order = ud[np.argsort(cache.stamp[ud])]
    t = cache.temp[order]
    # end of first heating ramp: first index where the running max stops growing
    # for several consecutive steps (robust to probe noise)
    run_max = np.maximum.accumulate(t)
    end = len(t)
    for k in range(1, len(t)):
        if run_max[k] - t[k] > 1.0:  # temperature dropped by > 1 deg from the peak
            end = k
            break
    end = int(np.argmax(t[:end])) + 1
    ref = order[:end]
    test = np.setdiff1d(np.arange(len(cache.temp)), ref)
    return ref, test
