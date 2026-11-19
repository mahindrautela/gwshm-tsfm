"""Smoke test: synthetic OGW-format data -> cache -> protocol -> metrics."""

import numpy as np

from gwtsfm import metrics, models, protocol
from gwtsfm.ogw import FreqCache, build_cache, pair_index, split_reference
from gwtsfm.synth import write_dataset


def test_end_to_end(tmp_path):
    write_dataset(tmp_path / "raw", freqs_khz=(100,), fs=1.25e6, n=1200, step=2.0,
                  n_udam_cycles=2, states=("udam", "D16"))
    c = build_cache(tmp_path / "raw", 100, verbose=False)
    c.save(tmp_path / "f100kHz.npz")
    c = FreqCache.load(tmp_path / "f100kHz.npz")
    ref, test = split_reference(c)
    assert np.all(np.diff(c.temp[ref]) > 0)          # first heating ramp only
    assert set(c.state[test]) == {"udam", "D16"}

    s = protocol.prepare(c)
    Y, sel, H = protocol.context(s, 40.0)
    assert Y.shape[0] == len(s.paths) * s.X.shape[-1] and H > 0

    fc = models.LastValue()(Y, H)
    df, _ = protocol.evaluate_forecast(s, 40.0, "Last value", fc)
    df = df._append(protocol.run_baselines(s, 40.0), ignore_index=True)
    assert set(df["method"]) == {"Last value", "OBS", "OBS+BSS", "Stretch extrapolation"}
    a = metrics.fused_auc(df, "di_res")
    assert a["auc"].between(0, 1).all()
    pp = metrics.per_path_auc(df)
    assert (pp["path"] == pair_index(4, 9)).any()
