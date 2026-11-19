"""Unified zero-shot forecasters.

Every forecaster maps a batch of univariate context series ``Y`` (B, L) to
predictive quantiles (q10, q50, q90), each (B, H).  Point-only models return
q10 = q50 = q90.  No model is trained or fine-tuned: weights are used exactly
as released.

Inputs are standardised per series (context mean / std) before being handed to
any model and de-standardised afterwards, so every model sees the same,
well-conditioned input regardless of its internal scaling scheme.
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass

import numpy as np

QS = (0.1, 0.5, 0.9)


@dataclass
class Forecast:
    q10: np.ndarray
    q50: np.ndarray
    q90: np.ndarray
    seconds: float = 0.0


class Forecaster:
    name = "base"
    hf_id: str | None = None
    probabilistic = True
    params = ""  # human-readable size for the paper table

    def load(self):  # lazy weight loading
        return self

    def _predict(self, Y: np.ndarray, H: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        raise NotImplementedError

    def __call__(self, Y: np.ndarray, H: int, chunk: int = 4096) -> Forecast:
        Y = np.asarray(Y, dtype=np.float32)
        mu = Y.mean(1, keepdims=True)
        sd = Y.std(1, keepdims=True) + 1e-8 * (np.abs(Y).max() + 1e-30)
        Z = (Y - mu) / sd
        t0 = time.time()
        outs = [self._predict(Z[i:i + chunk], H) for i in range(0, len(Z), chunk)]
        q10, q50, q90 = (np.concatenate([o[k] for o in outs]) for k in range(3))
        q10, q50, q90 = np.minimum(q10, q50), q50, np.maximum(q90, q50)
        f = lambda q: (q * sd + mu).astype(np.float32)  # noqa: E731
        return Forecast(f(q10), f(q50), f(q90), time.time() - t0)


# ----------------------------------------------------------------------------
# Non-learned forecasters along the temperature axis
# ----------------------------------------------------------------------------

class LastValue(Forecaster):
    """Persistence: the hottest available reference is the prediction."""
    name, probabilistic, params = "Last value", False, "-"

    def _predict(self, Y, H):
        q = np.repeat(Y[:, -1:], H, 1)
        return q, q, q


class LinearTrend(Forecaster):
    """Least-squares line through the last k context points, extrapolated."""
    name, probabilistic, params = "Linear trend", False, "-"

    def __init__(self, k: int = 8):
        self.k = k

    def _predict(self, Y, H):
        k = min(self.k, Y.shape[1])
        x = np.arange(k) - (k - 1) / 2
        y = Y[:, -k:]
        slope = (y - y.mean(1, keepdims=True)) @ x / (x @ x)
        h = np.arange(1, H + 1) + (k - 1) / 2
        q = y.mean(1, keepdims=True) + slope[:, None] * h[None]
        return q, q, q


class StatsETS(Forecaster):
    """AutoETS (statsforecast) per series, with 80 % prediction intervals."""
    name, params = "AutoETS", "-"

    def _predict(self, Y, H):
        import pandas as pd
        from statsforecast import StatsForecast
        from statsforecast.models import AutoETS

        B, L = Y.shape
        df = pd.DataFrame({"unique_id": np.repeat(np.arange(B), L),
                           "ds": np.tile(np.arange(L), B), "y": Y.ravel()})
        sf = StatsForecast(models=[AutoETS(season_length=1)], freq=1, n_jobs=os.cpu_count())
        fc = sf.forecast(df=df, h=H, level=[80]).sort_values(["unique_id", "ds"])
        g = lambda c: fc[c].to_numpy(np.float32).reshape(B, H)  # noqa: E731
        return g("AutoETS-lo-80"), g("AutoETS"), g("AutoETS-hi-80")


# ----------------------------------------------------------------------------
# Time series foundation models (zero-shot)
# ----------------------------------------------------------------------------

def _torch():
    import torch
    torch.set_num_threads(os.cpu_count() or 1)
    return torch


class Chronos2(Forecaster):
    name, hf_id, params = "Chronos-2", "amazon/chronos-2", "120M"

    def load(self):
        _torch()
        from chronos import BaseChronosPipeline
        self.pipe = BaseChronosPipeline.from_pretrained(self.hf_id, device_map="cpu")
        return self

    def _predict(self, Y, H):
        torch = _torch()
        with torch.inference_mode():
            q, _ = self.pipe.predict_quantiles(torch.from_numpy(Y)[:, None, :], prediction_length=H,
                                               quantile_levels=list(QS), batch_size=512)
        q = torch.stack([t[0] for t in q]).float().numpy()  # (B, H, 3)
        return q[..., 0], q[..., 1], q[..., 2]


class ChronosBolt(Forecaster):
    name, hf_id, params = "Chronos-Bolt (base)", "amazon/chronos-bolt-base", "205M"

    def load(self):
        _torch()
        from chronos import BaseChronosPipeline
        self.pipe = BaseChronosPipeline.from_pretrained(self.hf_id, device_map="cpu")
        return self

    def _predict(self, Y, H):
        torch = _torch()
        with torch.inference_mode():
            q, _ = self.pipe.predict_quantiles(torch.from_numpy(Y), prediction_length=H,
                                               quantile_levels=list(QS), limit_prediction_length=False)
        q = q.float().numpy()
        return q[..., 0], q[..., 1], q[..., 2]


class TimesFM25(Forecaster):
    name, hf_id, params = "TimesFM-2.5", "google/timesfm-2.5-200m-pytorch", "200M"

    def load(self):
        _torch()
        import timesfm
        self.model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(self.hf_id)
        self._compiled = None
        return self

    def _compile(self, H):
        import timesfm
        mh = int(math.ceil(H / 128) * 128)
        if self._compiled != mh:
            self.model.compile(timesfm.ForecastConfig(
                max_context=128, max_horizon=mh, normalize_inputs=True, per_core_batch_size=256,
                use_continuous_quantile_head=True, force_flip_invariance=True,
                infer_is_positive=False, fix_quantile_crossing=True))
            self._compiled = mh

    def _predict(self, Y, H):
        self._compile(H)
        _, q = self.model.forecast(horizon=H, inputs=list(Y))
        q = np.asarray(q)  # (B, H, 10): mean, q10..q90
        return q[..., 1], q[..., 5], q[..., 9]


class TimesFM3(Forecaster):
    name, hf_id, params = "TimesFM-3", "google/timesfm-3.0-pytorch", "330M"

    def load(self):
        _torch()
        import timesfm3
        cfg_cls = getattr(timesfm3, "ModelConfig")
        fc_cls = getattr(timesfm3, "TimesFM3Forecaster", None) or getattr(timesfm3, "TimesFM3Evaluator")
        self.fc = fc_cls(cfg_cls(checkpoint_path=self.hf_id, per_core_batch_size=256, device="cpu"))
        return self

    def _predict(self, Y, H):
        outs = list(self.fc.predict_batch(list(Y), horizon=H, return_quantiles=True,
                                          use_symmetric_averaging=True, make_positive=False))
        q = np.stack([o.quantiles for o in outs])  # (B, H, 9) for q0.1..q0.9
        return q[..., 0], q[..., 4], q[..., 8]


class TiRex(Forecaster):
    name, hf_id, params = "TiRex", "NX-AI/TiRex", "35M"

    def load(self):
        os.environ.setdefault("TIREX_NO_CUDA", "1")
        _torch()
        from tirex import load_model
        self.model = load_model(self.hf_id, device="cpu", backend="torch")
        return self

    def _predict(self, Y, H):
        torch = _torch()
        q, _ = self.model.forecast(context=torch.from_numpy(Y), prediction_length=H,
                                   output_type="numpy", batch_size=512)
        return q[..., 0], q[..., 4], q[..., 8]


class TiRex2(Forecaster):
    name, hf_id, params = "TiRex-2", "NX-AI/TiRex-2", "38M"

    def load(self):
        _torch()
        from tirex2 import load_model
        self.model = load_model(self.hf_id, device="cpu")
        return self

    def _predict(self, Y, H):
        torch = _torch()
        from tirex2 import TimeseriesType
        ts = [TimeseriesType(target=torch.from_numpy(y)[None], past_covariates=None,
                             future_covariates=None) for y in Y]
        out = self.model.forecast(ts, prediction_length=H, output_type="numpy", batch_size=512)
        q = out[0] if isinstance(out, tuple) else out
        q = np.asarray(q).reshape(len(Y), H, -1)  # (B, H, 9)
        return q[..., 0], q[..., 4], q[..., 8]


class Toto2(Forecaster):
    name, params = "Toto-2.0 (22M)", "22M"
    hf_id = "Datadog/Toto-2.0-22m"

    def load(self):
        torch = _torch()
        from toto2 import Toto2Model
        self.model = Toto2Model.from_pretrained(self.hf_id).to(torch.device("cpu")).eval()
        return self

    PATCH = 32

    def _predict(self, Y, H):
        torch = _torch()
        L = Y.shape[1]
        pad = (-L) % self.PATCH  # context must be a whole number of patches: left-pad, masked
        out = []
        for i in range(0, len(Y), 512):
            y = Y[i:i + 512]
            t = torch.from_numpy(np.ascontiguousarray(np.pad(y, ((0, 0), (pad, 0)))))[:, None, :]
            mask = torch.ones_like(t, dtype=torch.bool)
            mask[..., :pad] = False
            with torch.inference_mode():
                q = self.model.forecast({"target": t, "target_mask": mask,
                                         "series_ids": torch.zeros(t.shape[0], 1, dtype=torch.long)},
                                        horizon=H, has_missing_values=pad > 0)
            out.append(q[:, :, 0, :].float().numpy())  # (9, b, H)
        q = np.concatenate(out, 1)
        return q[0], q[4], q[8]


class Moirai2(Forecaster):
    name, hf_id, params = "Moirai-2.0 (small)", "Salesforce/moirai-2.0-R-small", "11M"

    def load(self):
        _torch()
        from uni2ts.model.moirai2 import Moirai2Module
        self.module = Moirai2Module.from_pretrained(self.hf_id)
        return self

    def _predict(self, Y, H):
        torch = _torch()
        from uni2ts.model.moirai2 import Moirai2Forecast
        m = Moirai2Forecast(module=self.module, prediction_length=H, context_length=Y.shape[1],
                            target_dim=1, feat_dynamic_real_dim=0, past_feat_dynamic_real_dim=0)
        out = []
        for i in range(0, len(Y), 256):
            with torch.inference_mode():
                out.append(np.asarray(m.predict([y[:, None] for y in Y[i:i + 256]])))
        q = np.concatenate(out).reshape(len(Y), -1, H)  # (B, n_q, H)
        ql = list(getattr(self.module, "quantile_levels", np.linspace(0.1, 0.9, q.shape[1])))
        idx = [int(np.argmin(np.abs(np.asarray(ql) - a))) for a in QS]
        return q[:, idx[0]], q[:, idx[1]], q[:, idx[2]]


class TTM(Forecaster):
    """Tiny Time Mixer r2 (point forecaster; short contexts are left-padded)."""
    name, hf_id, probabilistic, params = "TTM-R2", "ibm-granite/granite-timeseries-ttm-r2", False, "~1-5M"
    CONTEXTS = (52, 90, 180, 360, 512, 1024, 1536)

    def load(self):
        _torch()
        self.cache = {}
        return self

    def _get(self, L, H):
        from tsfm_public.toolkit.get_model import get_model
        key = (L, H)
        if key not in self.cache:
            # short-context TTMs have short native horizons: roll them forward
            m = get_model(self.hf_id, context_length=L, prediction_length=H, force_return="rolling")
            self.cache[key] = m.eval()
        return self.cache[key]

    def _predict(self, Y, H):
        torch = _torch()
        L = next((c for c in self.CONTEXTS if c >= Y.shape[1]), self.CONTEXTS[-1])
        if Y.shape[1] < L:  # left-pad by edge replication
            Y = np.concatenate([np.repeat(Y[:, :1], L - Y.shape[1], 1), Y], 1)
        Y = np.ascontiguousarray(Y[:, -L:])
        model = self._get(L, H)
        hm = int(getattr(model.config, "prediction_filter_length", None) or model.config.prediction_length)
        out = []
        for i in range(0, len(Y), 1024):
            ctx, preds = Y[i:i + 1024], []
            kw = {}
            if getattr(model.config, "resolution_prefix_tuning", False):
                kw["freq_token"] = torch.zeros(len(ctx), dtype=torch.long)  # 'oov' resolution
            while sum(p.shape[1] for p in preds) < H:
                with torch.inference_mode():
                    o = model(past_values=torch.from_numpy(np.ascontiguousarray(ctx))[:, :, None], **kw)
                p = o.prediction_outputs[:, :hm, 0].float().numpy()
                preds.append(p)
                ctx = np.concatenate([ctx, p], 1)[:, -L:]
            out.append(np.concatenate(preds, 1)[:, :H])
        q = np.concatenate(out)
        return q, q, q


REGISTRY = {
    "last": LastValue, "linear": LinearTrend, "ets": StatsETS,
    "chronos2": Chronos2, "chronos_bolt": ChronosBolt,
    "timesfm25": TimesFM25, "timesfm3": TimesFM3,
    "tirex": TiRex, "tirex2": TiRex2, "toto2": Toto2,
    "moirai2": Moirai2, "ttm": TTM,
}
TSFMS = ["chronos2", "chronos_bolt", "timesfm25", "timesfm3", "tirex", "tirex2",
         "toto2", "moirai2", "ttm"]


def get(name: str) -> Forecaster:
    return REGISTRY[name]().load()
