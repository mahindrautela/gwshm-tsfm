"""Frozen time-series foundation-model embeddings (no training, no fine-tuning).

Every embedder maps signals X (n, L) to a fixed-length feature matrix (n, d).
Pooling is mean-over-patches unless the model provides its own pooled output.
"""

from __future__ import annotations

import os
import time

import numpy as np


def _torch():
    import torch
    torch.set_num_threads(os.cpu_count() or 1)
    return torch


def resample(X: np.ndarray, n: int) -> np.ndarray:
    """Linear interpolation of each row onto n equally spaced points."""
    src = np.linspace(0, 1, X.shape[1])
    dst = np.linspace(0, 1, n)
    return np.stack([np.interp(dst, src, x) for x in X]).astype(np.float32)


class Embedder:
    name = "base"
    hf_id = ""
    params = ""

    def load(self):
        return self

    def _embed(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def __call__(self, X: np.ndarray, batch: int = 64) -> tuple[np.ndarray, float]:
        t0 = time.time()
        Z = np.concatenate([self._embed(np.ascontiguousarray(X[i:i + batch], dtype=np.float32))
                            for i in range(0, len(X), batch)])
        return Z.astype(np.float32), time.time() - t0


class Chronos2Emb(Embedder):
    name, hf_id, params = "Chronos-2", "amazon/chronos-2", "120M"

    def load(self):
        _torch()
        from chronos import BaseChronosPipeline
        self.pipe = BaseChronosPipeline.from_pretrained(self.hf_id, device_map="cpu")
        return self

    def _embed(self, X):
        torch = _torch()
        with torch.inference_mode():
            out = self.pipe.embed(torch.from_numpy(X)[:, None, :])
        embs = out[0] if isinstance(out, tuple) else out
        return torch.stack([e[0].float().mean(0) for e in embs]).numpy()


class ChronosBoltEmb(Embedder):
    name, hf_id, params = "Chronos-Bolt (base)", "amazon/chronos-bolt-base", "205M"

    def load(self):
        _torch()
        from chronos import BaseChronosPipeline
        self.pipe = BaseChronosPipeline.from_pretrained(self.hf_id, device_map="cpu")
        return self

    def _embed(self, X):
        torch = _torch()
        with torch.inference_mode():
            emb, _ = self.pipe.embed(torch.from_numpy(X))
        return emb.float().mean(1).numpy()


class TiRexEmb(Embedder):
    name, hf_id, params = "TiRex", "NX-AI/TiRex", "35M"

    def load(self):
        os.environ.setdefault("TIREX_NO_CUDA", "1")
        _torch()
        from tirex.models.embedding import TiRexEmbedding
        self.m = TiRexEmbedding(device="cpu", data_augmentation=False, batch_size=64)
        return self

    def _embed(self, X):
        torch = _torch()
        with torch.inference_mode():
            z = self.m(torch.from_numpy(X)[:, None, :])
        return z.reshape(len(X), -1).float().numpy()


def segments(X: np.ndarray, n: int = 512) -> np.ndarray:
    """Split (B, L) into consecutive n-sample windows -> (B * k, n), zero-padding the last."""
    k = int(np.ceil(X.shape[1] / n))
    Xp = np.pad(X, ((0, 0), (0, k * n - X.shape[1])))
    return Xp.reshape(len(X) * k, n), k


class MomentEmb(Embedder):
    name, hf_id, params = "MOMENT-1 (base)", "AutonLab/MOMENT-1-base", "125M"
    segment = False  # True: native-rate 512-sample windows, concatenated (no resampling)

    def load(self):
        _torch()
        from momentfm import MOMENTPipeline
        self.m = MOMENTPipeline.from_pretrained(self.hf_id, model_kwargs={"task_name": "embedding"})
        self.m.init()
        self.m.eval()
        return self

    def _embed(self, X):
        torch = _torch()
        if self.segment:
            x, k = segments(X)
        else:
            x, k = resample(X, 512), 1
        with torch.inference_mode():
            out = self.m(x_enc=torch.from_numpy(np.ascontiguousarray(x))[:, None, :],
                         input_mask=torch.ones(len(x), 512, dtype=torch.long))
        return out.embeddings.float().numpy().reshape(len(X), -1)


class MantisEmb(Embedder):
    name, hf_id, params = "Mantis-8M", "paris-noah/Mantis-8M", "8M"
    segment = False

    def load(self):
        _torch()
        from mantis.architecture import Mantis8M
        from mantis.trainer import MantisTrainer
        net = Mantis8M(device="cpu").from_pretrained(self.hf_id)
        self.tr = MantisTrainer(device="cpu", network=net)
        return self

    def _embed(self, X):
        x, k = segments(X) if self.segment else (resample(X, 512), 1)
        return np.asarray(self.tr.transform(x[:, None, :], batch_size=64)).reshape(len(X), -1)


class TTMEmb(Embedder):
    name, hf_id, params = "TTM-R2", "ibm-granite/granite-timeseries-ttm-r2", "~1-5M"
    L = 1024

    def load(self):
        _torch()
        from tsfm_public.toolkit.get_model import get_model
        self.m = get_model(self.hf_id, context_length=self.L, prediction_length=96).eval()
        return self

    def _embed(self, X):
        torch = _torch()
        if X.shape[1] < self.L:
            X = np.concatenate([np.repeat(X[:, :1], self.L - X.shape[1], 1), X], 1)
        X = X[:, -self.L:]
        with torch.inference_mode():
            o = self.m(past_values=torch.from_numpy(np.ascontiguousarray(X))[:, :, None])
        h = o.backbone_hidden_state  # (b, channels, patches, d)
        return h.float().mean(2).reshape(len(X), -1).numpy()


REGISTRY = {"chronos2": Chronos2Emb, "chronos_bolt": ChronosBoltEmb, "tirex": TiRexEmb,
            "moment": MomentEmb, "mantis": MantisEmb, "ttm": TTMEmb}


def get(name: str, segment: bool = False) -> Embedder:
    e = REGISTRY[name]()
    if hasattr(e, "segment"):
        e.segment = segment
    return e.load()
