"""The six frozen time series foundation models, used as feature extractors.

Every model maps a batch of signals X with shape (n, L) to embeddings with shape (n, d).
Nothing is trained or fine-tuned: we load the public checkpoint and read out its internal
representation of each signal, one univariate signal at a time.

How each model turns a signal into one vector:
  Chronos-2, Chronos-Bolt, TTM-R2   mean over the encoder's patch tokens
  TiRex                             the package's own embedding module
  MOMENT-1, Mantis-8M               the model's own embedding of a 512-sample window

MOMENT and Mantis only accept 512 samples. With ``segment=False`` the signal is resampled
to 512 points (used for GW-TCW). With ``segment=True`` it is cut into consecutive
512-sample windows at its native rate and the window embeddings are concatenated (used
for NASA and OGW-1, where resampling would alias the high-frequency content).

Usage:
    model = embed.get("moment", segment=True)
    Z, seconds = model(X)
"""

from __future__ import annotations

import os
import time

import numpy as np


def _torch():
    """Import torch lazily (it is slow to import) and use every CPU core."""
    import torch
    torch.set_num_threads(os.cpu_count() or 1)
    return torch


def resample(X: np.ndarray, n: int) -> np.ndarray:
    """Linearly interpolate each row of X onto n equally spaced points."""
    src = np.linspace(0, 1, X.shape[1])
    dst = np.linspace(0, 1, n)
    return np.stack([np.interp(dst, src, x) for x in X]).astype(np.float32)


def segments(X: np.ndarray, n: int = 512) -> tuple[np.ndarray, int]:
    """Cut each row into consecutive n-sample windows (the last one zero-padded).

    Returns the windows stacked as (rows * k, n) and the number of windows k per row.
    """
    k = int(np.ceil(X.shape[1] / n))
    Xp = np.pad(X, ((0, 0), (0, k * n - X.shape[1])))
    return Xp.reshape(len(X) * k, n), k


class Embedder:
    """Base class: subclasses load a checkpoint and implement ``_embed`` for one batch."""

    name = "base"      # display name
    hf_id = ""         # Hugging Face checkpoint
    params = ""        # parameter count of the loaded checkpoint

    def load(self):
        return self

    def _embed(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def __call__(self, X: np.ndarray, batch: int = 64) -> tuple[np.ndarray, float]:
        """Embed all rows of X in batches; returns (embeddings, seconds taken)."""
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
            out = self.pipe.embed(torch.from_numpy(X)[:, None, :])  # one series per item
        embs = out[0] if isinstance(out, tuple) else out
        # each item: (variates, tokens, d); average the tokens of the single variate
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
            emb, _ = self.pipe.embed(torch.from_numpy(X))   # (n, tokens, d)
        return emb.float().mean(1).numpy()                 # average over tokens


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


class MomentEmb(Embedder):
    name, hf_id, params = "MOMENT-1 (base)", "AutonLab/MOMENT-1-base", "110M"
    segment = False

    def load(self):
        _torch()
        from momentfm import MOMENTPipeline
        self.m = MOMENTPipeline.from_pretrained(self.hf_id, model_kwargs={"task_name": "embedding"})
        self.m.init()
        self.m.eval()
        return self

    def _embed(self, X):
        torch = _torch()
        x, k = segments(X) if self.segment else (resample(X, 512), 1)
        with torch.inference_mode():
            out = self.m(x_enc=torch.from_numpy(np.ascontiguousarray(x))[:, None, :],
                         input_mask=torch.ones(len(x), 512, dtype=torch.long))
        # (n * k windows, 768) -> (n, k * 768): window embeddings side by side
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
    name, hf_id, params = "TTM-R2", "ibm-granite/granite-timeseries-ttm-r2", "3M"
    L = 1024  # context length of the TTM variant we load

    def load(self):
        _torch()
        from tsfm_public.toolkit.get_model import get_model
        self.m = get_model(self.hf_id, context_length=self.L, prediction_length=96).eval()
        return self

    def _embed(self, X):
        torch = _torch()
        if X.shape[1] < self.L:  # pad on the left by repeating the first value
            X = np.concatenate([np.repeat(X[:, :1], self.L - X.shape[1], 1), X], 1)
        X = X[:, -self.L:]
        with torch.inference_mode():
            o = self.m(past_values=torch.from_numpy(np.ascontiguousarray(X))[:, :, None])
        h = o.backbone_hidden_state                       # (n, channels, patches, d)
        return h.float().mean(2).reshape(len(X), -1).numpy()  # average over patches


REGISTRY = {"chronos2": Chronos2Emb, "chronos_bolt": ChronosBoltEmb, "tirex": TiRexEmb,
            "moment": MomentEmb, "mantis": MantisEmb, "ttm": TTMEmb}


def get(name: str, segment: bool = False) -> Embedder:
    """Load a model by key (see config.MODELS). ``segment`` only matters for MOMENT and Mantis."""
    e = REGISTRY[name]()
    if hasattr(e, "segment"):
        e.segment = segment
    return e.load()
