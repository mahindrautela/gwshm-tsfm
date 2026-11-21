"""The frozen time series foundation models, used as feature extractors.

Every model maps a batch of signals X with shape (n, L) to embeddings with shape (n, d).
Nothing is trained or fine-tuned: we load the public checkpoint and read out its internal
representation of each signal, one univariate signal at a time.

How each model turns a signal into one vector:
  Chronos-2, Chronos-Bolt, TTM-R2   mean over the encoder's patch tokens
  TiRex                             the package's own embedding module
  MOMENT-1, Mantis-8M               the model's own embedding of a 512-sample window
  TimesFM 2.5, Sundial, Moirai 2    mean over the decoder's patch tokens (extended experiments)

MOMENT and Mantis only accept 512 samples. With ``segment=False`` the signal is resampled
to 512 points (used for GW-TCW). With ``segment=True`` it is cut into consecutive
512-sample windows at its native rate and the window embeddings are concatenated (used
for NASA and OGW-1, where resampling would alias the high-frequency content).

Every model runs on the CPU by default, as for the paper. ``device`` (e.g. "cuda" or "mps")
moves the model to an accelerator; the embeddings are always returned as CPU arrays.
Mantis is refused on Apple "mps": its scalar encoders apply LayerNorm(eps=1e-15) to
near-zero values, which the MPS kernel gets wrong (embeddings off by ~50% against cpu).

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
    device = "cpu"     # torch device the model runs on (set by ``get``)

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
        self.pipe = BaseChronosPipeline.from_pretrained(self.hf_id, device_map=self.device)
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
        self.pipe = BaseChronosPipeline.from_pretrained(self.hf_id, device_map=self.device)
        return self

    def _embed(self, X):
        torch = _torch()
        with torch.inference_mode():
            emb, _ = self.pipe.embed(torch.from_numpy(X))   # (n, tokens, d)
        return emb.float().mean(1).numpy()                 # average over tokens


class TiRexEmb(Embedder):
    name, hf_id, params = "TiRex", "NX-AI/TiRex", "35M"

    def load(self):
        if self.device == "cpu":
            os.environ.setdefault("TIREX_NO_CUDA", "1")
        _torch()
        from tirex.models.embedding import TiRexEmbedding
        self.m = TiRexEmbedding(device=self.device, data_augmentation=False, batch_size=64)
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
        self.m.to(self.device).eval()
        return self

    def _embed(self, X):
        torch = _torch()
        x, k = segments(X) if self.segment else (resample(X, 512), 1)
        with torch.inference_mode():
            out = self.m(x_enc=torch.from_numpy(np.ascontiguousarray(x))[:, None, :].to(self.device),
                         input_mask=torch.ones(len(x), 512, dtype=torch.long, device=self.device))
        # (n * k windows, 768) -> (n, k * 768): window embeddings side by side
        return out.embeddings.float().cpu().numpy().reshape(len(X), -1)


class MantisEmb(Embedder):
    name, hf_id, params = "Mantis-8M", "paris-noah/Mantis-8M", "8M"
    segment = False

    def load(self):
        if str(self.device).startswith("mps"):
            raise ValueError("Mantis gives wrong embeddings on mps (see the module docstring); use cpu")
        _torch()
        from mantis.architecture import Mantis8M
        from mantis.trainer import MantisTrainer
        net = Mantis8M(device=self.device).from_pretrained(self.hf_id)
        self.tr = MantisTrainer(device=self.device, network=net)
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
        self.m = get_model(self.hf_id, context_length=self.L, prediction_length=96).to(self.device).eval()
        return self

    def _embed(self, X):
        torch = _torch()
        if X.shape[1] < self.L:  # pad on the left by repeating the first value
            X = np.concatenate([np.repeat(X[:, :1], self.L - X.shape[1], 1), X], 1)
        X = X[:, -self.L:]
        with torch.inference_mode():
            o = self.m(past_values=torch.from_numpy(np.ascontiguousarray(X))[:, :, None].to(self.device))
        h = o.backbone_hidden_state                       # (n, channels, patches, d)
        return h.float().mean(2).reshape(len(X), -1).cpu().numpy()  # average over patches


# ----------------------------------------------------------------------------------------
# Models added by the extended experiments (E2). Each reads the encoder states the way the
# model's own forecasting code prepares its input (checked against the installed source),
# and averages them over the patch tokens of the signal. Checkpoints are pinned to the Hub
# revision that was used; ``params`` is the count measured in the loaded checkpoint.
# Moirai 2 needs the gwtsfm-moirai environment (uni2ts pins older torch and numpy).

class TimesFMEmb(Embedder):
    """TimesFM 2.5, decoder-only transformer (Google; timesfm package 3.0.2).

    As in TimesFM_2p5_200M_torch_module.decode: the signal is left-padded with masked zeros
    to a multiple of the 32-sample patch, each patch is normalised with the running mean and
    std of the context up to it, and the 20 transformer layers are run once (no decoding).
    """
    name, hf_id, params = "TimesFM 2.5", "google/timesfm-2.5-200m-pytorch", "231M"
    revision = "1d952420fba87f3c6dee4f240de0f1a0fbc790e3"

    def load(self):
        torch = _torch()
        import timesfm
        w = timesfm.TimesFM_2p5_200M_torch.from_pretrained(self.hf_id, revision=self.revision,
                                                            torch_compile=False)
        self.m = w.model.to(torch.device(self.device)).eval()
        return self

    def _embed(self, X):
        torch = _torch()
        from timesfm.torch.util import revin, update_running_stats
        p = self.m.p
        x = torch.from_numpy(X).to(self.device)
        pad = (-x.shape[1]) % p
        x = torch.nn.functional.pad(x, (pad, 0))
        mask = torch.zeros_like(x, dtype=torch.bool)
        mask[:, :pad] = True
        xp, mp = x.reshape(len(x), -1, p), mask.reshape(len(x), -1, p)
        n = mu = sigma = torch.zeros(len(x), device=x.device)
        mus, sigmas = [], []
        for i in range(xp.shape[1]):             # running statistics, patch by patch
            (n, mu, sigma), _ = update_running_stats(n, mu, sigma, xp[:, i], mp[:, i])
            mus.append(mu)
            sigmas.append(sigma)
        with torch.inference_mode():
            z = revin(xp, torch.stack(mus, 1), torch.stack(sigmas, 1), reverse=False)
            z = torch.where(mp, 0.0, z)
            (_, h, _, _), _ = self.m(z, mp)      # h: (n, patches, 1280)
        return h.float().mean(1).cpu().numpy()


class SundialEmb(Embedder):
    """Sundial, decoder-only transformer with a flow-matching head (THUML; remote code).

    As in SundialForPrediction.forward with revin=True: each signal is standardised, cut
    into 16-sample tokens (left zero-padding with a mask), and the hidden states after the
    final layer norm of SundialModel are averaged.
    """
    name, hf_id, params = "Sundial (base)", "thuml/sundial-base-128m", "128M"
    revision = "3212e42564493f520593e5414af4367fc4b49226"

    def load(self):
        torch = _torch()
        from transformers import AutoModelForCausalLM
        m = AutoModelForCausalLM.from_pretrained(self.hf_id, revision=self.revision,
                                                 trust_remote_code=True)
        # The rotary tables are non-persistent buffers, absent from the checkpoint. With
        # transformers 5 the model is built on the meta device and these buffers are left
        # uninitialised (random memory, sometimes NaN). Recompute them as SundialRotaryEmbedding
        # does in its __init__.
        for layer in m.model.layers:
            r = layer.self_attn.rotary_emb
            r.inv_freq = 1.0 / (r.base ** (torch.arange(0, r.dim, 2, dtype=torch.int64).float() / r.dim))
            r._set_cos_sin_cache(seq_len=r.max_position_embeddings, device=r.inv_freq.device,
                                 dtype=torch.get_default_dtype())
        self.m = m.model.to(torch.device(self.device)).eval()
        return self

    def _embed(self, X):
        torch = _torch()
        x = torch.from_numpy(X).to(self.device)
        mean = x.mean(1, keepdim=True)
        std = x.std(1, keepdim=True, unbiased=False)
        x = (x - mean) / torch.where(std > 1e-2, std, torch.ones_like(std))
        with torch.inference_mode():
            h = self.m(input_ids=x, use_cache=False, return_dict=True).last_hidden_state
        return h.float().mean(1).cpu().numpy()


class Moirai2Emb(Embedder):
    """Moirai 2.0-R small, decoder-only transformer (Salesforce; uni2ts 2.0.0).

    Moirai2Forecast._convert builds the packed input exactly as for forecasting (16-sample
    patches, observed mask, sample/time/variate ids, one prediction token); the module is
    then run as in Moirai2Module.forward up to the encoder output, and the context tokens
    (not the prediction token) are averaged. Runs in the gwtsfm-moirai environment, on cpu
    only: _convert uses cummax, which torch 2.4 does not implement on mps.
    """
    name, hf_id, params = "Moirai 2.0 (small)", "Salesforce/moirai-2.0-R-small", "11M"
    revision = "30f43ff08c8494f4943ae1521e9d4e94a0fbb389"

    def load(self):
        if self.device != "cpu":
            raise ValueError("Moirai 2 runs on cpu only (see the class docstring)")
        torch = _torch()
        from uni2ts.model.moirai2 import Moirai2Module
        self.mod = Moirai2Module.from_pretrained(self.hf_id, revision=self.revision)
        self.mod = self.mod.to(torch.device(self.device)).eval()
        return self

    def _embed(self, X):
        torch = _torch()
        from uni2ts.model.moirai2 import Moirai2Forecast
        from uni2ts.model.moirai2.module import packed_causal_attention_mask
        m = self.mod
        fc = Moirai2Forecast(module=m, prediction_length=m.patch_size, context_length=X.shape[1],
                             target_dim=1, feat_dynamic_real_dim=0, past_feat_dynamic_real_dim=0)
        x = torch.from_numpy(X)[:, :, None].to(self.device)
        with torch.inference_mode():
            t, om, sid, tid, vid, pm = fc._convert(m.patch_size, x, torch.ones_like(x, dtype=torch.bool),
                                                   torch.zeros(x.shape[:2], dtype=torch.bool, device=x.device))
            loc, scale = m.scaler(t, om * ~pm.unsqueeze(-1), sid, vid)
            tokens = torch.cat([(t - loc) / scale, om.to(torch.float32)], dim=-1)
            h = m.encoder(m.in_proj(tokens), packed_causal_attention_mask(sid, tid), time_id=tid, var_id=vid)
            keep = (~pm).float()[..., None]      # context tokens only
            z = (h * keep).sum(1) / keep.sum(1)
        return z.float().cpu().numpy()


REGISTRY = {"chronos2": Chronos2Emb, "chronos_bolt": ChronosBoltEmb, "tirex": TiRexEmb,
            "moment": MomentEmb, "mantis": MantisEmb, "ttm": TTMEmb,
            "timesfm": TimesFMEmb, "sundial": SundialEmb, "moirai2": Moirai2Emb}


def get(name: str, segment: bool = False, device: str = "cpu") -> Embedder:
    """Load a model by key (see config.MODELS). ``segment`` only matters for MOMENT and Mantis."""
    e = REGISTRY[name]()
    e.device = device
    if hasattr(e, "segment"):
        e.segment = segment
    return e.load()
