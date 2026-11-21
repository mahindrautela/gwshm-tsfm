"""Token states of every layer, pooled in several ways (E2 pooling ablation).

The embedders in gwtsfm.embed reduce a model to one vector, mostly by averaging the final
encoder tokens. Here forward hooks on the transformer (or xLSTM, MLP-mixer) blocks record
the residual stream after every block during the same forward pass, and each block's
tokens are reduced three ways:

    mean   average over the tokens
    max    elementwise maximum over the tokens
    last   the last token (the natural summary of causal models: TimesFM, Sundial,
           Moirai, TiRex)

Block outputs are taken before any final normalisation of the model, so "final block,
mean" is close to, but not the same as, the model's own embedding.

For TiRex, which left-pads every signal to its 2048-sample context, only the tokens of the
signal itself are kept (the last ceil(L / 32)), as TiRexEmbedding does. For MOMENT and
Mantis with ``segment=True``, each 512-sample window is pooled and the windows are
concatenated, as in their embedders.

Usage:
    with LayerPool(key, embedder) as lp:
        Z_model, _ = embedder(X)          # the model's own embedding, as before
    pooled = lp.result(len(X))            # {(layer, how): (n, d)}
"""

from __future__ import annotations

import dataclasses

import numpy as np

HOW = ("mean", "max", "last")


def _blocks(key, e):
    """(modules, post) for a loaded embedder: the blocks whose output is the residual
    stream, and a function (module, inputs, output) -> tensor (batch, tokens, d)."""
    first = _first_tensor
    if key in ("chronos2", "chronos_bolt"):
        return list(e.pipe.model.encoder.block), lambda m, i, o: first(o)
    if key == "moment":
        return list(e.m.encoder.block), lambda m, i, o: first(o)
    if key == "mantis":
        # ViT loop: x = attn(x) + x; x = ff(x) + x. The residual stream is ff's output + input.
        return [layer[1] for layer in e.tr.network.vit_unit.transformer.layers], \
            lambda m, i, o: first(o) + i[0]
    if key == "ttm":
        # (batch, channels, patches, d) -> (batch * channels, patches, d)
        return list(e.m.backbone.encoder.mlp_mixer_encoder.mixers), \
            lambda m, i, o: first(o).flatten(0, 1)
    if key == "tirex":
        return list(e.m.model.blocks), lambda m, i, o: first(o)
    if key == "timesfm":
        return list(e.m.stacked_xf), lambda m, i, o: first(o)
    if key == "sundial":
        return list(e.m.layers), lambda m, i, o: first(o)
    if key == "moirai2":
        return list(e.mod.encoder.layers), lambda m, i, o: first(o)
    raise KeyError(key)


def _first_tensor(o):
    import torch
    if torch.is_tensor(o):
        return o
    if isinstance(o, (tuple, list)):
        return _first_tensor(o[0])
    if dataclasses.is_dataclass(o):
        return _first_tensor(getattr(o, dataclasses.fields(o)[0].name))
    if hasattr(o, "values"):
        return _first_tensor(list(o.values())[0])
    raise TypeError(type(o))


class LayerPool:
    """Context manager that pools every block's tokens while the embedder runs."""

    def __init__(self, key: str, embedder, signal_length: int | None = None):
        self.key, self.e = key, embedder
        self.blocks, self.post = _blocks(key, embedder)
        self.keep = None
        if key == "tirex":                       # tokens of the signal, not of the padding
            self.keep = -(-signal_length // 32)
        self.store = {(l, h): [] for l in range(len(self.blocks)) for h in HOW}
        self.handles = []

    def __enter__(self):
        for l, b in enumerate(self.blocks):
            self.handles.append(b.register_forward_hook(self._hook(l)))
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()

    def _hook(self, layer):
        def hook(module, inputs, output):        # returns None: the output is not modified
            t = self.post(module, inputs, output).detach().float()
            if self.keep is not None:
                t = t[:, -self.keep:]
            self.store[(layer, "mean")].append(t.mean(1).cpu().numpy())
            self.store[(layer, "max")].append(t.amax(1).cpu().numpy())
            self.store[(layer, "last")].append(t[:, -1].cpu().numpy())
        return hook

    def result(self, n_signals: int) -> dict[tuple[int, str], np.ndarray]:
        """{(block index, how): (n_signals, d)}; windows of one signal are concatenated."""
        return {k: np.concatenate(v).reshape(n_signals, -1).astype(np.float32)
                for k, v in self.store.items()}

    @property
    def n_layers(self) -> int:
        return len(self.blocks)
