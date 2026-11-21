"""E6: light adaptation of the two best Task 1 models (MOMENT-1, TiRex) on the GPU (Apple mps).

Question: does adapting the backbone beat the paper's linear probe on held-out sensor
configurations (Task 1, GW-TCW)?

Same data, folds and sign symmetry as Task 1: the paper's 5 grouped folds; training on x and
-x; test score = mean of the probabilities for x and -x. Within each training fold, 20% of
the configurations (seed = fold) are held out for early stopping, so the test fold is never
used to choose anything.

Feature for the head: the mean over the signal's tokens of the last block's output (MOMENT:
its own 'embedding' output, on the signal resampled to 512 as in the paper; TiRex: the last
of its 12 sLSTM blocks, on the signal's own tokens). Variants:
  LP (sklearn)     the paper's logistic regression on these frozen features
  LP (sklearn, 80%) the same, fitted only on the 80% of the training configurations the
                   trained variants learn from (control: the others see 20% less data)
  LP (gradient)    a linear head trained with the loop below, backbone frozen (checks the loop)
  FT last 2        last 2 blocks unfrozen + linear head
  LoRA r=8         rank-8 adapters (peft) in the last 6 blocks + linear head; MOMENT: attention
                   q and v; TiRex: feed-forward projections (its sLSTM gates are custom layers)

TiRex pads every input to its 2,048-sample context with NaN; the first 3-4 blocks then get NaN
gradients (results/ext_mps/training.csv). Only the last 6 blocks are ever trained here, and
their gradients are finite.

Training (LP-then-FT): the head starts from the paper's logistic regression fitted on the
frozen features of the training configurations, so every variant starts at the linear
probe; this starting point is also scored on the validation configurations, and training
only replaces it if the validation AUC improves. Then AdamW (head 1e-3; backbone 1e-4 for
FT, 5e-4 for LoRA; weight decay 0.01), batch 32, binary cross-entropy, up to 20 epochs, the
epoch with the best validation AUC kept.

Outputs in results/ext_adapt/: folds.csv (test AUC per model, variant and fold), curves.csv

    python scripts/ext_adapt.py
"""

from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gwtsfm import embed  # noqa: E402
from gwtsfm.config import RESULTS  # noqa: E402
from embed import gwtcw_signals  # noqa: E402

import task1_gwtcw as t1  # noqa: E402

LORA = {"moment": ["q", "v"], "tirex": ["proj_up", "proj_up_gate", "proj_down"]}


def backbone(key, device):
    e = embed.get(key, device=device)
    net = e.m if key == "moment" else e.m.model
    for p in net.parameters():
        p.requires_grad_(False)
    return e, net


def blocks(key, net):
    return net.encoder.block if key == "moment" else net.blocks


def features(key, e, net, x):
    """Differentiable feature (batch, d) of signals x (batch, L) already on the device."""
    import torch
    if key == "moment":
        out = net(x_enc=x[:, None, :], input_mask=torch.ones(len(x), 512, dtype=torch.long, device=x.device))
        return out.embeddings
    fn = type(net)._embed_context.__wrapped__             # TiRex: the version without inference_mode
    h = fn(net, x)                                         # (batch, tokens, blocks, d), padding removed
    return h[:, :, -1].mean(1)                             # last block, mean over the signal's tokens


def set_trainable(key, net, variant):
    """Unfreeze the parts trained by a variant; returns the list of backbone parameters."""
    if variant == "FT last 2":
        params = [p for b in list(blocks(key, net))[-2:] for p in b.parameters()]
        for p in params:
            p.requires_grad_(True)
        return params
    if variant == "LoRA r=8":
        from peft import LoraConfig, inject_adapter_in_model
        n = len(blocks(key, net))
        prefix = "encoder.block" if key == "moment" else "blocks"
        targets = [name for name, _ in net.named_modules()
                   if any(name.endswith(t) for t in LORA[key])
                   and name.startswith(prefix) and int(name.split(".")[len(prefix.split("."))]) >= n - 6]
        inject_adapter_in_model(LoraConfig(r=8, lora_alpha=16, lora_dropout=0.0, target_modules=targets), net)
        params = [p for name, p in net.named_parameters() if "lora_" in name]
        for p in params:
            p.requires_grad_(True)
        return params
    return []


def inner_split(cfg, tr, fold):
    """Hold out 20% of the training configurations (seed = fold) for early stopping."""
    rng = np.random.default_rng(fold)
    cfgs = np.unique(cfg[tr])
    val_cfg = rng.choice(cfgs, int(round(0.2 * len(cfgs))), replace=False)
    return tr[np.isin(cfg[tr], val_cfg)], tr[~np.isin(cfg[tr], val_cfg)]


def train_eval(key, variant, X, y, cfg, tr, te, fold, device, epochs, log):
    import torch
    torch.manual_seed(fold)
    e, net = backbone(key, device)
    bb = set_trainable(key, net, variant)
    va, fit = inner_split(cfg, tr, fold)                   # grouped validation split
    Xt = torch.from_numpy(X).to(device)

    def feats(idx, sign, grad=False):
        with torch.set_grad_enabled(grad):
            return torch.cat([features(key, e, net, sign * Xt[idx[i:i + 64]]) for i in range(0, len(idx), 64)])

    net.train(False)
    with torch.no_grad():                                  # standardise with frozen training features
        f0 = torch.cat([feats(fit, 1.0), feats(fit, -1.0)])
        mu, sd = f0.mean(0), f0.std(0, unbiased=False) + 1e-6   # as StandardScaler
    head = torch.nn.Linear(f0.shape[1], 1).to(device)
    # the paper's head (StandardScaler + LogisticRegression(C=1)) on the frozen features
    lr0 = LogisticRegression(C=1.0, max_iter=4000).fit(((f0 - mu) / sd).cpu().numpy(), np.r_[y[fit], y[fit]])
    with torch.no_grad():                                  # start from the linear probe
        head.weight.copy_(torch.from_numpy(lr0.coef_).float())
        head.bias.copy_(torch.from_numpy(lr0.intercept_).float())
    groups = [{"params": head.parameters(), "lr": 1e-3}]
    if bb:
        groups.append({"params": bb, "lr": 1e-4 if variant == "FT last 2" else 5e-4})
    opt = torch.optim.AdamW(groups, weight_decay=0.01)
    lossf = torch.nn.BCEWithLogitsLoss()

    def score(idx):
        with torch.no_grad():
            p = [torch.sigmoid(head((feats(idx, s) - mu) / sd)).squeeze(1) for s in (1.0, -1.0)]
        return (0.5 * (p[0] + p[1])).cpu().numpy()

    train_idx = np.r_[fit, fit]
    train_sign = np.r_[np.ones(len(fit)), -np.ones(len(fit))]
    best = roc_auc_score(y[va], score(va))                # epoch 0 = the linear probe itself
    best_state = (copy.deepcopy(head.state_dict()), [p.detach().clone() for p in bb])
    log.append(dict(model=key, variant=variant, fold=fold, epoch=0, val_auc=best, loss=np.nan, seconds=0.0))
    for ep in range(1, epochs + 1):
        order = np.random.default_rng(1000 * fold + ep).permutation(len(train_idx))
        t0 = time.time()
        for i in range(0, len(order), 32):
            b = order[i:i + 32]
            xb = Xt[train_idx[b]] * torch.from_numpy(train_sign[b]).float().to(device)[:, None]
            logits = head((features(key, e, net, xb) - mu) / sd).squeeze(1)
            loss = lossf(logits, torch.from_numpy(y[train_idx[b]]).float().to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
        val = roc_auc_score(y[va], score(va))
        log.append(dict(model=key, variant=variant, fold=fold, epoch=ep, val_auc=val, loss=float(loss),
                        seconds=time.time() - t0))
        if val > best:
            best = val
            best_state = (copy.deepcopy(head.state_dict()), [p.detach().clone() for p in bb])
    head.load_state_dict(best_state[0])
    with torch.no_grad():
        for p, q in zip(bb, best_state[1]):
            p.copy_(q)
    return roc_auc_score(y[te], score(te)), best


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(RESULTS / "ext_adapt"))
    ap.add_argument("--device", default="mps")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--models", nargs="+", default=["moment", "tirex"])
    ap.add_argument("--probe-only", action="store_true",
                    help="only the two sklearn probes; replaces their rows in an existing folds.csv")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    import torch

    Xall, yall, cfgall = t1.load_gwtcw()
    keep = t1.first_repeat(len(Xall))
    y, cfg = yall[keep], cfgall[keep]
    X = gwtcw_signals()                                     # as embedded for the paper (unit std)
    assert np.allclose(X, Xall[keep])
    folds = t1.grouped_folds(cfg)
    rows, log = [], []
    for key in args.models:
        Xm = embed.resample(X, 512) if key == "moment" else X
        # the paper's head on the frozen feature used here
        e, net = backbone(key, args.device)
        with torch.no_grad():
            Xt = torch.from_numpy(Xm).to(args.device)
            Fp = torch.cat([features(key, e, net, Xt[i:i + 64]) for i in range(0, len(Xt), 64)]).cpu().numpy()
            Fn = torch.cat([features(key, e, net, -Xt[i:i + 64]) for i in range(0, len(Xt), 64)]).cpu().numpy()
        for i, (tr, te) in enumerate(folds):
            rows.append(dict(model=key, variant="LP (sklearn)", fold=i,
                             auc=roc_auc_score(y[te], t1.damage_score(t1.logistic, Fp, Fn, y, tr, te, True))))
            fit = inner_split(cfg, tr, i)[1]
            rows.append(dict(model=key, variant="LP (sklearn, 80%)", fold=i,
                             auc=roc_auc_score(y[te], t1.damage_score(t1.logistic, Fp, Fn, y, fit, te, True))))
        if args.probe_only:
            continue
        for variant in ["LP (gradient)", "FT last 2", "LoRA r=8"]:
            for i, (tr, te) in enumerate(folds):
                auc, val = train_eval(key, variant, Xm, y, cfg, tr, te, i, args.device, args.epochs, log)
                rows.append(dict(model=key, variant=variant, fold=i, auc=auc, best_val_auc=val))
                print(f"{key} {variant} fold {i}: test AUC {auc:.3f} (val {val:.3f})", flush=True)
                pd.DataFrame(rows).to_csv(out / "folds.csv", index=False)
                pd.DataFrame(log).to_csv(out / "curves.csv", index=False)
    R = pd.DataFrame(rows)
    if args.probe_only:                                    # keep the trained variants' rows
        old = pd.read_csv(out / "folds.csv")
        old = old[~(old.model.isin(args.models) & old.variant.str.startswith("LP (sklearn"))]
        R = pd.concat([R, old], ignore_index=True)
        R.to_csv(out / "folds.csv", index=False)
    print(R.groupby(["model", "variant"]).auc.agg(["mean", "std"]).round(3).to_string())


if __name__ == "__main__":
    main()
