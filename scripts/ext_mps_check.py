"""Extended experiments: does Apple's GPU backend (torch "mps") give the same embeddings as cpu?

Three checks for each of the paper's six models:
  1. numerics  relative difference ||z_mps - z_cpu|| / ||z_cpu|| per signal, on 256 real
               signals of each dataset (GW-TCW, NASA, OGW-1 residuals), and the speed of both
  2. downstream  NASA (Task 2) embedded completely on mps; the same-path detector is run on
               the cpu and the mps embeddings with the same reference draws, and the AUCs compared
  3. training  one forward and backward pass of MOMENT and TiRex (the two models adapted in
               E6) on mps; gradients compared with cpu. TiRex pads inputs shorter than its
               2048-sample training context with NaN, which makes part of its gradient NaN on
               every device; the comparison uses the finite entries and reports their share.

Mantis is refused on mps by gwtsfm.embed; here it is built directly to show why.

Outputs in results/ext_mps/: numerics.csv, downstream_task2.csv, training.csv
mps embeddings of NASA in data/embeddings_mps/nasa/<model>.npy

    python scripts/ext_mps_check.py
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gwtsfm import embed  # noqa: E402
from gwtsfm.config import DATA, EMBEDDINGS, PAPER_MODELS, RESULTS  # noqa: E402
from embed import array_signals, gwtcw_signals  # noqa: E402  (scripts/embed.py)
from task2_nasa import DAMAGED_CYCLES, HEALTHY_MAX_CYCLES, evaluation_plans, same_path_distance  # noqa: E402

OUT = RESULTS / "ext_mps"
MPS_EMB = DATA / "embeddings_mps"


def get_model(key, segment, device):
    """embed.get, except that Mantis is also built on mps (bypassing the guard on purpose)."""
    if key == "mantis" and device == "mps":
        import torch  # noqa: F401
        from mantis.architecture import Mantis8M
        from mantis.trainer import MantisTrainer
        e = embed.MantisEmb()
        e.device, e.segment = "mps", segment
        e.tr = MantisTrainer(device="mps", network=Mantis8M(device="mps").from_pretrained(e.hf_id))
        return e
    return embed.get(key, segment=segment, device=device)


def timed(model, X):
    model(X[:8])                       # warm-up (kernel compilation on mps)
    Z, t = model(X)
    return Z, t


def numerics(n=256):
    rng = np.random.default_rng(0)
    data = {"gwtcw": (gwtcw_signals(), False)}
    for d in ["nasa", "ogw1_residuals"]:
        X, _ = array_signals(d)
        data[d] = (X, True)
    rows = []
    for key in PAPER_MODELS:
        for d, (X, seg) in data.items():
            Xs = X[np.sort(rng.choice(len(X), min(n, len(X)), replace=False))]
            Zc, tc = timed(get_model(key, seg, "cpu"), Xs)
            Zm, tm = timed(get_model(key, seg, "mps"), Xs)
            rel = np.linalg.norm(Zm - Zc, axis=1) / np.linalg.norm(Zc, axis=1)
            rows.append(dict(model=PAPER_MODELS[key], dataset=d, n=len(Xs), rel_median=np.median(rel),
                             rel_max=rel.max(), ms_per_signal_cpu=1e3 * tc / len(Xs),
                             ms_per_signal_mps=1e3 * tm / len(Xs)))
            print(rows[-1], flush=True)
    return pd.DataFrame(rows)


def downstream():
    """Task 2 same-path detector on cpu vs mps embeddings of NASA (same reference draws)."""
    z = np.load(DATA / "nasa" / "layup2.npz")
    specimen, cycles = z["specimen"], z["cycles"]
    healthy = cycles <= HEALTHY_MAX_CYCLES
    plans = evaluation_plans(specimen, healthy, 5, np.random.default_rng(0))
    X, lead = array_signals("nasa")
    (MPS_EMB / "nasa").mkdir(parents=True, exist_ok=True)
    rows = []
    for key, name in PAPER_MODELS.items():
        if key == "mantis":
            continue                   # wrong on mps (see numerics.csv)
        f = MPS_EMB / "nasa" / f"{key}.npy"
        if not f.exists():
            Z, t = get_model(key, True, "mps")(X)
            np.save(f, Z.reshape(*lead, -1).astype(np.float16))
            print(f"{key}: NASA on mps in {t / 60:.1f} min", flush=True)
        F = {"cpu": np.load(EMBEDDINGS / "nasa" / f"{key}.npy").astype(np.float32),
             "mps": np.load(f).astype(np.float32)}
        for setting, draw, s, ref, test in plans:
            lab = np.where(cycles[test] <= HEALTHY_MAX_CYCLES, 0, np.where(cycles[test] == DAMAGED_CYCLES, 1, -1))
            m = lab >= 0
            row = dict(model=name, setting=setting, draw=draw, specimen=s)
            for dev, Fd in F.items():
                S = same_path_distance(Fd[ref], Fd[test])
                y = np.repeat(lab[m], S.shape[1] * S.shape[2])
                row[f"auc_{dev}"] = roc_auc_score(y, S[m].ravel())
                meas = S.reshape(len(test), -1).mean(1)
                row[f"detected_{dev}"] = float(meas[lab == 1].min() > meas[lab == 0].max())
            rows.append(row)
    return pd.DataFrame(rows)


def training():
    """One forward/backward pass of MOMENT and TiRex on cpu and on mps: same gradients?"""
    import torch
    X, _ = array_signals("nasa")
    x = X[np.random.default_rng(1).choice(len(X), 16, replace=False)][:, :512]
    rows = []
    for key in ["moment", "tirex"]:
        grads, secs = {}, {}
        for dev in ["cpu", "mps"]:
            e = embed.get(key, device=dev)
            net = e.m if key == "moment" else e.m.model
            net.train(False)
            for p in net.parameters():
                p.requires_grad_(True)
            net.zero_grad()
            xt = torch.from_numpy(x).to(dev)
            t0 = time.time()
            if key == "moment":
                out = e.m(x_enc=xt[:, None, :], input_mask=torch.ones(len(x), 512, dtype=torch.long, device=dev))
                loss = out.embeddings.pow(2).mean()
            else:
                # _embed_context is wrapped in torch.inference_mode(); call the undecorated function
                fn = type(e.m.model)._embed_context.__wrapped__
                loss = fn(e.m.model, xt).pow(2).mean()
            loss.backward()
            if dev == "mps":
                torch.mps.synchronize()
            secs[dev] = time.time() - t0
            grads[dev] = torch.cat([p.grad.detach().float().cpu().ravel() for p in net.parameters()
                                    if p.grad is not None])
        ok = torch.isfinite(grads["cpu"]) & torch.isfinite(grads["mps"])
        g0, g1 = grads["cpu"][ok], grads["mps"][ok]
        rows.append(dict(model=PAPER_MODELS[key], n_params_with_grad=len(ok),
                         finite_cpu=float(torch.isfinite(grads["cpu"]).float().mean()),
                         finite_mps=float(torch.isfinite(grads["mps"]).float().mean()),
                         grad_rel_diff=float((g1 - g0).norm() / g0.norm()),
                         grad_cosine=float(torch.nn.functional.cosine_similarity(g0, g1, dim=0)),
                         step_s_cpu=secs["cpu"], step_s_mps=secs["mps"]))
        print(rows[-1], flush=True)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="*", default=["numerics", "downstream", "training"])
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if "numerics" in args.only:
        numerics().to_csv(OUT / "numerics.csv", index=False)
    if "training" in args.only:
        training().to_csv(OUT / "training.csv", index=False)
    if "downstream" in args.only:
        D = downstream()
        D.to_csv(OUT / "downstream_task2.csv", index=False)
        print(D.groupby(["model", "setting"])[["auc_cpu", "auc_mps", "detected_cpu", "detected_mps"]]
              .mean().round(4).to_string())


if __name__ == "__main__":
    main()
