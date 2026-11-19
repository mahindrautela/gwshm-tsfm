# Frozen time series foundation models for guided-wave structural health monitoring

Benchmark of six pretrained time series foundation models (TSFMs) used **frozen**, as
feature extractors with only a light head, on three guided-wave SHM tasks:

| Task | Dataset | Setting |
|---|---|---|
| 1. Damage classification | GW-TCW tapered composite wing (Rautela et al. 2022) | supervised, test sensor paths unseen in training |
| 2. Delamination detection | NASA PCoE CFRP fatigue coupons, layup 2 | unsupervised: only healthy data is used |
| 3. Damage-zone localisation | Open Guided Waves dataset 1 (Moll et al. 2019), 28 positions in 7 zones | supervised; random, grouped and unseen-position splits; network-level (36-path) localisation |

Models: Chronos-2, Chronos-Bolt (base), TiRex, MOMENT-1 (base), Mantis-8M, TTM-R2.
No model is trained or fine-tuned.

## Results (frozen models, no fine-tuning)

| Task | Headline |
|---|---|
| 1. GW-TCW classification, unseen sensor paths | logistic regression on frozen embeddings: AUC 0.83–0.86 for five of six models (MOMENT, TiRex 0.86; TTM-R2 0.74); MiniRocket 0.90; 1D-CNN 0.69 (1.00 on the leaky random split) |
| 2. NASA delamination, healthy data only | same-path embedding distance: AUC 0.95–0.98 within a specimen for five of six models (Chronos-2 0.88), baseline subtraction 0.97; with references from other coupons Mantis (0.81) and Chronos-Bolt (0.67) flag all 4 damaged coupons at zero false alarms, raw methods ≤ 3/4 |
| 3. OGW-1 localisation, unseen damage positions | per path: near chance for every method; network level (36-path damage-index map): all six models localise 28/28 positions with 40/60/80 kHz combined, MOMENT and TiRex at every single frequency; physics residual energy 0.89 (0.25 at 80 kHz) |

Result tables behind the paper are in [`paper_results/`](paper_results).

## Setup

Python 3.12, CPU only.

```bash
uv venv -p 3.12 .venv && source .venv/bin/activate
uv pip install -r requirements.txt
uv pip install --no-deps momentfm && uv pip install mantis-tsfm aeon
```

## Data

```bash
# Task 1: GW-TCW (Hugging Face)
for f in IISHM_BaselineExpData.txt IISHM_DamageExpData.txt IISHM_timevector.txt; do
  curl -L -o data/gwtcw/$f https://huggingface.co/datasets/mahindrautela/GW-TCW/resolve/main/$f; done
python -c "import numpy as np; d='data/gwtcw/'; np.savez(d+'gwtcw.npz', \
  B=np.loadtxt(d+'IISHM_BaselineExpData.txt',delimiter=',').astype('f4'), \
  D=np.loadtxt(d+'IISHM_DamageExpData.txt',delimiter=',').astype('f4'), t=np.loadtxt(d+'IISHM_timevector.txt'))"

# Task 2: NASA PCoE composites (4.6 GB), then extract layup 2
curl -L -o data_ext/Composites.zip "https://phm-datasets.s3.amazonaws.com/NASA/2.+Composites.zip"
unzip -p data_ext/Composites.zip "2. Composites/Layup2.zip" > data_ext/Layup2.zip
python scripts/prepare_nasa.py --zip data_ext/Layup2.zip --out data/nasa/l2.npz

# Task 3: OGW dataset 1 (40/60/80 kHz streamed from the OGW Nextcloud share), residuals
python scripts/fetch_ogw1.py --out data/ogw1 --freqs 40 60 80
python scripts/prepare_ogw1_loc.py --cache data/ogw1 --out data/ogw1/loc.npz
```

## Run

```bash
# frozen embeddings
python scripts/gwtcw_embed.py   --data data/gwtcw/gwtcw.npz --out data/gwtcw_emb
python scripts/embed_signals.py --npz data/nasa/l2.npz  --key X --length 1024 --segment --out data/nasa_emb
python scripts/embed_signals.py --npz data/ogw1/loc.npz --key R --segment --out data/ogw1_emb_R
python scripts/embed_signals.py --npz data/ogw1/loc.npz --key B --segment --out data/ogw1_emb_B

# tasks
python scripts/gwtcw_classify.py --data data/gwtcw/gwtcw.npz --emb data/gwtcw_emb --out results_gwtcw
python scripts/nasa_anomaly.py   --data data/nasa/l2.npz --emb data/nasa_emb --out results_nasa
python scripts/ogw1_localize.py  --data data/ogw1/loc.npz --emb data/ogw1_emb_R --out results_ogw1
python scripts/ogw1_network.py   --data data/ogw1/loc.npz --embR data/ogw1_emb_R --embB data/ogw1_emb_B --out results_ogw1_network
python scripts/ogw1_spearman.py  --loc data/ogw1/loc.npz --di results_ogw1_network/di_maps.npz --out results_ogw1_network/spearman.csv

# tables, figures, and the copy kept in git
python scripts/summarize_gwtcw.py --results results_gwtcw --out report_gwtcw
python scripts/summarize_nasa.py  --results results_nasa
python scripts/export_results.py
```

## Layout

```
gwtsfm/embed.py      frozen TSFM embedders (mean-pooled tokens; MOMENT/Mantis on 512-sample windows)
gwtsfm/evalkit.py    heads (logistic regression, k-NN), one-class detectors, MiniRocket, 1D-CNN
gwtsfm/ogw.py        OGW H5 reader (channel order, sampling rate, excitation onset), decimation
scripts/             data preparation, embedding extraction, the three task evaluations
paper_results/       result tables behind the paper (per-fold metrics, summaries)
archive/forecasting/ an earlier exploratory study (forecasting along the temperature axis), not used
```
