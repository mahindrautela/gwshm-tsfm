# Frozen time series foundation models for guided-wave structural health monitoring

Code and results for the paper *Compare Like with Like: Frozen Time Series Foundation Models for
Guided-Wave Structural Health Monitoring* ([`paper/main.pdf`](paper/main.pdf), ICML 2026 format). The PDF and
an Overleaf-ready LaTeX source zip are also in [`docs/`](docs).

Six pretrained time series foundation models (Chronos-2, Chronos-Bolt, TiRex, MOMENT, Mantis, TTM) are
used **frozen**: each ultrasonic signal is turned into an embedding vector, and only a light head is
fitted on top. Nothing is trained or fine-tuned. The embeddings help when, as with classical baseline
subtraction, each signal is compared with healthy recordings of its own sensor path and the sensor
network is read as a whole.

## Results

| Task | Dataset | Headline |
|---|---|---|
| 1. Damage classification (supervised) | GW-TCW tapered composite wing | On held-out sensor configurations, logistic regression on frozen embeddings reaches AUC 0.83–0.86 for five of six models (TTM-R2 0.74); MiniRocket 0.90; 1D-CNN 0.69 |
| 2. Delamination detection (healthy data only) | NASA PCoE CFRP coupons, layup 2 | Same-path embedding distance: AUC 0.95–0.98 within a specimen for five of six models (Chronos-2 0.88), like baseline subtraction (0.97). With healthy references from other coupons, Mantis and Chronos-Bolt still flag all 4 damaged coupons at zero false alarms |
| 3. Damage-zone localisation (supervised) | Open Guided Waves, dataset 1 | One path at a time: near chance for every method. From the 36-path map of same-path distances: all six models locate all 28 held-out positions (three frequencies combined); residual energy 0.89 |

The tables behind the paper are in [`paper_results/`](paper_results).

## How the code is organised

```
public data ──prepare──▶ signal arrays ──embed──▶ embeddings ──task scripts──▶ results/ ──▶ paper/
```

**`gwtsfm/`**, the shared package:

| Module | What it holds |
|---|---|
| `config.py` | where data and results live, the six model names, the random seed |
| `embed.py` | the six frozen models: signals `(n, L)` → embeddings `(n, d)` |
| `heads.py` | logistic-regression and k-NN heads; unsupervised detectors (k-NN distance, PCA + one-class SVM) |
| `baselines.py` | references trained on the data: MiniRocket and the 1D-CNN |
| `ogw.py` | Open Guided Waves file reader, plate geometry, path-to-damage distance |

**`scripts/`**, one step each, in the order they run:

| Script | Step | Writes |
|---|---|---|
| `prepare_gwtcw.py` | download GW-TCW | `data/gwtcw/gwtcw.npz` |
| `prepare_nasa.py` | select the NASA measurements | `data/nasa/layup2.npz` |
| `fetch_ogw1.py` | download OGW-1 at 40, 60 and 80 kHz | `data/ogw1/ogw1_f*kHz.npz` |
| `prepare_ogw1.py` | damage and drift residuals | `data/ogw1/residuals.npz` |
| `embed.py` | frozen embeddings of every signal (the slow step) | `data/embeddings/<dataset>/<model>.np[yz]` |
| `task1_gwtcw.py` | Task 1: grouped cross-validation, label efficiency, leaky split | `results/task1_gwtcw/` |
| `task1_cnn_check.py` | Task 1: is the CNN's gap leakage or under-training? | printed |
| `task2_nasa.py` | Task 2: detection within and across specimens, damage growth | `results/task2_nasa/` |
| `task3_ogw1_paths.py` | Task 3: zone from one path at a time | `results/task3_ogw1/metrics.csv` |
| `task3_ogw1_network.py` | Task 3: zone from the 36-path damage-index map, control, RAPID, Spearman | `results/task3_ogw1/` |
| `audit_datasets.py` | dataset audits quoted in the paper | `paper_results/audits.json` |
| `export_results.py` | copy the result tables into git | `paper_results/` |

**`paper/`**: the LaTeX source, `make_tables.py` (every results table, straight from `results/`) and
`figures/make_figures.py`.

## Setup

Python 3.12, CPU only.

```bash
uv venv -p 3.12 .venv && source .venv/bin/activate
uv pip install -r requirements.txt
uv pip install --no-deps momentfm==0.1.4
```

## Reproduce

```bash
bash reproduce.sh
```

This runs every step above in order: downloads take about an hour, embeddings several hours and the
tasks about 1.5 hours on two CPU cores. Downloads and embeddings that already exist are skipped, and
every script can also be run on its own with default paths, for example `python scripts/task2_nasa.py`.

`python scripts/embed.py --device mps` (or `cuda`) embeds on a GPU; cpu stays the default.
[`REPRODUCTION.md`](REPRODUCTION.md) compares a full rerun with the paper. Further experiments
(temperature, three more models, other NASA layups, significance tests, coordinate regression, light
adaptation) are in `scripts/ext_*.py` and reported in [`EXTENSIONS.md`](EXTENSIONS.md), which also lists
the extra environments they need.

The earlier exploratory study, which used the models as forecasters along the temperature axis of
another OGW dataset, is kept on the
[`forecasting-archive`](https://github.com/mahindrautela/gwshm-tsfm/tree/forecasting-archive) branch.
