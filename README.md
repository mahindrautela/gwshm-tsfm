# Zero-shot guided-wave damage detection with time series foundation models

Benchmark of time series foundation models (TSFMs), used strictly zero-shot,
for temperature-compensated damage detection with ultrasonic guided waves on the
Open Guided Waves (OGW) CFRP temperature dataset (Moll et al., *Sci. Data* 6:191, 2019).

> **Status:** pipeline implemented and tested on synthetic data in the OGW file layout
> (`pytest tests`). Results on the real OGW data are pending.

## Idea

For one transducer pair and excitation frequency the received signal is x(t, T).
At each fixed time sample t_n, the pristine references ordered by temperature,
y_n[k] = x(t_n, T_k), form an ordinary, quasi-periodic time series on a 0.5 °C grid.
Given pristine references only up to a cut-off temperature T_c, each TSFM forecasts
all y_n beyond T_c. The forecast at the measured temperature of a new signal is the
predicted pristine signal; the prediction error is the damage index (DI). No model is
trained or fine-tuned.

Damage indices: residual energy ratio, 1 − correlation, and (for probabilistic
models) an uncertainty-normalised residual that uses the 10–90 % forecast interval.

Baselines on the same references: OBS, OBS + baseline signal stretch (BSS),
physics-based stretch extrapolation, last value, linear trend, AutoETS.

## Models

| key | model | weights |
|---|---|---|
| chronos2 | Chronos-2 | amazon/chronos-2 |
| chronos_bolt | Chronos-Bolt base | amazon/chronos-bolt-base |
| timesfm25 | TimesFM-2.5 | google/timesfm-2.5-200m-pytorch |
| timesfm3 | TimesFM-3 (non-commercial weights) | google/timesfm-3.0-pytorch |
| tirex | TiRex | NX-AI/TiRex |
| tirex2 | TiRex-2 (gated on Hugging Face) | NX-AI/TiRex-2 |
| toto2 | Toto-2.0 22M | Datadog/Toto-2.0-22m |
| moirai2 | Moirai-2.0 small | Salesforce/moirai-2.0-R-small |
| ttm | TTM-R2 (point forecasts) | ibm-granite/granite-timeseries-ttm-r2 |

## Setup

Python 3.12, CPU is enough.

```bash
uv venv -p 3.12 .venv && source .venv/bin/activate
uv pip install "torch==2.11.*" "numpy>=2.1.3,<2.4" scipy pandas h5py scikit-learn matplotlib \
    remotezip statsforecast pyarrow tabulate "chronos-forecasting>=2.3" tirex-ts tirex-2 granite-tsfm
git clone https://github.com/google-research/timesfm && uv pip install -e timesfm      # TimesFM 2.5 + 3
git clone https://github.com/DataDog/toto && uv pip install --no-deps -e toto/dd_unit_scaling -e toto/toto2
uv pip install --no-deps unit-scaling docstring_parser
git clone https://github.com/SalesforceAIResearch/uni2ts && uv pip install --no-deps -e uni2ts
uv pip install einops jaxtyping "gluonts[torch]>=0.16" lightning hydra-core python-dotenv orjson multiprocess safetensors
```

(The `--no-deps` installs relax version pins of Toto 2 and uni2ts that conflict with the
other packages; both import and run with torch 2.11.)

## Run

```bash
python scripts/download_ogw.py   --root data/raw   --freqs 40 100 160      # figshare, range requests
python scripts/prepare_cache.py  --root data/raw   --out data/cache --freqs 40 100 160
python scripts/run_benchmark.py  --cache data/cache --out results --freqs 40 100 160 \
       --tc 30 40 50 --methods baselines last linear ets chronos2 chronos_bolt timesfm25 \
       timesfm3 tirex toto2 moirai2 ttm
python scripts/summarize.py      --results results --cache data/cache --out report
```

`run_benchmark.py` is resumable: finished (method, frequency, cut-off) chunks are skipped.

Pipeline check without the real data:

```bash
python -c "from gwtsfm.synth import write_dataset; write_dataset('data/synth_raw', freqs_khz=(100,))"
python scripts/prepare_cache.py --root data/synth_raw --out data/synth_cache --freqs 100
python scripts/run_benchmark.py --cache data/synth_cache --out results_synth --freqs 100 --tc 30 45
```

## Layout

```
gwtsfm/ogw.py        dataset I/O, pair indexing, decimation, reference/test split
gwtsfm/synth.py      synthetic data in the OGW file layout (testing only)
gwtsfm/dsp.py        analysis window, time stretch, damage indices
gwtsfm/baselines.py  OBS, OBS+BSS, stretch extrapolation
gwtsfm/models.py     unified zero-shot forecaster wrappers
gwtsfm/protocol.py   temperature-axis forecasting protocol
gwtsfm/metrics.py    AUC (per path, fused), AUC vs gap, TPR@FPR, forecast error
scripts/             download, cache, run, summarize
```
