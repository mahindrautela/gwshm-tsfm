# Archived: forecasting along the temperature axis (not part of the study)

An early exploratory experiment on the OGW CFRP temperature dataset (Moll et al. 2019,
Sci. Data): pristine signals ordered by temperature were treated as time series and
forecast zero-shot beyond a cut-off temperature, the forecast serving as a virtual
baseline. Physics-based stretch compensation was far more accurate than the forecasting
models (Chronos-2 matched a linear trend), and forecasting was dropped as a framing for
SHM. Kept for reference only. Run scripts from the repository root, e.g.
`python archive/forecasting/scripts/run_benchmark.py ...`.
