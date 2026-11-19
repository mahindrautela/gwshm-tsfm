#!/usr/bin/env bash
# Reproduce every result of the paper, in order. Run from the repository root:
#     bash reproduce.sh
# Downloads and embeddings that already exist are skipped, so the script can be restarted.
# Rough CPU times (2 cores): downloads ~1 h, embeddings several hours, tasks ~1.5 h.
set -euo pipefail
PY=${PY:-python}

# 1. Data -------------------------------------------------------------------------------
$PY scripts/prepare_gwtcw.py                       # Task 1: GW-TCW from Hugging Face

mkdir -p data/nasa                                 # Task 2: NASA PCoE composites (4.6 GB)
if [ ! -f data/nasa/layup2.npz ]; then
  [ -f data/nasa/Layup2.zip ] || {
    curl -L -o data/nasa/Composites.zip "https://phm-datasets.s3.amazonaws.com/NASA/2.+Composites.zip"
    unzip -p data/nasa/Composites.zip "2. Composites/Layup2.zip" > data/nasa/Layup2.zip; }
  $PY scripts/prepare_nasa.py
fi

$PY scripts/fetch_ogw1.py                          # Task 3: OGW dataset 1, 40/60/80 kHz
$PY scripts/prepare_ogw1.py                        # damage and drift residuals

# 2. Frozen embeddings (the slow step) ----------------------------------------------------
$PY scripts/embed.py nasa ogw1_residuals ogw1_baselines
$PY scripts/embed.py gwtcw --models chronos2 chronos_bolt tirex
$PY scripts/embed.py gwtcw --models moment mantis ttm --all-repeats   # also used by the leaky-split check

# 3. The three tasks ---------------------------------------------------------------------
$PY scripts/task1_gwtcw.py
$PY scripts/task1_cnn_check.py
$PY scripts/task2_nasa.py
$PY scripts/task3_ogw1_paths.py
$PY scripts/task3_ogw1_network.py

# 4. Audits, result tables kept in git, and the paper -------------------------------------
$PY scripts/audit_datasets.py
$PY scripts/export_results.py
$PY paper/make_tables.py
$PY paper/figures/make_figures.py
(cd paper && latexmk -pdf main.tex)
