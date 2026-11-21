"""Paths, model names and the random seed shared by every script."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"            # downloaded data, prepared arrays and embeddings (not in git)
RESULTS = ROOT / "results"      # outputs of the three task scripts (not in git)
EMBEDDINGS = DATA / "embeddings"

SEED = 0                        # every random choice in the project starts from this seed

# Model key (used in file names and on the command line) -> name shown in result tables.
# The six models of the paper. Tasks 1-3, make_tables.py and make_figures.py use only these,
# so models added later cannot change the paper's results.
PAPER_MODELS = {
    "chronos2": "Chronos-2",
    "chronos_bolt": "Chronos-Bolt (base)",
    "tirex": "TiRex",
    "moment": "MOMENT-1 (base)",
    "mantis": "Mantis-8M",
    "ttm": "TTM-R2",
}

# Every model with an Embedder in gwtsfm.embed: the paper's six and those added by the
# extended experiments (scripts/ext_*.py).
MODELS = {
    **PAPER_MODELS,
}
