"""Frozen time series foundation models (TSFMs) for guided-wave structural health monitoring.

Modules
-------
config     where data and results live, model names, the random seed
embed      the six frozen TSFMs: signal -> embedding vector (no training)
heads      what is fitted on top: logistic regression, k-NN, and unsupervised detectors
baselines  references trained on the target data: MiniRocket and a 1D-CNN
ogw        Open Guided Waves file reader and the plate geometry used in Task 3
splits     grouped k-fold splits that are the same on every platform
"""

__version__ = "1.0.0"
