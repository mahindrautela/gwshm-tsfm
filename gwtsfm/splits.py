"""Grouped k-fold splits that are the same on every platform.

scikit-learn's GroupKFold (without shuffling) hands the groups out largest first, each to
the currently lightest fold. When groups have equal sizes, as all of ours do, their order
comes from ``np.argsort``, whose tie order depends on the NumPy build: the x86 kernels
used for the paper keep ties in index order, the ARM kernels (Apple Silicon) do not. The
folds then differ between machines although the seed is the same.

``grouped_kfold`` is the same algorithm with a stable sort. It gives the folds behind the
paper's results on every platform.
"""

from __future__ import annotations

import numpy as np


def grouped_kfold(groups: np.ndarray, k: int = 5) -> list[tuple[np.ndarray, np.ndarray]]:
    """(train, test) indices of k folds; all rows of a group fall in the same fold.

    Identical to ``list(GroupKFold(k).split(X, groups=groups))`` with ties between
    equal-sized groups broken by group order.
    """
    _, idx = np.unique(groups, return_inverse=True)
    size = np.bincount(idx)
    order = np.argsort(size, kind="stable")[::-1]      # largest groups first
    load = np.zeros(k)                                  # rows per fold so far
    fold_of = np.zeros(len(size), int)
    for g in order:                                     # next group -> lightest fold
        f = int(np.argmin(load))
        load[f] += size[g]
        fold_of[g] = f
    fold = fold_of[idx]
    return [(np.flatnonzero(fold != f), np.flatnonzero(fold == f)) for f in range(k)]
