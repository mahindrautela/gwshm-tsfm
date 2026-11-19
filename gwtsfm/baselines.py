"""References that are trained on the target data: MiniRocket and a 1D-CNN.

They show what a model fitted to the guided-wave data itself achieves under the same
splits as the frozen foundation models.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import RidgeClassifierCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import SEED


def sigmoid(d: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-d))


def softmax(d: np.ndarray) -> np.ndarray:
    e = np.exp(d - d.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


# ------------------------------------------------------------------------- MiniRocket
def minirocket_decision(X, y, train, test, extra=None, sign_symmetric=False) -> np.ndarray:
    """MiniRocket features + ridge classifier; returns decision values for the test rows.

    X: signals (n, L). ``extra``: optional features appended to the MiniRocket features
    (Task 3 appends the one-hot path and frequency). ``sign_symmetric``: also train on -x
    and average the decision over x and -x (Task 1). Decision values are 1-D for two
    classes and (n_test, n_classes) otherwise.
    """
    from aeon.transformations.collection.convolution_based import MiniRocket

    Xtr, ytr = X[train], y[train]
    if sign_symmetric:
        Xtr, ytr = np.r_[Xtr, -Xtr], np.r_[ytr, ytr]
    rocket = MiniRocket(random_state=SEED).fit(Xtr[:, None, :])

    def features(A, rows):
        Fa = rocket.transform(A[:, None, :])
        return Fa if extra is None else np.c_[Fa, extra[rows]]

    clf = make_pipeline(StandardScaler(), RidgeClassifierCV(alphas=np.logspace(-3, 3, 10)))
    clf.fit(features(Xtr, train), ytr)
    d = clf.decision_function(features(X[test], test))
    if sign_symmetric:
        d = 0.5 * (d + clf.decision_function(features(-X[test], test)))
    return d


# ------------------------------------------------------------------------------ 1D-CNN
def cnn_1d(length: int, n_outputs: int):
    """The 1D-CNN of the GW-TCW reference notebook (ported to PyTorch).

    Four blocks of convolution (kernel 3, no padding) and max-pooling with 16, 32, 64 and
    128 filters, then dense layers of 128 and 16 units.
    """
    import torch.nn as nn

    flat = length
    for _ in range(4):           # each block: conv shortens by 2, pooling halves
        flat = (flat - 2) // 2
    return nn.Sequential(
        nn.Conv1d(1, 16, 3), nn.MaxPool1d(2),
        nn.Conv1d(16, 32, 3), nn.ReLU(), nn.MaxPool1d(2),
        nn.Conv1d(32, 64, 3), nn.ReLU(), nn.MaxPool1d(2),
        nn.Conv1d(64, 128, 3), nn.ReLU(), nn.MaxPool1d(2),
        nn.Flatten(), nn.Linear(128 * flat, 128), nn.ReLU(),
        nn.Linear(128, 16), nn.ReLU(), nn.Linear(16, n_outputs))


def train_cnn(X, y, train, test, groups, n_classes=2, epochs=40, sign_symmetric=False,
              rescale=False, seed=SEED) -> np.ndarray:
    """Train the 1D-CNN on the training rows and score the test rows.

    Early stopping: 15% of the training *groups* (e.g. sensor configurations) are held
    out, and the weights with the lowest validation loss over ``epochs`` are kept. Pass
    one group per row to draw the validation rows at random instead.

    n_classes == 2   one output, binary cross-entropy; returns P(class 1), shape (n_test,)
    n_classes > 2    one output per class, cross-entropy; returns probabilities (n_test, C)
    sign_symmetric   also train on -x and average the test score over x and -x (Task 1)
    rescale          divide signals by the standard deviation of the fitting rows (Task 3)
    """
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    torch.set_num_threads(2)
    binary = n_classes == 2
    net = cnn_1d(X.shape[1], 1 if binary else n_classes)

    # hold out whole groups for early stopping
    rng = np.random.default_rng(seed)
    g = np.unique(groups[train])
    val_groups = rng.choice(g, max(1, int(0.15 * len(g))), replace=False)
    val = train[np.isin(groups[train], val_groups)]
    fit = train[~np.isin(groups[train], val_groups)]

    scale = X[fit].std() + 1e-12 if rescale else 1.0
    Xf, yf = X[fit], y[fit]
    if sign_symmetric:
        Xf, yf = np.r_[Xf, -Xf], np.r_[yf, yf]

    def tensor(A):  # (n, L) array -> (n, 1, L) float tensor
        return torch.from_numpy(np.ascontiguousarray(A / scale, dtype=np.float32))[:, None, :]

    def target(labels):
        return torch.from_numpy(labels.astype(np.float32 if binary else np.int64))

    def loss_on(A, labels):
        out = net(tensor(A))
        return loss_fn(out[:, 0], target(labels)) if binary else loss_fn(out, target(labels))

    loss_fn = nn.BCEWithLogitsLoss() if binary else nn.CrossEntropyLoss()
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    best, best_state = np.inf, None
    for _ in range(epochs):
        net.train()
        order = rng.permutation(len(Xf))
        for i in range(0, len(order), 32):          # mini-batches of 32
            b = order[i:i + 32]
            opt.zero_grad()
            loss_on(Xf[b], yf[b]).backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            val_loss = loss_on(X[val], y[val]).item()
        if val_loss < best:
            best, best_state = val_loss, {k: v.clone() for k, v in net.state_dict().items()}
    net.load_state_dict(best_state)

    with torch.no_grad():
        if not binary:
            return torch.softmax(net(tensor(X[test])), 1).numpy()
        s = torch.sigmoid(net(tensor(X[test]))[:, 0]).numpy()
        if sign_symmetric:
            s = 0.5 * (s + torch.sigmoid(net(tensor(-X[test]))[:, 0]).numpy())
        return s
