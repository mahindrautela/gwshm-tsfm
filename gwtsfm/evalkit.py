"""Shared heads, detectors and trained-from-scratch references for the three tasks."""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, RidgeClassifierCV
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

SEED = 0


# ----------------------------------------------------------------- supervised heads
def lr(C: float = 1.0):
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=4000))


def knn(k: int = 5):
    return make_pipeline(StandardScaler(), KNeighborsClassifier(k, metric="cosine"))


def fit_proba(make, F, y, tr, te):
    m = make().fit(F[tr], y[tr])
    return m.predict_proba(F[te])


# ------------------------------------------------------------ unsupervised detectors
def knn_distance(ref: np.ndarray, test: np.ndarray, k: int = 1) -> np.ndarray:
    """Mean distance to the k nearest reference vectors (standardised on the reference)."""
    sc = StandardScaler().fit(ref)
    nn = NearestNeighbors(n_neighbors=k).fit(sc.transform(ref))
    return nn.kneighbors(sc.transform(test))[0].mean(1)


def pca_ocsvm(ref: np.ndarray, test: np.ndarray, n: int = 10, nu: float = 0.1) -> np.ndarray:
    """PCA + one-class SVM (RBF), as in Rautela et al. (2022); higher = more anomalous."""
    sc = StandardScaler().fit(ref)
    p = PCA(min(n, len(ref) - 1, ref.shape[1]), random_state=SEED).fit(sc.transform(ref))
    z = lambda A: p.transform(sc.transform(A))  # noqa: E731
    oc = OneClassSVM(nu=nu, gamma="scale").fit(z(ref))
    return -oc.decision_function(z(test))


def pca_recon(ref: np.ndarray, test: np.ndarray, n: int = 10) -> np.ndarray:
    """Reconstruction error of a PCA subspace fit on the reference (a linear autoencoder)."""
    sc = StandardScaler().fit(ref)
    p = PCA(min(n, len(ref) - 1, ref.shape[1]), random_state=SEED).fit(sc.transform(ref))
    t = sc.transform(test)
    return np.mean((t - p.inverse_transform(p.transform(t))) ** 2, 1)


# --------------------------------------------------------- trained-from-scratch refs
def minirocket_proba(X, y, tr, te, extra=None):
    """MiniRocket features + ridge classifier; returns class scores (softmax of decision)."""
    from aeon.transformations.collection.convolution_based import MiniRocket
    mr = MiniRocket(random_state=SEED).fit(X[tr][:, None, :])
    f = lambda idx: mr.transform(X[idx][:, None, :])  # noqa: E731
    Ftr, Fte = f(tr), f(te)
    if extra is not None:
        Ftr, Fte = np.c_[Ftr, extra[tr]], np.c_[Fte, extra[te]]
    clf = make_pipeline(StandardScaler(), RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))).fit(Ftr, y[tr])
    d = clf.decision_function(Fte)
    if d.ndim == 1:
        p1 = 1 / (1 + np.exp(-d))
        return np.c_[1 - p1, p1]
    e = np.exp(d - d.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def cnn_proba(X, y, tr, te, groups, n_classes: int, epochs: int = 60, seed: int = SEED):
    """1D-CNN (conv 16-32-64-128, max-pool, dense 128-16), early stopping on held-out groups."""
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    L = X.shape[1]
    flat = L
    for _ in range(4):
        flat = (flat - 2) // 2
    net = nn.Sequential(
        nn.Conv1d(1, 16, 3), nn.MaxPool1d(2), nn.Conv1d(16, 32, 3), nn.ReLU(), nn.MaxPool1d(2),
        nn.Conv1d(32, 64, 3), nn.ReLU(), nn.MaxPool1d(2), nn.Conv1d(64, 128, 3), nn.ReLU(),
        nn.MaxPool1d(2), nn.Flatten(), nn.Linear(128 * flat, 128), nn.ReLU(), nn.Linear(128, 16),
        nn.ReLU(), nn.Linear(16, n_classes))
    rng = np.random.default_rng(seed)
    g = np.unique(groups[tr])
    vg = rng.choice(g, max(1, int(0.15 * len(g))), replace=False)
    va, fit = tr[np.isin(groups[tr], vg)], tr[~np.isin(groups[tr], vg)]
    s = X[fit].std() + 1e-12
    T = lambda A: torch.from_numpy(np.ascontiguousarray(A / s, dtype=np.float32))[:, None, :]  # noqa: E731
    Y = lambda idx: torch.from_numpy(y[idx].astype(np.int64))  # noqa: E731
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    lossf = nn.CrossEntropyLoss()
    best, state = np.inf, None
    for _ in range(epochs):
        net.train()
        p = rng.permutation(fit)
        for i in range(0, len(p), 32):
            b = p[i:i + 32]
            opt.zero_grad()
            lossf(net(T(X[b])), Y(b)).backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            vl = lossf(net(T(X[va])), Y(va)).item()
        if vl < best:
            best, state = vl, {k: v.clone() for k, v in net.state_dict().items()}
    net.load_state_dict(state)
    with torch.no_grad():
        return torch.softmax(net(T(X[te])), 1).numpy()
