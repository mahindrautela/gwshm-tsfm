"""What is fitted on top of the embeddings (or on raw features).

Supervised heads (Tasks 1 and 3): logistic regression and k-NN on standardised features.
Unsupervised detectors (Task 2): scores that grow the further a signal is from healthy
reference signals; they are fitted on healthy data only.
"""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from .config import SEED


# --------------------------------------------------------------------- supervised heads
def logistic_regression(C: float = 1.0):
    """Standardise features, then L2-regularised logistic regression."""
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=4000))


def knn_classifier(k: int = 5):
    """Standardise features, then a k-nearest-neighbour vote with cosine distance."""
    return make_pipeline(StandardScaler(), KNeighborsClassifier(k, metric="cosine"))


def predict_proba(make_model, F: np.ndarray, y: np.ndarray, train, test) -> np.ndarray:
    """Fit a fresh model on the training rows and return class probabilities for the test rows."""
    return make_model().fit(F[train], y[train]).predict_proba(F[test])


# --------------------------------------------------------------- unsupervised detectors
def knn_distance(ref: np.ndarray, test: np.ndarray, k: int = 1) -> np.ndarray:
    """Mean Euclidean distance to the k nearest reference rows (features standardised on ref)."""
    sc = StandardScaler().fit(ref)
    nn = NearestNeighbors(n_neighbors=k).fit(sc.transform(ref))
    return nn.kneighbors(sc.transform(test))[0].mean(1)


def pca_ocsvm(ref: np.ndarray, test: np.ndarray, n: int = 10, nu: float = 0.1) -> np.ndarray:
    """PCA (n components) + one-class SVM (RBF), as in Rautela et al. (2022).

    Higher score = more anomalous.
    """
    sc = StandardScaler().fit(ref)
    pca = PCA(min(n, len(ref) - 1, ref.shape[1]), random_state=SEED).fit(sc.transform(ref))

    def project(A):
        return pca.transform(sc.transform(A))

    svm = OneClassSVM(nu=nu, gamma="scale").fit(project(ref))
    return -svm.decision_function(project(test))
