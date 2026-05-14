# -*- coding: utf-8 -*-
"""
lib_autoencoder.py
Autoencoder lineal d -> k -> d  (k = dimension latente).

Estado x en R^{2dk} = [vec(E); vec(D)] con E (k x d) y D (d x k).
Forward:        Xhat = D E X.
Perdida (MSE):  f(x) = (1/2N) || D E X - X ||_F^2.
Gradientes analiticos (M = DEX - X):
    grad_D f = (1/N) M (E X)^T
    grad_E f = (1/N) D^T M X^T
"""

import numpy as np


def pack(E: np.ndarray, D: np.ndarray) -> np.ndarray:
    return np.concatenate([E.ravel(), D.ravel()])


def unpack(x: np.ndarray, d: int, k: int):
    nE = k * d
    E = x[:nE].reshape(k, d)
    D = x[nE:].reshape(d, k)
    return E, D


def forward(E: np.ndarray, D: np.ndarray, X: np.ndarray) -> np.ndarray:
    return D @ E @ X


def loss(x: np.ndarray, X: np.ndarray, k: int) -> float:
    d = X.shape[0]
    E, D = unpack(x, d, k)
    M = D @ E @ X - X
    N = X.shape[1]
    return 0.5 * float(np.sum(M * M)) / N


def grad(x: np.ndarray, X: np.ndarray, k: int) -> np.ndarray:
    d = X.shape[0]
    E, D = unpack(x, d, k)
    N = X.shape[1]
    M = D @ E @ X - X
    gD = (M @ (E @ X).T) / N
    gE = (D.T @ M @ X.T) / N
    return pack(gE, gD)
