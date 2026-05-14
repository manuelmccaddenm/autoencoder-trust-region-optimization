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
