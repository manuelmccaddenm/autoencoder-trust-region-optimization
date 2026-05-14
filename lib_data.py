import numpy as np


def make_data(d: int = 3, k: int = 2, N: int = 200,
              seed: int = 42, noise_std: float = 0.05) -> np.ndarray:
    rng = np.random.default_rng(seed)
    U = rng.standard_normal((k, N))
    P = rng.standard_normal((d, k))
    X = P @ U + noise_std * rng.standard_normal((d, N))
    X = X - X.mean(axis=1, keepdims=True)
    return X


if __name__ == "__main__":
    for d in (3, 5, 10, 100):
        X = make_data(d=d)
        print(f"d={d:3d}: X.shape={X.shape}, |media|_inf={np.max(np.abs(X.mean(1))):.2e}")
