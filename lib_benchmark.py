# -*- coding: utf-8 -*-
"""
lib_benchmark.py
Comparativa BFGS vs L-BFGS en autoencoder lineal d -> k -> d
para una rejilla de dimensiones d.

Reporta por cada (d, metodo):
  - n        = 2*d*k = numero de parametros
  - iter     = iteraciones realizadas
  - tiempo   = segundos pared
  - f(x*)    = perdida final
  - f* PCA   = optimo teorico (cota inferior por SVD rango k)
  - gap rel  = (f(x*) - f*) / f*
  - mem(B)   = memoria del estado del Hessiano aproximado
"""

import time
import numpy as np

from lib_data import make_data
from lib_autoencoder import loss, grad, pack
from lib_trust_region import trust_region


def pca_optimum(X: np.ndarray, k: int) -> float:
    """f* = (1/2N) * sum_{i>k} sigma_i^2 (cota inferior PCA rango k)."""
    _, S, _ = np.linalg.svd(X, full_matrices=False)
    return 0.5 * float(np.sum(S[k:] ** 2)) / X.shape[1]


def init_x0(d: int, k: int, seed: int = 0, scale: float = 0.1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    E0 = scale * rng.standard_normal((k, d))
    D0 = scale * rng.standard_normal((d, k))
    return pack(E0, D0)


def hessian_memory_floats(mode: str, n: int, m: int) -> int:
    """Memoria persistente del estado del Hessiano aproximado (en floats)."""
    if mode == "ibfgs":
        return n * n            # H explicita (n x n)
    return 2 * m * n            # m pares (s, gamma) de tamano n


def run_one(d: int, mode: str, lbfgs_m: int = 10, k: int = 2,
            N: int = 200, maxIter: int = 500, tol: float = 1e-8,
            data_seed: int = 42, init_seed: int = 0):
    X = make_data(d=d, k=k, N=N, seed=data_seed)
    x0 = init_x0(d=d, k=k, seed=init_seed)
    n = len(x0)

    f_fn = lambda x: loss(x, X, k)
    g_fn = lambda x: grad(x, X, k)

    t0 = time.perf_counter()
    x_star, hist = trust_region(
        f_fn, g_fn, x0,
        Delta0=1.0, Delta_max=10.0, eta=1e-4,
        tol=tol, maxIter=maxIter,
        hessian_mode=mode, lbfgs_m=lbfgs_m,
    )
    elapsed = time.perf_counter() - t0

    f_star = f_fn(x_star)
    f_pca = pca_optimum(X, k)

    return dict(
        d=d, k=k, n=n,
        mode=mode, m=lbfgs_m if mode == "lbfgs" else None,
        iters=len(hist), time=elapsed,
        time_per_iter=elapsed / max(len(hist), 1),
        f_star=f_star, f_pca=f_pca,
        gap=(f_star - f_pca) / f_pca,
        mem_floats=hessian_memory_floats(mode, n, lbfgs_m),
        history=hist,
    )


def benchmark(ds=(3, 5, 10, 100), lbfgs_m: int = 10, **kwargs):
    rows = []
    for d in ds:
        rows.append(run_one(d=d, mode="ibfgs", lbfgs_m=lbfgs_m, **kwargs))
        rows.append(run_one(d=d, mode="lbfgs", lbfgs_m=lbfgs_m, **kwargs))
    return rows


def print_table(rows):
    print(f"{'d':>4} {'n':>5} {'metodo':<13} {'iter':>5} {'time (s)':>9}"
          f" {'t/iter (ms)':>11} {'f(x*)':>12} {'f* PCA':>12}"
          f" {'gap':>10} {'mem(B)':>9}")
    print("-" * 100)
    for r in rows:
        label = r["mode"] + (f" m={r['m']}" if r["m"] is not None else "")
        mem_kb = r["mem_floats"] * 8 / 1024
        print(f"{r['d']:>4d} {r['n']:>5d} {label:<13} {r['iters']:>5d}"
              f" {r['time']:>9.3f} {r['time_per_iter']*1000:>11.2f}"
              f" {r['f_star']:>12.4e} {r['f_pca']:>12.4e}"
              f" {r['gap']:>+10.2e} {mem_kb:>7.1f}KB")
