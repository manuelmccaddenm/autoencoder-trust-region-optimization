# -*- coding: utf-8 -*-
"""
lib_ibfgs.py
Actualizacion BFGS sobre la inversa del Hessiano aproximado.

Mantiene H_k aproximando (nabla^2 f)^{-1} mediante la formula (iBFGS)
del handout de cuasi-Newton (Clase 24):

    H_{k+1} = (I - rho s gamma^T) H_k (I - rho gamma s^T) + rho s s^T,
    rho = 1 / (gamma^T s).

Costo: O(n^2). No invierte ninguna matriz. La preserva s.p.d. si se
cumple la condicion de curvatura s^T gamma > 0.
"""

import numpy as np


def ibfgs_update(H: np.ndarray, s: np.ndarray, gamma: np.ndarray) -> np.ndarray:
    sg = float(s @ gamma)
    rho = 1.0 / sg

    # Expansion directa: V^T H V + rho s s^T donde V = I - rho gamma s^T.
    # Equivalente a:
    # H_new = H - rho s (Hg)^T - rho (Hg) s^T + rho^2 (gamma^T H gamma) s s^T + rho s s^T.
    Hg = H @ gamma                       # n
    gHg = float(gamma @ Hg)              # escalar
    s_Hg = np.outer(s, Hg)               # n x n
    ss = np.outer(s, s)                  # n x n

    return H - rho * (s_Hg + s_Hg.T) + (rho * rho * gHg + rho) * ss
