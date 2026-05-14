import numpy as np


def ibfgs_update(H: np.ndarray, s: np.ndarray, gamma: np.ndarray) -> np.ndarray:
    sg = float(s @ gamma)
    rho = 1.0 / sg

    Hg = H @ gamma
    gHg = float(gamma @ Hg)
    s_Hg = np.outer(s, Hg)
    ss = np.outer(s, s)

    return H - rho * (s_Hg + s_Hg.T) + (rho * rho * gHg + rho) * ss
