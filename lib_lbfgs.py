from collections import deque
import numpy as np


class LBFGSMemory:
    def __init__(self, m: int = 10):
        assert m >= 1
        self.m = m
        self.pairs: deque = deque(maxlen=m)

    def push(self, s: np.ndarray, gamma: np.ndarray) -> None:
        self.pairs.append((s.copy(), gamma.copy()))

    def __len__(self) -> int:
        return len(self.pairs)

    def delta(self) -> float:
        if len(self.pairs) == 0:
            return 1.0
        s, gamma = self.pairs[-1]
        return float(s @ gamma) / float(gamma @ gamma)

    def two_loop(self, g: np.ndarray) -> np.ndarray:
        if len(self.pairs) == 0:
            return g.copy()

        m = len(self.pairs)
        alphas = [0.0] * m
        q = g.copy()

        for i in range(m - 1, -1, -1):
            s_i, gamma_i = self.pairs[i]
            rho_i = 1.0 / float(gamma_i @ s_i)
            alphas[i] = rho_i * float(s_i @ q)
            q = q - alphas[i] * gamma_i

        r = self.delta() * q

        for i in range(m):
            s_i, gamma_i = self.pairs[i]
            rho_i = 1.0 / float(gamma_i @ s_i)
            beta = rho_i * float(gamma_i @ r)
            r = r + (alphas[i] - beta) * s_i

        return r
