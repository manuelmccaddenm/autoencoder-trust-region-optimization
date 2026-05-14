import numpy as np

_EPS_A = 1e-12


def dogleg(g: np.ndarray, Delta: float, Hg_fn, f_fn, x_k: np.ndarray,
           f_k: float):
    g_norm = float(np.linalg.norm(g))
    if g_norm == 0.0:
        return np.zeros_like(g), 0.0

    g_sq = g_norm * g_norm

    t = min(1.0, Delta / g_norm)
    # underflow guard: when Delta has shrunk near machine epsilon, t*t = 0
    # and the parabola cannot be fit; take the boundary step in -g.
    if t * t < 1e-300:
        scale = Delta / g_norm
        return -scale * g, Delta * g_norm

    phi_t = float(f_fn(x_k - t * g))
    a = (phi_t - f_k + t * g_sq) / (t * t)

    if a <= _EPS_A * g_sq:
        scale = Delta / g_norm
        return -scale * g, Delta * g_norm

    alpha_u = g_sq / (2.0 * a)

    if alpha_u * g_norm >= Delta:
        scale = Delta / g_norm
        p = -scale * g
        pred = Delta * g_norm - a * (Delta * Delta) / g_sq
        return p, pred

    p_B = -Hg_fn(g)
    p_B_norm = float(np.linalg.norm(p_B))
    gHg = float(g @ (-p_B))

    if p_B_norm <= Delta:
        return p_B, 0.5 * gHg

    p_u = -alpha_u * g
    v = p_B - p_u
    a_q = float(v @ v)
    b_q = 2.0 * float(p_u @ v)
    c_q = float(p_u @ p_u) - Delta * Delta
    disc = b_q * b_q - 4.0 * a_q * c_q
    lam = (-b_q + np.sqrt(disc)) / (2.0 * a_q)
    p = p_u + lam * v

    neg_gp = (1.0 - lam) * alpha_u * g_sq + lam * gHg
    pBp = alpha_u * g_sq * (1.0 - lam * lam) + (lam * lam) * gHg
    pred = neg_gp - 0.5 * pBp
    return p, pred
