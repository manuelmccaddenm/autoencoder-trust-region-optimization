# -*- coding: utf-8 -*-
"""
lib_dogleg.py
Subproblema de Region de Confianza por la tecnica Dogleg, formulado
sin usar B explicita: solo se necesitan productos H @ v (con
H aproximando la inversa del Hessiano) y evaluaciones de f.

El punto de Cauchy se obtiene por interpolacion cuadratica de
phi(alpha) = f(x_k - alpha g) en tres datos:
    phi(0) = f_k,   phi'(0) = -||g||^2,   phi(t) = f(x_k - t g),
con t = min(1, Delta/||g||).

Salvaguardas:
  1. Si la parabola no abre hacia arriba (curvatura estimada <= 0),
     no hay minimo interior --> se va a la frontera en -g.
  2. Si el punto de prueba t cae mas alla de la frontera, se acorta.
  3. Si el punto Cauchy estimado quedaria mas alla de la frontera,
     se recorta a la frontera.

La reduccion predicha m_k(0) - m_k(p) se evalua sin formar B usando
las identidades algebraicas:
    p_u^T B p_u ~ alpha_u * ||g||^2,
    p_u^T B p_B   = alpha_u * ||g||^2   (por BH = I),
    p_B^T B p_B   = g^T H g            (por BH = I).
"""

import numpy as np

_EPS_A = 1e-12   # umbral para considerar curvatura nula en interpolacion


def dogleg(g: np.ndarray, Delta: float, Hg_fn, f_fn, x_k: np.ndarray,
           f_k: float):
    """
    Devuelve (p_k, pred_k) donde:
      p_k    : el paso dogleg.
      pred_k : la reduccion predicha m_k(0) - m_k(p_k).
    """
    g_norm = float(np.linalg.norm(g))
    if g_norm == 0.0:
        return np.zeros_like(g), 0.0

    g_sq = g_norm * g_norm

    # --- Interpolacion cuadratica para el punto de Cauchy ---
    t = min(1.0, Delta / g_norm)
    # Defensa contra underflow: si t es tan chico que t*t pierde
    # precision en float64, no se puede ajustar la parabola. La region
    # de confianza es entonces tan chica que el unico paso factible es
    # el borde en -g.
    if t * t < 1e-300:
        scale = Delta / g_norm
        return -scale * g, Delta * g_norm

    phi_t = float(f_fn(x_k - t * g))
    # I(alpha) = a alpha^2 + b alpha + c con b = -g_sq, c = f_k
    a = (phi_t - f_k + t * g_sq) / (t * t)

    # --- Salvaguarda 1: parabola no abre hacia arriba ---
    if a <= _EPS_A * g_sq:
        # No hay minimo interior del modelo en direccion -g.
        # Tomar el paso al borde de la region de confianza.
        scale = Delta / g_norm
        p = -scale * g
        # Solo se conserva el termino lineal (la curvatura estimada
        # es no positiva, no contribuye reduccion predicha):
        pred = Delta * g_norm
        return p, pred

    alpha_u = g_sq / (2.0 * a)

    # --- Caso 1: el Cauchy interpolado cae fuera de la region ---
    if alpha_u * g_norm >= Delta:            # salvaguarda 3
        scale = Delta / g_norm
        p = -scale * g
        # pred = -g^T p - 1/2 p^T B p
        #      = Delta * ||g|| - 1/2 (Delta^2 / ||g||^2) (2a)
        pred = Delta * g_norm - a * (Delta * Delta) / g_sq
        return p, pred

    # --- Paso de Newton: p_B = -H g ---
    p_B = -Hg_fn(g)
    p_B_norm = float(np.linalg.norm(p_B))
    gHg = float(g @ (-p_B))                  # g^T H g  (positivo si H s.p.d.)

    # --- Caso 2: el paso de Newton cae dentro de la region ---
    if p_B_norm <= Delta:
        # pred = 1/2 g^T H g  (usa BH = I para simplificar)
        return p_B, 0.5 * gHg

    # --- Caso 3: el segmento (p_u -> p_B) intersecta la frontera ---
    p_u = -alpha_u * g
    v = p_B - p_u
    a_q = float(v @ v)
    b_q = 2.0 * float(p_u @ v)
    c_q = float(p_u @ p_u) - Delta * Delta
    disc = b_q * b_q - 4.0 * a_q * c_q
    lam = (-b_q + np.sqrt(disc)) / (2.0 * a_q)
    p = p_u + lam * v

    # pred en el segmento, usando BH = I y la estimacion 2a de g^T B g:
    #   -g^T p   = (1-lam) alpha_u ||g||^2 + lam g^T H g
    #   p^T B p = alpha_u ||g||^2 (1 - lam^2) + lam^2 g^T H g
    neg_gp = (1.0 - lam) * alpha_u * g_sq + lam * gHg
    pBp = alpha_u * g_sq * (1.0 - lam * lam) + (lam * lam) * gHg
    pred = neg_gp - 0.5 * pBp
    return p, pred
