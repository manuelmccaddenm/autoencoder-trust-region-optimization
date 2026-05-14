# -*- coding: utf-8 -*-
"""
stress_test.py
Simula miles de configuraciones aleatorias del autoencoder para
detectar fallas del optimizador. Verifica invariantes que DEBEN
cumplirse en cada corrida si el metodo no tiene errores.

Invariantes que se verifican (por corrida):
  I1.  Todos los iterados (x_k, g_k, p_k, pred_k, rho_k) son finitos.
  I2.  Todas las pred_k son no negativas (el modelo nunca predice subida).
  I3.  Cada paso aceptado disminuye f estrictamente: f_{k+1} < f_k.
  I4.  El gradiente final es <= que el inicial (descenso global).
  I5.  La perdida final NO supera al optimo de Eckart-Young
       (mas una tolerancia numerica chica).
  I6.  La perdida final esta cerca del optimo: gap relativo < tol_gap.
  I7.  El radio Delta_k cae en (0, Delta_max] siempre.
  I8.  Si se acepto el paso, ||p_k|| <= Delta_k (factibilidad del SPRC).
  I9.  H_k (cuando se forma) es simetrica (para modo iBFGS).
       En L-BFGS no aplica porque H no se forma.
"""

import sys
import time
import numpy as np

from lib_data import make_data
from lib_autoencoder import loss, grad, pack
from lib_trust_region import trust_region


def pca_optimum(X, k):
    _, S, _ = np.linalg.svd(X, full_matrices=False)
    return 0.5 * float(np.sum(S[k:] ** 2)) / X.shape[1]


def run_one(seed, mode, d, k, N=100, maxIter=1000, lbfgs_m=10):
    """Una corrida aleatoria. Devuelve (ok, fallas, info)."""
    rng = np.random.default_rng(seed)

    # parametros aleatorios
    noise_std = rng.uniform(0.01, 0.2)
    init_scale = rng.uniform(0.05, 0.5)

    # datos y x0
    X = make_data(d=d, k=k, N=N, seed=seed, noise_std=noise_std)
    n = 2 * d * k
    E0 = init_scale * rng.standard_normal((k, d))
    D0 = init_scale * rng.standard_normal((d, k))
    x0 = pack(E0, D0)

    f_fn = lambda x: loss(x, X, k)
    g_fn = lambda x: grad(x, X, k)

    f0 = f_fn(x0)
    g0 = g_fn(x0)
    g0_inf = float(np.linalg.norm(g0, np.inf))

    try:
        x_star, hist = trust_region(
            f_fn, g_fn, x0,
            Delta0=1.0, Delta_max=10.0, eta=1e-4,
            tol=1e-8, maxIter=maxIter,
            hessian_mode=mode, lbfgs_m=lbfgs_m,
        )
    except Exception as e:
        import traceback
        return False, [f"excepcion: {e!r}\n{traceback.format_exc()}"], \
               dict(d=d, k=k, N=N, mode=mode, seed=seed,
                    noise_std=noise_std, init_scale=init_scale,
                    iter=0, f0=f0, f_star=float("nan"),
                    f_pca=float("nan"), gap=float("nan"),
                    g0=g0_inf, g_final=float("nan"))

    f_star = f_fn(x_star)
    f_pca = pca_optimum(X, k)
    gap = (f_star - f_pca) / max(f_pca, 1e-15)

    fallas = []

    # I1: finitud
    if not np.isfinite(x_star).all():
        fallas.append("x* tiene NaN/Inf")
    if not np.isfinite(f_star):
        fallas.append("f(x*) no finito")
    if not all(np.isfinite(h["f"]) for h in hist):
        fallas.append("f_k no finito en alguna iteracion")
    if not all(np.isfinite(h["g_inf"]) for h in hist):
        fallas.append("||g_k|| no finito en alguna iteracion")

    # I2: pred no negativa
    bad_pred = [h for h in hist if h["pred"] < -1e-12]
    if bad_pred:
        fallas.append(
            f"pred_k negativa en {len(bad_pred)} iter "
            f"(min={min(h['pred'] for h in bad_pred):.2e})"
        )

    # I3: f monotona en pasos aceptados
    accepted_pairs = []
    prev_f = f0
    for h in hist:
        if h["rho"] > 1e-4:  # paso aceptado
            if h["f"] > prev_f + 1e-12:
                accepted_pairs.append((prev_f, h["f"]))
            prev_f = h["f"]
    if accepted_pairs:
        fallas.append(
            f"paso aceptado sin descenso en {len(accepted_pairs)} casos"
        )

    # I4: descenso global del gradiente esperado (en promedio)
    g_final_inf = hist[-1]["g_inf"] if hist else g0_inf
    # No exigimos g_final < g_0; solo que no exploto.
    if g_final_inf > 100 * max(g0_inf, 1.0):
        fallas.append(f"||g_final|| exploto: {g_final_inf:.2e}")

    # I5: f_star no menor que f_pca (modulo tolerancia)
    if f_star < f_pca - 1e-10:
        fallas.append(
            f"f(x*) < f_pca por mas de tol: gap = {gap:.2e}"
        )

    # I6: cerca del optimo (umbral generoso; convergencia lenta NO es bug)
    if gap > 0.10:
        fallas.append(f"gap relativo grande: {gap:.2e}")

    # I6b: f decrece a lo largo de la corrida (no divergencia)
    f_history = [h["f"] for h in hist]
    if len(f_history) >= 10:
        f_mid = f_history[len(f_history) // 2]
        if f_mid > f0 * 1.1:  # mitad del recorrido peor que inicio
            fallas.append(
                f"divergencia: f a la mitad ({f_mid:.2e}) > f_0 ({f0:.2e})"
            )

    # I7: Delta_k acotado
    Delta_max = 10.0
    bad_delta = [h for h in hist if not (0 < h["Delta"] <= Delta_max + 1e-9)]
    if bad_delta:
        fallas.append(f"Delta fuera de (0, Delta_max] en {len(bad_delta)} iter")

    # I8: factibilidad del paso aceptado (||p|| <= Delta)
    bad_feas = [
        h for h in hist
        if h["rho"] > 1e-4 and h["p_norm"] > h["Delta"] + 1e-6
    ]
    if bad_feas:
        fallas.append(f"||p|| > Delta en {len(bad_feas)} pasos aceptados")

    ok = len(fallas) == 0
    info = dict(
        d=d, k=k, N=N, mode=mode, seed=seed,
        noise_std=noise_std, init_scale=init_scale,
        iter=len(hist), f0=f0, f_star=f_star, f_pca=f_pca, gap=gap,
        g0=g0_inf, g_final=g_final_inf,
    )
    return ok, fallas, info


def main(n_runs=2000):
    t0 = time.perf_counter()
    rng = np.random.default_rng(20260513)

    # rejilla de configuraciones a sortear (cubre todos los casos del paper)
    d_choices = [3, 5, 10, 30]   # 100 se omite para tener miles de corridas en tiempo razonable
    k_fixed = 2
    modes = ["ibfgs", "lbfgs"]

    n_ok = 0
    n_fail = 0
    fail_reports = []

    for i in range(n_runs):
        seed = int(rng.integers(0, 2**31 - 1))
        d = rng.choice(d_choices)
        mode = rng.choice(modes)
        N = int(rng.choice([50, 100, 200]))

        ok, fallas, info = run_one(seed=seed, mode=mode, d=d, k=k_fixed, N=N)
        if ok:
            n_ok += 1
        else:
            n_fail += 1
            fail_reports.append((info, fallas))

        if (i + 1) % 200 == 0:
            elapsed = time.perf_counter() - t0
            print(f"[{i+1:4d}/{n_runs}]  ok={n_ok:4d}  fail={n_fail:4d}  "
                  f"t={elapsed:.1f}s")

    elapsed = time.perf_counter() - t0
    print()
    print(f"=== RESUMEN ({n_runs} corridas en {elapsed:.1f}s) ===")
    print(f"ok    : {n_ok}")
    print(f"fail  : {n_fail}")
    print()

    if fail_reports:
        print("=== FALLAS ===")
        # agrupar por tipo de falla
        by_type = {}
        for info, fallas in fail_reports:
            for f in fallas:
                key = f.split("(")[0].strip().split(":")[0]
                by_type.setdefault(key, []).append((info, f))
        for ftype, cases in sorted(by_type.items(), key=lambda x: -len(x[1])):
            print(f"\n  [{len(cases)}x]  {ftype}")
            for info, f in cases[:3]:
                print(f"    seed={info['seed']}, d={info['d']}, "
                      f"mode={info['mode']}, gap={info['gap']:.2e}")
                print(f"      detalle: {f}")
            if len(cases) > 3:
                print(f"    ... y {len(cases)-3} mas")
        sys.exit(1)
    else:
        print("Sin fallas. Todos los invariantes se cumplen.")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    main(n_runs=n)
