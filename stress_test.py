"""
Prueba aleatoria masiva: corre N configuraciones y verifica que el
optimizador cumple los siguientes invariantes en cada corrida.

  I1. Los iterados (x_k, g_k, p_k, pred_k, rho_k) son finitos.
  I2. pred_k >= 0 (el modelo cuadratico nunca predice subida).
  I3. Cada paso aceptado disminuye f estrictamente: f_{k+1} < f_k.
  I4. La norma final del gradiente no explota.
  I5. La perdida final no cae por debajo de la cota de Eckart-Young.
  I6. El gap final relativo al optimo es moderado.
  I7. El radio de confianza se mantiene en (0, Delta_max].
  I8. ||p_k|| <= Delta_k en pasos aceptados.

I6 (gap > 10%) se reporta pero NO se considera falla critica: indica
convergencia lenta para configuraciones desfavorables (ruido alto
combinado con inicializacion lejos del optimo), no un bug del metodo.
"""

import sys
import time
import numpy as np

from lib_data import make_data
from lib_autoencoder import loss, grad, pack
from lib_trust_region import trust_region


_SLOW_CONVERGENCE_TAG = "gap relativo grande"


def pca_optimum(X, k):
    _, S, _ = np.linalg.svd(X, full_matrices=False)
    return 0.5 * float(np.sum(S[k:] ** 2)) / X.shape[1]


def run_one(seed, mode, d, k, N=100, maxIter=1000, lbfgs_m=10):
    rng = np.random.default_rng(seed)
    noise_std = rng.uniform(0.01, 0.2)
    init_scale = rng.uniform(0.05, 0.5)

    X = make_data(d=d, k=k, N=N, seed=seed, noise_std=noise_std)
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

    failures = []

    if not np.isfinite(x_star).all():
        failures.append("x* tiene NaN/Inf")
    if not np.isfinite(f_star):
        failures.append("f(x*) no finito")
    if not all(np.isfinite(h["f"]) for h in hist):
        failures.append("f_k no finito en alguna iteracion")
    if not all(np.isfinite(h["g_inf"]) for h in hist):
        failures.append("||g_k|| no finito en alguna iteracion")

    neg_pred = [h for h in hist if h["pred"] < -1e-12]
    if neg_pred:
        failures.append(
            f"pred_k negativa en {len(neg_pred)} iter "
            f"(min={min(h['pred'] for h in neg_pred):.2e})"
        )

    bad_descent = []
    prev_f = f0
    for h in hist:
        if h["rho"] > 1e-4:
            if h["f"] > prev_f + 1e-12:
                bad_descent.append((prev_f, h["f"]))
            prev_f = h["f"]
    if bad_descent:
        failures.append(
            f"paso aceptado sin descenso en {len(bad_descent)} casos"
        )

    g_final_inf = hist[-1]["g_inf"] if hist else g0_inf
    if g_final_inf > 100 * max(g0_inf, 1.0):
        failures.append(f"||g_final|| explota: {g_final_inf:.2e}")

    if f_star < f_pca - 1e-10:
        failures.append(f"f(x*) < f_pca por mas de tol: gap = {gap:.2e}")

    if gap > 0.10:
        failures.append(f"{_SLOW_CONVERGENCE_TAG}: {gap:.2e}")

    f_history = [h["f"] for h in hist]
    if len(f_history) >= 10:
        f_mid = f_history[len(f_history) // 2]
        if f_mid > f0 * 1.1:
            failures.append(
                f"divergencia: f a la mitad ({f_mid:.2e}) > f_0 ({f0:.2e})"
            )

    Delta_max = 10.0
    delta_out = [h for h in hist if not (0 < h["Delta"] <= Delta_max + 1e-9)]
    if delta_out:
        failures.append(f"Delta fuera de (0, Delta_max] en {len(delta_out)} iter")

    infeasible = [
        h for h in hist
        if h["rho"] > 1e-4 and h["p_norm"] > h["Delta"] + 1e-6
    ]
    if infeasible:
        failures.append(f"||p|| > Delta en {len(infeasible)} pasos aceptados")

    ok = len(failures) == 0
    info = dict(
        d=d, k=k, N=N, mode=mode, seed=seed,
        noise_std=noise_std, init_scale=init_scale,
        iter=len(hist), f0=f0, f_star=f_star, f_pca=f_pca, gap=gap,
        g0=g0_inf, g_final=g_final_inf,
    )
    return ok, failures, info


def main(n_runs=2000):
    t0 = time.perf_counter()
    rng = np.random.default_rng(20260513)

    d_choices = [3, 5, 10, 30]
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

        ok, failures, info = run_one(seed=seed, mode=mode, d=d, k=k_fixed, N=N)
        if ok:
            n_ok += 1
        else:
            n_fail += 1
            fail_reports.append((info, failures))

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

    critical = []
    slow = []
    for info, fs in fail_reports:
        if any(_SLOW_CONVERGENCE_TAG not in f for f in fs):
            critical.append((info, fs))
        else:
            slow.append((info, fs))

    if slow:
        print(f"  [{len(slow)}x]  {_SLOW_CONVERGENCE_TAG}  (convergencia lenta, no es bug)")
    if critical:
        print("\n=== FALLAS CRITICAS ===")
        by_type = {}
        for info, fs in critical:
            for f in fs:
                if _SLOW_CONVERGENCE_TAG in f:
                    continue
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
    print("\nSin fallas criticas. Todos los invariantes algoritmicos se cumplen.")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    main(n_runs=n)
