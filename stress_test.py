"""
Random stress test: runs N configurations and checks that the optimizer
satisfies the following invariants on every run.

  I1. Iterates (x_k, g_k, p_k, pred_k, rho_k) are finite.
  I2. pred_k >= 0 (the quadratic model never predicts an increase).
  I3. Accepted steps strictly decrease f: f_{k+1} < f_k.
  I4. Final gradient norm does not blow up.
  I5. Final loss is not below the Eckart-Young bound.
  I6. Final gap relative to the optimum is moderate.
  I7. Trust radius stays in (0, Delta_max].
  I8. ||p_k|| <= Delta_k for accepted steps.
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
        return False, [f"exception: {e!r}\n{traceback.format_exc()}"], \
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
        failures.append("x* has NaN/Inf")
    if not np.isfinite(f_star):
        failures.append("f(x*) not finite")
    if not all(np.isfinite(h["f"]) for h in hist):
        failures.append("f_k not finite at some iteration")
    if not all(np.isfinite(h["g_inf"]) for h in hist):
        failures.append("||g_k|| not finite at some iteration")

    bad_pred = [h for h in hist if h["pred"] < -1e-12]
    if bad_pred:
        failures.append(
            f"pred_k negative in {len(bad_pred)} iter "
            f"(min={min(h['pred'] for h in bad_pred):.2e})"
        )

    accepted_pairs = []
    prev_f = f0
    for h in hist:
        if h["rho"] > 1e-4:
            if h["f"] > prev_f + 1e-12:
                accepted_pairs.append((prev_f, h["f"]))
            prev_f = h["f"]
    if accepted_pairs:
        failures.append(
            f"accepted step without descent in {len(accepted_pairs)} cases"
        )

    g_final_inf = hist[-1]["g_inf"] if hist else g0_inf
    if g_final_inf > 100 * max(g0_inf, 1.0):
        failures.append(f"||g_final|| blew up: {g_final_inf:.2e}")

    if f_star < f_pca - 1e-10:
        failures.append(f"f(x*) < f_pca beyond tol: gap = {gap:.2e}")

    if gap > 0.10:
        failures.append(f"large relative gap: {gap:.2e}")

    f_history = [h["f"] for h in hist]
    if len(f_history) >= 10:
        f_mid = f_history[len(f_history) // 2]
        if f_mid > f0 * 1.1:
            failures.append(
                f"divergence: f midway ({f_mid:.2e}) > f_0 ({f0:.2e})"
            )

    Delta_max = 10.0
    bad_delta = [h for h in hist if not (0 < h["Delta"] <= Delta_max + 1e-9)]
    if bad_delta:
        failures.append(f"Delta out of (0, Delta_max] in {len(bad_delta)} iter")

    bad_feas = [
        h for h in hist
        if h["rho"] > 1e-4 and h["p_norm"] > h["Delta"] + 1e-6
    ]
    if bad_feas:
        failures.append(f"||p|| > Delta in {len(bad_feas)} accepted steps")

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
    print(f"=== SUMMARY ({n_runs} runs in {elapsed:.1f}s) ===")
    print(f"ok    : {n_ok}")
    print(f"fail  : {n_fail}")
    print()

    if fail_reports:
        print("=== FAILURES ===")
        by_type = {}
        for info, failures in fail_reports:
            for f in failures:
                key = f.split("(")[0].strip().split(":")[0]
                by_type.setdefault(key, []).append((info, f))
        for ftype, cases in sorted(by_type.items(), key=lambda x: -len(x[1])):
            print(f"\n  [{len(cases)}x]  {ftype}")
            for info, f in cases[:3]:
                print(f"    seed={info['seed']}, d={info['d']}, "
                      f"mode={info['mode']}, gap={info['gap']:.2e}")
                print(f"      detail: {f}")
            if len(cases) > 3:
                print(f"    ... and {len(cases)-3} more")
        sys.exit(1)
    else:
        print("No failures. All invariants hold.")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    main(n_runs=n)
