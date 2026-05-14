import numpy as np

from lib_dogleg import dogleg
from lib_ibfgs import ibfgs_update
from lib_lbfgs import LBFGSMemory


def trust_region(
    f,
    grad_fn,
    x0: np.ndarray,
    Delta0: float = 1.0,
    Delta_max: float = 100.0,
    eta: float = 1e-4,
    tol: float = 1e-6,
    maxIter: int = 500,
    verbose: bool = False,
    hessian_mode: str = "ibfgs",
    lbfgs_m: int = 10,
):
    assert 0.0 <= eta < 0.25
    assert 0.0 < Delta0 < Delta_max
    assert hessian_mode in ("ibfgs", "lbfgs")

    n = len(x0)
    x_k = x0.copy()
    Delta_k = Delta0
    g_k = grad_fn(x_k)
    f_k_cached = float(f(x_k))

    if hessian_mode == "ibfgs":
        H_k = np.eye(n)
        memory = None
        def Hg_fn(v):
            return H_k @ v
    else:
        H_k = None
        memory = LBFGSMemory(m=lbfgs_m)
        def Hg_fn(v):
            return memory.two_loop(v)

    Delta_min = 1e-14

    history = []
    k = 0
    while (np.linalg.norm(g_k, np.inf) > tol
           and k < maxIter
           and Delta_k > Delta_min):
        Delta_used = Delta_k

        p_k, pred = dogleg(g_k, Delta_k, Hg_fn, f, x_k, f_k_cached)

        f_kp = float(f(x_k + p_k))
        actual = f_k_cached - f_kp
        rho_k = actual / pred if pred != 0.0 else 0.0

        p_norm = float(np.linalg.norm(p_k))
        if rho_k < 0.25:
            Delta_k = 0.25 * Delta_k
        elif rho_k > 0.75 and abs(p_norm - Delta_k) < 1e-10:
            Delta_k = min(2.0 * Delta_k, Delta_max)

        if rho_k > eta:
            x_next = x_k + p_k
            g_next = grad_fn(x_next)
            s_k = x_next - x_k
            gamma_k = g_next - g_k
            if float(s_k @ gamma_k) > 1e-8:
                if hessian_mode == "ibfgs":
                    H_k = ibfgs_update(H_k, s_k, gamma_k)
                else:
                    memory.push(s_k, gamma_k)
            x_k = x_next
            g_k = g_next
            f_k_cached = f_kp

        history.append(
            dict(
                k=k,
                x=x_k.copy(),
                f=f_k_cached,
                g_inf=float(np.linalg.norm(g_k, np.inf)),
                Delta=Delta_used,
                Delta_next=Delta_k,
                rho=rho_k,
                p_norm=p_norm,
                pred=pred,
            )
        )
        if verbose:
            print(
                f"k={k:3d}  f={f_k_cached:.6e}  ||g||_inf={np.linalg.norm(g_k, np.inf):.3e}"
                f"  Delta={Delta_k:.3e}  rho={rho_k:+.3f}  ||p||={p_norm:.3e}"
            )
        k += 1

    return x_k, history
