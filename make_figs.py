import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from lib_data import make_data
from lib_autoencoder import loss, grad, pack, unpack, forward
from lib_trust_region import trust_region
from lib_benchmark import benchmark, init_x0, pca_optimum

FIGS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs")
os.makedirs(FIGS, exist_ok=True)
plt.rcParams.update({"font.size": 10, "figure.dpi": 130})

_VISTA = dict(elev=22, azim=-60)
_LIMITES = dict(xlim=(-1.0, 1.0), ylim=(-2.5, 2.5), zlim=(-1.5, 1.5))


def _aplicar_ejes(ax):
    ax.view_init(**_VISTA)
    ax.set_xlim(_LIMITES["xlim"]); ax.set_ylim(_LIMITES["ylim"]); ax.set_zlim(_LIMITES["zlim"])
    ax.set_xlabel("x_1"); ax.set_ylabel("x_2"); ax.set_zlabel("x_3")


def fig_datos_entrada():
    X = make_data(d=3, k=2, N=200, seed=42, noise_std=0.05)
    fig = plt.figure(figsize=(5.5, 4.5))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(X[0], X[1], X[2], s=10, alpha=0.65, c="tab:blue")
    _aplicar_ejes(ax)
    ax.set_title("Nube de N=200 puntos en R^3 (d=3, k=2)")
    plt.tight_layout()
    fig.savefig(os.path.join(FIGS, "input_3d.pdf"))
    plt.close(fig)


def fig_convergencia_d3():
    K = 2
    X = make_data(d=3, k=K, N=200, seed=42, noise_std=0.05)
    x0 = init_x0(d=3, k=K)
    f_fn = lambda x: loss(x, X, K)
    g_fn = lambda x: grad(x, X, K)

    _, history = trust_region(
        f_fn, g_fn, x0,
        Delta0=1.0, Delta_max=10.0, eta=1e-4,
        tol=1e-8, maxIter=500, hessian_mode="ibfgs",
    )

    ks  = [h["k"]     for h in history]
    fs  = [h["f"]     for h in history]
    gs  = [h["g_inf"] for h in history]
    ds  = [h["Delta"] for h in history]
    rs  = [h["rho"]   for h in history]

    f_pca = pca_optimum(X, K)

    fig, axes = plt.subplots(2, 2, figsize=(8.5, 5.5))
    axes[0, 0].semilogy(ks, fs, "b-", lw=1)
    axes[0, 0].axhline(f_pca, ls="--", c="gray", label=f"f* = {f_pca:.3e}")
    axes[0, 0].set_title("f(x_k)"); axes[0, 0].set_xlabel("k")
    axes[0, 0].grid(True, alpha=0.4); axes[0, 0].legend(fontsize=8)

    axes[0, 1].semilogy(ks, gs, "r-", lw=1)
    axes[0, 1].set_title("||g_k||_inf"); axes[0, 1].set_xlabel("k")
    axes[0, 1].grid(True, alpha=0.4)

    axes[1, 0].semilogy(ks, ds, "g-", lw=1)
    axes[1, 0].set_title("Delta_k"); axes[1, 0].set_xlabel("k")
    axes[1, 0].grid(True, alpha=0.4)

    axes[1, 1].plot(ks, rs, "m.", ms=2)
    axes[1, 1].axhline(0.25, ls=":", c="gray")
    axes[1, 1].axhline(0.75, ls=":", c="gray")
    axes[1, 1].set_title("rho_k"); axes[1, 1].set_xlabel("k")
    axes[1, 1].set_ylim(-0.5, 2.5); axes[1, 1].grid(True, alpha=0.4)

    plt.tight_layout()
    fig.savefig(os.path.join(FIGS, "convergence_d3.pdf"))
    plt.close(fig)


def fig_reconstruccion_d3():
    K = 2
    X = make_data(d=3, k=K, N=200, seed=42, noise_std=0.05)
    x0 = init_x0(d=3, k=K)
    f_fn = lambda x: loss(x, X, K)
    g_fn = lambda x: grad(x, X, K)

    x_star, _ = trust_region(
        f_fn, g_fn, x0,
        Delta0=1.0, Delta_max=10.0, eta=1e-4,
        tol=1e-8, maxIter=500, hessian_mode="ibfgs",
    )
    E, D = unpack(x_star, 3, K)
    Xhat = forward(E, D, X)

    fig = plt.figure(figsize=(7, 5.5))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(X[0], X[1], X[2], c="tab:blue", s=14, alpha=0.55, label="X")
    ax.scatter(Xhat[0], Xhat[1], Xhat[2], c="tab:orange", s=14, alpha=0.85,
               marker="^", label="X_hat = D E X")
    for i in range(X.shape[1]):
        ax.plot([X[0, i], Xhat[0, i]],
                [X[1, i], Xhat[1, i]],
                [X[2, i], Xhat[2, i]],
                c="gray", lw=0.3, alpha=0.4)

    rejilla_u = np.linspace(-2.5, 2.5, 12)
    rejilla_v = np.linspace(-2.5, 2.5, 12)
    U_, V_ = np.meshgrid(rejilla_u, rejilla_v)
    plano = D @ np.stack([U_.ravel(), V_.ravel()], axis=0)
    ax.plot_surface(plano[0].reshape(U_.shape),
                    plano[1].reshape(U_.shape),
                    plano[2].reshape(U_.shape),
                    alpha=0.18, color="tab:green", edgecolor="none")

    _aplicar_ejes(ax)
    ax.set_title("Original X (azul) vs. reconstruccion X_hat (naranja)\n"
                 "Subespacio aprendido en verde")
    ax.legend(loc="upper left", fontsize=9)
    plt.tight_layout()
    fig.savefig(os.path.join(FIGS, "reconstruction_d3.pdf"))
    plt.close(fig)


def fig_escalamiento():
    rows = benchmark(ds=(3, 5, 10, 100), lbfgs_m=10, maxIter=500)
    ibfgs = [r for r in rows if r["mode"] == "ibfgs"]
    lbfgs = [r for r in rows if r["mode"] == "lbfgs"]
    ds_grid = [r["d"] for r in ibfgs]

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))

    axes[0].loglog(ds_grid, [r["time"] for r in ibfgs], "o-", label="iBFGS")
    axes[0].loglog(ds_grid, [r["time"] for r in lbfgs], "s-", label="L-BFGS m=10")
    axes[0].set_title("Tiempo total (s, 500 iter)")
    axes[0].set_xlabel("d"); axes[0].grid(True, which="both", alpha=0.4); axes[0].legend(fontsize=8)

    axes[1].loglog(ds_grid, [r["mem_floats"] * 8 / 1024 for r in ibfgs], "o-", label="iBFGS")
    axes[1].loglog(ds_grid, [r["mem_floats"] * 8 / 1024 for r in lbfgs], "s-", label="L-BFGS m=10")
    axes[1].set_title("Memoria del Hessiano (KB)")
    axes[1].set_xlabel("d"); axes[1].grid(True, which="both", alpha=0.4); axes[1].legend(fontsize=8)

    axes[2].semilogy(ds_grid, [abs(r["gap"]) for r in ibfgs], "o-", label="iBFGS")
    axes[2].semilogy(ds_grid, [abs(r["gap"]) for r in lbfgs], "s-", label="L-BFGS m=10")
    axes[2].set_title("|gap| relativo a f*")
    axes[2].set_xlabel("d"); axes[2].grid(True, which="both", alpha=0.4); axes[2].legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(os.path.join(FIGS, "scaling.pdf"))
    plt.close(fig)


if __name__ == "__main__":
    fig_datos_entrada()
    fig_convergencia_d3()
    fig_reconstruccion_d3()
    fig_escalamiento()
    print(f"figuras guardadas en {FIGS}")
