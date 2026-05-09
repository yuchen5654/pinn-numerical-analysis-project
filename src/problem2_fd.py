"""
Problem 2.1: Finite-difference reference solution for the 1D heat equation.

PDE:  u_t = nu * u_xx,  x in (0,1),  t in (0, 0.5]
IC:   u(x,0) = sin(pi*x) + 0.5*sin(3*pi*x)
BC:   u(0,t) = u(1,t) = 0
nu = 0.01

Sections:
  (a) Forward Euler FD with dx=1/64, CFL r <= 1/2  -- report dt and r
  (b) L2 error at t=0.5
  (c) Heatmap of the solution
  extra: unstable run with r=0.6 (for Problem 2.4c)
"""
import os
import numpy as np
import matplotlib.pyplot as plt

_FIGURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
os.makedirs(_FIGURES_DIR, exist_ok=True)


NU = 0.01


def heat_exact(x, t):
    return (
        np.exp(-NU * np.pi**2 * t) * np.sin(np.pi * x)
        + 0.5 * np.exp(-9 * NU * np.pi**2 * t) * np.sin(3 * np.pi * x)
    )


def forward_euler_heat(dx=1 / 64, r_target=0.5, T_final=0.5):
    """Run the stable Forward Euler scheme and return (U, x, t, dt, r)."""
    x = np.arange(0, 1 + dx, dx)
    Nx = len(x) - 1

    dt_initial = r_target * dx**2 / NU
    Nt = int(np.ceil(T_final / dt_initial))
    dt = T_final / Nt
    r = NU * dt / dx**2

    t = np.linspace(0, T_final, Nt + 1)
    U = np.zeros((Nt + 1, Nx + 1))
    U[0, :] = np.sin(np.pi * x) + 0.5 * np.sin(3 * np.pi * x)
    U[:, 0] = 0.0
    U[:, -1] = 0.0

    for n in range(Nt):
        for j in range(1, Nx):
            U[n + 1, j] = U[n, j] + r * (U[n, j + 1] - 2 * U[n, j] + U[n, j - 1])
        U[n + 1, 0] = 0.0
        U[n + 1, -1] = 0.0

    return U, x, t, dt, r


def run_part_a():
    print("--- 2.1(a): Forward Euler FD ---")
    U, x, t, dt, r = forward_euler_heat()

    print(f"  Delta x  = {1/64:.8f}")
    print(f"  Delta t  = {dt:.8e}")
    print(f"  r        = {r:.8f}")
    print(f"  Time steps = {len(t) - 1}")
    return U, x, t


def run_part_b(U, x):
    print("\n--- 2.1(b): L2 error at t=0.5 ---")
    dx = x[1] - x[0]
    u_ex = heat_exact(x, 0.5)
    u_num = U[-1, :]
    L2_error = np.sqrt(dx * np.sum((u_num - u_ex) ** 2))
    print(f"  L2 error at t=0.5: {L2_error:.6e}")
    return L2_error


def run_part_c(U, x, t):
    print("\n--- 2.1(c): Heatmap ---")
    X, T = np.meshgrid(x, t)
    plt.figure(figsize=(8, 5))
    plt.pcolormesh(X, T, U, shading="auto", cmap="viridis")
    plt.xlabel("x")
    plt.ylabel("t")
    plt.title("Forward Euler FD — 1D Heat Equation")
    plt.colorbar(label="u(x,t)")
    plt.tight_layout()
    plt.savefig(os.path.join(_FIGURES_DIR, "2_1c_heat_fd_heatmap.png"), dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()


def run_unstable(r_unstable=0.6):
    """Run Forward Euler with r=0.6 (violates CFL) for Problem 2.4(c)."""
    print(f"\n--- Unstable FD run (r={r_unstable}) ---")
    dx = 1 / 64
    dt_unstable = r_unstable * dx**2 / NU
    T_final = 0.5
    Nt = int(np.ceil(T_final / dt_unstable))
    t = np.linspace(0, Nt * dt_unstable, Nt + 1)
    x = np.arange(0, 1 + dx, dx)
    Nx = len(x) - 1

    U = np.zeros((Nt + 1, Nx + 1))
    U[0, :] = np.sin(np.pi * x) + 0.5 * np.sin(3 * np.pi * x)
    U[:, 0] = 0.0
    U[:, -1] = 0.0

    for n in range(Nt):
        for j in range(1, Nx):
            U[n + 1, j] = U[n, j] + r_unstable * (
                U[n, j + 1] - 2 * U[n, j] + U[n, j - 1]
            )
        U[n + 1, 0] = 0.0
        U[n + 1, -1] = 0.0

    print(f"  dx={dx:.4f}, dt={dt_unstable:.4e}, r={r_unstable}, steps={Nt}")
    X, T = np.meshgrid(x, t)
    plt.figure(figsize=(8, 5))
    plt.pcolormesh(X, T, U, shading="auto", cmap="viridis")
    plt.xlabel("x")
    plt.ylabel("t")
    plt.title(f"Unstable Forward Euler FD (r={r_unstable})")
    plt.colorbar(label="u(x,t)")
    plt.tight_layout()
    plt.savefig(os.path.join(_FIGURES_DIR, f"2_1_unstable_fd_r{r_unstable}.png"), dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()


if __name__ == "__main__":
    U, x, t = run_part_a()
    L2 = run_part_b(U, x)
    run_part_c(U, x, t)
    run_unstable()
