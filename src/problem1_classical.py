"""
Problem 1.1: Classical numerical solutions for the first-order ODE.

ODE:  du/dt = -5u + 5cos(t) - sin(t),  t in [0,5],  u(0) = 0
Exact: u(t) = cos(t) - exp(-5t)

Sections:
  (a) Forward Euler with h = 0.01
  (b) RK4 with h = 0.01
  (c) Convergence table for h in {0.01, 0.005, 0.001}
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

_FIGURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
os.makedirs(_FIGURES_DIR, exist_ok=True)


def f(t, u):
    return -5 * u + 5 * np.cos(t) - np.sin(t)


def u_exact(t):
    return np.cos(t) - np.exp(-5 * t)


def forward_euler(h):
    a, b = 0, 5
    N = int((b - a) / h)
    t = np.linspace(a, b, N + 1)
    w = np.zeros(N + 1)
    for n in range(N):
        w[n + 1] = w[n] + h * f(t[n], w[n])
    return t, w


def rk4(h):
    a, b = 0, 5
    N = int((b - a) / h)
    t = np.linspace(a, b, N + 1)
    w = np.zeros(N + 1)
    for n in range(N):
        k1 = f(t[n], w[n])
        k2 = f(t[n] + h / 2, w[n] + (h / 2) * k1)
        k3 = f(t[n] + h / 2, w[n] + (h / 2) * k2)
        k4 = f(t[n] + h, w[n] + h * k3)
        w[n + 1] = w[n] + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    return t, w


def max_error(h, method):
    t, w = method(h)
    return np.max(np.abs(w - u_exact(t)))


def observed_orders(errors, h_values):
    orders = [None]
    for i in range(1, len(errors)):
        p = np.log(errors[i - 1] / errors[i]) / np.log(h_values[i - 1] / h_values[i])
        orders.append(round(p, 6))
    return orders


def run_part_a():
    print("--- 1.1(a): Forward Euler, h = 0.01 ---")
    h = 0.01
    t, w = forward_euler(h)
    u_true = u_exact(t)

    print(f"Maximum absolute error: {np.max(np.abs(u_true - w)):.6e}")

    plt.figure(figsize=(8, 5))
    plt.plot(t, u_true, label="Exact solution", linewidth=2)
    plt.plot(t, w, "--", label=f"Forward Euler, h={h}", linewidth=2)
    plt.xlabel("t")
    plt.ylabel("u(t)")
    plt.title("Forward Euler Approximation vs. Exact Solution")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(_FIGURES_DIR, "1_1a_forward_euler.png"), dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()


def run_part_b():
    print("--- 1.1(b): RK4, h = 0.01 ---")
    h = 0.01
    t, w = rk4(h)
    u_true = u_exact(t)

    print(f"Maximum absolute error for RK4: {np.max(np.abs(u_true - w)):.6e}")

    plt.figure(figsize=(8, 5))
    plt.plot(t, u_true, label="Exact solution", linewidth=2)
    plt.plot(t, w, "--", label=f"RK4, h={h}", linewidth=2)
    plt.xlabel("t")
    plt.ylabel("u(t)")
    plt.title("RK4 Approximation vs. Exact Solution")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(_FIGURES_DIR, "1_1b_rk4.png"), dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()


def run_part_c():
    print("--- 1.1(c): Convergence table ---")
    h_values = [0.01, 0.005, 0.001]

    euler_errors = [max_error(h, forward_euler) for h in h_values]
    rk4_errors = [max_error(h, rk4) for h in h_values]

    euler_orders = observed_orders(euler_errors, h_values)
    rk4_orders = observed_orders(rk4_errors, h_values)

    table = pd.DataFrame(
        {
            "h": h_values,
            "Forward Euler Error": euler_errors,
            "Euler Observed Order": euler_orders,
            "RK4 Error": rk4_errors,
            "RK4 Observed Order": rk4_orders,
        }
    )
    print(table.to_string(index=False))
    return euler_errors, rk4_errors, h_values


if __name__ == "__main__":
    run_part_a()
    run_part_b()
    run_part_c()
