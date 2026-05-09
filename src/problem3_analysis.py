"""
Problem 3: Analysis and Discussion helpers.

3(a) Summary table across all methods
3(b) Collocation point sweep (calls helpers in problem1_pinn / problem2_pinn)
3(c) Network size experiments
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch

import os
from model import PINN, device, train_pinn, plot_ode_comparison, plot_heat_comparison, FIGURES_DIR

NU = 0.01


# ------------------------------------------------------------------
# Exact solutions
# ------------------------------------------------------------------
def ode_exact(t):
    return np.cos(t) - np.exp(-5 * t)


def heat_exact(X, T):
    return (
        np.exp(-NU * np.pi**2 * T) * np.sin(np.pi * X)
        + 0.5 * np.exp(-9 * NU * np.pi**2 * T) * np.sin(3 * np.pi * X)
    )


# ------------------------------------------------------------------
# 3(c): Network size study
# ------------------------------------------------------------------
def _ode_ad_loss(model, N_r=500):
    t_r = 5.0 * torch.rand(N_r, 1, device=device)
    t_r.requires_grad_(True)
    u = model(t_r)
    du_dt = torch.autograd.grad(u, t_r, torch.ones_like(u), create_graph=True)[0]
    residual = du_dt + 5 * u - 5 * torch.cos(t_r) + torch.sin(t_r)
    t0 = torch.zeros(1, 1, device=device)
    return torch.mean(residual**2) + 50 * torch.mean(model(t0) ** 2)


def _ode_fdm_loss(model, N_r=500, eps=1e-3):
    t_r = eps + (5.0 - 2 * eps) * torch.rand(N_r, 1, device=device)
    u_p = model(t_r + eps)
    u_m = model(t_r - eps)
    u = model(t_r)
    du = (u_p - u_m) / (2 * eps)
    residual = du + 5 * u - 5 * torch.cos(t_r) + torch.sin(t_r)
    t0 = torch.zeros(1, 1, device=device)
    return torch.mean(residual**2) + 50 * torch.mean(model(t0) ** 2)


def _heat_ad_loss(model, Nr=10000):
    x_r = torch.rand(Nr, 1, device=device, requires_grad=True)
    t_r = 0.5 * torch.rand(Nr, 1, device=device, requires_grad=True)
    xt_r = torch.cat([x_r, t_r], dim=1)
    u = model(xt_r)
    grads = torch.autograd.grad(u, xt_r, torch.ones_like(u), create_graph=True)[0]
    u_x, u_t = grads[:, 0:1], grads[:, 1:2]
    u_xx = torch.autograd.grad(u_x, xt_r, torch.ones_like(u_x), create_graph=True)[0][:, 0:1]
    L_r = torch.mean((u_t - NU * u_xx) ** 2)
    x_ic = torch.rand(200, 1, device=device)
    xt_ic = torch.cat([x_ic, torch.zeros_like(x_ic)], dim=1)
    u_ic_true = torch.sin(torch.pi * x_ic) + 0.5 * torch.sin(3 * torch.pi * x_ic)
    L_ic = torch.mean((model(xt_ic) - u_ic_true) ** 2)
    t_bc = 0.5 * torch.rand(200, 1, device=device)
    L_bc = torch.mean(model(torch.cat([torch.zeros(200, 1, device=device), t_bc], 1)) ** 2) + \
           torch.mean(model(torch.cat([torch.ones(200, 1, device=device), t_bc], 1)) ** 2)
    return L_r + 20 * L_ic + 20 * L_bc


def _heat_fdm_loss(model, Nr=10000, eps=1e-3):
    x_r = eps + (1 - 2 * eps) * torch.rand(Nr, 1, device=device)
    t_r = eps + (0.5 - 2 * eps) * torch.rand(Nr, 1, device=device)
    xt = torch.cat([x_r, t_r], dim=1)
    u = model(xt)
    u_t = (model(torch.cat([x_r, t_r + eps], 1)) - model(torch.cat([x_r, t_r - eps], 1))) / (2 * eps)
    u_xx = (model(torch.cat([x_r + eps, t_r], 1)) - 2 * u + model(torch.cat([x_r - eps, t_r], 1))) / eps**2
    L_r = torch.mean((u_t - NU * u_xx) ** 2)
    x_ic = torch.rand(200, 1, device=device)
    xt_ic = torch.cat([x_ic, torch.zeros_like(x_ic)], dim=1)
    u_ic_true = torch.sin(torch.pi * x_ic) + 0.5 * torch.sin(3 * torch.pi * x_ic)
    L_ic = torch.mean((model(xt_ic) - u_ic_true) ** 2)
    t_bc = 0.5 * torch.rand(200, 1, device=device)
    L_bc = torch.mean(model(torch.cat([torch.zeros(200, 1, device=device), t_bc], 1)) ** 2) + \
           torch.mean(model(torch.cat([torch.ones(200, 1, device=device), t_bc], 1)) ** 2)
    return L_r + 20 * L_ic + 20 * L_bc


def run_network_size_study():
    """3(c): Compare small / medium / large networks for ODE and heat."""
    print("=" * 60)
    print("Problem 3(c): Network Size Study")
    print("=" * 60)

    configs = [
        ("Small  (2L-16H)", 2, 16),
        ("Medium (3L-32H)", 3, 32),
        ("Large  (5L-64H)", 5, 64),
    ]

    rows = []
    for label, n_layers, hidden in configs:
        print(f"\n  Config: {label}")

        slug = label.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")

        # ODE – AD
        m = PINN(1, hidden, n_layers).to(device)
        _, t = train_pinn(m, _ode_ad_loss, epochs=5000, lr=1e-3, log_every=5000)
        err = plot_ode_comparison(m, ode_exact, label=f"ODE AD {label}",
                                  save_path=os.path.join(FIGURES_DIR, f"3c_ode_ad_{slug}.png"))
        rows.append({"Problem": "ODE", "Method": "AD-PINN", "Network": label, "Error": float(err), "Time(s)": round(t, 1)})

        # ODE – FDM
        m = PINN(1, hidden, n_layers).to(device)
        _, t = train_pinn(m, _ode_fdm_loss, epochs=5000, lr=1e-3, log_every=5000)
        err = plot_ode_comparison(m, ode_exact, label=f"ODE FDM {label}",
                                  save_path=os.path.join(FIGURES_DIR, f"3c_ode_fdm_{slug}.png"))
        rows.append({"Problem": "ODE", "Method": "FDM-PINN", "Network": label, "Error": float(err), "Time(s)": round(t, 1)})

        # Heat – AD
        m = PINN(2, hidden, n_layers).to(device)
        _, t = train_pinn(m, _heat_ad_loss, epochs=10000, lr=1e-3, log_every=10000)
        err = plot_heat_comparison(m, heat_exact, label=f"Heat AD {label}",
                                   save_path=os.path.join(FIGURES_DIR, f"3c_heat_ad_{slug}.png"))
        rows.append({"Problem": "Heat", "Method": "AD-PINN", "Network": label, "Error": float(err), "Time(s)": round(t, 1)})

        # Heat – FDM
        m = PINN(2, hidden, n_layers).to(device)
        _, t = train_pinn(m, _heat_fdm_loss, epochs=10000, lr=1e-3, log_every=10000)
        err = plot_heat_comparison(m, heat_exact, label=f"Heat FDM {label}",
                                   save_path=os.path.join(FIGURES_DIR, f"3c_heat_fdm_{slug}.png"))
        rows.append({"Problem": "Heat", "Method": "FDM-PINN", "Network": label, "Error": float(err), "Time(s)": round(t, 1)})

    df = pd.DataFrame(rows)
    print("\n", df.to_string(index=False))
    return df


if __name__ == "__main__":
    run_network_size_study()
