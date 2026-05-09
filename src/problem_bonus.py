"""
Bonus [15 pts]: Inverse Problem for the 1D Heat Equation.

Assume nu is unknown. Add nu as a trainable parameter and supply
N_data = 50 noisy observations (sigma = 0.01) from the exact solution.
Recover nu and report |nu_rec - nu_true| / nu_true for both AD and FDM.
"""
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from model import PINN, device, FIGURES_DIR

NU_TRUE = 0.01
SIGMA   = 0.01
N_DATA  = 50


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def heat_exact_np(x, t):
    return (np.exp(-NU_TRUE * np.pi**2 * t) * np.sin(np.pi * x)
            + 0.5 * np.exp(-9 * NU_TRUE * np.pi**2 * t) * np.sin(3 * np.pi * x))


def generate_observations(seed=42):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 1,   N_DATA).astype(np.float32)
    t = rng.uniform(0, 0.5, N_DATA).astype(np.float32)
    u = (heat_exact_np(x, t)
         + SIGMA * rng.standard_normal(N_DATA).astype(np.float32))
    return (torch.tensor(x, device=device),
            torch.tensor(t, device=device),
            torch.tensor(u, device=device))


def _data_loss(model, x_d, t_d, u_d):
    xt = torch.stack([x_d, t_d], dim=1)
    return torch.mean((model(xt).squeeze() - u_d) ** 2)


# ------------------------------------------------------------------
# Training loops (custom, because nu is co-optimized with the network)
# ------------------------------------------------------------------
def run_inverse_ad(epochs=20000):
    print("=" * 60)
    print("Bonus: Inverse Problem — AD-PINN")
    print("=" * 60)

    x_d, t_d, u_d = generate_observations()
    model  = PINN(input_dim=2, hidden_dim=32, num_layers=3).to(device)
    # Parameterize nu = softplus(nu_raw) to guarantee positivity.
    # softplus(-2.2) ≈ 0.10, so we start about 10x above the true value.
    nu_raw = nn.Parameter(torch.tensor(-2.2, device=device))
    opt    = torch.optim.Adam(list(model.parameters()) + [nu_raw], lr=1e-3)

    loss_hist, nu_hist = [], []
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        opt.zero_grad()
        nu = F.softplus(nu_raw)

        # -- PDE residual (AD) --
        Nr   = 10000
        x_r  = torch.rand(Nr, 1, device=device, requires_grad=True)
        t_r  = 0.5 * torch.rand(Nr, 1, device=device, requires_grad=True)
        xt_r = torch.cat([x_r, t_r], dim=1)
        u    = model(xt_r)
        g    = torch.autograd.grad(u, xt_r, torch.ones_like(u), create_graph=True)[0]
        u_x, u_t = g[:, 0:1], g[:, 1:2]
        u_xx = torch.autograd.grad(
            u_x, xt_r, torch.ones_like(u_x), create_graph=True
        )[0][:, 0:1]
        L_r = torch.mean((u_t - nu * u_xx) ** 2)

        # -- IC --
        x_ic    = torch.rand(200, 1, device=device)
        xt_ic   = torch.cat([x_ic, torch.zeros_like(x_ic)], dim=1)
        u_ic_ex = torch.sin(torch.pi * x_ic) + 0.5 * torch.sin(3 * torch.pi * x_ic)
        L_ic    = torch.mean((model(xt_ic) - u_ic_ex) ** 2)

        # -- BC --
        t_bc = 0.5 * torch.rand(200, 1, device=device)
        L_bc = (torch.mean(model(torch.cat([torch.zeros(200, 1, device=device), t_bc], 1)) ** 2)
              + torch.mean(model(torch.cat([torch.ones(200, 1, device=device),  t_bc], 1)) ** 2))

        # -- Data --
        L_data = _data_loss(model, x_d, t_d, u_d)

        loss = L_r + 20 * L_ic + 20 * L_bc + 100 * L_data
        loss.backward()
        opt.step()

        with torch.no_grad():
            nu_val = F.softplus(nu_raw).item()
        loss_hist.append(loss.item())
        nu_hist.append(nu_val)

        if epoch % 2000 == 0:
            print(f"  Epoch {epoch}/{epochs}  loss={loss.item():.4e}  nu={nu_val:.6f}")

    elapsed = time.time() - t0
    nu_rec  = F.softplus(nu_raw).item()
    rel_err = abs(nu_rec - NU_TRUE) / NU_TRUE
    print(f"  Training time : {elapsed:.1f}s")
    print(f"  nu_true       = {NU_TRUE}")
    print(f"  nu_recovered  = {nu_rec:.6f}")
    print(f"  Relative error= {rel_err:.4e}")
    return nu_rec, rel_err, elapsed, loss_hist, nu_hist


def run_inverse_fdm(epochs=20000, epsilon=1e-3):
    print("=" * 60)
    print("Bonus: Inverse Problem — FDM-PINN")
    print("=" * 60)

    x_d, t_d, u_d = generate_observations()
    model  = PINN(input_dim=2, hidden_dim=32, num_layers=3).to(device)
    nu_raw = nn.Parameter(torch.tensor(-2.2, device=device))
    opt    = torch.optim.Adam(list(model.parameters()) + [nu_raw], lr=1e-3)

    eps = epsilon
    loss_hist, nu_hist = [], []
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        opt.zero_grad()
        nu = F.softplus(nu_raw)

        # -- PDE residual (FDM) --
        Nr  = 10000
        x_r = eps + (1 - 2 * eps) * torch.rand(Nr, 1, device=device)
        t_r = eps + (0.5 - 2 * eps) * torch.rand(Nr, 1, device=device)
        xt  = torch.cat([x_r, t_r], dim=1)
        u   = model(xt)
        u_t  = (model(torch.cat([x_r,       t_r + eps], 1))
              - model(torch.cat([x_r,       t_r - eps], 1))) / (2 * eps)
        u_xx = (model(torch.cat([x_r + eps, t_r],       1))
              - 2 * u
              + model(torch.cat([x_r - eps, t_r],       1))) / eps ** 2
        L_r = torch.mean((u_t - nu * u_xx) ** 2)

        # -- IC --
        x_ic    = torch.rand(200, 1, device=device)
        xt_ic   = torch.cat([x_ic, torch.zeros_like(x_ic)], dim=1)
        u_ic_ex = torch.sin(torch.pi * x_ic) + 0.5 * torch.sin(3 * torch.pi * x_ic)
        L_ic    = torch.mean((model(xt_ic) - u_ic_ex) ** 2)

        # -- BC --
        t_bc = 0.5 * torch.rand(200, 1, device=device)
        L_bc = (torch.mean(model(torch.cat([torch.zeros(200, 1, device=device), t_bc], 1)) ** 2)
              + torch.mean(model(torch.cat([torch.ones(200, 1, device=device),  t_bc], 1)) ** 2))

        # -- Data --
        L_data = _data_loss(model, x_d, t_d, u_d)

        loss = L_r + 20 * L_ic + 20 * L_bc + 100 * L_data
        loss.backward()
        opt.step()

        with torch.no_grad():
            nu_val = F.softplus(nu_raw).item()
        loss_hist.append(loss.item())
        nu_hist.append(nu_val)

        if epoch % 2000 == 0:
            print(f"  Epoch {epoch}/{epochs}  loss={loss.item():.4e}  nu={nu_val:.6f}")

    elapsed = time.time() - t0
    nu_rec  = F.softplus(nu_raw).item()
    rel_err = abs(nu_rec - NU_TRUE) / NU_TRUE
    print(f"  Training time : {elapsed:.1f}s")
    print(f"  nu_true       = {NU_TRUE}")
    print(f"  nu_recovered  = {nu_rec:.6f}")
    print(f"  Relative error= {rel_err:.4e}")
    return nu_rec, rel_err, elapsed, loss_hist, nu_hist


# ------------------------------------------------------------------
# Plotting
# ------------------------------------------------------------------
def plot_inverse_results(ad_loss, ad_nu_hist, fdm_loss, fdm_nu_hist, nu_ad, nu_fdm):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].semilogy(ad_loss,  label="AD-PINN")
    axes[0].semilogy(fdm_loss, label="FDM-PINN")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Inverse Problem: Training Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(ad_nu_hist,  label=f"AD-PINN  (ν={nu_ad:.5f})")
    axes[1].plot(fdm_nu_hist, label=f"FDM-PINN (ν={nu_fdm:.5f})")
    axes[1].axhline(NU_TRUE, color="k", linestyle="--", label=f"ν_true = {NU_TRUE}")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("ν")
    axes[1].set_title("ν Convergence During Training")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "bonus_inverse.png"), dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print(f"Bonus: Inverse Problem  (nu_true={NU_TRUE}, sigma={SIGMA}, N_data={N_DATA})")
    print("=" * 60)

    nu_ad,  err_ad,  t_ad,  loss_ad,  nu_h_ad  = run_inverse_ad(epochs=20000)
    nu_fdm, err_fdm, t_fdm, loss_fdm, nu_h_fdm = run_inverse_fdm(epochs=20000)

    plot_inverse_results(loss_ad, nu_h_ad, loss_fdm, nu_h_fdm, nu_ad, nu_fdm)

    table = pd.DataFrame({
        "Method":            ["AD-PINN",  "FDM-PINN"],
        "nu_true":           [NU_TRUE,    NU_TRUE],
        "nu_recovered":      [nu_ad,      nu_fdm],
        "Rel. Error":        [err_ad,     err_fdm],
        "Training Time (s)": [t_ad,       t_fdm],
    })
    print("\n" + "=" * 60)
    print("Bonus Summary")
    print("=" * 60)
    print(table.to_string(index=False))
