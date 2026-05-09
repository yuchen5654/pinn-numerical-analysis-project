"""
Problems 1.2, 1.3, 1.4: PINN solutions for the first-order ODE.

ODE:  du/dt = -5u + 5cos(t) - sin(t),  t in [0,5],  u(0) = 0
Exact: u(t) = cos(t) - exp(-5t)

Sections:
  1.2  AD-PINN  (automatic differentiation)
  1.3  FDM-PINN (finite-difference approximation)
  1.4  Comparison table + epsilon sweep
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch

from model import PINN, device, train_pinn, plot_loss_curve, plot_ode_comparison, FIGURES_DIR


# ------------------------------------------------------------------
# Exact solution
# ------------------------------------------------------------------
def ode_exact(t):
    return np.cos(t) - np.exp(-5 * t)


# ------------------------------------------------------------------
# Loss functions
# ------------------------------------------------------------------
def compute_loss_ode_ad(model):
    """PINN loss for the ODE using automatic differentiation."""
    N_r = 500
    t_r = 5.0 * torch.rand(N_r, 1, device=device)
    t_r.requires_grad_(True)

    u = model(t_r)
    du_dt = torch.autograd.grad(
        outputs=u,
        inputs=t_r,
        grad_outputs=torch.ones_like(u),
        create_graph=True,
    )[0]

    residual = du_dt + 5 * u - 5 * torch.cos(t_r) + torch.sin(t_r)
    loss_r = torch.mean(residual**2)

    t0 = torch.zeros(1, 1, device=device)
    loss_ic = torch.mean(model(t0) ** 2)

    return loss_r + 50 * loss_ic


def compute_loss_ode_fdm(model, epsilon=1e-3):
    """PINN loss for the ODE using central finite differences."""
    N_r = 500
    t_r = epsilon + (5.0 - 2.0 * epsilon) * torch.rand(N_r, 1, device=device)

    u_plus = model(t_r + epsilon)
    u_minus = model(t_r - epsilon)
    du_dt_fdm = (u_plus - u_minus) / (2.0 * epsilon)

    u = model(t_r)
    residual = du_dt_fdm + 5.0 * u - 5.0 * torch.cos(t_r) + torch.sin(t_r)
    loss_r = torch.mean(residual**2)

    t0 = torch.zeros(1, 1, device=device)
    loss_ic = torch.mean(model(t0) ** 2)

    return loss_r + 50.0 * loss_ic


# ------------------------------------------------------------------
# Section 1.2 – AD-PINN
# ------------------------------------------------------------------
def run_1_2():
    print("=" * 50)
    print("Problem 1.2: ODE PINN (Autograd)")
    print("=" * 50)

    model = PINN(1, 32, 3).to(device)
    loss_history, train_time = train_pinn(
        model, compute_loss_ode_ad, epochs=10000, lr=1e-3, log_every=2000
    )

    plot_loss_curve(loss_history, "ODE AD-PINN Training Loss (log scale)",
                    save_path=os.path.join(FIGURES_DIR, "1_2_ode_ad_loss.png"))
    error = plot_ode_comparison(model, ode_exact, label="PINN-AD",
                                save_path=os.path.join(FIGURES_DIR, "1_2_ode_ad_comparison.png"))

    print(f"\nTraining Time : {train_time:.2f} s")
    print(f"Max Abs Error : {error:.6e}")
    return model, error, train_time, loss_history


# ------------------------------------------------------------------
# Section 1.3 – FDM-PINN
# ------------------------------------------------------------------
def run_1_3(epsilon=1e-3):
    print("=" * 50)
    print("Problem 1.3: ODE PINN (FDM, epsilon={})".format(epsilon))
    print("=" * 50)

    model = PINN(1, 32, 3).to(device)
    loss_history, train_time = train_pinn(
        model,
        lambda m: compute_loss_ode_fdm(m, epsilon=epsilon),
        epochs=10000,
        lr=1e-3,
        log_every=2000,
    )

    plot_loss_curve(loss_history, f"ODE FDM-PINN Training Loss (eps={epsilon})",
                    save_path=os.path.join(FIGURES_DIR, "1_3_ode_fdm_loss.png"))
    error = plot_ode_comparison(model, ode_exact, label="PINN-FDM",
                                save_path=os.path.join(FIGURES_DIR, "1_3_ode_fdm_comparison.png"))

    print(f"\nTraining Time : {train_time:.2f} s")
    print(f"Max Abs Error : {error:.6e}")
    return model, error, train_time, loss_history


# ------------------------------------------------------------------
# Section 1.4(a) – comparison table
# ------------------------------------------------------------------
def run_1_4a(ad_results, fdm_results):
    print("=" * 50)
    print("Problem 1.4(a): Comparison Table")
    print("=" * 50)

    _, err_ad, time_ad, hist_ad = ad_results
    _, err_fdm, time_fdm, hist_fdm = fdm_results

    table = pd.DataFrame(
        {
            "Method": ["AD-PINN", "FDM-PINN"],
            "Final Training Loss": [hist_ad[-1], hist_fdm[-1]],
            "Max Absolute Error": [err_ad, err_fdm],
            "Training Time (s)": [time_ad, time_fdm],
        }
    )
    print(table.to_string(index=False))


# ------------------------------------------------------------------
# Section 1.4(b) – epsilon sweep
# ------------------------------------------------------------------
def run_1_4b():
    print("=" * 50)
    print("Problem 1.4(b): FDM-PINN epsilon sweep")
    print("=" * 50)

    eps_values = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
    fdm_errors = []
    fdm_losses = []
    fdm_times = []

    for eps in eps_values:
        print(f"\n  epsilon = {eps}")
        model_eps = PINN(1, 32, 3).to(device)
        loss_eps, time_eps = train_pinn(
            model_eps,
            lambda m, e=eps: compute_loss_ode_fdm(m, epsilon=e),
            epochs=10000,
            lr=1e-3,
            log_every=2000,
        )
        err_eps = plot_ode_comparison(
            model_eps, ode_exact, label=f"FDM eps={eps}",
            save_path=os.path.join(FIGURES_DIR, f"1_4b_ode_fdm_eps_{eps:.0e}.png")
        )
        fdm_errors.append(err_eps)
        fdm_losses.append(loss_eps[-1])
        fdm_times.append(time_eps)

    table = pd.DataFrame(
        {
            "epsilon": eps_values,
            "Final Training Loss": fdm_losses,
            "Max Absolute Error": fdm_errors,
            "Training Time (s)": fdm_times,
        }
    )
    print("\n", table.to_string(index=False))

    plt.figure(figsize=(6, 4))
    plt.loglog(eps_values, fdm_errors, marker="o")
    plt.xlabel(r"$\epsilon$")
    plt.ylabel("Max Absolute Error")
    plt.title(r"FDM-PINN Error vs. $\epsilon$")
    plt.grid(True, which="both")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "1_4b_ode_fdm_eps_sweep.png"), dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()


# ------------------------------------------------------------------
# Section 3(b) helper – vary Nr
# ------------------------------------------------------------------
def run_nr_sweep_ode():
    Nr_values = [100, 500, 2000, 10000]
    ad_errors, fdm_errors = [], []

    for Nr in Nr_values:
        print(f"\n  Nr = {Nr}")

        # AD
        def loss_ad(m, N=Nr):
            t_r = 5.0 * torch.rand(N, 1, device=device)
            t_r.requires_grad_(True)
            u = m(t_r)
            du_dt = torch.autograd.grad(u, t_r, torch.ones_like(u), create_graph=True)[0]
            residual = du_dt + 5 * u - 5 * torch.cos(t_r) + torch.sin(t_r)
            t0 = torch.zeros(1, 1, device=device)
            return torch.mean(residual**2) + 50 * torch.mean(m(t0) ** 2)

        m_ad = PINN(1, 32, 3).to(device)
        train_pinn(m_ad, loss_ad, epochs=5000, lr=1e-3, log_every=1000)
        ad_errors.append(plot_ode_comparison(
            m_ad, ode_exact, label=f"AD Nr={Nr}",
            save_path=os.path.join(FIGURES_DIR, f"3b_ode_ad_Nr{Nr}.png")))

        # FDM
        def loss_fdm(m, N=Nr, eps=1e-3):
            t_r = eps + (5.0 - 2 * eps) * torch.rand(N, 1, device=device)
            u_p = m(t_r + eps)
            u_m = m(t_r - eps)
            u = m(t_r)
            du = (u_p - u_m) / (2 * eps)
            residual = du + 5 * u - 5 * torch.cos(t_r) + torch.sin(t_r)
            t0 = torch.zeros(1, 1, device=device)
            return torch.mean(residual**2) + 50 * torch.mean(m(t0) ** 2)

        m_fdm = PINN(1, 32, 3).to(device)
        train_pinn(m_fdm, loss_fdm, epochs=5000, lr=1e-3, log_every=1000)
        fdm_errors.append(plot_ode_comparison(
            m_fdm, ode_exact, label=f"FDM Nr={Nr}",
            save_path=os.path.join(FIGURES_DIR, f"3b_ode_fdm_Nr{Nr}.png")))

    plt.figure(figsize=(6, 4))
    plt.loglog(Nr_values, ad_errors, marker="o", label="AD-PINN")
    plt.loglog(Nr_values, fdm_errors, marker="s", label="FDM-PINN")
    plt.xlabel(r"$N_r$")
    plt.ylabel("Max Absolute Error")
    plt.title("ODE PINN Error vs. Collocation Points")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "3b_ode_nr_sweep.png"), dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()


if __name__ == "__main__":
    ad_results = run_1_2()
    fdm_results = run_1_3()
    run_1_4a(ad_results, fdm_results)
    run_1_4b()
