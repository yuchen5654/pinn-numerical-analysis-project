"""
Problems 2.2, 2.3, 2.4: PINN solutions for the 1D heat equation.

PDE:  u_t = nu * u_xx,  x in (0,1),  t in (0,0.5]
IC:   u(x,0) = sin(pi*x) + 0.5*sin(3*pi*x)
BC:   u(0,t) = u(1,t) = 0
nu = 0.01

Sections:
  2.2  AD-PINN
  2.3  FDM-PINN
  2.4  Comparison table + epsilon sweep
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch

from model import PINN, device, train_pinn, plot_loss_curve, plot_heat_comparison, FIGURES_DIR


NU = 0.01


# ------------------------------------------------------------------
# Exact solution
# ------------------------------------------------------------------
def heat_exact(X, T):
    return (
        np.exp(-NU * np.pi**2 * T) * np.sin(np.pi * X)
        + 0.5 * np.exp(-9 * NU * np.pi**2 * T) * np.sin(3 * np.pi * X)
    )


# ------------------------------------------------------------------
# Loss functions
# ------------------------------------------------------------------
def compute_loss_heat_ad(model):
    """PINN loss for heat equation using automatic differentiation."""
    Nr = 10000
    x_r = torch.rand(Nr, 1, device=device, requires_grad=True)
    t_r = 0.5 * torch.rand(Nr, 1, device=device, requires_grad=True)
    xt_r = torch.cat([x_r, t_r], dim=1)

    u = model(xt_r)

    grads = torch.autograd.grad(
        u, xt_r, grad_outputs=torch.ones_like(u), create_graph=True
    )[0]
    u_x = grads[:, 0:1]
    u_t = grads[:, 1:2]

    u_x_grads = torch.autograd.grad(
        u_x, xt_r, grad_outputs=torch.ones_like(u_x), create_graph=True
    )[0]
    u_xx = u_x_grads[:, 0:1]

    residual = u_t - NU * u_xx
    L_r = torch.mean(residual**2)

    Nic = 200
    x_ic = torch.rand(Nic, 1, device=device)
    t_ic = torch.zeros_like(x_ic)
    xt_ic = torch.cat([x_ic, t_ic], dim=1)
    u_ic_pred = model(xt_ic)
    u_ic_true = torch.sin(torch.pi * x_ic) + 0.5 * torch.sin(3 * torch.pi * x_ic)
    L_ic = torch.mean((u_ic_pred - u_ic_true) ** 2)

    Nbc = 200
    t_bc = 0.5 * torch.rand(Nbc, 1, device=device)
    x_left = torch.zeros_like(t_bc)
    x_right = torch.ones_like(t_bc)
    xt_left = torch.cat([x_left, t_bc], dim=1)
    xt_right = torch.cat([x_right, t_bc], dim=1)
    L_bc = torch.mean(model(xt_left) ** 2) + torch.mean(model(xt_right) ** 2)

    return L_r + 20 * L_ic + 20 * L_bc


def compute_loss_heat_fdm(model, epsilon=1e-3):
    """PINN loss for heat equation using central finite differences."""
    N_r = 10000
    x_r = epsilon + (1.0 - 2.0 * epsilon) * torch.rand(N_r, 1, device=device)
    t_r = epsilon + (0.5 - 2.0 * epsilon) * torch.rand(N_r, 1, device=device)

    xt = torch.cat([x_r, t_r], dim=1)
    xt_t_plus = torch.cat([x_r, t_r + epsilon], dim=1)
    xt_t_minus = torch.cat([x_r, t_r - epsilon], dim=1)
    xt_x_plus = torch.cat([x_r + epsilon, t_r], dim=1)
    xt_x_minus = torch.cat([x_r - epsilon, t_r], dim=1)

    u = model(xt)
    u_t_fdm = (model(xt_t_plus) - model(xt_t_minus)) / (2.0 * epsilon)
    u_xx_fdm = (model(xt_x_plus) - 2.0 * u + model(xt_x_minus)) / (epsilon**2)

    residual = u_t_fdm - NU * u_xx_fdm
    loss_r = torch.mean(residual**2)

    N_ic = 200
    x_ic = torch.rand(N_ic, 1, device=device)
    t_ic = torch.zeros(N_ic, 1, device=device)
    xt_ic = torch.cat([x_ic, t_ic], dim=1)
    u_ic_pred = model(xt_ic)
    u_ic_true = torch.sin(torch.pi * x_ic) + 0.5 * torch.sin(3 * torch.pi * x_ic)
    loss_ic = torch.mean((u_ic_pred - u_ic_true) ** 2)

    N_bc = 200
    t_bc = 0.5 * torch.rand(N_bc, 1, device=device)
    x_left = torch.zeros(N_bc, 1, device=device)
    x_right = torch.ones(N_bc, 1, device=device)
    xt_left = torch.cat([x_left, t_bc], dim=1)
    xt_right = torch.cat([x_right, t_bc], dim=1)
    loss_bc = torch.mean(model(xt_left) ** 2) + torch.mean(model(xt_right) ** 2)

    return loss_r + 20.0 * loss_ic + 20.0 * loss_bc


# ------------------------------------------------------------------
# Section 2.2 – AD-PINN
# ------------------------------------------------------------------
def run_2_2():
    print("=" * 60)
    print("Problem 2.2: Heat PINN (Autograd)")
    print("=" * 60)

    model = PINN(input_dim=2, hidden_dim=32, num_layers=3).to(device)
    loss_history, train_time = train_pinn(
        model, compute_loss_heat_ad, epochs=20000, lr=1e-3, log_every=2000
    )

    plot_loss_curve(loss_history, "Heat AD-PINN Training Loss",
                    save_path=os.path.join(FIGURES_DIR, "2_2_heat_ad_loss.png"))
    rel_l2 = plot_heat_comparison(model, heat_exact, label="Heat AD-PINN",
                                  save_path=os.path.join(FIGURES_DIR, "2_2_heat_ad_comparison.png"))

    print(f"\nFinal training loss : {loss_history[-1]:.6e}")
    print(f"Relative L2 error   : {rel_l2:.6e}")
    print(f"Training time       : {train_time:.2f} s")
    return model, rel_l2, train_time, loss_history


# ------------------------------------------------------------------
# Section 2.3 – FDM-PINN
# ------------------------------------------------------------------
def run_2_3(epsilon=1e-3):
    print("=" * 60)
    print(f"Problem 2.3: Heat PINN (FDM, epsilon={epsilon})")
    print("=" * 60)

    model = PINN(input_dim=2, hidden_dim=32, num_layers=3).to(device)
    loss_history, train_time = train_pinn(
        model,
        lambda m: compute_loss_heat_fdm(m, epsilon=epsilon),
        epochs=20000,
        lr=1e-3,
        log_every=2000,
    )

    plot_loss_curve(loss_history, f"Heat FDM-PINN Training Loss (eps={epsilon})",
                    save_path=os.path.join(FIGURES_DIR, "2_3_heat_fdm_loss.png"))
    rel_l2 = plot_heat_comparison(model, heat_exact, label="Heat FDM-PINN",
                                  save_path=os.path.join(FIGURES_DIR, "2_3_heat_fdm_comparison.png"))

    print(f"\nFinal training loss : {loss_history[-1]:.6e}")
    print(f"Relative L2 error   : {rel_l2:.6e}")
    print(f"Training time       : {train_time:.2f} s")
    return model, rel_l2, train_time, loss_history


# ------------------------------------------------------------------
# Section 2.4(a) – comparison table
# ------------------------------------------------------------------
def run_2_4a(ad_results, fdm_results):
    print("=" * 60)
    print("Problem 2.4(a): Comparison Table")
    print("=" * 60)

    _, err_ad, time_ad, hist_ad = ad_results
    _, err_fdm, time_fdm, hist_fdm = fdm_results

    table = pd.DataFrame(
        {
            "Method": ["Heat AD-PINN", "Heat FDM-PINN"],
            "Final Training Loss": [hist_ad[-1], hist_fdm[-1]],
            "Relative L2 Error": [err_ad, err_fdm],
            "Training Time (s)": [time_ad, time_fdm],
        }
    )
    print(table.to_string(index=False))


# ------------------------------------------------------------------
# Section 2.4(b) – epsilon sweep
# ------------------------------------------------------------------
def run_2_4b():
    print("=" * 60)
    print("Problem 2.4(b): Heat FDM-PINN epsilon sweep")
    print("=" * 60)

    eps_values = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
    errors, losses, times = [], [], []

    for eps in eps_values:
        print(f"\n  epsilon = {eps}")
        model_eps = PINN(input_dim=2, hidden_dim=32, num_layers=3).to(device)
        loss_eps, time_eps = train_pinn(
            model_eps,
            lambda m, e=eps: compute_loss_heat_fdm(m, epsilon=e),
            epochs=20000,
            lr=1e-3,
            log_every=2000,
        )
        rel_l2 = plot_heat_comparison(
            model_eps, heat_exact, label=f"FDM eps={eps}",
            save_path=os.path.join(FIGURES_DIR, f"2_4b_heat_fdm_eps_{eps:.0e}.png")
        )
        errors.append(rel_l2)
        losses.append(loss_eps[-1])
        times.append(time_eps)

    table = pd.DataFrame(
        {
            "epsilon": eps_values,
            "Final Training Loss": losses,
            "Relative L2 Error": errors,
            "Training Time (s)": times,
        }
    )
    print("\n", table.to_string(index=False))

    plt.figure(figsize=(6, 4))
    plt.loglog(eps_values, errors, marker="o")
    plt.xlabel(r"$\epsilon$")
    plt.ylabel("Relative L2 Error")
    plt.title(r"Heat FDM-PINN Relative $L^2$ Error vs. $\epsilon$")
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "2_4b_heat_fdm_eps_sweep.png"), dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()


# ------------------------------------------------------------------
# Section 3(b) helper – vary Nr for heat
# ------------------------------------------------------------------
def run_nr_sweep_heat():
    Nr_values = [500, 2000, 10000, 50000]
    ad_errors, fdm_errors = [], []

    for Nr in Nr_values:
        print(f"\n  Nr = {Nr}")

        def loss_ad(m, N=Nr):
            x_r = torch.rand(N, 1, device=device, requires_grad=True)
            t_r = 0.5 * torch.rand(N, 1, device=device, requires_grad=True)
            xt_r = torch.cat([x_r, t_r], dim=1)
            u = m(xt_r)
            grads = torch.autograd.grad(u, xt_r, torch.ones_like(u), create_graph=True)[0]
            u_x, u_t = grads[:, 0:1], grads[:, 1:2]
            u_xx = torch.autograd.grad(u_x, xt_r, torch.ones_like(u_x), create_graph=True)[0][:, 0:1]
            L_r = torch.mean((u_t - NU * u_xx) ** 2)
            x_ic = torch.rand(200, 1, device=device)
            xt_ic = torch.cat([x_ic, torch.zeros_like(x_ic)], dim=1)
            u_ic_true = torch.sin(torch.pi * x_ic) + 0.5 * torch.sin(3 * torch.pi * x_ic)
            L_ic = torch.mean((m(xt_ic) - u_ic_true) ** 2)
            t_bc = 0.5 * torch.rand(200, 1, device=device)
            L_bc = torch.mean(m(torch.cat([torch.zeros(200, 1, device=device), t_bc], 1)) ** 2) + \
                   torch.mean(m(torch.cat([torch.ones(200, 1, device=device), t_bc], 1)) ** 2)
            return L_r + 20 * L_ic + 20 * L_bc

        m_ad = PINN(2, 32, 3).to(device)
        train_pinn(m_ad, loss_ad, epochs=10000, lr=1e-3, log_every=2000)
        ad_errors.append(plot_heat_comparison(
            m_ad, heat_exact, label=f"AD Nr={Nr}",
            save_path=os.path.join(FIGURES_DIR, f"3b_heat_ad_Nr{Nr}.png")))

        def loss_fdm(m, N=Nr, eps=1e-3):
            x_r = eps + (1 - 2 * eps) * torch.rand(N, 1, device=device)
            t_r = eps + (0.5 - 2 * eps) * torch.rand(N, 1, device=device)
            xt = torch.cat([x_r, t_r], dim=1)
            u = m(xt)
            u_t = (m(torch.cat([x_r, t_r + eps], 1)) - m(torch.cat([x_r, t_r - eps], 1))) / (2 * eps)
            u_xx = (m(torch.cat([x_r + eps, t_r], 1)) - 2 * u + m(torch.cat([x_r - eps, t_r], 1))) / eps**2
            L_r = torch.mean((u_t - NU * u_xx) ** 2)
            x_ic = torch.rand(200, 1, device=device)
            xt_ic = torch.cat([x_ic, torch.zeros_like(x_ic)], dim=1)
            u_ic_true = torch.sin(torch.pi * x_ic) + 0.5 * torch.sin(3 * torch.pi * x_ic)
            L_ic = torch.mean((m(xt_ic) - u_ic_true) ** 2)
            t_bc = 0.5 * torch.rand(200, 1, device=device)
            L_bc = torch.mean(m(torch.cat([torch.zeros(200, 1, device=device), t_bc], 1)) ** 2) + \
                   torch.mean(m(torch.cat([torch.ones(200, 1, device=device), t_bc], 1)) ** 2)
            return L_r + 20 * L_ic + 20 * L_bc

        m_fdm = PINN(2, 32, 3).to(device)
        train_pinn(m_fdm, loss_fdm, epochs=10000, lr=1e-3, log_every=2000)
        fdm_errors.append(plot_heat_comparison(
            m_fdm, heat_exact, label=f"FDM Nr={Nr}",
            save_path=os.path.join(FIGURES_DIR, f"3b_heat_fdm_Nr{Nr}.png")))

    plt.figure(figsize=(6, 4))
    plt.loglog(Nr_values, ad_errors, marker="o", label="AD-PINN")
    plt.loglog(Nr_values, fdm_errors, marker="s", label="FDM-PINN")
    plt.xlabel(r"$N_r$")
    plt.ylabel("Relative L2 Error")
    plt.title("Heat PINN Error vs. Collocation Points")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "3b_heat_nr_sweep.png"), dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()


if __name__ == "__main__":
    ad_results = run_2_2()
    fdm_results = run_2_3()
    run_2_4a(ad_results, fdm_results)
    run_2_4b()
