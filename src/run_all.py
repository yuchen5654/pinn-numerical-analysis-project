"""
run_all.py – Run the full project end-to-end and print a summary table.

This script ties together all four problem modules:
  - problem1_classical  (1.1 a/b/c)
  - problem1_pinn       (1.2, 1.3, 1.4)
  - problem2_fd         (2.1 a/b/c + unstable run)
  - problem2_pinn       (2.2, 2.3, 2.4)
  - problem3_analysis   (3c network size study)

Run from inside the src/ directory:
    python run_all.py
"""
import pandas as pd

# ------------------------------------------------------------------ #
# Section 1.1 – Classical ODE solvers                                 #
# ------------------------------------------------------------------ #
print("\n" + "=" * 60)
print("SECTION 1.1 – Classical ODE Solvers")
print("=" * 60)
from problem1_classical import run_part_a, run_part_b, run_part_c

run_part_a()
run_part_b()
euler_errors, rk4_errors, h_values = run_part_c()

# ------------------------------------------------------------------ #
# Section 1.2 / 1.3 / 1.4 – ODE PINNs                                #
# ------------------------------------------------------------------ #
print("\n" + "=" * 60)
print("SECTION 1.2 – ODE AD-PINN")
print("=" * 60)
from problem1_pinn import run_1_2, run_1_3, run_1_4a, run_1_4b

ad_ode = run_1_2()

print("\n" + "=" * 60)
print("SECTION 1.3 – ODE FDM-PINN")
print("=" * 60)
fdm_ode = run_1_3()

print("\n" + "=" * 60)
print("SECTION 1.4 – ODE Comparison")
print("=" * 60)
run_1_4a(ad_ode, fdm_ode)
run_1_4b()

# ------------------------------------------------------------------ #
# Section 2.1 – Finite-Difference Reference                           #
# ------------------------------------------------------------------ #
print("\n" + "=" * 60)
print("SECTION 2.1 – Heat Equation FD Reference")
print("=" * 60)
from problem2_fd import run_part_a as fd_a, run_part_b as fd_b, run_part_c as fd_c, run_unstable

U, x, t = fd_a()
L2_fd = fd_b(U, x)
fd_c(U, x, t)

# ------------------------------------------------------------------ #
# Section 2.2 / 2.3 / 2.4 – Heat PINNs                               #
# ------------------------------------------------------------------ #
print("\n" + "=" * 60)
print("SECTION 2.2 – Heat AD-PINN")
print("=" * 60)
from problem2_pinn import run_2_2, run_2_3, run_2_4a, run_2_4b

ad_heat = run_2_2()

print("\n" + "=" * 60)
print("SECTION 2.3 – Heat FDM-PINN")
print("=" * 60)
fdm_heat = run_2_3()

print("\n" + "=" * 60)
print("SECTION 2.4 – Heat Comparison")
print("=" * 60)
run_2_4a(ad_heat, fdm_heat)

print("\n" + "=" * 60)
print("SECTION 2.4(c) – Unstable FD run")
print("=" * 60)
run_unstable(r_unstable=0.6)

# ------------------------------------------------------------------ #
# Section 3(a) – Master summary table                                  #
# ------------------------------------------------------------------ #
print("\n" + "=" * 60)
print("SECTION 3(a) – Master Error Summary")
print("=" * 60)

_, err_ad_ode, time_ad_ode, hist_ad_ode = ad_ode
_, err_fdm_ode, time_fdm_ode, hist_fdm_ode = fdm_ode
_, err_ad_heat, time_ad_heat, hist_ad_heat = ad_heat
_, err_fdm_heat, time_fdm_heat, hist_fdm_heat = fdm_heat

summary = pd.DataFrame(
    {
        "Problem": ["ODE", "ODE", "ODE", "ODE", "Heat", "Heat", "Heat"],
        "Method": [
            "Forward Euler",
            "RK4",
            "AD-PINN",
            "FDM-PINN",
            "Forward Euler FD",
            "AD-PINN",
            "FDM-PINN",
        ],
        "Error Metric": [
            "Max abs error",
            "Max abs error",
            "Max abs error",
            "Max abs error",
            "L2 at t=0.5",
            "Relative L2",
            "Relative L2",
        ],
        "Error": [
            euler_errors[0],
            rk4_errors[0],
            err_ad_ode,
            err_fdm_ode,
            L2_fd,
            err_ad_heat,
            err_fdm_heat,
        ],
        "Training / Runtime (s)": [
            None, None,
            time_ad_ode, time_fdm_ode,
            None,
            time_ad_heat, time_fdm_heat,
        ],
    }
)
print(summary.to_string(index=False))

print("\nAll sections complete.")
