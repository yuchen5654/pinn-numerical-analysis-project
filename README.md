# PINN Final Project — EN 553.481/681 Numerical Analysis Spring 2026

Physics-Informed Neural Networks for solving a first-order ODE and the 1D heat equation, comparing automatic differentiation (AD) and finite-difference (FDM) derivative computation against classical numerical methods.

All experiments run on **NVIDIA GeForce RTX 5070 Laptop GPU** (CUDA 12.8, PyTorch 2.11.0+cu128).

---

## Project structure

```
pinn_project/
├── src/
│   ├── model.py                # Shared PINN class, training loop, plotting utilities
│   ├── problem1_classical.py   # §1.1  Forward Euler & RK4 for the ODE
│   ├── problem1_pinn.py        # §1.2–1.4  AD-PINN & FDM-PINN for the ODE
│   ├── problem2_fd.py          # §2.1  Forward Euler FD for the heat equation
│   ├── problem2_pinn.py        # §2.2–2.4  AD-PINN & FDM-PINN for heat equation
│   ├── problem3_analysis.py    # §3(c) Network size study
│   ├── problem_bonus.py        # Bonus: inverse problem (recover unknown ν)
│   └── run_all.py              # Runs every section sequentially
├── figures/                    # All saved plots (PNG, 150 dpi)
├── requirements.txt
└── README.md
```

---

## Setup

```bash
# 1. Create and activate a virtual environment (optional but recommended)
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS / Linux:
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. (GPU) Install PyTorch with CUDA support — replace cu128 with your CUDA version
pip install torch --index-url https://download.pytorch.org/whl/cu128

# 4. Move into the source directory
cd src
```

> **GPU note:** the code automatically uses CUDA if available (`torch.cuda.is_available()`). All experiments in the report were run on an RTX 5070 Laptop GPU.

---

## Running individual sections

All commands are run from inside the `src/` directory.

### Section 1.1 – Classical ODE solvers (Forward Euler & RK4)

```bash
python problem1_classical.py
```

Produces:
- Plot of Forward Euler vs exact solution (h = 0.01) → `figures/1_1a_forward_euler.png`
- Plot of RK4 vs exact solution (h = 0.01) → `figures/1_1b_rk4.png`
- Convergence table for h ∈ {0.01, 0.005, 0.001} with observed orders

---

### Sections 1.2 & 1.3 – ODE AD-PINN and FDM-PINN

```bash
python problem1_pinn.py
```

Produces:
- **1.2** AD-PINN: loss curve, solution vs exact, pointwise error → `figures/1_2_*.png`
- **1.3** FDM-PINN (ε = 10⁻³): same outputs → `figures/1_3_*.png`
- **1.4(a)** Comparison table
- **1.4(b)** ε sweep log-log error plot → `figures/1_4b_ode_fdm_eps_sweep.png`

---

### Section 2.1 – Heat equation finite-difference reference

```bash
python problem2_fd.py
```

Produces:
- Δt, r (CFL ratio), L² error at t = 0.5
- Heatmap of the stable FD solution → `figures/2_1c_heat_fd_heatmap.png`
- Heatmap of the **unstable** run (r = 0.6) → `figures/2_1_unstable_fd_r0.6.png`

---

### Sections 2.2 & 2.3 – Heat AD-PINN and FDM-PINN

```bash
python problem2_pinn.py
```

Produces:
- **2.2** AD-PINN (20 000 epochs): loss curve, heatmaps → `figures/2_2_*.png`
- **2.3** FDM-PINN (ε = 10⁻³, 20 000 epochs): same → `figures/2_3_*.png`
- **2.4(a)** Comparison table
- **2.4(b)** ε sweep plot → `figures/2_4b_heat_fdm_eps_sweep.png`

---

### Section 3(b) – Collocation-point Nr sweep

```bash
python -c "from problem1_pinn import run_nr_sweep_ode; run_nr_sweep_ode()"
python -c "from problem2_pinn import run_nr_sweep_heat; run_nr_sweep_heat()"
```

---

### Section 3(c) – Network size study

```bash
python problem3_analysis.py
```

Tests three network configurations (2L-16H, 3L-32H, 5L-64H) for both the ODE and heat equation, using both AD and FDM. Saves 12 comparison figures → `figures/3c_*.png`.

---

### Bonus – Inverse Problem (recover unknown ν)

```bash
python problem_bonus.py
```

Adds ν as a trainable parameter alongside the network weights. Supplies 50 noisy observations
(σ = 0.01) from the exact solution and recovers ν using both AD-PINN and FDM-PINN.
Reports `|ν_recovered − ν_true| / ν_true` and saves `figures/bonus_inverse.png`.

---

### Run everything at once

```bash
python run_all.py
```

Executes every section in order and prints the master error summary table (§3a) at the end.  
⚠️ This takes **several hours** on CPU because it trains multiple PINNs with 10 000–20 000 epochs each.

---

## Results

### Problem 1.1 — Classical ODE solvers (h = 0.01)

| Method | Max Absolute Error |
|--------|--------------------|
| Forward Euler | 1.003 × 10⁻² |
| RK4 | 2.165 × 10⁻⁸ |

**Convergence table:**

| h | FE Error | FE Order | RK4 Error | RK4 Order |
|---|----------|----------|-----------|-----------|
| 0.010 | 1.003e-02 | — | 2.165e-08 | — |
| 0.005 | 4.965e-03 | 1.014 | 1.326e-09 | 4.029 |
| 0.001 | 9.850e-04 | 1.005 | 2.087e-12 | 4.010 |

Observed orders match theory: O(h) for Forward Euler, O(h⁴) for RK4.

---

### Problems 1.2 & 1.3 — ODE PINNs (10 000 epochs, ε = 10⁻³)

| Method | Final Loss | Max Abs Error | Training Time |
|--------|-----------|---------------|---------------|
| AD-PINN | 1.342e-03 | 1.251e-02 | 48 s |
| FDM-PINN | 2.244e-04 | 2.868e-03 | 59 s |

FDM-PINN achieves ~4× lower error; both match the exact solution well across [0, 5].

---

### Problem 1.4(b) — ODE FDM-PINN ε sweep

| ε | Final Loss | Max Abs Error | Time (s) |
|---|-----------|---------------|---------|
| 0.1 | 2.442e-03 | 1.517e-02 | 65 |
| 0.01 | 1.149e-03 | 1.099e-02 | 57 |
| 0.001 | 3.374e-03 | 1.093e-02 | 52 |
| 0.0001 | 5.642e-04 | **6.760e-03** | 57 |
| 0.00001 | 7.274e-04 | 7.916e-03 | 58 |

Optimal ε ≈ 10⁻⁴. Below that, floating-point cancellation in the central-difference stencil raises round-off error faster than the truncation error decreases.

---

### Problem 2.1 — Heat equation forward-difference reference

- Δx = 1/64,  Δt = 1.220e-02,  r = 0.4995 (stable, r ≤ 1/2)
- **L² error at t = 0.5: 3.655 × 10⁻⁴**
- Unstable run at r = 0.6 blows up immediately (see `figures/2_1_unstable_fd_r0.6.png`)

---

### Problems 2.2 & 2.3 — Heat PINNs (20 000 epochs, ε = 10⁻³)

| Method | Final Loss | Relative L² Error | Training Time |
|--------|-----------|-------------------|---------------|
| AD-PINN | 1.611e-04 | 1.379e-03 | 377 s |
| FDM-PINN | 1.562e-04 | 1.300e-03 | 258 s |

Both PINNs achieve relative L² errors below 0.14%. FDM-PINN is ~1.5× faster because it avoids building the second-order autograd graph.

---

### Problem 2.4(b) — Heat FDM-PINN ε sweep

| ε | Final Loss | Relative L² Error | Time (s) |
|---|-----------|-------------------|---------|
| 0.1 | 4.056e-04 | 8.272e-03 | 173 |
| 0.01 | 2.156e-04 | 2.229e-03 | 178 |
| 0.001 | 1.634e-03 | 6.781e-03 | 168 |
| 0.0001 | 2.166e-02 | 2.640e-03 | 177 |
| 0.00001 | 8.196 | **5.82 × 10⁻¹** | 146 |

The trade-off is more pronounced for the PDE than the ODE: the second-derivative stencil divides by ε², so cancellation is catastrophic at ε = 10⁻⁵ (58% relative error).

---

### Problem 3(b) — Effect of collocation points Nr

**ODE** (5 000 epochs, max absolute error):

| Nr | AD-PINN | FDM-PINN |
|----|---------|----------|
| 100 | 1.589e-02 | 6.681e-03 |
| 500 | 1.177e-02 | 1.038e-02 |
| 2 000 | 1.048e-02 | 7.199e-03 |
| 10 000 | 1.125e-02 | 8.802e-03 |

**Heat** (10 000 epochs, relative L² error):

| Nr | AD-PINN | FDM-PINN |
|----|---------|----------|
| 500 | 8.717e-03 | 3.391e-03 |
| 2 000 | 6.289e-03 | 3.534e-03 |
| 10 000 | **1.347e-03** | 5.485e-03 |
| 50 000 | 6.005e-03 | 5.789e-03 |

Nr = 10 000 is the sweet spot for the heat AD-PINN. Beyond that, the per-epoch cost dominates without further accuracy gain (analogous to over-quadrature with diminishing returns).

---

### Problem 3(c) — Effect of network size

| Problem | Method | Network | Error | Time (s) |
|---------|--------|---------|-------|---------|
| ODE | AD-PINN | Small (2L-16H) | 8.52e-03 | 30 |
| ODE | FDM-PINN | Small (2L-16H) | 1.22e-02 | 41 |
| ODE | AD-PINN | Medium (3L-32H) | 8.86e-03 | 26 |
| ODE | FDM-PINN | Medium (3L-32H) | 7.48e-03 | 35 |
| ODE | AD-PINN | Large (5L-64H) | 1.52e-02 | 25 |
| ODE | FDM-PINN | Large (5L-64H) | 1.43e-02 | 29 |
| Heat | AD-PINN | Small (2L-16H) | 2.55e-03 | 88 |
| Heat | FDM-PINN | Small (2L-16H) | 2.37e-03 | 67 |
| Heat | AD-PINN | Medium (3L-32H) | 4.71e-03 | 93 |
| Heat | FDM-PINN | Medium (3L-32H) | **1.52e-03** | 88 |
| Heat | AD-PINN | Large (5L-64H) | 6.81e-03 | 120 |
| Heat | FDM-PINN | Large (5L-64H) | 4.88e-03 | 119 |

Larger networks do not consistently improve accuracy with fixed epochs — they require more epochs to converge, so medium (3L-32H) is the best trade-off here.

---

### Bonus — Inverse Problem: recovering unknown ν

Setup: true ν = 0.01, 50 noisy observations with σ = 0.01. ν initialized at ≈ 0.10 (10× off). Trained for 20 000 epochs with combined physics + data loss.

| Method | ν_true | ν_recovered | Relative Error | Time (s) |
|--------|--------|-------------|----------------|---------|
| AD-PINN | 0.01 | 0.009959 | **0.41%** | 236 |
| FDM-PINN | 0.01 | 0.009911 | **0.89%** | 208 |

Both methods recover ν to sub-1% accuracy from noisy data. AD-PINN is slightly more precise because exact derivatives give a cleaner gradient signal for the parameter update.

---

## Notes

- Random seeds are fixed (`torch.manual_seed(42)`, `np.random.seed(42)`) for reproducibility.
- The FDM ε sweep reveals the classic truncation-error vs. round-off trade-off: for the ODE (first derivative, ÷2ε) the optimal ε ≈ 10⁻⁴; for the heat PDE (second derivative, ÷ε²) catastrophic cancellation sets in at ε = 10⁻⁵.
- PINNs have no CFL stability restriction because they minimize a global loss over the entire space-time domain, rather than marching forward in time.
- All figures are saved to `figures/` at 150 dpi.
