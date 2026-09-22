"""
Experiment 6: Non-Centred High-Frequency Mean-Flow Diagnostics
===============================================================
Studies the O(h) gain regime, where the residuals follow a standardised t5
distribution (misspecified relative to the Gaussian score functions).

DGP:  Y_t = sqrt(h * exp(0)) * Z_t,  Z_t ~ t_5 / sqrt(5/3)
      (true log-variance = 0; t5 has var 5/(5-2)=5/3, so dividing by sqrt(5/3)
       gives unit variance)

Filter (O(h) gain):
  lambda_{t+1} = lambda_t - kappa*h*lambda_t + (alpha*h) * u_S(z_t)
  z_t = Y_t * exp(-lambda_t/2) / sqrt(h)   (h-standardised residual)
  alpha = 1,  kappa = 0.5

Mean-field function (drift map):
  m_S(lambda_f) = E_{t5}[u_S(z_std)]
  where z_std = Z * exp(-lambda_f/2),  Z ~ t5/sqrt(5/3)

This is evaluated on a fine lambda grid via Monte Carlo (large N sample).

Key findings
------------
- Gaussian-log: m_log(lambda) = exp(-lambda) - 1, crosses 0 at lambda=0.
- DP, CRPS: m_S(lambda) ≠ 0 at lambda=0 under t5; fixed point lambda* ≠ 0.
- The O(h) filter paths show nonzero drift for DP and CRPS but not log.

Parameters
----------
h_list = [2^{-k}, k=1,…,7]
R = 8000 replications, T = 30 steps
alpha = 1, kappa = 0.5
lambda grid: linspace(-2, 2, 101) for mean-field curves

Outputs
-------
results/exp6_drift_curves.png   – drift m_S(lambda) vs lambda for each criterion
results/exp6_paths.png          – sample filter paths under t5 DGP
results/exp6_meanfield.csv      – tabulated mean-field values
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------------------
# Score-function constants
# ---------------------------------------------------------------------------
BETA       = 0.15
C_BETA     = (2.0 * np.pi) ** (-BETA / 2.0)
KAPPA_BETA = BETA * C_BETA * (1.0 + BETA) ** (-0.5)

J_LOG   = 0.500
J_DP    = 0.357
J_CRPS  = 0.071

NU_DGP = 5                               # t5 DGP
SCALE_T5 = np.sqrt(NU_DGP / (NU_DGP - 2))   # sqrt(5/3); divide to get Var=1


def u_log(z: np.ndarray) -> np.ndarray:
    return z * z - 1.0


def u_dp(z: np.ndarray) -> np.ndarray:
    return (
        (1.0 + BETA) * C_BETA * np.exp(-0.5 * BETA * z * z) * (z * z - 1.0)
        + KAPPA_BETA
    )


def u_crps(z: np.ndarray) -> np.ndarray:
    return 1.0 / np.sqrt(np.pi) - np.sqrt(2.0 / np.pi) * np.exp(-0.5 * z * z)


SCORE_FNS = {
    "Gauss-log":  (u_log,  J_LOG),
    "Gauss-DP":   (u_dp,   J_DP),
    "Gauss-CRPS": (u_crps, J_CRPS),
}

# Gaussian log mean field: analytic
def m_log_analytic(lam_grid: np.ndarray) -> np.ndarray:
    """Under t5/sqrt(5/3): E[z^2-1] = exp(-lambda)*Var[Z]=exp(-lambda)-1."""
    return np.exp(-lam_grid) - 1.0


# ---------------------------------------------------------------------------
# Monte Carlo mean-field estimation
# ---------------------------------------------------------------------------
def compute_mean_field(
    u_fn,
    lam_grid: np.ndarray,
    N: int = 500_000,
    seed: int = 42,
) -> np.ndarray:
    """
    m_S(lambda_f) = E_{t5}[u_S(Z * exp(-lambda_f/2))]  for each lambda_f.
    Z ~ t5/sqrt(5/3)  (unit variance).
    """
    rng = np.random.default_rng(seed)
    Z = rng.standard_t(NU_DGP, size=N) / SCALE_T5   # unit-variance t5

    m = np.empty(len(lam_grid))
    for i, lf in enumerate(lam_grid):
        z_std = Z * np.exp(-0.5 * lf)
        m[i] = float(u_fn(z_std).mean())
    return m


# ---------------------------------------------------------------------------
# Fixed-point finder (drift = 0 ↔ lambda* where m_S(lambda*) = 0)
# ---------------------------------------------------------------------------
def find_fixed_point(m: np.ndarray, lam_grid: np.ndarray) -> float | None:
    """Return the lambda where m_S crosses zero (first sign change or exact zero)."""
    for i, mi in enumerate(m):
        if mi == 0.0:
            return float(lam_grid[i])
    signs = np.sign(m)
    for i in range(len(signs) - 1):
        if signs[i] != signs[i + 1] and signs[i] != 0 and signs[i + 1] != 0:
            alpha_interp = -m[i] / (m[i + 1] - m[i])
            return lam_grid[i] + alpha_interp * (lam_grid[i + 1] - lam_grid[i])
    return None


# ---------------------------------------------------------------------------
# Filter path simulation under t5 DGP
# ---------------------------------------------------------------------------
def simulate_paths(
    h: float,
    R: int,
    T: int,
    alpha: float,
    kappa: float,
    u_fn,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Simulate R paths of length T+1 using O(h) gain under t5 DGP.
    Returns lam_paths of shape (R, T+1).
    """
    sqrt_h = np.sqrt(h)
    Z = rng.standard_t(NU_DGP, size=(R, T)) / SCALE_T5
    lam = np.zeros((R, T + 1))

    for t in range(T):
        lam_t = lam[:, t]
        # y = sqrt(h * exp(0)) * Z = sqrt(h) * Z  (true log-var = 0)
        y_t   = sqrt_h * Z[:, t]
        z_std = y_t * np.exp(-0.5 * lam_t) / sqrt_h   # = Z[:,t]*exp(-lam_t/2)
        u_t   = u_fn(z_std)
        lam[:, t + 1] = lam_t - kappa * h * lam_t + alpha * h * u_t

    return lam


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def _plot_drift_curves(
    lam_grid: np.ndarray,
    mean_fields: dict,
    out_dir: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    colours = {
        "Gauss-log":           "#4C72B0",
        "Gauss-DP":            "#DD8452",
        "Gauss-CRPS":          "#55A868",
        "Gauss-log (analytic)":"#4C72B0",   # same colour, dashed style
    }
    styles = {
        "Gauss-log (analytic)": "--",
    }

    for name, m in mean_fields.items():
        col = colours.get(name, "#888888")
        ls  = styles.get(name, "-")
        ax.plot(lam_grid, m, color=col, linestyle=ls, label=name, linewidth=2)
        if ls == "-":   # only annotate MC curves
            fp = find_fixed_point(m, lam_grid)
            if fp is not None:
                ax.axvline(fp, color=col, linestyle=":", alpha=0.7)

    ax.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
    ax.axvline(0.0, color="grey",  linewidth=0.5, linestyle="--")
    ax.set_xlabel("Filter state λ")
    ax.set_ylabel(r"Mean field $m_S(\lambda)$")
    ax.set_title("Exp 6 – Mean Flow Under t5 DGP (true λ = 0)")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(out_dir, "exp6_drift_curves.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def _plot_paths(
    h: float,
    lam_paths: dict,
    T: int,
    out_dir: str,
) -> None:
    """Plot a handful of filter paths per criterion at the largest h."""
    n_show = 30
    time_ax = np.arange(T + 1) * h

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    colours = {"Gauss-log": "#4C72B0", "Gauss-DP": "#DD8452", "Gauss-CRPS": "#55A868"}

    for ax, (name, paths) in zip(axes, lam_paths.items()):
        col = colours[name]
        for i in range(min(n_show, paths.shape[0])):
            ax.plot(time_ax, paths[i], color=col, alpha=0.2, linewidth=0.7)
        ax.axhline(0.0, color="black", linewidth=1.0, linestyle="--")
        ax.set_title(name)
        ax.set_xlabel("Time")

    axes[0].set_ylabel("λ")
    fig.suptitle(f"Exp 6 – Filter Paths Under t5 DGP  (h={h:.3f})", y=1.01)
    fig.tight_layout()
    path = os.path.join(out_dir, "exp6_paths.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    print("Experiment 6 – Non-Centred Mean-Flow Diagnostics (t5 DGP)")

    out_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(out_dir, exist_ok=True)

    # ---- Mean-field curves ----
    lam_grid = np.linspace(-2.0, 2.0, 201)
    print("\nComputing mean-field curves via Monte Carlo (N=500k)…")
    mean_fields = {}
    fp_rows = []
    for name, (u_fn, J_S) in SCORE_FNS.items():
        m = compute_mean_field(u_fn, lam_grid, N=500_000, seed=100 + list(SCORE_FNS).index(name))
        mean_fields[name] = m
        fp = find_fixed_point(m, lam_grid)
        fp_str = f"{fp:.4f}" if fp is not None else "none in [-2,2]"
        print(f"  {name}: m_S(0)={m[len(lam_grid)//2]:.5f}  fixed point λ*={fp_str}")
        fp_rows.append({"criterion": name, "m_at_0": float(m[len(lam_grid)//2]), "fixed_point": fp})

    # Analytic Gaussian-log for comparison
    mean_fields["Gauss-log (analytic)"] = m_log_analytic(lam_grid)

    _plot_drift_curves(lam_grid, mean_fields, out_dir)

    # Save mean-field table
    mf_df = pd.DataFrame({"lambda": lam_grid})
    for name, m in mean_fields.items():
        mf_df[name] = m
    mf_path = os.path.join(out_dir, "exp6_meanfield.csv")
    mf_df.to_csv(mf_path, index=False)
    print(f"  Mean-field table saved to {mf_path}")

    fp_df = pd.DataFrame(fp_rows)
    print("\nFixed points (where drift = 0):")
    print(fp_df.to_string(index=False, float_format=lambda x: f"{x:.5f}"))

    # ---- Filter paths at h=0.5 (largest h for visual clarity) ----
    h_paths = 0.5
    R_paths = 200
    T_paths = 30
    alpha, kappa = 1.0, 0.5
    rng = np.random.default_rng(20_260_432)
    print(f"\nSimulating paths at h={h_paths}, R={R_paths}, T={T_paths}…")
    lam_paths = {}
    for name, (u_fn, _) in SCORE_FNS.items():
        lam_paths[name] = simulate_paths(h_paths, R_paths, T_paths, alpha, kappa, u_fn, rng)

    _plot_paths(h_paths, lam_paths, T_paths, out_dir)

    # ---- Terminal mean of filter state across h values ----
    h_list = [2.0 ** (-k) for k in range(1, 8)]
    R_sim, T_sim = 8_000, 30
    rows = []
    rng2 = np.random.default_rng(20_260_433)
    print(f"\nRunning paths (R={R_sim}, T={T_sim}) across h values…")
    for name, (u_fn, _) in SCORE_FNS.items():
        for h in h_list:
            paths = simulate_paths(h, R_sim, T_sim, alpha, kappa, u_fn, rng2)
            terminal_mean = float(paths[:, -1].mean())
            terminal_std  = float(paths[:, -1].std())
            rows.append({
                "criterion": name, "h": h,
                "terminal_mean": terminal_mean,
                "terminal_std":  terminal_std,
            })
            print(f"  {name}  h={h:.4f}  terminal mean={terminal_mean:.4f}  std={terminal_std:.4f}")

    paths_df = pd.DataFrame(rows)
    paths_csv = os.path.join(out_dir, "exp6_paths_summary.csv")
    paths_df.to_csv(paths_csv, index=False)
    print(f"\n  Path summary saved to {paths_csv}")
    print("\nExperiment 6 complete.")


if __name__ == "__main__":
    main()
