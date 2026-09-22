"""
Experiment 4: Finite-Sample Filtering Robustness (Monte Carlo)
==============================================================
Compares five score-driven scale filters — Gaussian-log, density-power (DP),
CRPS, MMD-RBF, and Student-t6-log — across clean and 2%-contaminated
observation paths drawn from a log-volatility DGP.

DGP (latent):  x_t = phi_x * x_{t-1} + sigma_x * eta_t,  eta_t ~ N(0,1)
               phi_x = 0.97, sigma_x = 0.10  (chosen to give realistic
               persistence; the paper leaves these unspecified).
Observation:   y_t = exp(x_t/2) * eps_t
Contaminated:  eps_t -> eps_t * 6  with probability 0.02 (Bernoulli).

Filter recursion (curvature-normalised):
   lambda_{t+1} = omega + varphi * lambda_t + (alpha / J_S) * u_S(z_t)
   z_t = y_t * exp(-lambda_t / 2)        (standardised residual)
   alpha = 0.10,  varphi = 0.97,  omega = 0

J_S constants (from paper):
   J_log=0.500, J_DP=0.357, J_CRPS=0.071, J_MMD=0.034
   J_t6 computed numerically under the t6 distribution.

Usage
-----
  python exp/exp_4_mc_filtering.py          # full run  R=20000
  python exp/exp_4_mc_filtering.py --small  # quick run R=500
"""

import argparse
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import numba
import pandas as pd
from scipy import integrate, stats

# ---------------------------------------------------------------------------
# Score-function constants
# ---------------------------------------------------------------------------
BETA = 0.15
C_BETA = (2.0 * np.pi) ** (-BETA / 2.0)
KAPPA_BETA = BETA * C_BETA * (1.0 + BETA) ** (-0.5)

# Curvature-normalisation constants (from paper)
J_LOG   = 0.500
J_DP    = 0.357
J_CRPS  = 0.071
J_MMD   = 0.034


def _compute_J_t6() -> float:
    """Fisher information for the t6 scale parameter: E_t6[(u_t6(Z)/2)^2]."""
    nu = 6
    def integrand(z):
        u = (nu + 1) * z * z / (nu + z * z) - 1.0
        return 0.25 * u * u * stats.t.pdf(z, nu)
    val, _ = integrate.quad(integrand, -80.0, 80.0, limit=200)
    return val


J_T6 = _compute_J_t6()

# ---------------------------------------------------------------------------
# Numba-jitted filter loops (one per score type for nopython compat.)
# ---------------------------------------------------------------------------
@numba.jit(nopython=True)
def _filter_log(y: np.ndarray, varphi: float, alpha_eff: float) -> np.ndarray:
    """Gaussian-log score filter."""
    T = len(y)
    lam = np.empty(T)
    lam_t = 0.0
    for t in range(T):
        lam[t] = lam_t
        z = y[t] * np.exp(-0.5 * lam_t)
        u = z * z - 1.0
        lam_t = varphi * lam_t + alpha_eff * u
    return lam


@numba.jit(nopython=True)
def _filter_dp(
    y: np.ndarray,
    varphi: float,
    alpha_eff: float,
    beta: float,
    c_beta: float,
    kappa_beta: float,
) -> np.ndarray:
    """Density-power score filter."""
    T = len(y)
    lam = np.empty(T)
    lam_t = 0.0
    for t in range(T):
        lam[t] = lam_t
        z = y[t] * np.exp(-0.5 * lam_t)
        u = (1.0 + beta) * c_beta * np.exp(-0.5 * beta * z * z) * (z * z - 1.0) + kappa_beta
        lam_t = varphi * lam_t + alpha_eff * u
    return lam


@numba.jit(nopython=True)
def _filter_crps(y: np.ndarray, varphi: float, alpha_eff: float) -> np.ndarray:
    """Gaussian-CRPS score filter."""
    T = len(y)
    lam = np.empty(T)
    lam_t = 0.0
    inv_sqrt_pi = 1.0 / np.sqrt(np.pi)
    sqrt_2_over_pi = np.sqrt(2.0 / np.pi)
    for t in range(T):
        lam[t] = lam_t
        z = y[t] * np.exp(-0.5 * lam_t)
        u = inv_sqrt_pi - sqrt_2_over_pi * np.exp(-0.5 * z * z)
        lam_t = varphi * lam_t + alpha_eff * u
    return lam


@numba.jit(nopython=True)
def _filter_mmd(y: np.ndarray, varphi: float, alpha_eff: float) -> np.ndarray:
    """Gaussian MMD-RBF (ell=2, D0=5) score filter."""
    T = len(y)
    lam = np.empty(T)
    lam_t = 0.0
    a = 2.0 * (2.0 / np.sqrt(5.0))          # 4/sqrt(5)
    const = 2.0 * 2.0 / (6.0 ** 1.5)         # centering constant
    for t in range(T):
        lam[t] = lam_t
        z = y[t] * np.exp(-0.5 * lam_t)
        u = a * np.exp(-z * z / 10.0) * (z * z / 25.0 - 0.2) + const
        lam_t = varphi * lam_t + alpha_eff * u
    return lam


@numba.jit(nopython=True)
def _filter_t6(y: np.ndarray, varphi: float, alpha_eff: float) -> np.ndarray:
    """Student-t6 log-score filter (benchmark)."""
    T = len(y)
    lam = np.empty(T)
    lam_t = 0.0
    nu = 6.0
    for t in range(T):
        lam[t] = lam_t
        z = y[t] * np.exp(-0.5 * lam_t)
        u = (nu + 1.0) * z * z / (nu + z * z) - 1.0
        lam_t = varphi * lam_t + alpha_eff * u
    return lam


# ---------------------------------------------------------------------------
# DGP simulation (vectorised in NumPy, called per replication)
# ---------------------------------------------------------------------------
def _simulate_dgp(
    rng: np.random.Generator,
    T: int,
    phi_x: float,
    sigma_x: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (x, y_clean, y_cont) arrays of length T."""
    # Start latent process from stationionary distribution
    sig_stat = sigma_x / np.sqrt(1.0 - phi_x ** 2)
    x = np.empty(T)
    x[0] = rng.normal(0.0, sig_stat)
    eta = rng.standard_normal(T - 1)
    for t in range(1, T):
        x[t] = phi_x * x[t - 1] + sigma_x * eta[t - 1]

    eps = rng.standard_normal(T)
    # Contamination: I_t ~ Bernoulli(0.02), multiplier 6
    contam = rng.random(T) < 0.02
    eps_cont = eps * np.where(contam, 6.0, 1.0)

    y_clean = np.exp(0.5 * x) * eps
    y_cont  = np.exp(0.5 * x) * eps_cont
    return x, y_clean, y_cont


# ---------------------------------------------------------------------------
# Per-replication metrics
# ---------------------------------------------------------------------------
def _metrics(lam: np.ndarray, x: np.ndarray) -> tuple[float, float, float]:
    diff = lam - x
    bias = float(diff.mean())
    rmse = float(np.sqrt((diff ** 2).mean()))
    maxdev = float(np.abs(diff).max())
    return bias, rmse, maxdev


# ---------------------------------------------------------------------------
# Main Monte Carlo loop
# ---------------------------------------------------------------------------
def run_mc(
    R: int = 20_000,
    T: int = 4_000,
    seed: int = 20_260_430,
    phi_x: float = 0.97,
    sigma_x: float = 0.10,
    alpha: float = 0.10,
    varphi: float = 0.97,
    omega: float = 0.0,
) -> dict:
    """
    Run R replications for each filter (clean + contaminated).

    Returns a dict mapping criterion name -> DataFrame with columns
    [bias, rmse, maxdev] for clean and contaminated scenarios.
    """
    filters = {
        "Gauss-log":  (J_LOG,  lambda y, ae: _filter_log(y, varphi, ae)),
        "Gauss-DP":   (J_DP,   lambda y, ae: _filter_dp(y, varphi, ae, BETA, C_BETA, KAPPA_BETA)),
        "Gauss-CRPS": (J_CRPS, lambda y, ae: _filter_crps(y, varphi, ae)),
        "Gauss-MMD":  (J_MMD,  lambda y, ae: _filter_mmd(y, varphi, ae)),
        "Student-t6": (J_T6,   lambda y, ae: _filter_t6(y, varphi, ae)),
    }

    results = {
        name: {
            "clean": np.empty((R, 3)),
            "cont":  np.empty((R, 3)),
        }
        for name in filters
    }

    rng = np.random.default_rng(seed)

    t0 = time.time()
    for r in range(R):
        x, y_clean, y_cont = _simulate_dgp(rng, T, phi_x, sigma_x)
        for name, (J_S, filt_fn) in filters.items():
            alpha_eff = alpha / J_S
            lam_c = filt_fn(y_clean,  alpha_eff)
            lam_k = filt_fn(y_cont,   alpha_eff)
            results[name]["clean"][r] = _metrics(lam_c, x)
            results[name]["cont"][r]  = _metrics(lam_k, x)

        if (r + 1) % max(1, R // 20) == 0:
            elapsed = time.time() - t0
            pct = 100 * (r + 1) / R
            eta = elapsed / (r + 1) * (R - r - 1)
            print(f"  {r+1:>6}/{R}  ({pct:.0f}%)  elapsed {elapsed:.1f}s  ETA {eta:.0f}s")

    return results


def _summary_table(results: dict) -> pd.DataFrame:
    rows = []
    for name, d in results.items():
        for scen, key in [("Clean", "clean"), ("Contam", "cont")]:
            arr = d[key]
            rows.append({
                "Criterion":  name,
                "Scenario":   scen,
                "Bias":       arr[:, 0].mean(),
                "RMSE":       arr[:, 1].mean(),
                "MaxDev":     arr[:, 2].mean(),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def _plot_rmse(df: pd.DataFrame, out_dir: str) -> None:
    criteria = df["Criterion"].unique()
    x_pos = np.arange(len(criteria))
    width = 0.35

    clean_rmse = df[df["Scenario"] == "Clean" ].set_index("Criterion").loc[criteria, "RMSE"]
    cont_rmse  = df[df["Scenario"] == "Contam"].set_index("Criterion").loc[criteria, "RMSE"]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x_pos - width / 2, clean_rmse.values, width, label="Clean",        color="#4C72B0")
    ax.bar(x_pos + width / 2, cont_rmse.values,  width, label="Contaminated", color="#DD8452")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(criteria, rotation=15, ha="right")
    ax.set_ylabel("Mean RMSE")
    ax.set_title("Exp 4 – Filter RMSE: Clean vs Contaminated")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(out_dir, "exp4_rmse.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Exp 4 MC Filtering Robustness")
    parser.add_argument("--small", action="store_true",
                        help="Use R=500 instead of R=20000 for quick testing")
    parser.add_argument("--R", type=int, default=None,
                        help="Override number of replications")
    args = parser.parse_args()

    R = 500 if args.small else (args.R if args.R else 20_000)
    print(f"Experiment 4 – MC Filtering Robustness  (R={R}, T=4000)")
    print(f"  J_T6 (computed) = {J_T6:.4f}")

    out_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(out_dir, exist_ok=True)

    # Warm-up numba JIT
    print("  Warming up numba JIT…")
    _dummy = np.ones(10, dtype=np.float64)
    _filter_log(_dummy, 0.97, 0.2)
    _filter_dp(_dummy, 0.97, 0.28, BETA, C_BETA, KAPPA_BETA)
    _filter_crps(_dummy, 0.97, 1.4)
    _filter_mmd(_dummy, 0.97, 2.9)
    _filter_t6(_dummy, 0.97, 0.2)
    print("  JIT warm-up done.")

    print(f"\nRunning {R} replications…")
    results = run_mc(R=R)
    df = _summary_table(results)

    print("\n" + "=" * 72)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("=" * 72)

    csv_path = os.path.join(out_dir, "exp4_table.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n  Table saved to {csv_path}")

    _plot_rmse(df, out_dir)
    print("\nExperiment 4 complete.")


if __name__ == "__main__":
    main()
