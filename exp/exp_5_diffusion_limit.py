"""
Experiment 5: Centred High-Frequency Diffusion-Limit Diagnostics
================================================================
Studies the O(sqrt(h)) gain regime as the time step h → 0.

DGP at step h:  Y = sqrt(h * exp(lambda)) * Z,  Z ~ N(0,1)
Filter:  lambda_{t+1} = lambda_t - kappa*h*lambda_t + sqrt(h)*u_S(z_t)
         z_t = Y_t * exp(-lambda_t/2) / sqrt(h)   (h-standardised residual)

Parameters
----------
alpha = 1  (curvature-normalised; absorbed into J_S scaling)
kappa = 0.5  (mean-reversion speed)
R = 8000 replications,  T = 200 time steps per path
h values: 2^{-1}, 2^{-2}, …, 2^{-7}
Criteria: Gaussian-log, density-power (beta=0.15), Gaussian-CRPS

Diagnostics per (h, criterion)
-------------------------------
1. E[u_S(z_t)]:              should be ≈ 0  (centering)
2. h^{-1} * E[(Δλ)^2]:      should converge to K_S  (diffusion variance)
3. h^{-1} * E[(Δλ)^4]:      Lindeberg check – should → 0 as h → 0
4. log–log slope of h^{-1}*E[(Δλ)^2] vs h: should be ≈ 0 (flat = converged)

K_S = Var_{N(0,1)}[u_S(Z)] is computed via numerical quadrature.

Outputs
-------
results/exp5_diag_table.csv   – full (h, criterion, diagnostic) table
results/exp5_scaling.png      – second-moment scaling log–log plot
results/exp5_centering.png    – centering diagnostic plot
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import integrate

# ---------------------------------------------------------------------------
# Score-function constants
# ---------------------------------------------------------------------------
BETA     = 0.15
C_BETA   = (2.0 * np.pi) ** (-BETA / 2.0)
KAPPA_BETA = BETA * C_BETA * (1.0 + BETA) ** (-0.5)

J_LOG   = 0.500
J_DP    = 0.357
J_CRPS  = 0.071


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


# ---------------------------------------------------------------------------
# Theoretical K_S = Var_{N(0,1)}[u_S(Z)]
# ---------------------------------------------------------------------------
def _K_via_quadrature(u_fn) -> float:
    """Var_{N(0,1)}[u_fn(Z)] via Gauss–Hermite quadrature."""
    # E[u^2] (centering ensures mean=0 so Var=E[u^2])
    pts, wts = np.polynomial.hermite.hermgauss(200)
    z = pts * np.sqrt(2.0)          # N(0,1) nodes
    w = wts / np.sqrt(np.pi)        # normalised weights
    u_vals = u_fn(z)
    mean_u  = float(np.dot(w, u_vals))
    mean_u2 = float(np.dot(w, u_vals ** 2))
    return mean_u2 - mean_u ** 2


K_LOG  = _K_via_quadrature(u_log)
K_DP   = _K_via_quadrature(u_dp)
K_CRPS = _K_via_quadrature(u_crps)

K_VALUES = {
    "Gauss-log":  K_LOG,
    "Gauss-DP":   K_DP,
    "Gauss-CRPS": K_CRPS,
}


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------
def run_one_h(
    h: float,
    R: int,
    T: int,
    kappa: float,
    u_fn,
    rng: np.random.Generator,
) -> dict:
    """
    Simulate R paths of length T for a given h and score function.
    Returns empirical moments of the update and the step increment.
    """
    sqrt_h = np.sqrt(h)
    # Vectorise over replications: shape (R, T)
    Z = rng.standard_normal((R, T))             # innovations
    lam = np.zeros((R, T + 1))                  # lambda paths
    updates_all = np.empty((R, T))

    for t in range(T):
        # h-standardised residual: Y_t*exp(-lambda_t/2)/sqrt(h) = Z_t (at true lam)
        # Here Y_t = sqrt(h*exp(lam[:,t]))*Z[:,t], so z_std = Z[:,t]*exp((0-lam[:,t])/2)
        # (true lambda is 0 throughout for this diagnostic)
        lam_t = lam[:, t]
        y_t   = sqrt_h * np.exp(0.5 * lam_t) * Z[:, t]   # sqrt(h*exp(lam))*Z
        z_std = y_t * np.exp(-0.5 * lam_t) / sqrt_h      # = Z[:,t]

        u_t = u_fn(z_std)
        updates_all[:, t] = u_t
        delta = -kappa * h * lam_t + sqrt_h * u_t
        lam[:, t + 1] = lam_t + delta

    # Flatten all updates and increments
    u_flat    = updates_all.ravel()
    delta_lam = (lam[:, 1:] - lam[:, :-1]).ravel()

    return {
        "E_u":          float(u_flat.mean()),
        "E_du2_over_h": float((delta_lam ** 2).mean()) / h,
        "E_du4_over_h": float((delta_lam ** 4).mean()) / h,
    }


def run_experiment(
    h_list: list,
    R: int = 8_000,
    T: int = 200,
    kappa: float = 0.5,
    seed: int = 20_260_431,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for name, (u_fn, J_S) in SCORE_FNS.items():
        K_S = K_VALUES[name]
        print(f"  {name}  K_S={K_S:.4f}")
        for h in h_list:
            d = run_one_h(h, R, T, kappa, u_fn, rng)
            rows.append({
                "criterion":     name,
                "h":             h,
                "E_u":           d["E_u"],
                "E_du2_over_h":  d["E_du2_over_h"],
                "E_du4_over_h":  d["E_du4_over_h"],
                "K_S_theory":    K_S,
                "ratio_2nd":     d["E_du2_over_h"] / K_S,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def _plot_scaling(df: pd.DataFrame, h_list: list, out_dir: str) -> None:
    """Log–log plot of h^{-1}*E[(Δλ)^2] vs h."""
    fig, ax = plt.subplots(figsize=(7, 5))
    colours = ["#4C72B0", "#DD8452", "#55A868"]
    for (name, grp), col in zip(df.groupby("criterion", sort=False), colours):
        h_vals = grp["h"].values
        y_vals = grp["E_du2_over_h"].values
        K_S    = grp["K_S_theory"].iloc[0]
        ax.plot(h_vals, y_vals, "o-", color=col, label=name)
        ax.axhline(K_S, color=col, linestyle="--", alpha=0.4)

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("h  (log₂ scale)")
    ax.set_ylabel(r"$h^{-1}\,\mathbb{E}[(\Delta\lambda)^2]$")
    ax.set_title("Exp 5 – Second-Moment Scaling vs h")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(out_dir, "exp5_scaling.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def _plot_centering(df: pd.DataFrame, out_dir: str) -> None:
    """Plot E[u_S] vs h for each criterion — should be near 0."""
    fig, ax = plt.subplots(figsize=(7, 4))
    colours = ["#4C72B0", "#DD8452", "#55A868"]
    for (name, grp), col in zip(df.groupby("criterion", sort=False), colours):
        ax.plot(grp["h"].values, grp["E_u"].values, "o-", color=col, label=name)
    ax.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("h  (log₂ scale)")
    ax.set_ylabel(r"$\hat{\mathbb{E}}[u_S(z_t)]$")
    ax.set_title("Exp 5 – Centering Diagnostic (should be ≈ 0)")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(out_dir, "exp5_centering.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def _log_log_slope(h_list: list, df: pd.DataFrame) -> pd.DataFrame:
    """OLS slope of log(E_du2_over_h) ~ log(h) per criterion (should be ≈ 0)."""
    rows = []
    log_h = np.log(h_list)
    for name, grp in df.groupby("criterion", sort=False):
        y = np.log(grp["E_du2_over_h"].values)
        slope = float(np.polyfit(log_h, y, 1)[0])
        rows.append({"criterion": name, "log_log_slope": slope})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    print("Experiment 5 – Centred Diffusion-Limit Diagnostics")
    print(f"  K_log={K_LOG:.4f}  K_DP={K_DP:.4f}  K_CRPS={K_CRPS:.4f}")

    h_list = [2.0 ** (-k) for k in range(1, 8)]   # 1/2, 1/4, …, 1/128
    R, T = 8_000, 200

    out_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(out_dir, exist_ok=True)

    print(f"\nRunning R={R} replications, T={T} steps, {len(h_list)} h values…")
    df = run_experiment(h_list, R=R, T=T)

    # Log–log slopes
    slope_df = _log_log_slope(h_list, df)
    print("\nLog–log slope of h^{-1}*E[(Δλ)^2] vs h  (target ≈ 0):")
    print(slope_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Full table
    print("\n" + "=" * 72)
    print(df[["criterion", "h", "E_u", "E_du2_over_h", "K_S_theory", "ratio_2nd"]].to_string(
        index=False, float_format=lambda x: f"{x:.5f}"
    ))
    print("=" * 72)

    csv_path = os.path.join(out_dir, "exp5_diag_table.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n  Diagnostics saved to {csv_path}")

    _plot_scaling(df, h_list, out_dir)
    _plot_centering(df, out_dir)
    print("\nExperiment 5 complete.")


if __name__ == "__main__":
    main()
