"""
Experiment 7: Local OU Tracking Approximation
==============================================
Studies convergence of the normalised tracking error to a limiting OU process
as the step size h → 0.

Tracking error recursion (derived by cancelling common lambda / mu terms):
  e^h_{k+1} = e^h_k + rho*sqrt(h)*u_S(Z_{k+1}*exp(-e^h_k/2))
              - Gamma*sqrt(h)*W_{k+1}
  Z_{k+1} ~ N(0,1),  W_{k+1} ~ N(0,1),  independent

Moving target: mu^h_{kh} = mu^h_{(k-1)h} + Gamma*sqrt(h)*W_k  (Brownian target)
The observation is Y_{(k+1)h} = sqrt(h*exp(mu^h_{kh})) * Z_{k+1}, so
  z_std = Y / (sqrt(h) * exp(lambda/2)) = Z_{k+1} * exp(-e^h_k / 2).

Stopping:  |e^h_k| > K_stop  (path truncated at first exceedance).
Terminal:  Z_{S,h} = h^{-1/4} * e^h_{floor(M/h)}  (or at truncation time)
Normalised variance should converge to V^bullet_S as h -> 0.

Theoretical OU stationary variance:
  Var[h^{-1/4} * e^h] → V^bullet_S  as h → 0

  V^bullet_S = (rho^2 * K_S + Gamma^2) / (4 * rho * J_S)

Derivation:
  OU dynamics: e_{k+1} ≈ (1 - B_S * rho * sqrt(h)) * e_k + noise_k
  where B_S = (1/2)*E_N[Z^2 * u_S(Z)] = 2*J_S  (proven via Gaussian integration-by-parts)
  and noise variance per step = h * (rho^2 * K_S + Gamma^2).
  OU stationary variance = h * q^2 / (2 * B_S * rho * sqrt(h)) = sqrt(h) * q^2 / (4*rho*J_S)
  Normalised: Var[h^{-1/4}*e] = h^{-1/2} * sqrt(h) * q^2/(4*rho*J_S) = q^2/(4*rho*J_S).

K_S = Var_{N(0,1)}[u_S(Z)];  J_S = curvature normalization constants (from paper).

Parameters
----------
M       = 3   (local horizon in time units)
rho     = 0.8
Gamma   = 0.65
K_stop  = 3.5
R       = 12000 replications  (main arm)
h_list  = [2^{-4}, 2^{-5}, …, 2^{-12}]
Companion arm: h = 2^{-16},  R_comp=3000,  e_0 = h^{1/4} * Z_0, Z_0 ~ U[-1,1]

Outputs
-------
results/exp7_variance_ratios.csv   – empirical var / V^bullet_S per (h, criterion)
results/exp7_variance_plot.png     – variance ratio vs h
results/exp7_ks_companion.csv      – KS distances for companion arm
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

# Experiment parameters
M       = 3       # local time horizon (time units)
RHO     = 0.8
GAMMA   = 0.65
K_STOP  = 3.5


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
# K_S = Var_{N(0,1)}[u_S(Z)] via Gauss–Hermite quadrature
# ---------------------------------------------------------------------------
def _K_via_quadrature(u_fn) -> float:
    pts, wts = np.polynomial.hermite.hermgauss(200)
    z = pts * np.sqrt(2.0)
    w = wts / np.sqrt(np.pi)
    u_vals = u_fn(z)
    mean_u  = float(np.dot(w, u_vals))
    mean_u2 = float(np.dot(w, u_vals ** 2))
    return mean_u2 - mean_u ** 2


K_LOG  = _K_via_quadrature(u_log)
K_DP   = _K_via_quadrature(u_dp)
K_CRPS = _K_via_quadrature(u_crps)

K_VALUES = {"Gauss-log": K_LOG, "Gauss-DP": K_DP, "Gauss-CRPS": K_CRPS}


def V_bullet(K_S: float, J_S: float) -> float:
    """
    Theoretical OU stationary variance of h^{-1/4}*e as h → 0.

    V^bullet_S = (rho^2*K_S + Gamma^2) / (4*rho*J_S)

    Derivation: OU mean-reversion coefficient is b_S = 2*rho*J_S  (the factor 2
    follows from B_S = 2*J_S, where B_S = (1/2)*E_N[Z^2*u_S(Z)] proven via
    Gaussian integration-by-parts for all score functions in the scale family).
    """
    q2 = RHO ** 2 * K_S + GAMMA ** 2
    return q2 / (4.0 * RHO * J_S)


# ---------------------------------------------------------------------------
# Core tracking-error simulation (vectorised over R)
# ---------------------------------------------------------------------------
def simulate_tracking(
    h: float,
    R: int,
    u_fn,
    rng: np.random.Generator,
    e0: np.ndarray | None = None,
) -> np.ndarray:
    """
    Simulate R tracking-error paths for step size h.

    Parameters
    ----------
    e0 : initial tracking error (array of shape R); 0 if None.

    Returns
    -------
    Z_terminal : h^{-1/4} * e at the terminal (or stopping) time, shape (R,).
    """
    sqrt_h   = np.sqrt(h)
    n_steps  = int(round(M / h))
    hinv14   = h ** (-0.25)

    e = np.zeros(R) if e0 is None else e0.copy()
    active = np.ones(R, dtype=bool)    # paths not yet stopped
    terminal = np.empty(R)
    terminal[:] = np.nan

    for _ in range(n_steps):
        if not active.any():
            break
        Zk = rng.standard_normal(R)    # observation noise
        Wk = rng.standard_normal(R)    # target drift noise

        z_std = Zk * np.exp(-0.5 * e)
        u_val = u_fn(z_std)

        e_new = e + RHO * sqrt_h * u_val - GAMMA * sqrt_h * Wk

        # Check stopping (only for active paths); clip to boundary at stop time
        newly_stopped = active & (np.abs(e_new) > K_STOP)
        terminal[newly_stopped] = hinv14 * K_STOP * np.sign(e_new[newly_stopped])
        active[newly_stopped] = False

        e = e_new

    # Paths that never stopped: use terminal value
    terminal[active] = hinv14 * e[active]
    return terminal


# ---------------------------------------------------------------------------
# Main simulation loop
# ---------------------------------------------------------------------------
def run_main_arm(
    h_list: list,
    R: int = 12_000,
    seed: int = 20_260_434,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for name, (u_fn, J_S) in SCORE_FNS.items():
        K_S  = K_VALUES[name]
        V_th = V_bullet(K_S, J_S)
        print(f"  {name}  K_S={K_S:.4f}  J_S={J_S:.3f}  V_theory={V_th:.4f}")
        for h in h_list:
            Z_term = simulate_tracking(h, R, u_fn, rng)
            emp_var = float(np.nanvar(Z_term))
            ratio   = emp_var / V_th
            stop_frac = float(np.isnan(Z_term).sum()) / R   # paths that were NaN (shouldn't happen)
            rows.append({
                "criterion": name,
                "h":         h,
                "emp_var":   emp_var,
                "V_theory":  V_th,
                "ratio":     ratio,
                "stop_frac": stop_frac,
            })
            print(f"    h=2^{{{int(round(np.log2(h))):+d}}}  emp_var={emp_var:.4f}  "
                  f"ratio={ratio:.4f}")
    return pd.DataFrame(rows)


def run_companion_arm(
    R_comp: int = 3_000,
    h_comp: float = 2.0 ** (-16),
    seed: int = 20_260_435,
) -> pd.DataFrame:
    """
    Companion arm: initial condition e_0 = h^{1/4} * Z_0, Z_0 ~ U[-1,1].
    KS distance between empirical terminal distribution and N(0, V^bullet_S).
    """
    rng = np.random.default_rng(seed)
    Z0 = rng.uniform(-1.0, 1.0, size=R_comp)
    e0 = (h_comp ** 0.25) * Z0

    rows = []
    for name, (u_fn, J_S) in SCORE_FNS.items():
        K_S  = K_VALUES[name]
        V_th = V_bullet(K_S, J_S)
        # Generate fresh rng state for each criterion
        rng_c = np.random.default_rng(seed + hash(name) % (2**32))
        Z_term = simulate_tracking(h_comp, R_comp, u_fn, rng_c, e0=e0)
        # KS test vs N(0, V_th)
        sigma_th = np.sqrt(V_th)
        ks_stat, ks_pval = stats.kstest(Z_term, "norm", args=(0.0, sigma_th))
        rows.append({
            "criterion": name,
            "h":         h_comp,
            "V_theory":  V_th,
            "emp_var":   float(np.nanvar(Z_term)),
            "ks_stat":   float(ks_stat),
            "ks_pval":   float(ks_pval),
        })
        print(f"  {name}  KS={ks_stat:.4f}  p={ks_pval:.4f}  V_th={V_th:.4f}")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def _plot_variance_ratios(df: pd.DataFrame, out_dir: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    colours = {"Gauss-log": "#4C72B0", "Gauss-DP": "#DD8452", "Gauss-CRPS": "#55A868"}
    for name, grp in df.groupby("criterion", sort=False):
        log2_h = np.log2(grp["h"].values)
        ax.plot(log2_h, grp["ratio"].values, "o-", label=name, color=colours[name])
    ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--", label="target=1")
    ax.set_xlabel("log₂(h)")
    ax.set_ylabel("Empirical Var / V⁰_S")
    ax.set_title("Exp 7 – Normalised Tracking-Error Variance Ratio")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(out_dir, "exp7_variance_plot.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    print("Experiment 7 – Local OU Tracking Approximation")
    print(f"  M={M}, rho={RHO}, Gamma={GAMMA}, K_stop={K_STOP}")
    print(f"  K_log={K_LOG:.4f}  K_DP={K_DP:.4f}  K_CRPS={K_CRPS:.4f}")

    out_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(out_dir, exist_ok=True)

    h_list = [2.0 ** (-k) for k in range(4, 13)]   # 2^{-4} … 2^{-12}
    R = 12_000

    print(f"\nMain arm: R={R}, h in [2^-4, ..., 2^-12]")
    df_main = run_main_arm(h_list, R=R)

    csv_main = os.path.join(out_dir, "exp7_variance_ratios.csv")
    df_main.to_csv(csv_main, index=False)
    print(f"\n  Variance ratios saved to {csv_main}")

    print("\nSummary (smallest h per criterion):")
    print(
        df_main.sort_values("h").groupby("criterion").first()[["emp_var", "V_theory", "ratio"]]
        .to_string(float_format=lambda x: f"{x:.4f}")
    )

    _plot_variance_ratios(df_main, out_dir)

    # Companion arm at h=2^{-16}
    print("\nCompanion arm: h=2^{-16}, R=3000, Z_0 ~ U[-1,1]")
    df_comp = run_companion_arm(R_comp=3_000, h_comp=2.0 ** (-16))

    csv_comp = os.path.join(out_dir, "exp7_ks_companion.csv")
    df_comp.to_csv(csv_comp, index=False)
    print(f"  KS companion table saved to {csv_comp}")
    print("\nKS distances (companion arm vs N(0, V^bullet_S)):")
    print(df_comp[["criterion", "V_theory", "emp_var", "ks_stat", "ks_pval"]].to_string(
        index=False, float_format=lambda x: f"{x:.5f}"
    ))

    print("\nExperiment 7 complete.")


if __name__ == "__main__":
    main()
