"""
Experiment 2: Pseudo-True Target Shifts Under Variance Contamination.

Studies how the pseudo-true variance target v* depends on the scoring criterion
when the true law is a contaminated Gaussian mixture:

    Q_{eps,tau} = (1-eps)*N(0,v0) + eps*N(0,tau*v0)

with eps = 0.05 and tau varying over a display grid.

For the Gaussian log score the exact closed-form is:
    v*_log = v0 * (1 + eps*(tau - 1))

For bounded criteria (density-power, CRPS, MMD) the pseudo-true target is
found numerically via a 6000-point scan of the contaminated risk, then
optionally refined.

Expected outcome: Gaussian-log target grows linearly in tau; bounded criteria
exhibit saturation (flattening) for large tau.
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import brentq

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.score_drivers import u_log, u_DP, u_CRPS, u_MMD, _b, _c_b, _kappa_b

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Constants ─────────────────────────────────────────────────────────────────
EPS  = 0.05
V0   = 1.0
N_SCAN = 6000
N_NODES = 96          # Gauss-Hermite nodes for each component


# ── Score functions S(P_lambda, y) ──────────────────────────────────────────

def score_log(y: np.ndarray, lam: float) -> np.ndarray:
    """Gaussian log score: S_log = 0.5*lambda + 0.5*exp(-lambda)*y^2."""
    return 0.5 * lam + 0.5 * np.exp(-lam) * y**2


def score_DP(y: np.ndarray, lam: float) -> np.ndarray:
    """Density-power score for Gaussian scale family.

    S_beta = c_b * exp(-b*lambda/2) * [(1+b)^(-1/2) - (1+1/b)*exp(-b*z^2/2)]
    where z = y*exp(-lambda/2).
    """
    z = y * np.exp(-lam / 2.0)
    term1 = (1.0 + _b) ** (-0.5)
    term2 = -(1.0 + 1.0 / _b) * np.exp(-_b * z**2 / 2.0)
    return _c_b * np.exp(-_b * lam / 2.0) * (term1 + term2)


def score_CRPS(y: np.ndarray, lam: float) -> np.ndarray:
    """Gaussian CRPS score.

    S_CRPS = exp(lambda/2) * [z*(2*Phi(z)-1) + 2*phi(z) - pi^(-1/2)]
    """
    from scipy.special import ndtr as _Phi
    from scipy.stats import norm as _norm
    z = y * np.exp(-lam / 2.0)
    phi_z = _norm.pdf(z)
    Phi_z = _Phi(z)
    inner = z * (2.0 * Phi_z - 1.0) + 2.0 * phi_z - 1.0 / np.sqrt(np.pi)
    return np.exp(lam / 2.0) * inner


def score_MMD(y: np.ndarray, lam: float) -> np.ndarray:
    """MMD-RBF score with fixed bandwidth ell=2.

    S_MMD = 1 - 2*(ell/sqrt(ell^2+v))*exp(-y^2/(2*(ell^2+v))) + ell/sqrt(ell^2+2v)
    where v = exp(lambda).
    """
    ell = 2.0
    v   = np.exp(lam)
    term1 = -2.0 * (ell / np.sqrt(ell**2 + v)) * np.exp(-y**2 / (2.0 * (ell**2 + v)))
    term2 = ell / np.sqrt(ell**2 + 2.0 * v)
    return 1.0 + term1 + term2


# ── Quadrature expectation under N(0, sigma_sq) ────────────────────────────

def _gh_expect_component(score_fn, lam: float, sigma_sq: float,
                          n_nodes: int = N_NODES) -> float:
    """E_{Y ~ N(0, sigma_sq)}[score_fn(Y, lam)] via Gauss-Hermite (scalar lam)."""
    xi, wi = np.polynomial.hermite.hermgauss(n_nodes)
    y = np.sqrt(2.0 * sigma_sq) * xi
    vals = score_fn(y, lam)
    return float(np.dot(wi, vals) / np.sqrt(np.pi))


def _gh_expect_vec(score_fn, lam_grid: np.ndarray, sigma_sq: float,
                   n_nodes: int = N_NODES) -> np.ndarray:
    """Vectorised E_{Y~N(0,sigma_sq)}[score_fn(Y, lam)] for every lam in lam_grid.

    Uses numpy broadcasting so no Python loop over lambda values.
    score_fn must support broadcasting: y shape (n_nodes,) × lam shape (n_scan,1).

    Returns array of shape (len(lam_grid),).
    """
    xi, wi = np.polynomial.hermite.hermgauss(n_nodes)
    y = np.sqrt(2.0 * sigma_sq) * xi                       # (n_nodes,)
    # Expand dims: y → (1, n_nodes), lam → (n_scan, 1)
    vals = score_fn(y[np.newaxis, :], lam_grid[:, np.newaxis])  # (n_scan, n_nodes)
    return (vals @ wi) / np.sqrt(np.pi)                     # (n_scan,)


def contaminated_risk(score_fn, lam: float, eps: float, tau: float,
                      v0: float, n_nodes: int = N_NODES) -> float:
    """R_S(lambda) = (1-eps)*E_{N(0,v0)}[S] + eps*E_{N(0,tau*v0)}[S] (scalar lam)."""
    r1 = _gh_expect_component(score_fn, lam, v0, n_nodes)
    r2 = _gh_expect_component(score_fn, lam, tau * v0, n_nodes)
    return (1.0 - eps) * r1 + eps * r2


def _contaminated_risk_vec(score_fn, lam_grid: np.ndarray, eps: float,
                            tau: float, v0: float,
                            n_nodes: int = N_NODES) -> np.ndarray:
    """Vectorised contaminated risk over a grid of lambda values (no Python loop)."""
    r1 = _gh_expect_vec(score_fn, lam_grid, v0, n_nodes)
    r2 = _gh_expect_vec(score_fn, lam_grid, tau * v0, n_nodes)
    return (1.0 - eps) * r1 + eps * r2


def find_pseudo_true(score_fn, eps: float, tau: float, v0: float,
                     lam_lo: float = -5.0, lam_hi: float = 5.0,
                     n_scan: int = N_SCAN, n_nodes: int = N_NODES) -> float:
    """Find lambda* minimising the contaminated risk via vectorised scan + refinement.

    Returns v* = exp(lambda*).
    """
    lam_grid = np.linspace(lam_lo, lam_hi, n_scan)
    # Vectorised scan — single matrix operation instead of a Python loop
    risks = _contaminated_risk_vec(score_fn, lam_grid, eps, tau, v0, n_nodes)
    best_idx = int(np.argmin(risks))
    lam_star_scan = lam_grid[best_idx]

    # Refine in a narrow bracket around the scan minimum
    step = (lam_hi - lam_lo) / n_scan
    lo = max(lam_lo, lam_star_scan - 3.0 * step)
    hi = min(lam_hi, lam_star_scan + 3.0 * step)

    # Scalar gradient via central finite differences for Brent root-finding
    def grad(lam: float) -> float:
        dl = 1e-5
        return (contaminated_risk(score_fn, lam + dl, eps, tau, v0, n_nodes)
                - contaminated_risk(score_fn, lam - dl, eps, tau, v0, n_nodes)) / (2.0 * dl)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lam_star = brentq(grad, lo, hi, xtol=1e-9, maxiter=200)
    except (ValueError, RuntimeError):
        lam_star = lam_star_scan  # fallback to scan minimum

    return float(np.exp(lam_star))


# ── Main computation ──────────────────────────────────────────────────────────

def run():
    """Compute v*/v0 vs tau for all criteria and produce Figure 2 equivalent."""

    tau_grid = np.concatenate([
        np.linspace(1.0, 5.0,  20),
        np.linspace(5.0, 20.0, 20),
        np.linspace(20.0, 60.0, 20),
    ])
    tau_grid = np.unique(tau_grid)

    scores = {
        "Gaussian-Log":  score_log,
        "Density-Power": score_DP,
        "CRPS":          score_CRPS,
        "MMD-RBF":       score_MMD,
    }
    colors = {"Gaussian-Log": "C0", "Density-Power": "C1", "CRPS": "C2", "MMD-RBF": "C3"}

    results = {name: [] for name in scores}
    # Closed-form for log
    log_exact = [V0 * (1.0 + EPS * (tau - 1.0)) / V0 for tau in tau_grid]

    print("Computing pseudo-true targets...")
    for name, sfn in scores.items():
        for tau in tau_grid:
            v_star = find_pseudo_true(sfn, EPS, tau, V0)
            results[name].append(v_star / V0)
        print(f"  {name}: done.  v*/v0 at tau=20 = {results[name][-20]:.4f}")

    # Validate log score: numerical vs closed-form
    log_numerical = np.array(results["Gaussian-Log"])
    log_cf        = np.array(log_exact)
    max_err = float(np.max(np.abs(log_numerical - log_cf)))
    print(f"\nGaussian-log max error (numerical vs closed-form): {max_err:.2e}")
    assert max_err < 1e-3, f"Log pseudo-true target error too large: {max_err:.2e}"

    # ── Plot ──────────────────────────────────────────────────────────────────
    sns.set_theme(style="whitegrid", font_scale=1.1)
    fig, ax = plt.subplots(figsize=(9, 5))

    # Exact log line
    ax.plot(tau_grid, log_exact, color="C0", lw=2.0, label="Gaussian-Log (exact)")
    ax.plot(tau_grid, results["Gaussian-Log"], color="C0", lw=1.0, ls="--",
            alpha=0.7, label="Gaussian-Log (numerical)")
    for name in ["Density-Power", "CRPS", "MMD-RBF"]:
        ax.plot(tau_grid, results[name], color=colors[name], lw=2.0, label=name)

    ax.set_xlabel("Contamination severity τ")
    ax.set_ylabel("v*/v₀")
    ax.set_title("Pseudo-True Variance Target vs. Contamination Severity\n"
                 "Q = (1−ε)·N(0,v₀) + ε·N(0,τv₀),  ε=0.05")
    ax.legend(fontsize=9)
    ax.set_xlim(tau_grid[0], min(tau_grid[-1], 60))
    ax.set_ylim(0.9, min(log_exact[-1] * 1.05, 5.0))

    fig_path = os.path.join(RESULTS_DIR, "exp2_pseudo_true_targets.png")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=120)
    plt.close()
    print(f"Saved: {fig_path}")

    # Full-range plot (log scale for y axis)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(tau_grid, log_exact, color="C0", lw=2.0, label="Gaussian-Log (exact)")
    for name in ["Density-Power", "CRPS", "MMD-RBF"]:
        ax.plot(tau_grid, results[name], color=colors[name], lw=2.0, label=name)
    ax.set_yscale("log")
    ax.set_xlabel("τ")
    ax.set_ylabel("v*/v₀  (log scale)")
    ax.set_title("Pseudo-True Targets (log scale) — Saturation of Bounded Criteria")
    ax.legend(fontsize=9)
    fig2_path = os.path.join(RESULTS_DIR, "exp2_pseudo_true_logscale.png")
    plt.tight_layout()
    plt.savefig(fig2_path, dpi=120)
    plt.close()
    print(f"Saved: {fig2_path}")

    # Save CSV
    df = pd.DataFrame({"tau": tau_grid, "log_exact": log_exact, **results})
    csv_path = os.path.join(RESULTS_DIR, "exp2_pseudo_true_table.csv")
    df.to_csv(csv_path, index=False, float_format="%.6f")
    print(f"Saved: {csv_path}")

    # Saturation check: bounded criteria should grow much less than log
    for name in ["Density-Power", "CRPS", "MMD-RBF"]:
        arr = np.array(results[name])
        # v*/v0 at tau=60 should be well below the log target
        log_at_60 = float(log_exact[-1])
        arr_at_60 = float(arr[-1])
        assert arr_at_60 < log_at_60 * 0.8, (
            f"{name}: should saturate below log target ({arr_at_60:.3f} vs {log_at_60:.3f})"
        )
    print("✓ Saturation assertions passed (bounded criteria < Gaussian-log).")

    return df


if __name__ == "__main__":
    print("=" * 60)
    print("Experiment 2: Pseudo-True Target Shifts")
    print("=" * 60)
    df = run()
    print("\nSample output (tau in {1,5,10,20,40}):")
    mask = df["tau"].isin([1.0, 5.0, 10.0, 20.0, 40.0])
    if not mask.any():
        mask = (df["tau"] - df["tau"].iloc[0]).abs() < 0.1
    print(df[mask][["tau", "log_exact", "Gaussian-Log", "Density-Power",
                     "CRPS", "MMD-RBF"]].to_string(index=False))
