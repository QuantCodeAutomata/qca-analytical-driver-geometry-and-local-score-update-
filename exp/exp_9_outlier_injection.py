"""
Experiment 9: Controlled S&P 500 Outlier Injection.

Injects a synthetic one-day outlier into a fixed historical S&P 500 return
path and compares downstream filter perturbations across criteria with all
parameters frozen.

Model cells:
  1. Gaussian log
  2. Student-t6 log       (bounded, as density-change benchmark)
  3. Student-t6 density-power  (approximated via bounded DP score)
  4. Student-t6 MMD        (approximated via bounded MMD score)

Parameters estimated on 2012-01-01 to 2016-12-31, frozen throughout 2017.
Outlier injected on 2017-07-03.

Validation targets (Figure 4, Table):
  At B = 100:
    Gaussian-log peak  ≈ 541.64
    Student-log        ≈ 1.29
    Student-DP         ≈ 0.25
    Student-MMD        ≈ 0.42
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
from scipy.optimize import minimize

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.score_drivers import u_log, u_DP, u_CRPS, u_MMD, J_LOG, J_DP, J_CRPS, J_MMD

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

NU = 6.0   # Student degrees of freedom


# ── Student-t6 score driver ───────────────────────────────────────────────────

def u_student_log(z: np.ndarray, nu: float = NU) -> np.ndarray:
    """Student-t log score update (bounded).

    u_t(z, nu) = (nu+1)*z^2/(nu+z^2) - 1
    This is bounded in (-1, nu) for all z.
    """
    return (nu + 1.0) * z**2 / (nu + z**2) - 1.0


def u_student_DP(z: np.ndarray, nu: float = NU, beta: float = 0.15) -> np.ndarray:
    """Density-power score driver under Student-t6 working density (approximation).

    Under a Student-t working density the density-power update is bounded.
    We approximate by combining the Student redescending shape with the
    density-power weighting.  The key property (bounded update) is preserved.
    """
    # Use density-power weighting applied to the Student standardised residual
    from src.score_drivers import _c_b, _kappa_b
    b = beta
    c_b = _c_b
    kappa_b = _kappa_b
    # Weight by the Student tail-robustified version of the DP update
    w = (nu / (nu + z**2)) ** ((nu + 1) / 2.0)   # Student-t weighting kernel
    base_DP = (1 + b) * c_b * np.exp(-b * z**2 / 2.0) * (z**2 - 1.0) + kappa_b
    # Blend: weight the DP update by the Student density ratio to suppress outliers
    return w * base_DP + (1.0 - w) * kappa_b


def u_student_MMD(z: np.ndarray, nu: float = NU) -> np.ndarray:
    """MMD score driver under Student-t6 working density (approximation).

    The MMD score depends only on the predictive distribution through the kernel,
    so it is approximately density-agnostic; we use the standard Gaussian MMD driver
    re-weighted by the Student density ratio to suppress leverage for extreme z.
    """
    w = (nu / (nu + z**2)) ** ((nu + 1) / 2.0)
    return w * u_MMD(z)


# ── Data loading ──────────────────────────────────────────────────────────────

def load_spx(start: str = "2010-01-01", end: str = "2018-01-01") -> pd.DataFrame:
    """Load S&P 500 OHLCV from Yahoo Finance."""
    try:
        import yfinance as yf
        df = yf.download("^GSPC", start=start, end=end,
                         auto_adjust=True, progress=False)
        if df.empty:
            raise ValueError("Empty Yahoo Finance response")
        return df
    except Exception as e:
        warnings.warn(f"Yahoo Finance download failed: {e}. Using synthetic data.")
        return pd.DataFrame()


def make_synthetic_spx(start: str = "2010-01-01", end: str = "2018-01-01",
                        seed: int = 42) -> pd.DataFrame:
    """Synthetic S&P 500 replacement for testing."""
    rng   = np.random.default_rng(seed)
    dates = pd.date_range(start, end, freq="B")
    n     = len(dates)
    log_ret = rng.normal(0.03, 1.0, n)       # daily log returns (percent)
    log_price = 2000.0 * np.exp(np.cumsum(log_ret) / 100.0)
    df = pd.DataFrame({
        "Open":   log_price * (1 - 0.003 * rng.random(n)),
        "High":   log_price * (1 + 0.005 * rng.random(n)),
        "Low":    log_price * (1 - 0.005 * rng.random(n)),
        "Close":  log_price,
        "Volume": rng.integers(1_000_000, 5_000_000, n).astype(float),
    }, index=dates)
    return df


# ── Filter infrastructure ─────────────────────────────────────────────────────

_DRIVERS = {
    "Gaussian-Log":   u_log,
    "Student-Log":    lambda z: u_student_log(z, NU),
    "Student-DP":     lambda z: u_student_DP(z, NU),
    "Student-MMD":    lambda z: u_student_MMD(z, NU),
}
_COLORS = {
    "Gaussian-Log": "C0",
    "Student-Log":  "C1",
    "Student-DP":   "C2",
    "Student-MMD":  "C3",
}


def run_filter(returns: np.ndarray, omega: float, phi: float, alpha: float,
               driver_fn: callable, lam0: float | None = None) -> np.ndarray:
    """Run score-driven filter recursion, return lambda path (length T+1)."""
    T   = len(returns)
    lam = np.empty(T + 1)
    lam[0] = np.log(np.var(returns) + 1e-8) if lam0 is None else lam0
    for t in range(T):
        z      = returns[t] * np.exp(-lam[t] / 2.0)
        lam[t + 1] = omega + phi * lam[t] + alpha * driver_fn(np.array([z]))[0]
    return lam


def fit_filter(returns: np.ndarray, driver_fn: callable,
               n_starts: int = 4) -> dict:
    """Fit omega, phi, alpha via Gaussian log-likelihood, multi-start L-BFGS-B."""
    lam_var = np.log(np.var(returns) + 1e-8)

    def neg_ll(params):
        omega, logit_phi, log_alpha = params
        phi   = 0.999 / (1.0 + np.exp(-logit_phi))
        alpha = np.exp(log_alpha)
        lam   = run_filter(returns, omega, phi, alpha, driver_fn,
                           lam0=lam_var)
        ll    = -0.5 * (lam[:-1] + np.exp(-lam[:-1]) * returns**2)
        if not np.all(np.isfinite(ll)):
            return 1e10
        return -float(np.sum(ll))

    best_val = np.inf
    best_phi = 0.97
    best_alpha = 0.05
    best_omega = lam_var * 0.03

    for phi_i in [0.90, 0.95, 0.97, 0.99][:n_starts]:
        logit_phi = np.log(phi_i / (0.999 - phi_i))
        x0 = [lam_var * (1 - phi_i), logit_phi, np.log(0.05)]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                res = minimize(neg_ll, x0, method="L-BFGS-B",
                               bounds=[(-10, 10), (-6, 6), (-6, 2)],
                               options={"maxiter": 400, "ftol": 1e-10})
                if res.fun < best_val:
                    best_val   = res.fun
                    phi_fit    = 0.999 / (1.0 + np.exp(-res.x[1]))
                    alpha_fit  = np.exp(res.x[2])
                    best_phi   = phi_fit
                    best_alpha = alpha_fit
                    best_omega = res.x[0]
            except Exception:
                pass

    return {"omega": best_omega, "phi": best_phi, "alpha": best_alpha}


# ── Main experiment ───────────────────────────────────────────────────────────

def run(B_grid: list | None = None, seed: int = 42):
    """Run outlier-injection experiment.

    Parameters
    ----------
    B_grid : list of shock magnitudes (in sigma_ref units)
    """
    if B_grid is None:
        B_grid = [0, 5, 10, 20, 35, 60, 100]

    # ── Data ──────────────────────────────────────────────────────────────────
    print("Loading S&P 500 data...")
    spx = load_spx("2010-01-01", "2018-12-31")
    if spx.empty:
        spx = make_synthetic_spx("2010-01-01", "2018-12-31", seed=seed)
        print("  Using synthetic data.")

    # Handle multi-level columns from yfinance
    if isinstance(spx.columns, pd.MultiIndex):
        spx.columns = spx.columns.get_level_values(0)

    close = spx["Close"].dropna()
    log_ret = np.log(close).diff().dropna() * 100.0   # log returns in percent

    # Estimation window: 2012-01-01 to 2016-12-31
    train_mask = (log_ret.index >= "2012-01-01") & (log_ret.index <= "2016-12-31")
    train_ret  = log_ret[train_mask].values
    train_mean = float(np.mean(train_ret))
    train_ret_dm = train_ret - train_mean

    # 2017 evaluation period
    eval_mask  = (log_ret.index >= "2017-01-01") & (log_ret.index <= "2017-12-31")
    eval_ret   = log_ret[eval_mask]
    eval_ret_dm = eval_ret.values - train_mean

    print(f"Train: n={len(train_ret_dm)},  Eval 2017: n={len(eval_ret_dm)}")

    # Injection date: 2017-07-03 (or nearest trading day)
    inject_dates = eval_ret.index[eval_ret.index >= "2017-07-03"]
    if len(inject_dates) == 0:
        inject_dates = eval_ret.index[eval_ret.index >= "2017-06-30"]
    inject_date = inject_dates[0]
    inject_idx  = int(np.where(eval_ret.index == inject_date)[0][0])
    print(f"Injection date: {inject_date.date()}  (index {inject_idx})")

    # sigma_ref: RMS of previous 60 sessions
    pre_inject = eval_ret.iloc[max(0, inject_idx - 60): inject_idx].values - train_mean
    if len(pre_inject) < 5:
        pre_inject = train_ret_dm[-60:]
    sigma_ref = float(np.sqrt(np.mean(pre_inject**2)))
    print(f"sigma_ref = {sigma_ref:.4f}")

    # ── Fit frozen models on training data ────────────────────────────────────
    params_fitted = {}
    for name, u_fn in _DRIVERS.items():
        print(f"  Fitting {name}...", end=" ", flush=True)
        p = fit_filter(train_ret_dm, u_fn, n_starts=4)
        params_fitted[name] = p
        print(f"phi={p['phi']:.4f}  alpha={p['alpha']:.4f}")

    # ── Compute lam0 from end of training ─────────────────────────────────────
    lam0_per_model = {}
    for name, u_fn in _DRIVERS.items():
        p = params_fitted[name]
        lam_train = run_filter(train_ret_dm, p["omega"], p["phi"], p["alpha"], u_fn)
        lam0_per_model[name] = float(lam_train[-1])

    # ── Run perturbation study ─────────────────────────────────────────────────
    peak_perturb = {name: [] for name in _DRIVERS}
    perturb_paths = {B: {} for B in B_grid}

    for B in B_grid:
        shock = B * sigma_ref
        eval_perturbed = eval_ret_dm.copy()
        eval_perturbed[inject_idx] += shock

        for name, u_fn in _DRIVERS.items():
            p    = params_fitted[name]
            lam0 = lam0_per_model[name]

            lam_orig = run_filter(eval_ret_dm,    p["omega"], p["phi"], p["alpha"],
                                  u_fn, lam0=lam0)
            lam_pert = run_filter(eval_perturbed, p["omega"], p["phi"], p["alpha"],
                                  u_fn, lam0=lam0)

            delta = lam_pert[:-1] - lam_orig[:-1]   # lambda difference path
            peak  = float(np.max(np.abs(delta)))
            peak_perturb[name].append(peak)
            perturb_paths[B][name] = delta

    # ── Plots ──────────────────────────────────────────────────────────────────
    sns.set_theme(style="whitegrid", font_scale=1.1)

    # Peak perturbation vs B
    fig, ax = plt.subplots(figsize=(9, 5))
    for name, col in _COLORS.items():
        ax.semilogy(B_grid, [p if p > 0 else 1e-6 for p in peak_perturb[name]],
                    color=col, lw=2.0, marker="o", ms=5, label=name)
    ax.set_xlabel("Shock size B (multiples of σ_ref)")
    ax.set_ylabel("Peak |Δλ|  (log scale)")
    ax.set_title("Peak Filter Perturbation vs. Outlier Size B\n"
                 "S&P 500, parameters frozen on 2012-2016, shock on 2017-07-03")
    ax.legend(fontsize=9)
    fig_path = os.path.join(RESULTS_DIR, "exp9_peak_perturbation.png")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=120)
    plt.close()
    print(f"Saved: {fig_path}")

    # Perturbation path at B=100
    if 100 in B_grid:
        B_show = 100
    else:
        B_show = max(B_grid)

    fig, ax = plt.subplots(figsize=(11, 5))
    dates_ev = eval_ret.index
    for name, col in _COLORS.items():
        delta = perturb_paths[B_show][name]
        ax.plot(dates_ev[:len(delta)], delta, color=col, lw=1.6, label=name)
    ax.axvline(inject_date, color="red", ls="--", lw=1.0, label="Injection date")
    ax.axhline(0, color="k", lw=0.6, ls="--")
    ax.set_xlabel("Date")
    ax.set_ylabel("Δλ (perturbed − original)")
    ax.set_title(f"Filter Perturbation Path  (B={B_show}, shock = {B_show}×σ_ref)")
    ax.legend(fontsize=9)
    fig2_path = os.path.join(RESULTS_DIR, "exp9_perturbation_path.png")
    plt.tight_layout()
    plt.savefig(fig2_path, dpi=120)
    plt.close()
    print(f"Saved: {fig2_path}")

    # ── Summary table ──────────────────────────────────────────────────────────
    df_rows = []
    for i, B in enumerate(B_grid):
        row = {"B": B}
        for name in _DRIVERS:
            row[name] = peak_perturb[name][i]
        df_rows.append(row)
    df = pd.DataFrame(df_rows)
    csv_path = os.path.join(RESULTS_DIR, "exp9_peak_table.csv")
    df.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"Saved: {csv_path}")

    print("\nPeak perturbations:")
    print(df.to_string(index=False))

    # ── Validation ────────────────────────────────────────────────────────────
    # Key qualitative check: Gaussian-log peak should be much larger
    if 100 in B_grid:
        idx100 = B_grid.index(100)
        log_peak = peak_perturb["Gaussian-Log"][idx100]
        for name in ["Student-Log", "Student-DP", "Student-MMD"]:
            other_peak = peak_perturb[name][idx100]
            assert log_peak > other_peak * 2, (
                f"Gaussian-log peak ({log_peak:.2f}) should >> {name} ({other_peak:.2f})"
            )
        print(f"\n✓ Gaussian-log peak ({log_peak:.2f}) >> bounded criteria at B=100")

    # Gaussian-log peak should grow faster than linearly (at least super-linear in B)
    if len(B_grid) >= 3:
        B_arr  = np.array([b for b in B_grid if b > 0])
        log_peaks = np.array([peak_perturb["Gaussian-Log"][i]
                               for i, b in enumerate(B_grid) if b > 0])
        if len(B_arr) >= 2:
            log_B = np.log(B_arr + 1e-8)
            log_P = np.log(log_peaks + 1e-8)
            slope = float(np.polyfit(log_B, log_P, 1)[0])
            print(f"Gaussian-log log-log slope: {slope:.2f}  (target ≈ 2.0 for quadratic)")

    return df


if __name__ == "__main__":
    print("=" * 60)
    print("Experiment 9: Controlled S&P 500 Outlier Injection")
    print("=" * 60)
    B_grid = [0, 5, 10, 20, 35, 60, 100]
    df = run(B_grid=B_grid)
