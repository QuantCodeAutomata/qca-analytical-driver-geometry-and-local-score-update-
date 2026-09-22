"""
Experiment 14: Gaussian Location Filtering on the Nile River

Data: Annual Nile flow 1871-1970 (100 observations, statsmodels nile dataset),
units 10^8 m³.

Model: Y_t ~ N(μ_t, σ²)
  μ_{t+1} = ω + φ·μ_t + κ·u_loc(z_t)
  z_t = (y_t − μ_t) / σ   (standardised residual)
  κ = α̃·σ  (gain in flow units)

Three location score drivers (functions of z):
  Log  : u_log(z)  = z                     (unbounded)
  CRPS : u_CRPS(z) = 2·Φ(z) − 1           (bounded in [−1, 1])
  DP   : u_DP(z)   = (1+b)·c_b·e^{−b z²/2}·z   (b=0.15; Huber-like, soft bound)

Procedure
---------
1. Load Nile data (years 1871-1970).
2. Fit each filter on 1871-1944 (indices 0-73, 74 observations).
3. Freeze all parameters (ω, φ, κ, σ̂) after training.
4. Generate clean continuation path 1945-1970 with frozen filter.
5. Inject +6·σ̂ outlier at year 1945; re-run frozen filter from 1945.
6. Report displacement (μ_1946_contam − μ_1946_clean) / σ̂.

Validation targets:
  Log  ≈ 1.615·σ̂,  CRPS ≈ 0.465·σ̂,  DP ≈ 0.066·σ̂

Outputs
-------
results/exp14_nile_clean.png
results/exp14_nile_injected.png
"""

import sys
import os
import warnings
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import ndtr as Phi   # standard normal CDF
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import statsmodels.api as sm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# DP constants
_B = 0.15
_CB = (2 * np.pi) ** (-_B / 2.0)


# ---------------------------------------------------------------------------
# Location score drivers  (functions of dimensionless z = r / σ)
# ---------------------------------------------------------------------------

def u_log_z(z):
    """Unbounded: standardised residual."""
    return z


def u_CRPS_z(z):
    """Bounded in [−1, 1]: 2·Φ(z) − 1."""
    return 2.0 * Phi(z) - 1.0


def u_DP_z(z):
    """Soft-bounded DP driver (b=0.15)."""
    return (1 + _B) * _CB * np.exp(-_B * z * z / 2.0) * z


DRIVERS = {
    'log':  u_log_z,
    'CRPS': u_CRPS_z,
    'DP':   u_DP_z,
}


# ---------------------------------------------------------------------------
# Filter
# ---------------------------------------------------------------------------

def run_location_filter(y, omega, phi, alpha_tilde, sigma, mu0=None):
    """Run location filter; return μ path of length len(y)+1.

    Update: μ_{t+1} = ω + φ·μ_t + κ·u(z_t)
    where κ = α̃·σ,  z_t = (y_t − μ_t)/σ.
    """
    raise NotImplementedError   # replaced by driver-specific below


def make_filter_runner(driver_fn):
    """Return a runner closure for a given score driver."""
    def run(y, omega, phi, kappa, sigma, mu0=None):
        T = len(y)
        mu = np.empty(T + 1)
        mu[0] = float(np.mean(y)) if mu0 is None else float(mu0)
        for t in range(T):
            z = (y[t] - mu[t]) / sigma
            mu[t + 1] = omega + phi * mu[t] + kappa * driver_fn(z)
        return mu
    return run


# Build per-driver runners
_runners = {name: make_filter_runner(fn) for name, fn in DRIVERS.items()}


def neg_loglik(params, y, driver_name, runner):
    """Negative Gaussian log-likelihood."""
    omega, phi, log_kappa, log_sigma = params
    kappa = np.exp(log_kappa)
    sigma = np.exp(log_sigma)
    mu = runner(y, omega, phi, kappa, sigma)
    r = y - mu[:-1]
    ll = -0.5 * np.log(2 * np.pi * sigma ** 2) - 0.5 * (r / sigma) ** 2
    if not np.isfinite(ll).all():
        return 1e12
    return float(-np.sum(ll))


def fit_filter(y_train, driver_name, phi0=0.88, alpha0=0.25):
    """Fit location filter on training data by Gaussian MLE."""
    runner = _runners[driver_name]
    sigma0 = float(np.std(y_train))
    mean0 = float(np.mean(y_train))
    omega0 = mean0 * (1 - phi0)
    kappa0 = alpha0 * sigma0
    x0 = [omega0, phi0, np.log(kappa0), np.log(sigma0)]
    bounds = [(-1e5, 1e5), (0.50, 0.9995), (-10.0, 10.0), (1.0, 10.0)]
    best = None
    for method in ['L-BFGS-B', 'Nelder-Mead']:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                res = minimize(neg_loglik, x0, args=(y_train, driver_name, runner),
                               method=method, bounds=bounds if method == 'L-BFGS-B' else None,
                               options={'maxiter': 5000, 'ftol': 1e-12, 'xatol': 1e-8})
            if best is None or res.fun < best.fun:
                best = res
        except Exception:
            continue
    omega, phi, log_kappa, log_sigma = best.x
    return {
        'omega': omega,
        'phi': phi,
        'kappa': np.exp(log_kappa),
        'sigma': np.exp(log_sigma),
        'alpha_tilde': np.exp(log_kappa) / np.exp(log_sigma),
        'nll': best.fun,
    }


# ---------------------------------------------------------------------------
# Outlier injection experiment
# ---------------------------------------------------------------------------

def displacement_experiment(y_full, years, train_cutoff=1944, outlier_year=1945,
                             outlier_sigma_mult=6.0):
    """
    For each driver:
    1. Fit on training data (≤ train_cutoff).
    2. Run frozen filter on full data (clean path).
    3. Inject +6σ̂ at outlier_year, run frozen filter (contaminated path).
    4. Compute displacement at the year after outlier_year.

    Returns dict with driver → displacement / σ̂.
    """
    train_mask = years <= train_cutoff
    y_train = y_full[train_mask]

    results = {}
    for name in DRIVERS:
        params = fit_filter(y_train, name)
        omega, phi, kappa, sigma_hat = (params['omega'], params['phi'],
                                        params['kappa'], params['sigma'])
        runner = _runners[name]
        alpha_tilde = params['alpha_tilde']

        # Clean path through entire series
        mu_clean = runner(y_full, omega, phi, kappa, sigma_hat)

        # Contaminated: inject at outlier_year
        y_contam = y_full.copy()
        inj_idx = int(np.where(years == outlier_year)[0][0])
        y_contam[inj_idx] = mu_clean[inj_idx] + outlier_sigma_mult * sigma_hat

        # Re-run from start; frozen parameters; mu_0 same
        mu_contam = runner(y_contam, omega, phi, kappa, sigma_hat)

        # Displacement at year after outlier (index inj_idx + 1 in mu path)
        disp = mu_contam[inj_idx + 1] - mu_clean[inj_idx + 1]
        disp_norm = disp / sigma_hat

        results[name] = {
            'mu_clean': mu_clean,
            'mu_contam': mu_contam,
            'sigma_hat': sigma_hat,
            'alpha_tilde': alpha_tilde,
            'kappa': kappa,
            'displacement_raw': float(disp),
            'displacement_norm': float(disp_norm),
        }
        print(f'  {name:5s}: σ̂={sigma_hat:.1f}, α̃={alpha_tilde:.4f}, '
              f'κ={kappa:.2f},  Δμ/σ̂={disp_norm:.4f}')

    return results


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_clean(years, y_full, results, path):
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(years, y_full, color='grey', alpha=0.5, linewidth=0.8, label='Observed')
    colours = {'log': '#1f77b4', 'CRPS': '#ff7f0e', 'DP': '#2ca02c'}
    for name, res in results.items():
        mu = res['mu_clean'][:-1]  # μ_t for t=1…T
        ax.plot(years, mu, color=colours[name], linewidth=1.5, label=f'μ̂ ({name})')
    ax.set_xlabel('Year')
    ax.set_ylabel('Annual flow (10⁸ m³)')
    ax.set_title('Exp 14: Nile location filter — clean paths')
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f'  Saved {path}')


def plot_injected(years, y_full, results, outlier_year, path):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    colours = {'log': '#1f77b4', 'CRPS': '#ff7f0e', 'DP': '#2ca02c'}
    for ax, (name, res) in zip(axes, results.items()):
        ax.plot(years, y_full, color='grey', alpha=0.4, linewidth=0.8)
        ax.plot(years, res['mu_clean'][:-1], color=colours[name],
                linewidth=1.5, label='Clean', linestyle='--')
        ax.plot(years, res['mu_contam'][:-1], color=colours[name],
                linewidth=1.5, label='Contaminated')
        ax.axvline(outlier_year, color='red', linewidth=0.9, linestyle=':')
        disp = res['displacement_norm']
        ax.set_title(f'{name}  (Δμ/σ̂ = {disp:.3f})')
        ax.set_xlabel('Year')
        ax.legend(fontsize=8)
    axes[0].set_ylabel('Annual flow (10⁸ m³)')
    fig.suptitle(f'Exp 14: Injected +6σ outlier at {outlier_year}', fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f'  Saved {path}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Load Nile data
    nile_data = sm.datasets.nile.load_pandas().data
    years = nile_data['year'].values.astype(int)
    y_full = nile_data['volume'].values.astype(float)
    print(f'Nile data: {len(y_full)} observations, years {years[0]}–{years[-1]}')

    TRAIN_CUTOFF = 1944
    OUTLIER_YEAR = 1945
    n_train = int(np.sum(years <= TRAIN_CUTOFF))
    print(f'Training: {years[0]}–{TRAIN_CUTOFF} ({n_train} obs), '
          f'Test: {OUTLIER_YEAR}–{years[-1]}')

    print('\nFitting location filters…')
    results = displacement_experiment(y_full, years,
                                      train_cutoff=TRAIN_CUTOFF,
                                      outlier_year=OUTLIER_YEAR,
                                      outlier_sigma_mult=6.0)

    print('\nDisplacement summary (normalised by σ̂):')
    rows = []
    for name, res in results.items():
        print(f'  {name:5s}: {res["displacement_norm"]:.4f}  '
              f'(target: log≈1.615, CRPS≈0.465, DP≈0.066)')
        rows.append({'driver': name,
                     'sigma_hat': res['sigma_hat'],
                     'alpha_tilde': res['alpha_tilde'],
                     'displacement_norm': res['displacement_norm']})

    df = pd.DataFrame(rows)
    csv_path = os.path.join(RESULTS_DIR, 'exp14_displacement.csv')
    df.to_csv(csv_path, index=False)
    print(f'\nSaved {csv_path}')

    plot_clean(years, y_full, results,
               os.path.join(RESULTS_DIR, 'exp14_nile_clean.png'))
    plot_injected(years, y_full, results, OUTLIER_YEAR,
                  os.path.join(RESULTS_DIR, 'exp14_nile_injected.png'))

    print('\nExp 14 complete.')
    return df


if __name__ == '__main__':
    main()
