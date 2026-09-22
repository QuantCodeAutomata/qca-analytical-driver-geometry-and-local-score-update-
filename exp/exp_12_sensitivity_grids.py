"""
Experiment 12: Empirical Robustness and Sensitivity Grids

Four sensitivity sweeps on a subset of 3 representative equity markets
(^GSPC, ^FTSE, ^N225) to keep computation tractable.

Grid A: GARCH(1,1)-N vs GARCH(1,1)-t  –  QLIKE and VaR coverage
Grid B: Student ν sweep (ν ∈ {4,5,6,8,10})  –  score-driven filter
Grid C: Train-fraction sweep (0.60, 0.70, 0.80)  –  ranking stability
Grid D: MMD bandwidth multiplier (0.5, 1.0, 2.0)  –  score scaling effect

Outputs
-------
results/exp12_grid_A.png, results/exp12_grid_B.png
results/exp12_grid_C.png, results/exp12_grid_D.png
results/exp12_sensitivity_table.csv
"""

import sys
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from arch import arch_model

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.score_drivers import u_log, u_DP, u_CRPS, u_MMD, u_student

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

MARKETS = ['^GSPC', '^FTSE', '^N225']
START_DATE = '2000-01-01'
END_DATE = '2023-12-31'
BASE_TRAIN_FRAC = 0.70
RNG = np.random.default_rng(42)


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def download_returns(tickers, start=START_DATE, end=END_DATE):
    """Download adjusted-close log returns.  Falls back to synthetic if unavailable."""
    import yfinance as yf
    returns = {}
    for t in tickers:
        try:
            raw = yf.download(t, start=start, end=end, progress=False, auto_adjust=True)
            if raw.empty:
                raise ValueError('empty')
            px = raw['Close'].dropna()
            r = np.log(px / px.shift(1)).dropna().values * 100  # percent returns
            returns[t] = r
        except Exception as exc:
            warnings.warn(f'{t}: download failed ({exc}), using synthetic data')
            returns[t] = _synthetic_garch(1200)
    return returns


def _synthetic_garch(n, omega=0.05, alpha=0.05, beta=0.90, rng=None):
    """Simulate a GARCH(1,1)-N series (returns in percent)."""
    rng = RNG if rng is None else rng
    h = np.empty(n)
    y = np.empty(n)
    h[0] = omega / (1 - alpha - beta)
    y[0] = np.sqrt(h[0]) * rng.standard_normal()
    for t in range(1, n):
        h[t] = omega + alpha * y[t - 1] ** 2 + beta * h[t - 1]
        y[t] = np.sqrt(h[t]) * rng.standard_normal()
    return y


def split(arr, frac):
    n = int(len(arr) * frac)
    return arr[:n], arr[n:]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def qlike(sigma2_hat, r):
    """QLIKE = mean(log σ² + r²/σ²)."""
    return float(np.mean(np.log(sigma2_hat) + r ** 2 / sigma2_hat))


def var_coverage(sigma_hat, r, alpha=0.05):
    """Empirical coverage of 1-sided VaR_{1-alpha}.

    Returns fraction of observations that violate the VaR bound.
    Theoretical level = alpha.
    """
    var = -norm.ppf(alpha) * sigma_hat
    violations = np.mean(r < -var)
    return float(violations)


# ---------------------------------------------------------------------------
# Score-driven filter  (fixed parameters, varying driver)
# ---------------------------------------------------------------------------

def run_sdf(y, driver, omega=0.0, phi=0.97, alpha=0.05, nu=5.0, mmd_mult=1.0):
    """Run score-driven scale filter with fixed parameters.

    Returns in-sample conditional variance sigma_t^2 for t=1…T.
    """
    T = len(y)
    lam = np.empty(T + 1)
    lam[0] = np.log(np.var(y) + 1e-8)
    for t in range(T):
        z = y[t] * np.exp(-lam[t] / 2.0)
        if driver == 'log':
            u = u_log(z)
        elif driver == 'DP':
            u = u_DP(z)
        elif driver == 'CRPS':
            u = u_CRPS(z)
        elif driver == 'MMD':
            u = mmd_mult * u_MMD(z)
        elif driver == 'student':
            u = u_student(z, nu)
        else:
            raise ValueError(f'Unknown driver: {driver}')
        lam[t + 1] = omega + phi * lam[t] + alpha * u
    return np.exp(lam[:-1])  # sigma_t^2


# ---------------------------------------------------------------------------
# Grid A: GARCH-N vs GARCH-t
# ---------------------------------------------------------------------------

def grid_a(returns_dict, train_frac=BASE_TRAIN_FRAC):
    print('\n=== Grid A: GARCH-N vs GARCH-t ===')
    rows = []
    for ticker, r in returns_dict.items():
        train, test = split(r, train_frac)
        for dist, label in [('normal', 'GARCH-N'), ('t', 'GARCH-t')]:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                am = arch_model(train, vol='GARCH', p=1, q=1, dist=dist, rescale=False)
                res = am.fit(disp='off')
            fc = res.forecast(horizon=1, reindex=False)
            # rolling 1-step-ahead variance on test set
            # re-run on full data to get test-period variance
            am2 = arch_model(r, vol='GARCH', p=1, q=1, dist=dist, rescale=False)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                res2 = am2.fit(disp='off', starting_values=res.params.values)
            cond_var = res2.conditional_volatility ** 2
            test_var = cond_var[len(train):]
            test_r = test
            ql = qlike(test_var, test_r)
            cov = var_coverage(np.sqrt(test_var), test_r)
            rows.append({'ticker': ticker, 'model': label,
                         'QLIKE': ql, 'VaR_coverage': cov})
            print(f'  {ticker:8s} {label}: QLIKE={ql:.4f}, VaR_cov={cov:.4f}')
    return pd.DataFrame(rows)


def plot_grid_a(df, path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, metric in zip(axes, ['QLIKE', 'VaR_coverage']):
        pivot = df.pivot(index='ticker', columns='model', values=metric)
        pivot.plot(kind='bar', ax=ax)
        ax.set_title(metric)
        ax.set_xlabel('')
        ax.tick_params(axis='x', rotation=30)
    fig.suptitle('Grid A: GARCH-N vs GARCH-t', fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f'  Saved {path}')


# ---------------------------------------------------------------------------
# Grid B: Student ν sweep
# ---------------------------------------------------------------------------

def grid_b(returns_dict, nu_list=(4, 5, 6, 8, 10), train_frac=BASE_TRAIN_FRAC):
    print('\n=== Grid B: Student ν sweep ===')
    rows = []
    for ticker, r in returns_dict.items():
        train, test = split(r, train_frac)
        for nu in nu_list:
            sv = run_sdf(test, "student", nu=nu)
            ql = qlike(sv, test)
            cov = var_coverage(np.sqrt(sv), test)
            rows.append({'ticker': ticker, 'nu': nu,
                         'QLIKE': ql, 'VaR_coverage': cov})
            print(f'  {ticker:8s} nu={nu}: QLIKE={ql:.4f}, VaR_cov={cov:.4f}')
    return pd.DataFrame(rows)


def plot_grid_b(df, path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, metric in zip(axes, ['QLIKE', 'VaR_coverage']):
        for t, grp in df.groupby('ticker'):
            ax.plot(grp['nu'], grp[metric], marker='o', label=t)
        ax.set_xlabel('ν')
        ax.set_title(metric)
        ax.legend(fontsize=8)
    fig.suptitle('Grid B: Student ν sweep', fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f'  Saved {path}')


# ---------------------------------------------------------------------------
# Grid C: Train-fraction sweep
# ---------------------------------------------------------------------------

def grid_c(returns_dict, frac_list=(0.60, 0.70, 0.80)):
    print('\n=== Grid C: Train-fraction sweep ===')
    drivers = ['log', 'DP', 'CRPS', 'MMD']
    rows = []
    for ticker, r in returns_dict.items():
        for frac in frac_list:
            train, test = split(r, frac)
            for drv in drivers:
                sv = run_sdf(test, drv)
                ql = qlike(sv, test)
                rows.append({'ticker': ticker, 'frac': frac,
                             'driver': drv, 'QLIKE': ql})
            ql_vals = {d: rows[-len(drivers) + i]['QLIKE']
                       for i, d in enumerate(drivers)}
            ranked = sorted(ql_vals, key=ql_vals.get)
            print(f'  {ticker:8s} frac={frac}: ranking = {ranked}')
    return pd.DataFrame(rows)


def plot_grid_c(df, path):
    tickers = df['ticker'].unique()
    fig, axes = plt.subplots(1, len(tickers), figsize=(5 * len(tickers), 4),
                             sharey=False)
    if len(tickers) == 1:
        axes = [axes]
    for ax, t in zip(axes, tickers):
        sub = df[df['ticker'] == t]
        pivot = sub.pivot_table(index='frac', columns='driver', values='QLIKE')
        pivot.plot(ax=ax, marker='o')
        ax.set_title(t)
        ax.set_xlabel('Train fraction')
    axes[0].set_ylabel('QLIKE')
    fig.suptitle('Grid C: Train-fraction sweep', fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f'  Saved {path}')


# ---------------------------------------------------------------------------
# Grid D: MMD bandwidth multiplier
# ---------------------------------------------------------------------------

def grid_d(returns_dict, mult_list=(0.5, 1.0, 2.0), train_frac=BASE_TRAIN_FRAC):
    print('\n=== Grid D: MMD bandwidth multipliers ===')
    rows = []
    for ticker, r in returns_dict.items():
        _, test = split(r, train_frac)
        for mult in mult_list:
            sv = run_sdf(test, 'MMD', mmd_mult=mult)
            ql = qlike(sv, test)
            cov = var_coverage(np.sqrt(sv), test)
            rows.append({'ticker': ticker, 'mult': mult,
                         'QLIKE': ql, 'VaR_coverage': cov})
            print(f'  {ticker:8s} mult={mult}: QLIKE={ql:.4f}, VaR_cov={cov:.4f}')
    return pd.DataFrame(rows)


def plot_grid_d(df, path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, metric in zip(axes, ['QLIKE', 'VaR_coverage']):
        for t, grp in df.groupby('ticker'):
            ax.plot(grp['mult'], grp[metric], marker='s', label=t)
        ax.set_xlabel('Bandwidth multiplier')
        ax.set_title(metric)
        ax.legend(fontsize=8)
    fig.suptitle('Grid D: MMD bandwidth multiplier', fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f'  Saved {path}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print('Downloading market data (3 markets, 2000-2023)…')
    returns = download_returns(MARKETS)

    df_a = grid_a(returns)
    df_b = grid_b(returns)
    df_c = grid_c(returns)
    df_d = grid_d(returns)

    plot_grid_a(df_a, os.path.join(RESULTS_DIR, 'exp12_grid_A.png'))
    plot_grid_b(df_b, os.path.join(RESULTS_DIR, 'exp12_grid_B.png'))
    plot_grid_c(df_c, os.path.join(RESULTS_DIR, 'exp12_grid_C.png'))
    plot_grid_d(df_d, os.path.join(RESULTS_DIR, 'exp12_grid_D.png'))

    # Aggregate sensitivity table
    df_a['grid'] = 'A'
    df_b['grid'] = 'B'
    df_c['grid'] = 'C'
    df_d['grid'] = 'D'
    summary = pd.concat([
        df_a[['grid', 'ticker', 'model', 'QLIKE', 'VaR_coverage']].rename(columns={'model': 'variant'}),
        df_b[['grid', 'ticker', 'nu', 'QLIKE', 'VaR_coverage']].assign(
            variant=df_b['nu'].astype(str)).drop(columns='nu'),
        df_c[['grid', 'ticker', 'driver', 'frac', 'QLIKE']].assign(
            variant=df_c['driver'] + '@' + df_c['frac'].astype(str)).drop(columns=['driver', 'frac']),
        df_d[['grid', 'ticker', 'mult', 'QLIKE', 'VaR_coverage']].assign(
            variant='mult=' + df_d['mult'].astype(str)).drop(columns='mult'),
    ], ignore_index=True)

    csv_path = os.path.join(RESULTS_DIR, 'exp12_sensitivity_table.csv')
    summary.to_csv(csv_path, index=False)
    print(f'\nSaved {csv_path}')
    print('\nExp 12 complete.')


if __name__ == '__main__':
    main()
