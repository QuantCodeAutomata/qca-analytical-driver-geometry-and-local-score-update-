"""
Experiment 8: Descriptive Event-Path Robustness Studies.

Estimates score-driven filters on pre-event windows only, freezes parameters,
and compares annualised volatility paths around two market events:

  1. Thai baht float:    1997-07-02  (FRED series DEXTHUS)
  2. S&P 500 episode:    2008-09-29  (Yahoo Finance ^GSPC)

Validation targets (paper Figure G.10):
  Thai baht:  Gaussian-log peak ≈ 10^7.17; robust alternatives ≈ 10^1.63–1.70
  S&P 2008:   All four peaks roughly 10^1.81–1.88
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
from src.filters import ScaleDrivenFilter

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

ANNUALIZE = 252.0   # trading days per year

CRITERIA = {
    "Gaussian-Log":  ("log",  "C0"),
    "Density-Power": ("DP",   "C1"),
    "CRPS":          ("CRPS", "C2"),
    "MMD-RBF":       ("MMD",  "C3"),
}


# ── Data loading ──────────────────────────────────────────────────────────────

def load_yahoo(ticker: str, start: str, end: str) -> pd.Series:
    """Load daily adjusted close from Yahoo Finance, return as pd.Series."""
    try:
        import yfinance as yf
        df = yf.download(ticker, start=start, end=end,
                         auto_adjust=True, progress=False)
        if df.empty:
            raise ValueError(f"Empty data for {ticker}")
        close = df["Close"].squeeze()
        if hasattr(close, 'columns'):
            close = close.iloc[:, 0]
        close = close.dropna()
        close.name = ticker
        return close
    except Exception as e:
        warnings.warn(f"Yahoo Finance download failed for {ticker}: {e}")
        return pd.Series(dtype=float)


def load_fred(series_id: str, start: str, end: str) -> pd.Series:
    """Load FRED series via pandas_datareader."""
    try:
        import pandas_datareader as pdr
        data = pdr.DataReader(series_id, "fred", start, end)
        s = data[series_id].dropna()
        s.name = series_id
        return s
    except Exception as e:
        warnings.warn(f"FRED download failed for {series_id}: {e}")
        return pd.Series(dtype=float)


def compute_log_returns(prices: pd.Series) -> pd.Series:
    """Compute close-to-close log returns in percent."""
    return np.log(prices).diff().dropna() * 100.0


# ── Filter estimation ─────────────────────────────────────────────────────────

def fit_filter(returns: np.ndarray, driver: str,
               phi0: float = 0.97, alpha0: float = 0.05) -> dict:
    """Fit score-driven filter on returns array.

    Returns dict of fitted parameters (omega, phi, alpha).
    """
    filt = ScaleDrivenFilter(driver=driver)

    # Use log-likelihood objective regardless of scoring criterion for fitting
    # (consistent with frozen-parameter event study)
    lam_var = np.log(np.var(returns) + 1e-8)
    omega0  = lam_var * (1.0 - phi0)

    def neg_loglik(params):
        omega, phi, alpha = params
        if not (-1.0 < phi < 1.0) or alpha <= 0:
            return 1e9
        lam = filt.run(returns, omega, phi, alpha)
        ll = -0.5 * (lam[:-1] + np.exp(-lam[:-1]) * returns**2)
        return -np.sum(ll)

    best_nll = np.inf
    best_x   = np.array([omega0, phi0, alpha0])
    for phi_init in [0.90, 0.95, 0.97, 0.99]:
        for alpha_init in [0.01, 0.05, 0.10]:
            x0 = [lam_var * (1 - phi_init), phi_init, alpha_init]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = minimize(neg_loglik, x0, method="L-BFGS-B",
                               bounds=[(-10, 10), (0.5, 0.9999), (1e-4, 1.0)],
                               options={"maxiter": 300})
            if res.fun < best_nll:
                best_nll = res.fun
                best_x   = res.x

    return {"omega": best_x[0], "phi": best_x[1], "alpha": best_x[2]}


def filter_path(returns: np.ndarray, params: dict, driver: str,
                lam0: float | None = None) -> np.ndarray:
    """Run score-driven filter with frozen params. Returns lambda path."""
    filt = ScaleDrivenFilter(driver=driver)
    return filt.run(returns, params["omega"], params["phi"], params["alpha"],
                    lam0=lam0)


def annualized_vol(lam: np.ndarray, annualize: float = ANNUALIZE) -> np.ndarray:
    """Convert log-variance lambda to annualised volatility in percent."""
    return np.sqrt(annualize * np.exp(lam)) * 100.0


# ── Event study runner ────────────────────────────────────────────────────────

def run_event_study(event_name: str, prices: pd.Series,
                    event_date: str, train_years: int = 5,
                    event_window_days: int = 60):
    """Run one event study: estimate on pre-event window, freeze, filter through event.

    Parameters
    ----------
    event_name      : label for the event
    prices          : pd.Series of prices indexed by date
    event_date      : event date string 'YYYY-MM-DD'
    train_years     : pre-event training years
    event_window_days : days to show after the event
    """
    ed = pd.Timestamp(event_date)
    prices = prices.sort_index()

    # Training window: strictly before the event date
    train_end   = prices.index[prices.index < ed]
    if len(train_end) == 0:
        warnings.warn(f"No training data before {event_date}")
        return None

    train_end_date = train_end[-1]
    train_start_dt = train_end_date - pd.DateOffset(years=train_years)
    train_prices   = prices.loc[train_start_dt:train_end_date]

    if len(train_prices) < 50:
        warnings.warn(f"Too few training observations: {len(train_prices)}")
        return None

    train_ret = compute_log_returns(train_prices)
    # Demean using training mean only
    train_mean = float(train_ret.mean())
    train_ret_dm = train_ret - train_mean

    print(f"  Training: {train_ret.index[0].date()} to {train_ret.index[-1].date()}"
          f"  (n={len(train_ret)})")

    # Event window: from the event date forward
    event_prices = prices.loc[ed:]
    if len(event_prices) < 2:
        warnings.warn(f"No post-event data for {event_date}")
        return None

    event_ret   = compute_log_returns(event_prices)
    event_ret_dm = event_ret - train_mean

    # Combine for filtering (train + event)
    all_ret_dm = pd.concat([train_ret_dm, event_ret_dm])

    # Fit and filter each criterion
    fitted = {}
    lam_event = {}

    for cname, (driver, col) in CRITERIA.items():
        print(f"    Fitting {cname}...", end=" ", flush=True)
        try:
            params = fit_filter(train_ret_dm.values, driver=driver)
            fitted[cname] = params

            # Initial lambda from end of training
            lam_train = filter_path(train_ret_dm.values, params, driver)
            lam0      = float(lam_train[-1])

            # Filter through event window with frozen params
            lam_ev = filter_path(event_ret_dm.values, params, driver, lam0=lam0)
            lam_event[cname] = lam_ev[:-1]   # contemporaneous state
            print(f"omega={params['omega']:.3f} phi={params['phi']:.3f} alpha={params['alpha']:.4f}")
        except Exception as e:
            warnings.warn(f"  Filter {cname} failed: {e}")
            lam_event[cname] = None

    # ── Plot ──────────────────────────────────────────────────────────────────
    sns.set_theme(style="whitegrid", font_scale=1.0)
    fig, ax = plt.subplots(figsize=(11, 5))

    show_n = min(event_window_days, len(event_ret_dm))

    for cname, (driver, col) in CRITERIA.items():
        lam = lam_event[cname]
        if lam is None:
            continue
        vol = annualized_vol(lam[:show_n])
        vol_log10 = np.log10(np.maximum(vol, 1e-6))
        dates_ev = event_ret_dm.index[:show_n]
        ax.plot(dates_ev, vol_log10, color=col, lw=1.6, label=cname)

    ax.axvline(ed, color="red", ls="--", lw=1.2, alpha=0.8, label="Event date")
    ax.set_xlabel("Date")
    ax.set_ylabel("log₁₀(Annualised Volatility %)")
    ax.set_title(f"Event Study: {event_name}\nEvent date: {event_date}")
    ax.legend(fontsize=9)

    fig_path = os.path.join(RESULTS_DIR, f"exp8_{event_name.replace(' ', '_').lower()}.png")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=120)
    plt.close()
    print(f"  Saved: {fig_path}")

    return {"fitted": fitted, "lam_event": lam_event, "event_dates": event_ret_dm.index}


# ── Main ──────────────────────────────────────────────────────────────────────

def run():
    """Run both event studies."""

    # ── Event 1: Thai Baht Float (1997-07-02) ─────────────────────────────────
    print("\n--- Event 1: Thai Baht Float (1997-07-02) ---")
    thb = load_fred("DEXTHUS", "1990-01-01", "2001-01-01")

    if thb.empty:
        print("  FRED data unavailable; generating synthetic replacement.")
        rng   = np.random.default_rng(42)
        dates = pd.date_range("1990-01-01", "2001-01-01", freq="B")
        # Simulate a stable period then a jump
        p     = np.ones(len(dates)) * 25.0
        shock_idx = int(np.where(dates >= "1997-07-01")[0][0])
        p[shock_idx:] *= 1.5 + rng.random(len(dates) - shock_idx) * 0.3
        thb = pd.Series(p, index=dates, name="DEXTHUS").dropna()

    res_thb = run_event_study(
        "Thai Baht Float", thb, "1997-07-02",
        train_years=5, event_window_days=120,
    )

    # ── Event 2: S&P 500 2008 Episode (2008-09-29) ────────────────────────────
    print("\n--- Event 2: S&P 500 2008 Episode (2008-09-29) ---")
    spx = load_yahoo("^GSPC", "2001-01-01", "2012-01-01")

    if spx.empty:
        print("  Yahoo Finance unavailable; generating synthetic replacement.")
        rng   = np.random.default_rng(99)
        dates = pd.date_range("2001-01-01", "2012-01-01", freq="B")
        log_ret = rng.normal(0, 1.0, len(dates))
        shock_idx = int(np.where(dates >= "2008-09-01")[0][0])
        log_ret[shock_idx:shock_idx + 40] *= 5.0
        prices = 1000.0 * np.exp(np.cumsum(log_ret) / 100)
        spx = pd.Series(prices, index=dates, name="^GSPC")

    res_spx = run_event_study(
        "SP500 2008 Episode", spx, "2008-09-29",
        train_years=5, event_window_days=120,
    )

    print("\n✓ Event studies complete.  Figures saved to results/.")
    return res_thb, res_spx


if __name__ == "__main__":
    print("=" * 60)
    print("Experiment 8: Descriptive Event-Path Robustness Studies")
    print("=" * 60)
    run()
