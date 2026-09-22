"""
Experiment 11: Empirical Density-by-Criterion Factorial Across 12 Equity Indices.

8-cell factorial: {Gaussian, Student-t6} x {Log, Density-Power, CRPS, MMD}.
Evaluation metrics: QLIKE, VaR coverage (1% and 5%), PIT diagnostics.
70/30 train/test split per market.

Indices: ^GSPC, ^IXIC, ^DJI, ^FTSE, ^GDAXI, ^FCHI, ^STOXX50E, ^N225,
          ^HSI, ^GSPTSE, ^AXJO, ^BVSP

Note: In test mode (--test flag), runs on ^GSPC and ^FTSE only with 2 starts.
"""

import os
import sys
import warnings
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.score_drivers import (
    u_log, u_DP, u_CRPS, u_MMD,
    J_LOG, J_DP, J_CRPS, J_MMD,
    _b, _c_b, _kappa_b,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

ALL_TICKERS = [
    "^GSPC", "^IXIC", "^DJI", "^FTSE", "^GDAXI", "^FCHI",
    "^STOXX50E", "^N225", "^HSI", "^GSPTSE", "^AXJO", "^BVSP",
]
NU_STUDENT = 6.0


# ── Student-t6 score driver ──────────────────────────────────────────────────

def u_student_log(z: np.ndarray, nu: float = NU_STUDENT) -> np.ndarray:
    """Student-t log score update."""
    return (nu + 1.0) * z**2 / (nu + z**2) - 1.0


def u_student_DP(z: np.ndarray, nu: float = NU_STUDENT, beta: float = 0.15) -> np.ndarray:
    """Student density-power update (approximation under Student working density)."""
    w = (nu / (nu + z**2)) ** ((nu + 1.0) / 2.0)
    base = (1 + beta) * _c_b * np.exp(-beta * z**2 / 2.0) * (z**2 - 1.0) + _kappa_b
    return w * base + (1.0 - w) * _kappa_b


def u_student_MMD(z: np.ndarray, nu: float = NU_STUDENT) -> np.ndarray:
    """Student MMD update (approximation)."""
    w = (nu / (nu + z**2)) ** ((nu + 1.0) / 2.0)
    return w * u_MMD(z)


# 8 model cells: (density, criterion) -> update function
CELLS = {
    "Gauss-Log":       u_log,
    "Gauss-DP":        u_DP,
    "Gauss-CRPS":      u_CRPS,
    "Gauss-MMD":       u_MMD,
    "Student-Log":     u_student_log,
    "Student-DP":      u_student_DP,
    "Student-CRPS":    u_CRPS,     # CRPS score same regardless of density (approximation)
    "Student-MMD":     u_student_MMD,
}
IS_STUDENT = {k: k.startswith("Student") for k in CELLS}
CELL_COLORS = {
    "Gauss-Log":    "C0", "Gauss-DP":    "C1", "Gauss-CRPS":   "C2", "Gauss-MMD":   "C3",
    "Student-Log":  "C4", "Student-DP":  "C5", "Student-CRPS": "C6", "Student-MMD":  "C7",
}


# ── Data loading ──────────────────────────────────────────────────────────────

def load_index(ticker: str, start: str = "2000-01-01",
               end: str = "2024-01-01") -> pd.DataFrame:
    """Load OHLCV from Yahoo Finance. Returns DataFrame with OHLC columns."""
    try:
        import yfinance as yf
        df = yf.download(ticker, start=start, end=end,
                         auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.dropna(subset=["Close"])
        df.index.name = "Date"
        return df
    except Exception as e:
        warnings.warn(f"Failed to download {ticker}: {e}")
        return pd.DataFrame()


def make_synthetic_index(n: int = 5000, seed: int = 0) -> pd.DataFrame:
    """Synthetic OHLC data for testing."""
    rng = np.random.default_rng(seed)
    ret = rng.normal(0.0, 1.0, n)
    close = 1000.0 * np.exp(np.cumsum(ret) / 100.0)
    hi  = close * (1.0 + 0.003 * np.abs(rng.normal(0, 1, n)))
    lo  = close * (1.0 - 0.003 * np.abs(rng.normal(0, 1, n)))
    dates = pd.date_range("2000-01-01", periods=n, freq="B")
    return pd.DataFrame({"Open": close, "High": hi, "Low": lo, "Close": close},
                         index=dates)


# ── Return and proxy computation ───────────────────────────────────────────────

def compute_returns_and_proxies(df: pd.DataFrame) -> pd.DataFrame:
    """Compute log returns (%) and range-based volatility proxies."""
    close = df["Close"]
    ret   = np.log(close).diff() * 100.0

    # Parkinson estimator (if High/Low available)
    if "High" in df.columns and "Low" in df.columns:
        pk = (np.log(df["High"] / df["Low"]))**2 / (4.0 * np.log(2.0))
    else:
        pk = ret**2

    # Garman-Klass estimator (if Open/High/Low/Close available)
    if all(c in df.columns for c in ["Open", "High", "Low", "Close"]):
        co = np.log(close / df["Open"])
        hl = np.log(df["High"] / df["Low"])
        gk = 0.5 * hl**2 - (2.0 * np.log(2.0) - 1.0) * co**2
        gk = gk.clip(lower=0)
    else:
        gk = ret**2

    out = pd.DataFrame({
        "ret":  ret,
        "ret2": ret**2,
        "PK":   pk,
        "GK":   gk,
    }).dropna()
    return out


# ── MMD bandwidth selection ───────────────────────────────────────────────────

def select_mmd_bandwidth(train_ret: np.ndarray, cap: int = 1024) -> float:
    """Training-sample median absolute distance among returns (capped at 1024)."""
    n_sub = min(len(train_ret), cap)
    step  = max(1, len(train_ret) // n_sub)
    sub   = train_ret[::step][:n_sub]
    diffs = np.abs(sub[:, None] - sub[None, :])
    return float(np.median(diffs[diffs > 0]))


# ── MMD score with variable bandwidth ─────────────────────────────────────────

def u_MMD_bw(z: np.ndarray, ell: float) -> np.ndarray:
    """MMD driver with arbitrary bandwidth ell."""
    r_lam = ell   # at lambda=0
    D_lam = r_lam**2 + 1.0
    A = 2.0 * (r_lam / np.sqrt(D_lam))
    B = 2.0 * r_lam / (r_lam**2 + 2.0)**1.5
    return A * np.exp(-z**2 / (2.0 * D_lam)) * (z**2 / D_lam - 1.0) + B


# ── Filter ────────────────────────────────────────────────────────────────────

def run_filter(returns: np.ndarray, u_fn: callable,
               omega: float, phi: float, alpha: float,
               lam0: float = 0.0) -> np.ndarray:
    """Score-driven filter recursion. Returns lambda path (T+1,)."""
    T   = len(returns)
    lam = np.empty(T + 1)
    lam[0] = lam0
    for t in range(T):
        z       = returns[t] * np.exp(-lam[t] / 2.0)
        lam[t + 1] = omega + phi * lam[t] + alpha * float(u_fn(np.array([z]))[0])
    return lam


def fit_filter(returns: np.ndarray, u_fn: callable,
               n_starts: int = 8, seed: int = 0) -> dict:
    """Fit filter parameters via Gaussian log-likelihood, multi-start."""
    rng     = np.random.default_rng(seed)
    lam_var = float(np.log(np.var(returns) + 1e-8))

    def neg_ll(params):
        omega, logit_phi, log_alpha = params
        phi   = 0.999 / (1.0 + np.exp(-logit_phi))
        alpha = np.exp(log_alpha)
        lam   = run_filter(returns, u_fn, omega, phi, alpha, lam0=lam_var)
        ll    = -0.5 * (lam[:-1] + np.exp(-lam[:-1]) * returns**2)
        if not np.all(np.isfinite(ll)):
            return 1e10
        return float(-np.sum(ll))

    best_val = np.inf
    best_params = {"omega": lam_var * 0.03, "phi": 0.97, "alpha": 0.05}

    phi_starts = rng.uniform(0.85, 0.999, n_starts)
    alpha_starts = np.exp(rng.uniform(-4, 0, n_starts))

    for phi_i, alpha_i in zip(phi_starts, alpha_starts):
        logit_phi = np.log(phi_i / (0.999 - phi_i))
        x0 = [lam_var * (1.0 - phi_i), logit_phi, np.log(alpha_i)]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                res = minimize(neg_ll, x0, method="L-BFGS-B",
                               bounds=[(-15, 15), (-8, 8), (-8, 2)],
                               options={"maxiter": 500, "ftol": 1e-10})
                if res.fun < best_val and np.all(np.isfinite(res.x)):
                    best_val = res.fun
                    best_params = {
                        "omega":  res.x[0],
                        "phi":    0.999 / (1.0 + np.exp(-res.x[1])),
                        "alpha":  np.exp(res.x[2]),
                    }
            except Exception:
                pass

    return best_params


# ── Metrics ───────────────────────────────────────────────────────────────────

def qlike(v: np.ndarray, q: np.ndarray) -> float:
    """Forecast-dependent QLIKE: E[log(v) + q/v]."""
    mask = np.isfinite(v) & np.isfinite(q) & (v > 0)
    return float(np.mean(np.log(v[mask]) + q[mask] / v[mask]))


def var_coverage(v: np.ndarray, y: np.ndarray, p: float,
                 is_student: bool = False, nu: float = NU_STUDENT) -> float:
    """VaR coverage rate at level p.

    VaR_t(p) = q_p * sqrt(v_t) where q_p is the quantile of the density.
    """
    if is_student:
        q_p = np.sqrt((nu - 2.0) / nu) * stats.t.ppf(p, df=nu)
    else:
        q_p = stats.norm.ppf(p)
    var_t = q_p * np.sqrt(v)
    return float(np.mean(y < var_t))


def pit_values(v: np.ndarray, y: np.ndarray,
               is_student: bool = False, nu: float = NU_STUDENT) -> np.ndarray:
    """PIT (probability integral transform) values."""
    sigma = np.sqrt(v)
    if is_student:
        return stats.t.cdf(y / sigma, df=nu)
    else:
        return stats.norm.cdf(y / sigma)


# ── Market analysis ────────────────────────────────────────────────────────────

def analyze_market(ticker: str, n_starts: int = 4) -> pd.DataFrame:
    """Fit and evaluate all 8 cells for one market. Returns metrics DataFrame."""
    print(f"  {ticker}", end="", flush=True)

    df_raw = load_index(ticker)
    if df_raw.empty or len(df_raw) < 100:
        print(" [skipped: no data]")
        return pd.DataFrame()

    data   = compute_returns_and_proxies(df_raw)
    n      = len(data)
    n_tr   = int(0.70 * n)

    ret_all = data["ret"].values
    train_ret = ret_all[:n_tr]
    test_ret  = ret_all[n_tr:]

    train_mean   = float(np.mean(train_ret))
    train_ret_dm = train_ret - train_mean
    test_ret_dm  = test_ret  - train_mean

    # Proxies on test
    q2_test = data["ret2"].values[n_tr:]
    pk_test = data["PK"].values[n_tr:]
    gk_test = data["GK"].values[n_tr:]

    # MMD bandwidth from training
    ell_train = select_mmd_bandwidth(train_ret_dm)

    # Build effective u_fn for MMD cells with training bandwidth
    def u_mmd_fit(z):
        return u_MMD_bw(z, ell_train)

    def u_student_mmd_fit(z):
        nu = NU_STUDENT
        w  = (nu / (nu + z**2)) ** ((nu + 1.0) / 2.0)
        return w * u_mmd_fit(z)

    cells_market = {
        "Gauss-Log":    u_log,
        "Gauss-DP":     u_DP,
        "Gauss-CRPS":   u_CRPS,
        "Gauss-MMD":    u_mmd_fit,
        "Student-Log":  u_student_log,
        "Student-DP":   u_student_DP,
        "Student-CRPS": u_CRPS,
        "Student-MMD":  u_student_mmd_fit,
    }

    rows = []
    for cname, u_fn in cells_market.items():
        print(f" {cname}", end="", flush=True)
        try:
            params = fit_filter(train_ret_dm, u_fn, n_starts=n_starts, seed=42)

            # Filter training to get lam0
            lam_tr  = run_filter(train_ret_dm, u_fn,
                                  params["omega"], params["phi"], params["alpha"],
                                  lam0=np.log(np.var(train_ret_dm) + 1e-8))
            lam0_te = float(lam_tr[-1])

            # Filter test
            lam_te  = run_filter(test_ret_dm, u_fn,
                                  params["omega"], params["phi"], params["alpha"],
                                  lam0=lam0_te)[:-1]

            if not np.all(np.isfinite(lam_te)):
                raise ValueError("Non-finite lambda on test")

            v_te = np.exp(lam_te)    # conditional variance

            # QLIKE with three proxies
            ql_r2 = qlike(v_te, q2_test[:len(v_te)])
            ql_pk = qlike(v_te, pk_test[:len(v_te)])
            ql_gk = qlike(v_te, gk_test[:len(v_te)])

            # VaR coverage
            is_stu = IS_STUDENT[cname]
            cov1   = var_coverage(v_te, test_ret_dm[:len(v_te)], 0.01, is_stu)
            cov5   = var_coverage(v_te, test_ret_dm[:len(v_te)], 0.05, is_stu)

            # PIT
            pit = pit_values(v_te, test_ret_dm[:len(v_te)], is_stu)
            pit_mean = float(np.mean(pit))
            pit_std  = float(np.std(pit))

            rows.append({
                "Ticker":    ticker,
                "Cell":      cname,
                "IsStudent": is_stu,
                "omega":     params["omega"],
                "phi":       params["phi"],
                "alpha":     params["alpha"],
                "QLIKE_r2":  ql_r2,
                "QLIKE_PK":  ql_pk,
                "QLIKE_GK":  ql_gk,
                "Cov1pct":   cov1,
                "Cov5pct":   cov5,
                "PIT_mean":  pit_mean,
                "PIT_std":   pit_std,
                "n_train":   n_tr,
                "n_test":    len(test_ret_dm),
            })

        except Exception as e:
            warnings.warn(f"  {ticker}/{cname}: {e}")
            rows.append({
                "Ticker":    ticker,
                "Cell":      cname,
                "IsStudent": IS_STUDENT[cname],
                "QLIKE_r2":  float("nan"),
                "QLIKE_PK":  float("nan"),
                "QLIKE_GK":  float("nan"),
                "Cov1pct":   float("nan"),
                "Cov5pct":   float("nan"),
                "PIT_mean":  float("nan"),
                "PIT_std":   float("nan"),
                "n_train":   0,
                "n_test":    0,
            })

    print()
    return pd.DataFrame(rows)


# ── Main ──────────────────────────────────────────────────────────────────────

def run(test_mode: bool = False, n_starts: int | None = None):
    """Run the empirical factorial.

    Parameters
    ----------
    test_mode : if True, use only 2 tickers and 2 starts.
    """
    if test_mode:
        tickers = ["^GSPC", "^FTSE"]
        starts  = 2
        print("TEST MODE: 2 tickers, 2 starts")
    else:
        tickers = ALL_TICKERS
        starts  = n_starts or 4

    all_dfs = []
    for ticker in tickers:
        df_m = analyze_market(ticker, n_starts=starts)
        if not df_m.empty:
            all_dfs.append(df_m)

    if not all_dfs:
        print("No results!")
        return pd.DataFrame()

    df = pd.concat(all_dfs, ignore_index=True)

    # ── Panel summary ──────────────────────────────────────────────────────────
    metrics = ["QLIKE_r2", "QLIKE_PK", "QLIKE_GK", "Cov1pct", "Cov5pct"]
    panel   = df.groupby("Cell")[metrics].mean()
    print("\nPanel average metrics:")
    print(panel.to_string())

    # ── VaR coverage table ─────────────────────────────────────────────────────
    cov_tbl = df.pivot_table(values="Cov1pct", index="Ticker", columns="Cell")
    print("\n1% VaR Coverage rates:")
    print(cov_tbl.round(4).to_string())

    # ── Plot: QLIKE heatmap ────────────────────────────────────────────────────
    sns.set_theme(style="whitegrid", font_scale=0.9)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for i, metric in enumerate(["QLIKE_r2", "QLIKE_PK", "QLIKE_GK"]):
        piv = df.pivot_table(values=metric, index="Ticker", columns="Cell")
        # Rank within each market (lower QLIKE = better)
        ranks = piv.rank(axis=1)
        sns.heatmap(ranks, ax=axes[i], cmap="RdYlGn_r", annot=piv.round(2),
                    fmt=".2f", linewidths=0.3, cbar_kws={"label": "Rank"})
        axes[i].set_title(f"QLIKE ({metric.split('_')[1]})\nValue shown, color=rank")
        axes[i].set_xlabel("")
        axes[i].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    fig_path = os.path.join(RESULTS_DIR, "exp11_qlike_heatmap.png")
    plt.savefig(fig_path, dpi=100)
    plt.close()
    print(f"Saved: {fig_path}")

    # ── Plot: VaR coverage ─────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 6))
    cells_ordered = list(CELLS.keys())
    x = np.arange(len(tickers))
    width = 0.1
    for j, cell in enumerate(cells_ordered[:min(8, len(cells_ordered))]):
        cov_vals = []
        for tk in tickers:
            row = df[(df["Ticker"] == tk) & (df["Cell"] == cell)]
            if len(row) > 0:
                cov_vals.append(float(row["Cov1pct"].values[0]))
            else:
                cov_vals.append(float("nan"))
        offset = (j - 3.5) * width
        ax.bar(x + offset, cov_vals, width, label=cell, color=list(CELL_COLORS.values())[j], alpha=0.7)

    ax.axhline(0.01, color="red", ls="--", lw=1.2, label="Nominal 1%")
    ax.set_xticks(x)
    ax.set_xticklabels(tickers, rotation=45, ha="right")
    ax.set_ylabel("Empirical 1% VaR coverage rate")
    ax.set_title("1% VaR Coverage Across Markets and Cells\n(target = 0.01)")
    ax.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    fig2_path = os.path.join(RESULTS_DIR, "exp11_var_coverage.png")
    plt.savefig(fig2_path, dpi=100)
    plt.close()
    print(f"Saved: {fig2_path}")

    # ── Save CSVs ──────────────────────────────────────────────────────────────
    csv_path = os.path.join(RESULTS_DIR, "exp11_results.csv")
    df.to_csv(csv_path, index=False, float_format="%.6f")
    panel_csv = os.path.join(RESULTS_DIR, "exp11_panel_summary.csv")
    panel.to_csv(panel_csv, float_format="%.6f")
    print(f"Saved: {csv_path}")

    # ── Validation ────────────────────────────────────────────────────────────
    # Student cells should have better VaR coverage (closer to 1%)
    stu_cov = df[df["IsStudent"] == True]["Cov1pct"].dropna()
    gas_cov = df[df["IsStudent"] == False]["Cov1pct"].dropna()
    if len(stu_cov) > 0 and len(gas_cov) > 0:
        print(f"\nStudent 1% coverage: {stu_cov.mean():.4f}  "
              f"Gaussian: {gas_cov.mean():.4f}  (Student should be closer to 0.01)")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Run in test mode (2 tickers)")
    parser.add_argument("--n-starts", type=int, default=None, help="Number of multistart runs")
    args = parser.parse_args()

    print("=" * 60)
    print("Experiment 11: Empirical Density-by-Criterion Factorial")
    print("=" * 60)
    df = run(test_mode=args.test, n_starts=args.n_starts)
