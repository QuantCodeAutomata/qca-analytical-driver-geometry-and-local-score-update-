"""Tests for Experiments 8-11: Event Studies, Outlier Injection, Stress Tests, Factorial."""

import sys
import os
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.score_drivers import u_log, u_DP, u_CRPS, u_MMD, J_LOG, J_DP, J_CRPS, J_MMD


# ============================================================
# Shared filter infrastructure used in exps 8-11
# ============================================================

def _run_filter(returns: np.ndarray, u_fn, omega: float, phi: float,
                alpha: float, lam0: float = 0.0) -> np.ndarray:
    T   = len(returns)
    lam = np.empty(T + 1)
    lam[0] = lam0
    for t in range(T):
        z      = returns[t] * np.exp(-lam[t] / 2.0)
        lam[t + 1] = omega + phi * lam[t] + alpha * float(u_fn(np.array([z]))[0])
    return lam


# ============================================================
# Experiment 8: Event Studies
# ============================================================

class TestExp8:
    """Tests for event-path robustness helpers."""

    def test_log_returns_computation(self):
        """Close-to-close log returns in percent."""
        from exp.exp_8_event_studies import compute_log_returns
        prices = pd.Series([100.0, 101.0, 99.0, 101.0],
                            index=pd.date_range("2020-01-01", periods=4, freq="B"))
        ret = compute_log_returns(prices)
        assert len(ret) == 3
        assert abs(float(ret.iloc[0]) - np.log(101 / 100) * 100) < 1e-8

    def test_annualized_vol_positive(self):
        """Annualised volatility should be positive."""
        from exp.exp_8_event_studies import annualized_vol
        lam = np.array([-1.0, 0.0, 1.0])
        vol = annualized_vol(lam)
        assert np.all(vol > 0)

    def test_gaussian_log_larger_vol_after_jump(self):
        """After a large outlier, Gaussian-log vol should diverge more than DP."""
        rng  = np.random.default_rng(0)
        n    = 200
        ret  = rng.normal(0, 1.0, n)
        ret[100] = 20.0  # inject outlier

        omega, phi, alpha = 0.0, 0.97, 0.05
        lam_log = _run_filter(ret, u_log, omega, phi, alpha)
        lam_dp  = _run_filter(ret, u_DP,  omega, phi, alpha)

        # At t=101 and t=102, log filter should be more perturbed
        for t in [100, 101, 102]:
            diff_log = abs(lam_log[t + 1] - lam_log[99])
            diff_dp  = abs(lam_dp[t + 1]  - lam_dp[99])
            assert diff_log >= diff_dp, (
                f"t={t}: log diff={diff_log:.3f} should >= DP diff={diff_dp:.3f}"
            )

    def test_filter_persistence_after_outlier_decays(self):
        """After an outlier, the perturbation decays geometrically."""
        rng   = np.random.default_rng(1)
        n     = 100
        ret   = np.zeros(n)
        ret[10] = 10.0  # outlier at t=10

        omega, phi, alpha = 0.0, 0.97, 0.05
        lam = _run_filter(ret, u_DP, omega, phi, alpha)
        # From t=15 onward, differences should decay
        diffs = [abs(lam[t + 1] - lam[t]) for t in range(12, 20)]
        assert diffs[-1] < diffs[0], "Perturbation should decay after outlier"


# ============================================================
# Experiment 9: Outlier Injection
# ============================================================

class TestExp9:
    """Tests for controlled outlier injection."""

    def test_u_student_log_bounded(self):
        """Student-log update should be bounded: u in (-1, nu)."""
        from exp.exp_9_outlier_injection import u_student_log
        z = np.array([-100.0, -10.0, 0.0, 10.0, 100.0])
        u = u_student_log(z, nu=6.0)
        assert np.all(u > -1.0 - 1e-10), f"u_student_log not bounded below: min={u.min()}"
        assert np.all(u < 6.0 + 1e-10),  f"u_student_log not bounded above: max={u.max()}"

    def test_perturbation_grows_with_B(self):
        """Perturbation should grow with shock size B for all models."""
        from exp.exp_9_outlier_injection import run_filter, u_student_log
        rng   = np.random.default_rng(42)
        n_tr  = 200
        n_ev  = 50
        ret   = rng.normal(0, 1.0, n_tr + n_ev)
        omega, phi, alpha = 0.0, 0.97, 0.05
        inject_idx = 10

        for B1, B2 in [(1.0, 5.0), (5.0, 20.0)]:
            ret1 = ret[n_tr:].copy()
            ret2 = ret[n_tr:].copy()
            ret1[inject_idx] += B1
            ret2[inject_idx] += B2

            for u_fn in [u_log, u_student_log]:
                lam0 = 0.0
                lam_orig = run_filter(ret[n_tr:], omega, phi, alpha, u_fn, lam0=lam0)
                lam1     = run_filter(ret1,        omega, phi, alpha, u_fn, lam0=lam0)
                lam2     = run_filter(ret2,        omega, phi, alpha, u_fn, lam0=lam0)
                d1 = float(np.max(np.abs(lam1[:-1] - lam_orig[:-1])))
                d2 = float(np.max(np.abs(lam2[:-1] - lam_orig[:-1])))
                assert d2 >= d1, f"Peak perturbation should grow with B"

    def test_gaussian_log_more_sensitive_than_student(self):
        """Gaussian-log perturbation >> Student-log at large B."""
        from exp.exp_9_outlier_injection import run_filter, u_student_log
        rng   = np.random.default_rng(7)
        n     = 100
        ret   = rng.normal(0, 1.0, n)
        omega, phi, alpha = 0.0, 0.97, 0.05
        lam0  = 0.0

        perturbed = ret.copy()
        perturbed[50] += 30.0  # B=30

        lam_log_clean = run_filter(ret, omega, phi, alpha, u_log, lam0=lam0)
        lam_log_pert  = run_filter(perturbed, omega, phi, alpha, u_log, lam0=lam0)

        lam_stu_clean = run_filter(ret, omega, phi, alpha, u_student_log, lam0=lam0)
        lam_stu_pert  = run_filter(perturbed, omega, phi, alpha, u_student_log, lam0=lam0)

        peak_log = float(np.max(np.abs(lam_log_pert[:-1] - lam_log_clean[:-1])))
        peak_stu = float(np.max(np.abs(lam_stu_pert[:-1] - lam_stu_clean[:-1])))
        assert peak_log > peak_stu * 2, (
            f"Gaussian-log peak={peak_log:.3f} should >> Student-log peak={peak_stu:.3f}"
        )

    def test_sigma_ref_positive(self):
        """sigma_ref computed from pre-shock window is positive."""
        rng = np.random.default_rng(99)
        pre_inject = rng.normal(0, 1.0, 60)
        sigma_ref  = float(np.sqrt(np.mean(pre_inject**2)))
        assert sigma_ref > 0


# ============================================================
# Experiment 10: Heavy-Tail Stress Test
# ============================================================

class TestExp10:
    """Tests for heavy-tail stress test."""

    def test_innovation_scale_gaussian(self):
        """For Gaussian, scale should give MAD=MAD_TARGET."""
        from exp.exp_10_heavy_tail_stress import innovation_scale_for_mad, MAD_TARGET
        scale  = innovation_scale_for_mad(np.inf)
        from scipy.stats import norm
        pop_mad = scale * float(norm.ppf(0.75))
        assert abs(pop_mad - MAD_TARGET) < 1e-8, (
            f"Gaussian MAD={pop_mad:.6f} != target {MAD_TARGET}"
        )

    def test_innovation_scale_cauchy(self):
        """For Cauchy (nu=1), scale gives MAD=MAD_TARGET."""
        from exp.exp_10_heavy_tail_stress import innovation_scale_for_mad, MAD_TARGET
        from scipy.stats import t
        scale   = innovation_scale_for_mad(1.0)
        pop_mad = scale * float(t.ppf(0.75, df=1.0))
        assert abs(pop_mad - MAD_TARGET) < 1e-8

    def test_tracking_correlation_range(self):
        """Tracking correlation should be in [-1, 1]."""
        from exp.exp_10_heavy_tail_stress import tracking_correlation
        rng   = np.random.default_rng(0)
        lam   = rng.normal(0, 1, 100)
        x_true = rng.normal(0, 1, 100)
        corr  = tracking_correlation(lam, x_true)
        assert -1.0 <= corr <= 1.0

    def test_admissibility_gate_rejects_large_values(self):
        """Inadmissible if lambda path contains large values."""
        from exp.exp_10_heavy_tail_stress import is_admissible
        lam_ok  = np.linspace(-5, 5, 100)
        lam_bad = np.array([0.0] * 50 + [100.0])
        assert is_admissible(lam_ok)
        assert not is_admissible(lam_bad)

    def test_admissibility_gate_rejects_nan(self):
        """Inadmissible if lambda path contains NaN."""
        from exp.exp_10_heavy_tail_stress import is_admissible
        lam_nan = np.array([0.0, 1.0, float("nan")])
        assert not is_admissible(lam_nan)

    def test_generate_innovations_shape(self):
        """generate_innovations returns array of correct length."""
        from exp.exp_10_heavy_tail_stress import generate_innovations
        rng = np.random.default_rng(42)
        eps = generate_innovations(200, 5.0, rng)
        assert len(eps) == 200

    def test_generate_innovations_mad_close(self):
        """Empirical MAD of large sample is close to target."""
        from exp.exp_10_heavy_tail_stress import generate_innovations, MAD_TARGET
        rng = np.random.default_rng(0)
        eps = generate_innovations(100_000, 3.0, rng)
        mad = float(np.median(np.abs(eps)))
        assert abs(mad - MAD_TARGET) < 0.05, f"Empirical MAD={mad:.4f}, target={MAD_TARGET}"


# ============================================================
# Experiment 11: Empirical Factorial
# ============================================================

class TestExp11:
    """Tests for empirical factorial helpers."""

    def test_qlike_definition(self):
        """QLIKE = E[log(v) + q/v] with q=v gives log(v)+1."""
        from exp.exp_11_empirical_factorial import qlike
        v = np.array([1.0, 2.0, 4.0])
        q = v.copy()
        ql = qlike(v, q)
        expected = np.mean(np.log(v) + 1.0)
        assert abs(ql - expected) < 1e-10

    def test_qlike_larger_for_worse_forecast(self):
        """Worse forecast (v further from q) gives larger QLIKE."""
        from exp.exp_11_empirical_factorial import qlike
        q  = np.ones(100)
        v1 = np.ones(100)          # perfect forecast
        v2 = np.ones(100) * 4.0   # over-estimate
        assert qlike(v1, q) < qlike(v2, q)

    def test_var_coverage_target(self):
        """1% VaR coverage should be near 0.01 for well-calibrated model."""
        from exp.exp_11_empirical_factorial import var_coverage
        rng   = np.random.default_rng(5)
        n     = 50_000
        v     = np.ones(n)          # unit variance
        y     = rng.normal(0, 1, n)
        cov   = var_coverage(v, y, 0.01, is_student=False)
        assert abs(cov - 0.01) < 0.005, f"Coverage={cov:.4f}, expected near 0.01"

    def test_pit_uniform_under_correct_model(self):
        """PIT values should be near-uniform under the correct Gaussian model."""
        from exp.exp_11_empirical_factorial import pit_values
        rng = np.random.default_rng(9)
        n   = 10_000
        v   = np.ones(n)
        y   = rng.normal(0, 1, n)
        pit = pit_values(v, y, is_student=False)
        assert pit.min() >= 0.0 and pit.max() <= 1.0
        # KS statistic should be small
        from scipy.stats import kstest
        ks_stat, _ = kstest(pit, "uniform")
        assert ks_stat < 0.05, f"PIT KS statistic = {ks_stat:.4f}, expected < 0.05"

    def test_mmd_bandwidth_positive(self):
        """MMD training bandwidth selection returns a positive number."""
        from exp.exp_11_empirical_factorial import select_mmd_bandwidth
        rng = np.random.default_rng(0)
        ret = rng.normal(0, 1, 500)
        bw  = select_mmd_bandwidth(ret)
        assert bw > 0

    def test_compute_returns_and_proxies(self):
        """compute_returns_and_proxies returns correct columns and length."""
        from exp.exp_11_empirical_factorial import compute_returns_and_proxies, make_synthetic_index
        df_raw = make_synthetic_index(n=200, seed=0)
        data   = compute_returns_and_proxies(df_raw)
        assert set(["ret", "ret2", "PK", "GK"]).issubset(data.columns)
        assert len(data) == 199  # one fewer than prices

    def test_ret2_equals_ret_squared(self):
        """ret2 column should equal ret squared."""
        from exp.exp_11_empirical_factorial import compute_returns_and_proxies, make_synthetic_index
        df_raw = make_synthetic_index(n=100, seed=1)
        data   = compute_returns_and_proxies(df_raw)
        np.testing.assert_allclose(data["ret2"].values, data["ret"].values**2)

    def test_pk_nonnegative(self):
        """Parkinson estimator should be non-negative."""
        from exp.exp_11_empirical_factorial import compute_returns_and_proxies, make_synthetic_index
        df_raw = make_synthetic_index(n=200, seed=2)
        data   = compute_returns_and_proxies(df_raw)
        assert np.all(data["PK"].values >= 0)

    def test_student_var_coverage_closer_to_nominal(self):
        """Student VaR coverage should be closer to 0.01 than Gaussian for heavy-tailed data."""
        from exp.exp_11_empirical_factorial import var_coverage
        from scipy.stats import t
        rng   = np.random.default_rng(77)
        n     = 100_000
        nu    = 6.0
        v     = np.ones(n)
        y     = t.rvs(df=nu, size=n, random_state=int(rng.integers(1000))) * np.sqrt((nu - 2) / nu)

        cov_gauss   = var_coverage(v, y, 0.01, is_student=False)
        cov_student = var_coverage(v, y, 0.01, is_student=True, nu=nu)
        # Student coverage should be closer to 1%
        assert abs(cov_student - 0.01) < abs(cov_gauss - 0.01), (
            f"Student={cov_student:.4f} should be closer to 0.01 than Gauss={cov_gauss:.4f}"
        )
