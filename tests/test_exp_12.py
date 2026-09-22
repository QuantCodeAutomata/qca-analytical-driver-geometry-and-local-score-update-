"""
Tests for Experiment 12: Sensitivity Grids.

Uses synthetic data so no network calls are required.
"""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.score_drivers import u_log, u_DP, u_CRPS, u_MMD, u_student
from exp.exp_12_sensitivity_grids import (
    _synthetic_garch, qlike, var_coverage, run_sdf, split,
    grid_b, grid_c, grid_d,
)


# ---------------------------------------------------------------------------
# Score driver sanity checks
# ---------------------------------------------------------------------------

class TestScoreDrivers:
    def test_u_log_at_one(self):
        assert abs(u_log(1.0)) < 1e-10

    def test_u_log_positive_for_large_z(self):
        assert u_log(3.0) > 0

    def test_u_DP_mean_approx_zero(self):
        rng = np.random.default_rng(0)
        z = rng.standard_normal(100_000)
        assert abs(np.mean(u_DP(z))) < 0.01

    def test_u_CRPS_bounded(self):
        z = np.linspace(-10, 10, 1000)
        vals = u_CRPS(z)
        # CRPS driver is bounded above by 1/sqrt(pi) and below by -sqrt(2/pi)
        assert np.all(vals <= 1.0 / np.sqrt(np.pi) + 1e-12)

    def test_u_MMD_returns_array(self):
        z = np.array([0.0, 1.0, -2.0])
        out = u_MMD(z)
        assert out.shape == (3,)

    def test_u_student_approaches_u_log_for_large_nu(self):
        z = np.linspace(-3, 3, 50)
        assert np.allclose(u_student(z, nu=1e6), u_log(z), atol=1e-4)


# ---------------------------------------------------------------------------
# GARCH fit test (arch package)
# ---------------------------------------------------------------------------

class TestGARCHFit:
    def setup_method(self):
        self.y = _synthetic_garch(600, rng=np.random.default_rng(42))

    def test_arch_model_normal_fits(self):
        from arch import arch_model
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            am = arch_model(self.y, vol='GARCH', p=1, q=1, dist='normal',
                            rescale=False)
            res = am.fit(disp='off')
        assert res is not None
        params = res.params
        assert set(['mu', 'omega', 'alpha[1]', 'beta[1]']).issubset(params.index)
        assert params['alpha[1]'] >= 0
        assert params['beta[1]'] >= 0

    def test_arch_model_student_fits(self):
        from arch import arch_model
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            am = arch_model(self.y, vol='GARCH', p=1, q=1, dist='t',
                            rescale=False)
            res = am.fit(disp='off')
        assert 'nu' in res.params.index
        nu = res.params['nu']
        assert nu > 2.0, 'Estimated nu must be > 2 for finite variance'

    def test_garch_conditional_vol_positive(self):
        from arch import arch_model
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            am = arch_model(self.y, vol='GARCH', p=1, q=1, dist='normal',
                            rescale=False)
            res = am.fit(disp='off')
        assert np.all(res.conditional_volatility > 0)


# ---------------------------------------------------------------------------
# Metric tests
# ---------------------------------------------------------------------------

class TestMetrics:
    def test_qlike_is_positive(self):
        rng = np.random.default_rng(1)
        r = rng.standard_normal(500)
        sigma2 = np.abs(rng.standard_normal(500)) + 0.1
        assert qlike(sigma2, r) > 0

    def test_var_coverage_in_unit_interval(self):
        rng = np.random.default_rng(2)
        r = rng.standard_normal(1000)
        sigma = np.ones(1000)
        cov = var_coverage(sigma, r)
        assert 0.0 <= cov <= 1.0

    def test_var_coverage_near_alpha_for_normal(self):
        rng = np.random.default_rng(3)
        r = rng.standard_normal(10000)
        sigma = np.ones(10000)
        cov = var_coverage(sigma, r, alpha=0.05)
        assert abs(cov - 0.05) < 0.02


# ---------------------------------------------------------------------------
# Score-driven filter tests
# ---------------------------------------------------------------------------

class TestScoreDrivenFilter:
    def setup_method(self):
        self.y = _synthetic_garch(500, rng=np.random.default_rng(7))

    def test_sdf_log_returns_positive_variance(self):
        sv = run_sdf(self.y, 'log')
        assert np.all(sv > 0)
        assert len(sv) == len(self.y)

    def test_sdf_DP_returns_positive_variance(self):
        sv = run_sdf(self.y, 'DP')
        assert np.all(sv > 0)

    def test_sdf_CRPS_returns_positive_variance(self):
        sv = run_sdf(self.y, 'CRPS')
        assert np.all(sv > 0)

    def test_sdf_MMD_multiplier_scales_variance(self):
        sv1 = run_sdf(self.y, 'MMD', mmd_mult=1.0)
        sv2 = run_sdf(self.y, 'MMD', mmd_mult=2.0)
        # Different multipliers should give different paths
        assert not np.allclose(sv1, sv2)

    def test_sdf_student_varies_with_nu(self):
        sv4 = run_sdf(self.y, 'student', nu=4.0)
        sv10 = run_sdf(self.y, 'student', nu=10.0)
        assert not np.allclose(sv4, sv10)


# ---------------------------------------------------------------------------
# Grid dimension tests
# ---------------------------------------------------------------------------

class TestGridDimensions:
    def setup_method(self):
        rng = np.random.default_rng(99)
        self.returns = {
            '^GSPC': _synthetic_garch(500, rng=rng),
            '^FTSE': _synthetic_garch(500, rng=rng),
        }

    def test_grid_b_has_correct_rows(self):
        nu_list = [4, 6, 10]
        df = grid_b(self.returns, nu_list=nu_list)
        assert len(df) == len(self.returns) * len(nu_list)
        assert set(df['nu'].unique()) == set(nu_list)

    def test_grid_c_has_correct_rows(self):
        frac_list = [0.60, 0.70, 0.80]
        df = grid_c(self.returns, frac_list=frac_list)
        drivers = ['log', 'DP', 'CRPS', 'MMD']
        assert len(df) == len(self.returns) * len(frac_list) * len(drivers)

    def test_grid_d_has_correct_rows(self):
        mult_list = [0.5, 1.0, 2.0]
        df = grid_d(self.returns, mult_list=mult_list)
        assert len(df) == len(self.returns) * len(mult_list)

    def test_split_train_size(self):
        arr = np.arange(100)
        train, test = split(arr, 0.70)
        assert len(train) == 70
        assert len(test) == 30
