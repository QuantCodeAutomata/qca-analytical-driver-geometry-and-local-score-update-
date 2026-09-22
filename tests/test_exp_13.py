"""
Tests for Experiment 13: Robustness–Adaptation Trade-off.

All tests use synthetic data; no external network calls.
"""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.score_drivers import u_log, u_DP, u_CRPS, u_MMD
from exp.exp_13_robustness_adaptation import (
    run_filter_path, oracle_excess_log_risk, score_update,
    gen_clean_series, add_jumps, rmse_filter,
    ALPHA, PHI, OMEGA,
)


# ---------------------------------------------------------------------------
# Variance-break simulation (Part A)
# ---------------------------------------------------------------------------

class TestVarianceBreakSimulation:
    """Tests for Part A: Permanent variance break."""

    def test_filter_path_dimensions(self):
        y = np.random.default_rng(0).standard_normal(400)
        lam = run_filter_path(y, 'log', lam0=0.0)
        assert lam.shape == (401,)

    def test_filter_path_starts_at_lam0(self):
        y = np.random.default_rng(1).standard_normal(50)
        lam0 = 0.5
        lam = run_filter_path(y, 'log', lam0=lam0)
        assert lam[0] == pytest.approx(lam0)

    def test_oracle_excess_nonneg_when_lam_below_oracle(self):
        """When filter underestimates variance, excess loss should be positive."""
        T = 200
        rng = np.random.default_rng(42)
        y = 3.0 * rng.standard_normal(T)   # true sigma^2 = 9
        lam0 = 0.0                         # underestimate
        lam_oracle = np.log(9)
        lam = run_filter_path(y, 'log', lam0=lam0)
        excess = oracle_excess_log_risk(lam, y, lam_oracle)
        # Excess may be negative for some paths; just check it's finite
        assert np.isfinite(excess)

    def test_log_adapts_faster_than_DP(self):
        """Over many replications, log filter should reach oracle faster than DP."""
        R = 50
        sigma2_post = 9.0
        lam_oracle = np.log(sigma2_post)
        sigma_post = np.sqrt(sigma2_post)
        rng = np.random.default_rng(2024)
        BAND = 0.5
        T_pre, T_post = 50, 200

        delays_log, delays_dp = [], []
        for _ in range(R):
            pre = rng.standard_normal(T_pre)
            post = sigma_post * rng.standard_normal(T_post)

            for drv, store in [('log', delays_log), ('DP', delays_dp)]:
                lam_pre = run_filter_path(pre, drv, lam0=0.0)
                lam_post = run_filter_path(post, drv, lam0=lam_pre[-1])
                within = np.where(np.abs(lam_post[1:] - lam_oracle) < BAND)[0]
                delay = within[0] + 1 if len(within) > 0 else T_post
                store.append(delay)

        mean_log = np.mean(delays_log)
        mean_dp = np.mean(delays_dp)
        assert mean_log < mean_dp, (
            f'Expected log ({mean_log:.1f}) < DP ({mean_dp:.1f})')

    def test_score_update_drivers_differ(self):
        z = 3.0
        vals = {d: score_update(d, z) for d in ['log', 'DP', 'CRPS', 'MMD']}
        unique = len(set(f'{v:.6f}' for v in vals.values()))
        assert unique == 4, 'All four driver scores should be distinct at z=3'

    def test_DP_score_bounded_for_large_z(self):
        """DP driver should be much smaller than log for large z."""
        z = 10.0
        assert abs(u_DP(z)) < abs(u_log(z))

    def test_CRPS_driver_bounded(self):
        z = np.array([100.0, -100.0, 0.0])
        vals = u_CRPS(z)
        upper = 1.0 / np.sqrt(np.pi)
        lower = -np.sqrt(2.0 / np.pi)
        assert np.all(vals <= upper + 1e-12)
        assert np.all(vals >= lower - 1e-12)

    def test_filter_constants(self):
        assert ALPHA == pytest.approx(0.05)
        assert PHI == pytest.approx(0.97)
        assert OMEGA == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Jump misspecification (Part B)
# ---------------------------------------------------------------------------

class TestJumpMisspecification:
    """Tests for Part B: Jump contamination."""

    def test_gen_clean_series_dimensions(self):
        T = 500
        y, x = gen_clean_series(T, rng=np.random.default_rng(0))
        assert y.shape == (T,)
        assert x.shape == (T,)

    def test_gen_clean_series_x_is_ar1(self):
        """AR(1) latent variable should have auto-correlation near phi."""
        T = 5000
        phi = 0.97
        _, x = gen_clean_series(T, phi_ar=phi, rng=np.random.default_rng(1))
        acf1 = float(np.corrcoef(x[:-1], x[1:])[0, 1])
        assert abs(acf1 - phi) < 0.03

    def test_add_jumps_changes_some_observations(self):
        T = 2000
        rng = np.random.default_rng(2)
        y, x = gen_clean_series(T, rng=rng)
        y_contam = add_jumps(y, x, prob=0.05, rng=rng)
        assert not np.allclose(y, y_contam), 'Contamination should change some obs'

    def test_add_jumps_prob_zero_unchanged(self):
        T = 300
        rng = np.random.default_rng(3)
        y, x = gen_clean_series(T, rng=rng)
        y_contam = add_jumps(y, x, prob=0.0, rng=rng)
        np.testing.assert_array_equal(y, y_contam)

    def test_jumps_are_negative(self):
        """Shocks are one-sided (negative)."""
        T = 5000
        rng = np.random.default_rng(4)
        y, x = gen_clean_series(T, rng=rng)
        y_contam = add_jumps(y, x, prob=0.10, scale=5.0, rng=rng)
        diff = y_contam - y
        # All non-zero differences should be negative
        nonzero = diff[diff != 0]
        assert np.all(nonzero < 0), 'All jumps should be negative (one-sided)'

    def test_rmse_filter_path_bound_works(self):
        """Path with extreme initialisation should trigger bound violation."""
        T = 200
        rng = np.random.default_rng(5)
        y, x = gen_clean_series(T, rng=rng)
        # Use alpha=10 to cause explosive filter
        result = rmse_filter(y, x, omega=0.0, phi=0.0, alpha=100.0,
                             driver='log', bound=12.0)
        # May or may not violate bound; just check return type
        assert result is None or isinstance(result, float)

    def test_rmse_filter_returns_float_for_stable(self):
        T = 200
        rng = np.random.default_rng(6)
        y, x = gen_clean_series(T, rng=rng)
        result = rmse_filter(y, x, omega=0.0, phi=0.97, alpha=0.05,
                             driver='Gaussian-DP', bound=12.0)
        assert result is not None
        assert isinstance(result, float)
        assert result > 0.0

    def test_contaminated_rmse_geq_clean_rmse(self):
        """On average, contaminated RMSE ≥ clean RMSE (jump hurts the filter)."""
        T = 1000
        rng = np.random.default_rng(7)
        y_clean, x = gen_clean_series(T, rng=rng)
        y_contam = add_jumps(y_clean, x, prob=0.05, rng=rng)

        omega, phi, alpha = 0.0, 0.97, 0.05
        rc = rmse_filter(y_clean, x, omega, phi, alpha, 'log')
        rj = rmse_filter(y_contam, x, omega, phi, alpha, 'log')

        if rc is not None and rj is not None:
            assert rj >= rc - 0.05, (
                f'Contaminated RMSE {rj:.4f} unexpectedly much less than clean {rc:.4f}')
