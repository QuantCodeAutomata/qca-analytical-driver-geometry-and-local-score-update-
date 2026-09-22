"""
Tests for Experiment 14: Gaussian Location Filtering on the Nile River.

Tests cover score boundedness, filter mechanics, and outlier sensitivity ordering.
"""

import sys
import os
import numpy as np
import pytest
from scipy.special import ndtr as Phi

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.score_drivers import u_log_loc, u_CRPS_loc, u_DP_loc
from exp.exp_14_nile_location import (
    u_log_z, u_CRPS_z, u_DP_z, make_filter_runner,
    neg_loglik, fit_filter, displacement_experiment,
)


# ---------------------------------------------------------------------------
# Location score driver properties
# ---------------------------------------------------------------------------

class TestLocationScoreDrivers:
    """Unit tests for the three location score drivers."""

    # ---- u_log_z ----------------------------------------------------------

    def test_u_log_z_is_identity(self):
        z = np.linspace(-5, 5, 100)
        np.testing.assert_array_almost_equal(u_log_z(z), z)

    def test_u_log_z_zero_at_zero(self):
        assert u_log_z(0.0) == pytest.approx(0.0)

    def test_u_log_z_unbounded(self):
        assert abs(u_log_z(100.0)) > 50.0

    # ---- u_CRPS_z ----------------------------------------------------------

    def test_u_CRPS_z_bounded_by_one(self):
        z = np.linspace(-20, 20, 500)
        vals = u_CRPS_z(z)
        assert np.all(np.abs(vals) <= 1.0 + 1e-12), 'CRPS loc score bounded in [-1,1]'

    def test_u_CRPS_z_zero_at_zero(self):
        assert u_CRPS_z(0.0) == pytest.approx(0.0, abs=1e-12)

    def test_u_CRPS_z_positive_for_positive_z(self):
        z = np.array([0.5, 1.0, 3.0, 10.0])
        assert np.all(u_CRPS_z(z) > 0)

    def test_u_CRPS_z_negative_for_negative_z(self):
        z = np.array([-0.5, -1.0, -3.0, -10.0])
        assert np.all(u_CRPS_z(z) < 0)

    def test_u_CRPS_z_antisymmetric(self):
        z = np.linspace(-5, 5, 50)
        np.testing.assert_allclose(u_CRPS_z(-z), -u_CRPS_z(z), atol=1e-12)

    def test_u_CRPS_z_approaches_one_for_large_z(self):
        assert u_CRPS_z(20.0) == pytest.approx(1.0, abs=1e-6)

    # ---- u_DP_z ------------------------------------------------------------

    def test_u_DP_z_zero_at_zero(self):
        assert u_DP_z(0.0) == pytest.approx(0.0, abs=1e-12)

    def test_u_DP_z_mean_approx_zero(self):
        rng = np.random.default_rng(0)
        z = rng.standard_normal(200_000)
        assert abs(np.mean(u_DP_z(z))) < 0.005

    def test_u_DP_z_smaller_than_u_log_z_for_large_z(self):
        """DP driver must be softer than log for large z (robustness)."""
        z = 6.0
        assert abs(u_DP_z(z)) < abs(u_log_z(z))

    def test_u_DP_z_antisymmetric(self):
        z = np.linspace(-4, 4, 40)
        np.testing.assert_allclose(u_DP_z(-z), -u_DP_z(z), atol=1e-12)

    def test_robustness_ordering_at_outlier_z(self):
        """At z=6 (outlier), |u_DP| < |u_CRPS| < |u_log|."""
        z = 6.0
        log_val = abs(u_log_z(z))
        crps_val = abs(u_CRPS_z(z))
        dp_val = abs(u_DP_z(z))
        assert dp_val < crps_val < log_val, (
            f'Expected |DP|={dp_val:.4f} < |CRPS|={crps_val:.4f} < |log|={log_val:.4f}')


# ---------------------------------------------------------------------------
# Filter runner tests
# ---------------------------------------------------------------------------

class TestLocationFilterRunner:
    def setup_method(self):
        rng = np.random.default_rng(42)
        # Simple AR(1) data with some variation
        self.y = 900.0 + 50.0 * rng.standard_normal(80)
        self.runners = {name: make_filter_runner(fn)
                        for name, fn in [('log', u_log_z),
                                         ('CRPS', u_CRPS_z),
                                         ('DP', u_DP_z)]}

    def test_runner_output_length(self):
        for name, runner in self.runners.items():
            mu = runner(self.y, omega=100.0, phi=0.9, kappa=50.0, sigma=100.0)
            assert mu.shape == (len(self.y) + 1,), f'{name}: wrong shape'

    def test_runner_mu0_respected(self):
        mu0 = 950.0
        for name, runner in self.runners.items():
            mu = runner(self.y, omega=100.0, phi=0.9, kappa=50.0,
                        sigma=100.0, mu0=mu0)
            assert mu[0] == pytest.approx(mu0), f'{name}: mu0 not respected'

    def test_runner_tracks_mean(self):
        """With high gain, filter should stay near sample mean."""
        for name, runner in self.runners.items():
            mu = runner(self.y, omega=0.0, phi=0.99, kappa=100.0, sigma=100.0)
            # Filter should not diverge far from data range
            data_range = self.y.max() - self.y.min()
            assert abs(mu[-1] - np.mean(self.y)) < 3 * data_range, \
                f'{name}: filter diverged'


# ---------------------------------------------------------------------------
# Negative log-likelihood test
# ---------------------------------------------------------------------------

class TestNegLoglik:
    def setup_method(self):
        rng = np.random.default_rng(10)
        self.y = 900.0 + 150.0 * rng.standard_normal(74)

    def test_neg_loglik_is_finite(self):
        from exp.exp_14_nile_location import _runners
        runner = _runners['log']
        params = [100.0, 0.9, np.log(50.0), np.log(150.0)]
        val = neg_loglik(params, self.y, 'log', runner)
        assert np.isfinite(val)
        assert val > 0.0

    def test_neg_loglik_worse_for_bad_sigma(self):
        """Sigma far from truth should give higher NLL."""
        from exp.exp_14_nile_location import _runners
        runner = _runners['log']
        true_sigma = np.std(self.y)
        params_good = [np.mean(self.y) * 0.1, 0.9, np.log(true_sigma * 0.3),
                       np.log(true_sigma)]
        params_bad = [np.mean(self.y) * 0.1, 0.9, np.log(true_sigma * 0.3),
                      np.log(true_sigma * 10)]
        nll_good = neg_loglik(params_good, self.y, 'log', runner)
        nll_bad = neg_loglik(params_bad, self.y, 'log', runner)
        assert nll_good < nll_bad


# ---------------------------------------------------------------------------
# Displacement ordering test (core validation)
# ---------------------------------------------------------------------------

class TestDisplacementOrdering:
    """The key qualitative result: robustness ordering holds."""

    @pytest.fixture(scope='class')
    def nile_displacements(self):
        import statsmodels.api as sm
        nile_data = sm.datasets.nile.load_pandas().data
        years = nile_data['year'].values.astype(int)
        y_full = nile_data['volume'].values.astype(float)
        results = displacement_experiment(y_full, years,
                                          train_cutoff=1944,
                                          outlier_year=1945,
                                          outlier_sigma_mult=6.0)
        return {name: res['displacement_norm'] for name, res in results.items()}

    def test_log_displacement_largest(self, nile_displacements):
        d = nile_displacements
        assert d['log'] > d['CRPS'], (
            f"log ({d['log']:.4f}) should exceed CRPS ({d['CRPS']:.4f})")
        assert d['log'] > d['DP'], (
            f"log ({d['log']:.4f}) should exceed DP ({d['DP']:.4f})")

    def test_DP_displacement_smallest(self, nile_displacements):
        d = nile_displacements
        assert d['DP'] < d['CRPS'], (
            f"DP ({d['DP']:.4f}) should be less than CRPS ({d['CRPS']:.4f})")

    def test_log_displacement_approx_target(self, nile_displacements):
        d = nile_displacements
        assert 0.5 < d['log'] < 5.0, (
            f"log displacement/σ̂ = {d['log']:.4f}, expected ~1.615")

    def test_DP_displacement_approx_target(self, nile_displacements):
        d = nile_displacements
        assert 0.01 < d['DP'] < 1.0, (
            f"DP displacement/σ̂ = {d['DP']:.4f}, expected ~0.066")

    def test_all_displacements_positive(self, nile_displacements):
        """A positive outlier should produce a positive displacement for all drivers."""
        for name, disp in nile_displacements.items():
            assert disp > 0.0, f'{name}: displacement should be positive for +6σ injection'


# ---------------------------------------------------------------------------
# Nile data loading test
# ---------------------------------------------------------------------------

class TestNileData:
    def test_nile_dataset_loads(self):
        import statsmodels.api as sm
        data = sm.datasets.nile.load_pandas().data
        assert len(data) == 100
        assert 'year' in data.columns
        assert 'volume' in data.columns

    def test_nile_years_range(self):
        import statsmodels.api as sm
        data = sm.datasets.nile.load_pandas().data
        years = data['year'].values.astype(int)
        assert years[0] == 1871
        assert years[-1] == 1970

    def test_train_test_split_sizes(self):
        import statsmodels.api as sm
        data = sm.datasets.nile.load_pandas().data
        years = data['year'].values.astype(int)
        n_train = int(np.sum(years <= 1944))
        n_test = int(np.sum(years >= 1945))
        assert n_train == 74
        assert n_test == 26
        assert n_train + n_test == 100
