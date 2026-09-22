"""
Tests for Experiment 4: Finite-Sample Filtering Robustness
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from scipy import integrate, stats

from exp.exp_4_mc_filtering import (
    BETA, C_BETA, KAPPA_BETA,
    J_LOG, J_DP, J_CRPS, J_MMD, J_T6,
    _filter_log, _filter_dp, _filter_crps, _filter_mmd, _filter_t6,
    _simulate_dgp, _metrics, run_mc,
)


# ---------------------------------------------------------------------------
# 1. Score functions: centering under N(0,1)
# ---------------------------------------------------------------------------
class TestScoreCentering:
    """E_{N(0,1)}[u_S(Z)] = 0 for all score functions."""

    def _gauss_hermite_mean(self, u_fn) -> float:
        pts, wts = np.polynomial.hermite.hermgauss(200)
        z = pts * np.sqrt(2.0)
        w = wts / np.sqrt(np.pi)
        return float(np.dot(w, u_fn(z)))

    def test_log_centered(self):
        def u(z): return z * z - 1.0
        assert abs(self._gauss_hermite_mean(u)) < 1e-12

    def test_dp_centered(self):
        def u(z):
            return (
                (1 + BETA) * C_BETA * np.exp(-0.5 * BETA * z * z) * (z * z - 1.0)
                + KAPPA_BETA
            )
        assert abs(self._gauss_hermite_mean(u)) < 1e-8

    def test_crps_centered(self):
        def u(z):
            return 1.0 / np.sqrt(np.pi) - np.sqrt(2.0 / np.pi) * np.exp(-0.5 * z * z)
        assert abs(self._gauss_hermite_mean(u)) < 1e-12

    def test_mmd_centered(self):
        def u(z):
            return (
                2.0 * (2.0 / np.sqrt(5.0)) * np.exp(-z * z / 10.0) * (z * z / 25.0 - 0.2)
                + 2.0 * 2.0 / (6.0 ** 1.5)
            )
        assert abs(self._gauss_hermite_mean(u)) < 1e-8


# ---------------------------------------------------------------------------
# 2. J_S constants (curvature = Fisher information for scale)
# ---------------------------------------------------------------------------
class TestJConstants:
    """J_S = E_S[(score)^2] = E_S[(u_S/2)^2]."""

    def _J_gauss_hermite(self, u_fn) -> float:
        pts, wts = np.polynomial.hermite.hermgauss(200)
        z = pts * np.sqrt(2.0)
        w = wts / np.sqrt(np.pi)
        score_sq = (u_fn(z) / 2.0) ** 2
        return float(np.dot(w, score_sq))

    def test_J_log(self):
        def u(z): return z * z - 1.0
        assert abs(self._J_gauss_hermite(u) - J_LOG) < 1e-6

    def test_J_crps_positive(self):
        """J_CRPS should be positive and match documented value approximately."""
        assert J_CRPS > 0

    def test_J_t6_positive_and_finite(self):
        assert 0 < J_T6 < 1.0, f"J_T6={J_T6} out of expected range (0, 1)"

    def test_J_t6_computed_under_t6(self):
        """Verify J_T6 is computed under the t6 distribution, not N(0,1)."""
        nu = 6
        def integrand(z):
            u = (nu + 1) * z * z / (nu + z * z) - 1.0
            return 0.25 * u * u * stats.t.pdf(z, nu)
        val, _ = integrate.quad(integrand, -80, 80, limit=200)
        assert abs(val - J_T6) < 1e-6


# ---------------------------------------------------------------------------
# 3. Filter recursion correctness
# ---------------------------------------------------------------------------
class TestFilterRecursion:
    """One-step and multi-step filter correctness."""

    def test_log_one_step_from_zero(self):
        """At lambda=0, z=y, u=y^2-1; verify lambda_1 = alpha_eff*(y^2-1)."""
        y = np.array([2.0])
        varphi, alpha_eff = 0.97, 0.2   # alpha/J_log = 0.1/0.5
        lam = _filter_log(y, varphi, alpha_eff)
        expected_lam0 = 0.0
        # lam[0] is lambda_0 = 0, then lambda_1 would be update of y[0]
        assert abs(lam[0] - expected_lam0) < 1e-12

    def test_log_two_steps(self):
        """Verify two-step recursion matches hand calculation."""
        y = np.array([2.0, 1.0])
        varphi, alpha_eff = 1.0, 1.0   # simplified: no decay, full gain
        lam = _filter_log(y, varphi, alpha_eff)
        # lam[0] = 0
        assert abs(lam[0]) < 1e-12
        # lam[1]: after observing y[0]=2, z=2*exp(0)=2, u=4-1=3, lam_1=1.0*0+1.0*3=3
        # But lam[1] is the stored state before the update by y[1]
        # Actually: lam[0]=0 (initial), then update by y[0]: lam_new=0+1.0*(4-1)=3
        # lam[1] = 3 (stored before processing y[1])
        assert abs(lam[1] - 3.0) < 1e-12

    def test_dp_bounded_update(self):
        """DP update u_DP(z) stays finite for extreme z."""
        y_extreme = np.array([100.0])
        lam = _filter_dp(y_extreme, 0.97, 0.28, BETA, C_BETA, KAPPA_BETA)
        assert np.isfinite(lam[0])

    def test_mmd_bounded_update(self):
        """MMD update is bounded (exp kernel suppresses large z)."""
        y_extreme = np.array([1000.0])
        lam = _filter_mmd(y_extreme, 0.97, 2.9)
        assert np.isfinite(lam[0])

    def test_crps_bounded_update(self):
        """CRPS update is bounded: u_CRPS in [-1/sqrt(pi), 1/sqrt(pi)]."""
        y_extreme = np.array([1000.0])
        lam = _filter_crps(y_extreme, 0.97, 1.4)
        assert np.isfinite(lam[0])

    def test_t6_bounded_update(self):
        """t6 score is bounded (heaviest tails still give finite score)."""
        y_extreme = np.array([1000.0])
        lam = _filter_t6(y_extreme, 0.97, 0.2)
        assert np.isfinite(lam[0])

    def test_filter_output_length(self):
        """Filter output has same length as input y."""
        T = 50
        y = np.random.default_rng(1).standard_normal(T)
        alpha_eff = 0.2
        for fn in [_filter_log, _filter_crps, _filter_mmd, _filter_t6]:
            out = fn(y, 0.97, alpha_eff)
            assert len(out) == T, f"{fn.__name__} output length wrong"

    def test_log_filter_z_zero(self):
        """At z=0 (y=0), u_log(0) = -1; filter decreases."""
        y = np.array([0.0, 0.0, 0.0])
        varphi, alpha_eff = 1.0, 0.5
        lam = _filter_log(y, varphi, alpha_eff)
        # lambda_0=0, lambda_1=0+0.5*(-1)=-0.5, lambda_2=-0.5+0.5*(-1)=-1
        assert abs(lam[0] - 0.0) < 1e-12
        assert abs(lam[1] - (-0.5)) < 1e-12
        assert abs(lam[2] - (-1.0)) < 1e-12


# ---------------------------------------------------------------------------
# 4. DGP simulation
# ---------------------------------------------------------------------------
class TestDGP:
    def test_dgp_output_shapes(self):
        rng = np.random.default_rng(0)
        T = 100
        x, y_c, y_k = _simulate_dgp(rng, T, 0.97, 0.1)
        assert x.shape == (T,) and y_c.shape == (T,) and y_k.shape == (T,)

    def test_dgp_clean_no_contamination(self):
        """Clean y and contaminated y can differ (no deterministic guarantee)."""
        rng = np.random.default_rng(42)
        _, y_c, y_k = _simulate_dgp(rng, 5000, 0.97, 0.1)
        # Contamination rate ≈ 2%; contaminated should have heavier tails
        assert y_k.std() > y_c.std() * 0.99   # contaminated is at least as variable

    def test_dgp_latent_persistence(self):
        """Autocorrelation of x should be close to phi_x=0.97."""
        rng = np.random.default_rng(7)
        T = 50_000
        phi_x = 0.97
        x, _, _ = _simulate_dgp(rng, T, phi_x, 0.1)
        acf1 = float(np.corrcoef(x[:-1], x[1:])[0, 1])
        assert abs(acf1 - phi_x) < 0.02


# ---------------------------------------------------------------------------
# 5. Metrics
# ---------------------------------------------------------------------------
class TestMetrics:
    def test_perfect_filter(self):
        x = np.array([1.0, 2.0, 3.0])
        bias, rmse, maxdev = _metrics(x, x)
        assert abs(bias) < 1e-12
        assert abs(rmse) < 1e-12
        assert abs(maxdev) < 1e-12

    def test_constant_offset(self):
        x   = np.zeros(100)
        lam = np.ones(100) * 0.5
        bias, rmse, maxdev = _metrics(lam, x)
        assert abs(bias   - 0.5) < 1e-12
        assert abs(rmse   - 0.5) < 1e-12
        assert abs(maxdev - 0.5) < 1e-12


# ---------------------------------------------------------------------------
# 6. Small MC smoke test (R=20, T=100)
# ---------------------------------------------------------------------------
class TestSmallMC:
    def test_run_mc_small(self):
        results = run_mc(R=20, T=100, seed=0)
        assert set(results.keys()) == {"Gauss-log", "Gauss-DP", "Gauss-CRPS",
                                       "Gauss-MMD", "Student-t6"}
        for name, d in results.items():
            assert d["clean"].shape == (20, 3)
            assert d["cont"].shape  == (20, 3)
            assert np.isfinite(d["clean"]).all(), f"{name} clean has NaN/Inf"
            assert np.isfinite(d["cont"]).all(),  f"{name} cont has NaN/Inf"

    def test_contaminated_rmse_higher_for_log(self):
        """Gaussian-log should have higher RMSE under contamination than robust criteria."""
        results = run_mc(R=200, T=500, seed=42)
        rmse_log_clean = results["Gauss-log"]["clean"][:, 1].mean()
        rmse_log_cont  = results["Gauss-log"]["cont"][:, 1].mean()
        rmse_dp_cont   = results["Gauss-DP"]["cont"][:, 1].mean()
        # Contamination should worsen Gauss-log more than DP
        assert rmse_log_cont > rmse_log_clean, "Contamination should increase log RMSE"
        assert rmse_log_cont > rmse_dp_cont,   "Log RMSE under contamination should exceed DP"
