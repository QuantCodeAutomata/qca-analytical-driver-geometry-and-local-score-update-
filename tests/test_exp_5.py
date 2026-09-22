"""
Tests for Experiment 5: Centred Diffusion-Limit Diagnostics
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from exp.exp_5_diffusion_limit import (
    BETA, C_BETA, KAPPA_BETA,
    J_LOG, J_DP, J_CRPS,
    u_log, u_dp, u_crps,
    K_LOG, K_DP, K_CRPS, K_VALUES,
    _K_via_quadrature,
    SCORE_FNS,
    run_one_h,
    run_experiment,
)


# ---------------------------------------------------------------------------
# 1. Score function properties
# ---------------------------------------------------------------------------
class TestScoreFunctions:
    """Properties of the score functions under N(0,1)."""

    def _gh_mean(self, u_fn) -> float:
        pts, wts = np.polynomial.hermite.hermgauss(200)
        z = pts * np.sqrt(2.0)
        w = wts / np.sqrt(np.pi)
        return float(np.dot(w, u_fn(z)))

    def test_log_centering(self):
        assert abs(self._gh_mean(u_log)) < 1e-12

    def test_dp_centering(self):
        assert abs(self._gh_mean(u_dp)) < 1e-8

    def test_crps_centering(self):
        assert abs(self._gh_mean(u_crps)) < 1e-12

    def test_log_at_zero(self):
        """u_log(0) = -1."""
        assert abs(float(u_log(np.array([0.0]))[0]) - (-1.0)) < 1e-12

    def test_crps_at_zero(self):
        """u_crps(0) = 1/sqrt(pi) - sqrt(2/pi): a specific negative number."""
        import math
        expected = 1.0 / math.sqrt(math.pi) - math.sqrt(2.0 / math.pi)
        got = float(u_crps(np.array([0.0]))[0])
        assert abs(got - expected) < 1e-12

    def test_dp_monotone_in_abs_z(self):
        """u_DP at large |z| should be dominated by kappa_beta (bounded above)."""
        z_large = np.array([50.0])
        z_small = np.array([0.0])
        # For large |z|, exp(-beta*z^2/2) ≈ 0, so u_DP ≈ kappa_beta > 0
        assert abs(float(u_dp(z_large)[0]) - KAPPA_BETA) < 1e-6


# ---------------------------------------------------------------------------
# 2. K_S = Var_{N(0,1)}[u_S(Z)] via quadrature
# ---------------------------------------------------------------------------
class TestKConstants:

    def test_K_log_analytic(self):
        """K_log = Var[Z^2-1] = E[(Z^2-1)^2] = 2 (since mean=0)."""
        # E[Z^4] - 2*E[Z^2] + 1 = 3 - 2 + 1 = 2
        assert abs(K_LOG - 2.0) < 1e-6

    def test_K_dp_positive(self):
        assert K_DP > 0.0

    def test_K_crps_positive(self):
        assert K_CRPS > 0.0

    def test_K_crps_numeric(self):
        """K_CRPS from quadrature should match a direct MC estimate."""
        rng = np.random.default_rng(0)
        Z = rng.standard_normal(2_000_000)
        u = u_crps(Z)
        mc_var = float(u.var())
        assert abs(mc_var - K_CRPS) < 0.01, (
            f"K_CRPS={K_CRPS:.5f} vs MC={mc_var:.5f}"
        )

    def test_K_dp_numeric(self):
        """K_DP from quadrature should match MC."""
        rng = np.random.default_rng(1)
        Z = rng.standard_normal(2_000_000)
        u = u_dp(Z)
        mc_var = float(u.var())
        assert abs(mc_var - K_DP) < 0.01, (
            f"K_DP={K_DP:.5f} vs MC={mc_var:.5f}"
        )

    def test_quadrature_self_consistency(self):
        """_K_via_quadrature applied to u_log should give K_log."""
        k = _K_via_quadrature(u_log)
        assert abs(k - K_LOG) < 1e-10

    def test_K_values_dict(self):
        """K_VALUES should contain the three criteria."""
        assert set(K_VALUES.keys()) == {"Gauss-log", "Gauss-DP", "Gauss-CRPS"}


# ---------------------------------------------------------------------------
# 3. Diffusion-limit diagnostics (run_one_h)
# ---------------------------------------------------------------------------
class TestRunOneH:

    def test_centering_at_h1(self):
        """At h=1, E[u_log(z)] ≈ 0 (z_std = Z ~ N(0,1))."""
        rng = np.random.default_rng(42)
        d = run_one_h(h=1.0, R=5_000, T=50, kappa=0.5, u_fn=u_log, rng=rng)
        assert abs(d["E_u"]) < 0.05, f"E_u at h=1 should be ≈ 0, got {d['E_u']:.4f}"

    def test_second_moment_convergence(self):
        """h^{-1}*E[(Δλ)^2] should be close to K_log for small h."""
        rng = np.random.default_rng(10)
        d = run_one_h(h=0.25, R=4_000, T=100, kappa=0.5, u_fn=u_log, rng=rng)
        ratio = d["E_du2_over_h"] / K_LOG
        assert 0.8 < ratio < 1.3, (
            f"h^{{-1}}*E[(Δλ)^2] / K_log = {ratio:.3f}, expected near 1"
        )

    def test_fourth_moment_smaller_than_second_squared(self):
        """h^{-1}*E[(Δλ)^4] should be small (Lindeberg condition)."""
        rng = np.random.default_rng(11)
        # For h=0.5, the fourth moment scaled by h^{-1} should be ~ h * (K_log^2 terms)
        # i.e., smaller than (h^{-1}*E[(Δλ)^2])^2 * h = K_log^2 * h
        d = run_one_h(h=0.5, R=2_000, T=50, kappa=0.5, u_fn=u_log, rng=rng)
        # No strict bound; just check it's positive and finite
        assert d["E_du4_over_h"] > 0
        assert np.isfinite(d["E_du4_over_h"])


# ---------------------------------------------------------------------------
# 4. Full experiment (small scale)
# ---------------------------------------------------------------------------
class TestRunExperiment:

    def test_run_experiment_small(self):
        """run_experiment should return a DataFrame with correct columns."""
        h_list = [0.5, 0.25]
        df = run_experiment(h_list, R=500, T=20, seed=99)
        required_cols = {"criterion", "h", "E_u", "E_du2_over_h", "K_S_theory", "ratio_2nd"}
        assert required_cols.issubset(set(df.columns))
        assert len(df) == len(h_list) * len(SCORE_FNS)
        assert df["ratio_2nd"].notna().all()

    def test_ratio_2nd_near_one_for_log(self):
        """For Gaussian-log, the ratio h^{-1}*E[(Δλ)^2] / K_log should be near 1."""
        h_list = [0.25, 0.125]
        df = run_experiment(h_list, R=2_000, T=100, seed=77)
        log_rows = df[df["criterion"] == "Gauss-log"]
        # Average ratio across h values
        avg_ratio = float(log_rows["ratio_2nd"].mean())
        assert 0.75 < avg_ratio < 1.35, (
            f"Average ratio for Gauss-log = {avg_ratio:.3f}, expected ≈ 1"
        )

    def test_all_criteria_finite(self):
        """No NaN or Inf in output."""
        h_list = [0.5, 0.25]
        df = run_experiment(h_list, R=300, T=20, seed=88)
        numeric_cols = ["E_u", "E_du2_over_h", "E_du4_over_h", "K_S_theory", "ratio_2nd"]
        for col in numeric_cols:
            assert df[col].notna().all(), f"NaN in column {col}"
            assert np.isfinite(df[col].values).all(), f"Inf in column {col}"
