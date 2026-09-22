"""Tests for Experiment 3: Additive Outlier Impulse-Response Analysis."""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.score_drivers import u_log, u_DP, u_CRPS, u_MMD, J_LOG, J_DP, J_CRPS, J_MMD
from exp.exp_3_impulse_response import (
    g_norm, impulse_response, sup_g, cumulative_envelope_bound,
    ALPHA, VARPHI, B_SHOCK,
)


# ── Exact recursion-difference identity ───────────────────────────────────────

def test_exact_recursion_difference():
    """Delta_k = alpha * varphi^k * [g_S(B) - g_S(0)] for Gaussian log."""
    B       = 5.0
    horizon = 20
    delta   = impulse_response(u_log, J_LOG, B, ALPHA, VARPHI, horizon)
    g_B     = g_norm(u_log, J_LOG, np.array([B]))[0]
    g_0     = g_norm(u_log, J_LOG, np.array([0.0]))[0]

    for k in range(horizon + 1):
        expected = ALPHA * (VARPHI**k) * (g_B - g_0)
        assert abs(delta[k] - expected) < 1e-12, (
            f"k={k}: delta={delta[k]:.8e}, expected={expected:.8e}"
        )


def test_delta_decays_exponentially():
    """After the shock, delta decays geometrically at rate varphi."""
    delta = impulse_response(u_log, J_LOG, B_SHOCK, ALPHA, VARPHI, horizon=30)
    for k in range(5, 25):
        ratio = abs(delta[k + 1]) / (abs(delta[k]) + 1e-15)
        assert abs(ratio - VARPHI) < 1e-6, f"k={k}: ratio={ratio:.6f}, varphi={VARPHI}"


# ── Gaussian-log: unbounded / large initial response ─────────────────────────

def test_gaussian_log_large_B8():
    """Gaussian-log Delta_0 at B=8 should be large (>> bounded criteria)."""
    delta = impulse_response(u_log, J_LOG, 8.0, ALPHA, VARPHI, horizon=0)
    assert abs(delta[0]) > 1.0, f"Gaussian-log Delta_0={delta[0]:.4f}, expected > 1"


def test_gaussian_log_delta0_grows_with_B():
    """Gaussian-log initial displacement grows (roughly quadratic) with B."""
    d5  = abs(impulse_response(u_log, J_LOG, 5.0, ALPHA, VARPHI, 0)[0])
    d10 = abs(impulse_response(u_log, J_LOG, 10.0, ALPHA, VARPHI, 0)[0])
    assert d10 > d5, "Gaussian-log Delta_0 should grow with B"


# ── Bounded criteria: smaller initial response ─────────────────────────────────

@pytest.mark.parametrize("u_fn,J,name", [
    (u_DP,   J_DP,   "Density-Power"),
    (u_CRPS, J_CRPS, "CRPS"),
    (u_MMD,  J_MMD,  "MMD"),
])
def test_bounded_criteria_smaller_than_log(u_fn, J, name):
    """Bounded criteria initial Delta_0 < Gaussian-log Delta_0."""
    log_delta = abs(impulse_response(u_log, J_LOG, B_SHOCK, ALPHA, VARPHI, 0)[0])
    bnd_delta = abs(impulse_response(u_fn,  J,     B_SHOCK, ALPHA, VARPHI, 0)[0])
    assert bnd_delta < log_delta, (
        f"{name}: bnd_delta={bnd_delta:.4f} should < log_delta={log_delta:.4f}"
    )


# ── Supremum and envelope bound ───────────────────────────────────────────────

def test_sup_g_DP_bounded():
    """sup_z |g_DP(z)| is finite."""
    M = sup_g(u_DP, J_DP, z_max=50)
    assert np.isfinite(M), f"DP sup should be finite, got {M}"
    assert M < 10.0, f"DP sup={M:.4f} seems too large"


def test_sup_g_log_grows_with_z_max():
    """Gaussian-log sup grows without bound."""
    M10 = sup_g(u_log, J_LOG, z_max=10)
    M50 = sup_g(u_log, J_LOG, z_max=50)
    assert M50 > M10 * 3, f"Gaussian-log sup should grow: M10={M10:.1f}, M50={M50:.1f}"


def test_envelope_bound_finite_for_bounded():
    """Cumulative envelope bound is finite for DP, CRPS, MMD."""
    for u_fn, J, name in [(u_DP, J_DP, "DP"), (u_CRPS, J_CRPS, "CRPS"), (u_MMD, J_MMD, "MMD")]:
        M     = sup_g(u_fn, J)
        bound = cumulative_envelope_bound(M, ALPHA, VARPHI)
        assert np.isfinite(bound), f"{name} envelope bound should be finite"
        assert bound > 0, f"{name} bound should be positive"


# ── Baseline residual ─────────────────────────────────────────────────────────

def test_baseline_z0_impulse():
    """With baseline_z=0, g_S(baseline)=g_S(0). Delta_0=alpha*[g_S(B)-g_S(0)]."""
    B = 3.0
    delta = impulse_response(u_DP, J_DP, B, ALPHA, VARPHI, horizon=0)
    g_B = g_norm(u_DP, J_DP, np.array([B]))[0]
    g_0 = g_norm(u_DP, J_DP, np.array([0.0]))[0]
    expected = ALPHA * (g_B - g_0)
    assert abs(delta[0] - expected) < 1e-12


# ── Edge cases ────────────────────────────────────────────────────────────────

def test_zero_shock():
    """With B=0 (baseline=0), Delta_k=0 for all k."""
    delta = impulse_response(u_log, J_LOG, 0.0, ALPHA, VARPHI, horizon=10,
                              baseline_z=0.0)
    assert np.all(np.abs(delta) < 1e-12)


def test_horizon_length():
    """impulse_response returns array of length horizon+1."""
    delta = impulse_response(u_CRPS, J_CRPS, 2.0, ALPHA, VARPHI, horizon=50)
    assert len(delta) == 51


def test_g_norm_consistency_with_u():
    """g_norm(z) = u_S(z) / (2*J_S) at lambda=0."""
    z = np.array([-2.0, 0.0, 1.5, 3.0])
    for u_fn, J in [(u_log, J_LOG), (u_DP, J_DP), (u_CRPS, J_CRPS)]:
        g    = g_norm(u_fn, J, z)
        g_ex = u_fn(z) / (2.0 * J)
        assert np.allclose(g, g_ex), "g_norm should equal u_S / (2*J_S)"
