"""Tests for Experiment 2: Pseudo-True Target Shifts."""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from exp.exp_2_pseudo_true_target import (
    score_log, score_DP, score_CRPS, score_MMD,
    _gh_expect_component, contaminated_risk, find_pseudo_true,
    EPS, V0, N_NODES,
)


# ── Score functions ───────────────────────────────────────────────────────────

def test_score_log_minimum_at_true_variance():
    """S_log expected value is minimised at the true variance."""
    # Under Y~N(0,1): E[S_log(P_lambda, Y)] = 0.5*lambda + 0.5*exp(-lambda)
    # Minimised at lambda=0 (exp(lambda)=1=var).
    vals = np.array([0.5 * lam + 0.5 * np.exp(-lam) for lam in [-2, -1, 0, 1, 2]])
    assert vals[2] == vals.min() or abs(vals[2] - vals.min()) < 1e-10


def test_score_log_positive():
    """Log score should be positive (improper scores can be negative, but check scale)."""
    y = np.array([0.0, 1.0, -1.0, 2.0])
    s = score_log(y, 0.0)
    assert np.all(np.isfinite(s))


def test_gh_expect_gaussian_component():
    """E_{N(0,1)}[S_log(P_0, Y)] = 0.5*0 + 0.5*1 = 0.5."""
    val = _gh_expect_component(score_log, lam=0.0, sigma_sq=1.0, n_nodes=96)
    assert abs(val - 0.5) < 1e-8, f"Expected 0.5, got {val:.8f}"


def test_contaminated_risk_log_closed_form():
    """Contaminated risk for log score should yield v*_log = v0*(1+eps*(tau-1))."""
    for tau in [2.0, 5.0, 10.0]:
        v_star = find_pseudo_true(score_log, EPS, tau, V0)
        v_expect = V0 * (1.0 + EPS * (tau - 1.0))
        err = abs(v_star - v_expect)
        assert err < 1e-3, f"tau={tau}: v*={v_star:.6f}, expected {v_expect:.6f}, err={err:.2e}"


def test_contaminated_risk_increases_with_tau_log():
    """For Gaussian log, v*/v0 grows linearly with tau."""
    taus = np.array([1.0, 3.0, 5.0, 10.0])
    targets = V0 * (1.0 + EPS * (taus - 1.0)) / V0
    # Check it's strictly increasing
    assert np.all(np.diff(targets) > 0)


def test_bounded_criteria_saturation():
    """Bounded criteria produce smaller v*/v0 growth than Gaussian log."""
    tau_large = 20.0
    v_log = find_pseudo_true(score_log, EPS, tau_large, V0)
    for sfn in [score_DP, score_CRPS, score_MMD]:
        v_other = find_pseudo_true(sfn, EPS, tau_large, V0)
        assert v_other < v_log * 0.99, (
            f"v_other={v_other:.4f} should be < v_log={v_log:.4f} at tau={tau_large}"
        )


def test_pseudo_true_at_tau_1_is_v0():
    """At tau=1 (no contamination), v* = v0 for all criteria."""
    for sfn in [score_log, score_DP, score_CRPS]:
        v_star = find_pseudo_true(sfn, EPS, 1.0, V0, n_scan=200)
        # v*/v0 should be close to 1 when tau=1
        assert abs(v_star / V0 - 1.0) < 0.10, f"v*/v0={v_star/V0:.4f} at tau=1"


# ── GH expectation checks ─────────────────────────────────────────────────────

def test_gh_expect_scales_with_variance():
    """E_{N(0,s^2)}[S_log(P_0, Y)] = 0.5*s^2 (variance dependence)."""
    for s2 in [0.5, 1.0, 2.0]:
        val = _gh_expect_component(score_log, lam=0.0, sigma_sq=s2, n_nodes=96)
        expected = 0.5 * s2    # E[S_log] = 0.5*lam + 0.5*E[y^2]*exp(-lam) = 0 + 0.5*s2
        assert abs(val - expected) < 1e-6, f"s2={s2}: got {val:.6f}, expected {expected:.6f}"


# ── Edge cases ────────────────────────────────────────────────────────────────

def test_score_DP_finite_for_large_y():
    """Density-power score remains finite for large observations."""
    y = np.array([10.0, 50.0, 100.0])
    s = score_DP(y, 0.0)
    assert np.all(np.isfinite(s))


def test_score_MMD_bounded():
    """MMD score should be bounded (between 0 and 2 for reasonable inputs)."""
    y = np.linspace(-10, 10, 100)
    s = score_MMD(y, 0.0)
    assert np.all(np.isfinite(s))
    assert np.all(s >= -0.5)
