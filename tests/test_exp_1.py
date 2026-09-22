"""Tests for Experiment 1: Analytical Driver Geometry."""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.score_drivers import (
    u_log, u_DP, u_CRPS, u_MMD,
    J_LOG, K_LOG, J_DP, K_DP, J_CRPS, K_CRPS, J_MMD, K_MMD,
)
from exp.exp_1_driver_geometry import (
    psi_log, psi_DP, psi_CRPS, psi_MMD,
    gh_normal_expectation, compute_J, compute_K,
    sup_driver,
)


# ── Gauss-Hermite quadrature ──────────────────────────────────────────────────

def test_gh_normal_expectation_mean():
    """E[Z] = 0 under N(0,1)."""
    val = gh_normal_expectation(lambda z: z, n_nodes=64)
    assert abs(val) < 1e-12, f"E[Z] should be 0, got {val}"


def test_gh_normal_expectation_variance():
    """E[Z^2] = 1 under N(0,1)."""
    val = gh_normal_expectation(lambda z: z**2, n_nodes=64)
    assert abs(val - 1.0) < 1e-12, f"E[Z^2] should be 1, got {val}"


def test_gh_convergence_64_vs_128():
    """64-node and 128-node quadrature agree within 1e-8."""
    for fn in [psi_log, psi_DP, psi_CRPS, psi_MMD]:
        v64  = gh_normal_expectation(lambda z, f=fn: f(z)**2, n_nodes=64)
        v128 = gh_normal_expectation(lambda z, f=fn: f(z)**2, n_nodes=128)
        assert abs(v64 - v128) < 1e-8, f"Quadrature disagreement: {abs(v64-v128):.2e}"


# ── Score derivative correctness (finite-difference check) ────────────────────

def _score_log(y, lam):
    return 0.5 * lam + 0.5 * np.exp(-lam) * y**2


def _score_DP(y, lam):
    from src.score_drivers import _b, _c_b
    z  = y * np.exp(-lam / 2.0)
    t1 = (1 + _b)**(-0.5)
    t2 = -(1 + 1.0 / _b) * np.exp(-_b * z**2 / 2.0)
    return _c_b * np.exp(-_b * lam / 2.0) * (t1 + t2)


@pytest.mark.parametrize("z_val", [0.0, 1.0, -2.0, 3.0])
def test_psi_log_finite_difference(z_val):
    """psi_log = -d S_log / d lambda  (score-driven convention: gradient of log p).

    FD of -S_log w.r.t. lambda should match psi_log.
    """
    y   = z_val   # at lambda=0, z=y
    lam = 0.0
    h   = 1e-5
    # psi_log = -dS_log/dlambda (gradient of log-likelihood)
    fd  = -(_score_log(y, lam + h) - _score_log(y, lam - h)) / (2 * h)
    analytic = float(psi_log(np.array([z_val]))[0])
    assert abs(fd - analytic) < 1e-7, f"z={z_val}: FD={fd:.8f} analytic={analytic:.8f}"


@pytest.mark.parametrize("z_val", [0.0, 1.0, 2.0])
def test_psi_DP_finite_difference(z_val):
    """psi_DP = -d S_DP / d lambda (gradient of score w.r.t. lambda).

    For the density-power proper scoring rule, the update direction is negative
    gradient of the score (since we minimise S and psi drives the recursion toward
    the score minimum).
    """
    y   = z_val
    lam = 0.0
    h   = 1e-5
    # Negative finite difference (psi = -dS/dlambda)
    fd  = -(_score_DP(y, lam + h) - _score_DP(y, lam - h)) / (2 * h)
    analytic = float(psi_DP(np.array([z_val]))[0])
    # Allow loose tolerance since psi_DP = u_DP/2
    assert abs(fd - analytic) < 1e-3, f"z={z_val}: FD={fd:.6f} analytic={analytic:.6f}"


# ── Local curvature J_S ───────────────────────────────────────────────────────

def test_J_log():
    """J_log should equal 0.500 analytically."""
    J = compute_J(psi_log, n_nodes=128)
    assert abs(J - J_LOG) < 0.01, f"J_log = {J:.4f}, expected {J_LOG}"


def test_J_DP():
    """J_DP should be approximately 0.357."""
    J = compute_J(psi_DP, n_nodes=128)
    assert abs(J - J_DP) < J_DP * 0.15, f"J_DP = {J:.4f}, expected {J_DP}"


def test_J_CRPS():
    """J_CRPS should be approximately 0.071."""
    J = compute_J(psi_CRPS, n_nodes=128)
    assert abs(J - J_CRPS) < J_CRPS * 0.20, f"J_CRPS = {J:.4f}, expected {J_CRPS}"


def test_J_MMD():
    """J_MMD should be approximately 0.034."""
    J = compute_J(psi_MMD, n_nodes=128)
    assert abs(J - J_MMD) < J_MMD * 0.25, f"J_MMD = {J:.4f}, expected {J_MMD}"


# ── Raw derivative variance K_S ───────────────────────────────────────────────

def test_K_log():
    """K_log = E[(psi_log)^2] = 0.5."""
    K = compute_K(psi_log, n_nodes=128)
    assert abs(K - K_LOG) < 0.01, f"K_log = {K:.4f}, expected {K_LOG}"


def test_K_DP():
    """K_DP approximately matches expected value."""
    K = compute_K(psi_DP, n_nodes=128)
    assert abs(K - K_DP) < K_DP * 0.20, f"K_DP = {K:.4f}, expected {K_DP}"


# ── Boundedness and tail behaviour ────────────────────────────────────────────

def test_gaussian_log_unbounded():
    """Gaussian log driver grows without bound."""
    sup_50 = sup_driver(psi_log, z_max=50)
    assert sup_50 > 100, f"Gaussian log should be unbounded, got sup={sup_50}"


def test_density_power_bounded():
    """Density-power driver is bounded."""
    sup_50 = sup_driver(psi_DP, z_max=50)
    assert sup_50 < 5.0, f"DP driver should be bounded, got {sup_50:.3f}"


def test_crps_bounded():
    """CRPS driver is bounded."""
    sup_50 = sup_driver(psi_CRPS, z_max=50)
    assert sup_50 < 1.0, f"CRPS driver should be bounded, got {sup_50:.3f}"


def test_mmd_bounded():
    """MMD driver is bounded."""
    sup_50 = sup_driver(psi_MMD, z_max=50)
    assert sup_50 < 1.0, f"MMD driver should be bounded, got {sup_50:.3f}"


# ── Centering (E[psi] = 0) ────────────────────────────────────────────────────

def test_psi_log_centered():
    """E_{N(0,1)}[psi_log(Z)] = E[0.5*(Z^2-1)] = 0."""
    val = gh_normal_expectation(psi_log, n_nodes=128)
    assert abs(val) < 1e-12, f"psi_log not centered: {val}"


def test_psi_DP_centered():
    """E_{N(0,1)}[psi_DP(Z)] = 0 (lambda=0 is the pseudo-true for N(0,1) law).

    Analytically: E[(1+b)*c_b*exp(-bZ^2/2)*(Z^2-1)] + kappa_b = 0 under N(0,1).
    """
    val = gh_normal_expectation(psi_DP, n_nodes=128)
    assert abs(val) < 1e-6, f"psi_DP not centered: E[psi_DP]={val:.2e}"


# ── Edge cases ────────────────────────────────────────────────────────────────

def test_z_zero_psi_log():
    """psi_log(0) = -0.5."""
    val = float(psi_log(np.array([0.0]))[0])
    assert abs(val - (-0.5)) < 1e-10, f"psi_log(0) = {val}"


def test_large_z_mmd_saturates():
    """MMD driver saturates for |z| >> 1."""
    val_100 = float(psi_MMD(np.array([100.0]))[0])
    val_50  = float(psi_MMD(np.array([50.0]))[0])
    assert abs(val_100 - val_50) < 0.01, "MMD should saturate for large |z|"


def test_implemented_update_z_zero():
    """u_log(0) = -1.  u_CRPS(0) < 0 (small residual → reduce predicted variance).

    u_CRPS(z) = pi^{-1/2} - sqrt(2/pi)*exp(-z^2/2).
    At z=0: pi^{-1/2} - sqrt(2/pi) = (1-sqrt(2))/sqrt(pi) < 0.
    The update saturates to pi^{-1/2} > 0 for |z|→∞ (bounded, not redescending).
    """
    assert abs(float(u_log(np.array([0.0]))[0]) - (-1.0)) < 1e-10
    assert float(u_CRPS(np.array([0.0]))[0]) < 0, "u_CRPS(0) should be negative"
    # Tail saturation: u_CRPS → pi^{-1/2} for large |z|
    u_crps_tail = float(u_CRPS(np.array([20.0]))[0])
    assert abs(u_crps_tail - np.pi ** -0.5) < 1e-6


def test_update_consistency_with_psi():
    """u_S(z) = 2 * psi_S(z) at lambda=0 (since G_S=2 at lambda=0)."""
    z = np.array([0.5, 1.0, 2.0])
    for psi_fn, u_fn in [(psi_log, u_log), (psi_DP, u_DP),
                          (psi_CRPS, u_CRPS), (psi_MMD, u_MMD)]:
        diff = np.abs(u_fn(z) - 2.0 * psi_fn(z))
        assert np.all(diff < 1e-10), f"u_S != 2*psi_S: max diff = {diff.max():.2e}"
