"""
Score drivers for the Gaussian working-scale family.

Working-scale transformation: z = y * exp(-lambda / 2)
where lambda = log(sigma^2) is the log-variance parameter.

Each driver u(z) gives the score update; J and K are the
Fisher information and curvature constants, respectively.
"""

import numpy as np
from scipy.special import ndtr as _Phi  # CDF of N(0,1)
from scipy.stats import norm as _norm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_b = 0.15
_c_b = (2 * np.pi) ** (-_b / 2)
_kappa_b = _b * _c_b * (1 + _b) ** (-0.5)

# J and K values from analytical derivations
J_LOG = 0.500
K_LOG = 0.500
J_DP = 0.357
K_DP = 0.268
J_CRPS = 0.071
K_CRPS = 0.012
J_MMD = 0.034
K_MMD = 0.002

# MMD constants at lambda=0, ell=2
_MMD_A = 2.0 * (2.0 / np.sqrt(5.0))       # leading coefficient
_MMD_CONST = 2.0 * 2.0 / (6.0 ** 1.5)    # centering constant


# ---------------------------------------------------------------------------
# Scale score drivers  (z = y * exp(-lambda/2))
# ---------------------------------------------------------------------------

def u_log(z):
    """Gaussian log score driver: u_log(z) = z^2 - 1."""
    return z * z - 1.0


def u_DP(z):
    """Density-power score driver (b=0.15).

    u_DP(z) = (1+b)*c_b*exp(-b*z^2/2)*(z^2-1) + kappa_b
    """
    return (1 + _b) * _c_b * np.exp(-_b * z * z / 2.0) * (z * z - 1.0) + _kappa_b


def u_CRPS(z):
    """CRPS score driver.

    u_CRPS(z) = 1/sqrt(pi) - sqrt(2/pi)*exp(-z^2/2)
    """
    return 1.0 / np.sqrt(np.pi) - np.sqrt(2.0 / np.pi) * np.exp(-z * z / 2.0)


def u_MMD(z):
    """MMD score driver at lambda=0, ell=2.

    u_MMD(z) = 2*(2/sqrt(5))*exp(-z^2/10)*(z^2/25 - 1/5) + 2*2/(6)^(3/2)
    """
    return _MMD_A * np.exp(-z * z / 10.0) * (z * z / 25.0 - 0.2) + _MMD_CONST


def u_student(z, nu):
    """Score driver under Student-t(nu) for the log-variance parameter.

    Derived from d/d(lambda) log t_nu(y; 0, exp(lambda/2)).
    u_student(z, nu) = (nu+1)*z^2/(nu + z^2) - 1
    """
    return (nu + 1.0) * z * z / (nu + z * z) - 1.0


# ---------------------------------------------------------------------------
# Location score drivers  (r = y - mu, z_loc = r / sigma)
# ---------------------------------------------------------------------------

def u_log_loc(r, sigma):
    """Gaussian log location score (normalized by J=1/sigma^2 → yields residual).

    Curvature-normalised update: g = r  (standardised residual * sigma).
    Raw score: r / sigma^2; divided by J_log_loc = 1/sigma^2 gives r.
    """
    return r / (sigma * sigma)


def u_CRPS_loc(r, sigma):
    """CRPS location score under N(mu, sigma^2).

    u_CRPS_loc = 2*Phi(r/sigma) - 1
    Bounded in [-1, 1].  Positive when y > mu (pushes mu upward).
    """
    return 2.0 * _Phi(r / sigma) - 1.0


def u_DP_loc(r, sigma):
    """Density-power location score (beta=0.15) under N(mu, sigma^2).

    u_DP_loc = (1+b)*c_b*exp(-b*(r/sigma)^2/2)*(r/sigma^2)
    Self-centering: E[u_DP_loc]=0 by symmetry; no kappa correction needed.
    """
    z = r / sigma
    return (1 + _b) * _c_b * np.exp(-_b * z * z / 2.0) * z / sigma
