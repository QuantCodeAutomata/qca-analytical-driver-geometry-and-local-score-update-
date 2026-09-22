"""
Score-driven filter implementations.

ScaleDrivenFilter: log-variance recursion  lambda_{t+1} = omega + phi*lambda_t + alpha*u(z_t)
LocationFilter   : location recursion       mu_{t+1}     = omega + phi*mu_t    + alpha*u(r_t, sigma)
"""

import numpy as np
from scipy.optimize import minimize

from src.score_drivers import (
    u_log, u_DP, u_CRPS, u_MMD, u_student,
    u_log_loc, u_CRPS_loc, u_DP_loc,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _loglik_gaussian(y, lam):
    """Gaussian log-likelihood: -0.5*(log(2pi) + lambda + y^2*exp(-lambda))."""
    return -0.5 * (np.log(2 * np.pi) + lam + y * y * np.exp(-lam))


def _qlike(sigma2, r2):
    """QLIKE loss: log(sigma^2) + r^2/sigma^2."""
    return np.log(sigma2) + r2 / sigma2


# ---------------------------------------------------------------------------
# Scale-driven filter (log-variance)
# ---------------------------------------------------------------------------

class ScaleDrivenFilter:
    """Score-driven log-variance filter.

    Parameters
    ----------
    driver : str
        One of 'log', 'DP', 'CRPS', 'MMD', 'student'.
    nu : float
        Degrees of freedom for Student driver (ignored otherwise).
    mmd_mult : float
        Bandwidth multiplier applied to the MMD score (default 1.0).
    """

    DRIVERS = {
        'log': u_log,
        'DP': u_DP,
        'CRPS': u_CRPS,
        'MMD': u_MMD,
    }

    def __init__(self, driver='log', nu=5.0, mmd_mult=1.0):
        self.driver = driver
        self.nu = nu
        self.mmd_mult = mmd_mult
        self.omega_ = None
        self.phi_ = None
        self.alpha_ = None

    def _score(self, z):
        if self.driver == 'student':
            return u_student(z, self.nu)
        elif self.driver == 'MMD':
            return self.mmd_mult * u_MMD(z)
        else:
            return self.DRIVERS[self.driver](z)

    def run(self, y, omega, phi, alpha, lam0=None):
        """Run the filter forward and return lambda path.

        Parameters
        ----------
        y      : 1-D array of observations
        omega, phi, alpha : filter parameters
        lam0   : initial lambda (default: log variance of y)

        Returns
        -------
        lam : array of shape (len(y)+1,), lambda_0 … lambda_T
        """
        T = len(y)
        lam = np.empty(T + 1)
        lam[0] = np.log(np.var(y)) if lam0 is None else lam0
        for t in range(T):
            z = y[t] * np.exp(-lam[t] / 2.0)
            lam[t + 1] = omega + phi * lam[t] + alpha * self._score(z)
        return lam

    def neg_loglik(self, params, y):
        omega, phi, alpha = params
        lam = self.run(y, omega, phi, alpha)
        ll = _loglik_gaussian(y, lam[:-1])
        return -np.sum(ll)

    def fit(self, y, phi0=0.97, alpha0=0.05):
        """Fit omega, phi, alpha by Gaussian log-likelihood."""
        lam_var = np.log(np.var(y))
        omega0 = lam_var * (1 - phi0)
        x0 = [omega0, phi0, alpha0]
        bounds = [(-10, 10), (0.80, 0.9999), (1e-4, 2.0)]
        res = minimize(self.neg_loglik, x0, args=(y,), method='L-BFGS-B',
                       bounds=bounds, options={'maxiter': 500, 'ftol': 1e-9})
        self.omega_ = res.x[0]
        self.phi_ = res.x[1]
        self.alpha_ = res.x[2]
        return self

    def predict_variance(self, y, omega=None, phi=None, alpha=None):
        """Return in-filter conditional variance sigma_t^2 = exp(lambda_t)."""
        omega = self.omega_ if omega is None else omega
        phi = self.phi_ if phi is None else phi
        alpha = self.alpha_ if alpha is None else alpha
        lam = self.run(y, omega, phi, alpha)
        return np.exp(lam[:-1])


# ---------------------------------------------------------------------------
# Location-driven filter
# ---------------------------------------------------------------------------

class LocationFilter:
    """Score-driven location filter under N(mu_t, sigma^2).

    mu_{t+1} = omega + phi*mu_t + alpha * u_loc(y_t - mu_t, sigma)

    Parameters
    ----------
    driver : str
        One of 'log', 'CRPS', 'DP'.
    """

    LOC_DRIVERS = {
        'log':  u_log_loc,
        'CRPS': u_CRPS_loc,
        'DP':   u_DP_loc,
    }

    def __init__(self, driver='log'):
        self.driver = driver
        self.omega_ = None
        self.phi_ = None
        self.alpha_ = None
        self.sigma_ = None

    def _score(self, r, sigma):
        return self.LOC_DRIVERS[self.driver](r, sigma)

    def run(self, y, omega, phi, alpha, sigma, mu0=None):
        """Run the filter forward.

        Returns mu path of shape (len(y)+1,): mu_0, mu_1, …, mu_T.
        """
        T = len(y)
        mu = np.empty(T + 1)
        mu[0] = np.mean(y) if mu0 is None else mu0
        for t in range(T):
            r = y[t] - mu[t]
            mu[t + 1] = omega + phi * mu[t] + alpha * self._score(r, sigma)
        return mu

    def neg_loglik(self, params, y):
        omega, phi, alpha, log_sigma = params
        sigma = np.exp(log_sigma)
        mu = self.run(y, omega, phi, alpha, sigma)
        r = y - mu[:-1]
        ll = -0.5 * (np.log(2 * np.pi * sigma ** 2) + (r / sigma) ** 2)
        return -np.sum(ll)

    def fit(self, y, phi0=0.90, alpha0=0.20):
        """Fit parameters by Gaussian log-likelihood (same for all drivers)."""
        sigma0 = np.std(y)
        mean0 = np.mean(y)
        omega0 = mean0 * (1 - phi0)
        x0 = [omega0, phi0, alpha0, np.log(sigma0)]
        bounds = [(-1e4, 1e4), (0.50, 0.9999), (1e-4, 5.0), (-10, 10)]
        res = minimize(self.neg_loglik, x0, args=(y,), method='L-BFGS-B',
                       bounds=bounds, options={'maxiter': 1000, 'ftol': 1e-10})
        self.omega_ = res.x[0]
        self.phi_ = res.x[1]
        self.alpha_ = res.x[2]
        self.sigma_ = np.exp(res.x[3])
        return self

    def predict(self, y, omega=None, phi=None, alpha=None, sigma=None, mu0=None):
        """Return in-filter location mu_t path."""
        omega = self.omega_ if omega is None else omega
        phi = self.phi_ if phi is None else phi
        alpha = self.alpha_ if alpha is None else alpha
        sigma = self.sigma_ if sigma is None else sigma
        return self.run(y, omega, phi, alpha, sigma, mu0=mu0)
