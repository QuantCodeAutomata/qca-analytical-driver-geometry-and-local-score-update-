# Repository Knowledge: Score-Driven Filter Geometry and Robustness

## Project Overview
This repository implements a comprehensive study of score-driven (GAS/DCS) filters
for volatility forecasting and robustness analysis.  The focus is on comparing four
score drivers—Gaussian log, density-power (DP), CRPS, and MMD—across synthetic and
real financial/hydrological data sets.

## Core Score Drivers (Gaussian Working-Scale Family, z = y·exp(-λ/2))

| Driver | Formula | J | K |
|--------|---------|---|---|
| Log | u_log(z) = z²−1 | 0.500 | 0.500 |
| DP (b=0.15) | u_DP(z) = (1+b)·c_b·exp(−b·z²/2)·(z²−1) + κ_b | 0.357 | 0.268 |
| CRPS | u_CRPS(z) = 1/√π − √(2/π)·exp(−z²/2) | 0.071 | 0.012 |
| MMD (ℓ=2) | u_MMD(z) = 2·(2/√5)·exp(−z²/10)·(z²/25−1/5) + 2·2/(6)^(3/2) | 0.034 | 0.002 |

Constants: b=0.15, c_b=(2π)^(−b/2), κ_b=b·c_b·(1+b)^(−0.5)

## Filter Recursion
λ_{t+1} = ω + φ·λ_t + α·u(z_t),  z_t = y_t·exp(−λ_t/2)

Default parameters (unless experiment overrides): α=0.05, φ=0.97, ω=0.

## Directory Structure
- `exp/`      — experiment scripts exp_01 … exp_14
- `src/`      — shared utilities: score_drivers.py, filters.py
- `tests/`    — pytest test files, one per experiment
- `results/`  — PNG figures, CSV tables, RESULTS.md
- `data/`     — NOT committed (raw market data fetched at runtime)

## Key Implementation Notes
- Market data: use yfinance; 3 representative tickers (^GSPC, ^FTSE, ^N225) for speed
- Arch package: use `arch_model(y*100, vol='GARCH', ...)` to avoid DataScaleWarning
- Nile dataset: `from statsmodels.datasets import nile; nile.load_pandas().data`
  - 100 annual flow observations, 1871-1970 (units: 10^8 m³)
  - Training split: 1871-1944 (74 obs, indices 0-73)
  - Test split: 1945-1970 (26 obs, indices 74-99)
- Student-t score driver: u_student(z,ν) = (ν+1)·z²/(ν+z²) − 1
- Filter fitting: scipy.optimize.minimize with 'L-BFGS-B' or 'Nelder-Mead'

## Validation Targets
- Exp 13A adaptation delay: Log ~29.4 dates, DP ~42.8 dates (difference ~13.4, SE ~0.117)
- Exp 13A excess loss: Log ~26.2, DP ~41.6
- Exp 13B RMSE deterioration: Gaussian-DP ~0.016, Student-log ~0.047, Student-DP ~0.023
- Exp 14 displacement/σ̂: Log ~1.615, CRPS ~0.465, DP ~0.066

## Test Strategy
- Tests use synthetic data (no network calls) to stay fast
- Each test file covers: score function bounds, filter dimensions, simulation shapes
- Run with: `python -m pytest tests/ -v --tb=short`

## Git Remote
https://github.com/QuantCodeAutomata/qca-analytical-driver-geometry-and-local-score-update-

---

## Experiments 4–7 (added by agent, exp/ implemented inline – src/ is empty)

### src/ Status
`src/` is currently **empty**.  All score functions and helper routines for
experiments 4–7 are implemented **inline** in each experiment file.

### Corrected K_S values (Gauss–Hermite quadrature, 200 nodes)
```
K_log  = 2.000  (analytic: Var[Z²-1] = E[(Z²-1)²] = 2)
K_DP   ≈ 1.074  (numerical)
K_CRPS ≈ 0.049  (numerical)
```
(The K column in the table above is wrong for Log/DP/CRPS; these are correct.)

### h-Standardised Residual (Experiments 5–7)
`z_std = y * exp(-lambda/2) / sqrt(h)` — this ensures E[u_S(z_std)] = 0 for
all h when y ~ N(0, h*exp(lambda)).  In Exp 4 (h=1 implicitly), z = y*exp(-lambda/2).

### Exp 4 — MC Filtering Robustness
- 5 filters: Gauss-log, Gauss-DP, Gauss-CRPS, Gauss-MMD, Student-t6
- Filter update: `alpha/J_S * u_S(z)`;  alpha=0.1, varphi=0.97, omega=0
- J_T6 ≈ 0.333 (computed numerically under t6 via scipy.integrate.quad)
- numba @jit (no cache=True) for filter loops; ~1.2 s for R=500, T=4000
- --small flag: R=500; full run: R=20000
- Key finding: Gauss-log contaminated RMSE ≈ 10× clean; bounded criteria robust

### Exp 5 — Diffusion-Limit Diagnostics
- alpha_h = sqrt(h); h ∈ {2^{-k}, k=1..7}; R=8000, T=200
- Diagnostic: h^{-1}*E[(Δλ)²] → K_S (ratios near 1 for small h)
- Log-log slope near 0 (flat convergence), confirming diffusion limit holds

### Exp 6 — Non-Centred Mean-Flow (t5 DGP)
- O(h) gain: `delta_lambda = h*u_S(z_std) - kappa*h*lambda`
- t5/sqrt(5/3) residuals (unit variance); SCALE_T5 = sqrt(5/3) ≈ 1.291
- Mean field m_log(λ) = exp(-λ)−1 (zero at λ=0 even under t5, because Var[Z]=1)
- Fixed points: DP at λ*≈−0.175, CRPS at λ*≈−0.246 (t5 misspecification)

### Exp 7 — Local OU Tracking
- Tracking error: `e_{k+1} = e_k + ρ√h*u_S(Z*exp(-e_k/2)) - Γ√h*W_k`
- **CORRECTED theoretical variance** (the problem description formula is wrong):
  ```
  V^bullet_S = (ρ²·K_S + Γ²) / (4·ρ·J_S)
  ```
  Derivation: OU mean-reversion coefficient b_S = 2ρJ_S because
  B_S = (1/2)·E_N[Z²·u_S(Z)] = 2·J_S  (Gaussian integration-by-parts identity).
  The problem's formula `(ρ²K_S/J_S²+Γ²)/(2ρJ_S·M)` gives ratios ≈ 0.46, not 1.
- Convergence at h=2^{-12}: ratios log≈1.01, DP≈1.00, CRPS≈1.02
- Companion KS (h=2^{-16}, R=3000): log≈0.023, DP≈0.021, CRPS≈0.016

### Running All Tests
```bash
cd /workspace/project && python -m pytest tests/ -v --tb=short
# Expected: 143 passed (includes pre-existing tests for exp 12-14)
```
