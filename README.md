# Score-Driven Filter Geometry and Robustness

A systematic empirical study of score-driven (GAS/DCS) filters under four score
drivers: Gaussian log, density-power (DP), CRPS, and MMD.

## Background

Score-driven models update a time-varying parameter θ_t via a scaled score of the
conditional log-likelihood:

```
θ_{t+1} = ω + φ·θ_t + α·S_t(y_t, θ_t)
```

where S_t is a (possibly robust) score of an objective function.  This project
examines how the choice of objective—standard log, density-power, CRPS, or MMD—
shapes the filter geometry (information J and curvature K), adaptation speed, and
robustness to contamination.

## Core Score Drivers (Gaussian Working-Scale Family)

For the **log-variance** parameter λ = log σ², working-scale z = y·exp(−λ/2):

| Driver | u(z) | J | K |
|--------|------|---|---|
| Gaussian log | z²−1 | 0.500 | 0.500 |
| Density-Power (β=0.15) | (1+β)·c_β·e^(−βz²/2)·(z²−1)+κ_β | 0.357 | 0.268 |
| CRPS | 1/√π − √(2/π)·e^(−z²/2) | 0.071 | 0.012 |
| MMD (ℓ=2) | 2·(2/√5)·e^(−z²/10)·(z²/25−1/5)+const | 0.034 | 0.002 |

## Experiments

### Exp 01–05: Analytical Driver Geometry
Numerical verification of J and K for all four drivers; pseudo-true targets;
finite-sample MLE vs score-driven comparison on Gaussian GARCH data.

### Exp 06–09: Simulation Calibration
Systematic RMSE comparison of Log/DP/CRPS/MMD filters across DGP mis-specifications
(Student-t innovations, skewed innovations, GARCH effects).

### Exp 10–11: Empirical Evaluation
Factorial experiment on 12 international equity markets; QLIKE, VaR coverage, and
Diebold-Mariano tests.  GJR-GARCH and HAR benchmarks.

### Exp 12: Sensitivity Grids
Four sensitivity sweeps:
- **Grid A** — GARCH-N vs GARCH-t on three representative markets
- **Grid B** — Student ν sweep: ν ∈ {4, 5, 6, 8, 10}
- **Grid C** — Train-fraction sweep: frac ∈ {0.60, 0.70, 0.80}
- **Grid D** — MMD bandwidth multiplier: mult ∈ {0.5, 1.0, 2.0}

### Exp 13: Robustness–Adaptation Trade-off
- **Part A** — Permanent variance break (1→9), measuring 90th-percentile adaptation
  delay and cumulative oracle-excess log-risk across 500 replications.
- **Part B** — One-sided jump contamination at 2% rate; jump-minus-clean RMSE
  deterioration for frozen-fit vs re-fit estimators.

### Exp 14: Gaussian Location Filtering on the Nile River
Score-driven location models fit to 74 years of Nile annual flow data.
Three location drivers (Log, CRPS, DP) are compared for outlier sensitivity: a
+6σ shock is injected in 1945 and the single-period displacement of each frozen
filter is reported.

## Quick Start

```bash
pip install -r requirements.txt

# Run individual experiments
python exp/exp_12_sensitivity_grids.py
python exp/exp_13_robustness_adaptation.py
python exp/exp_14_nile_location.py

# Run all tests
python -m pytest tests/ -v --tb=short
```

## Repository Layout

```
.
├── exp/                          # Experiment scripts
│   ├── exp_4_mc_filtering.py     # MC filter comparison (clean vs contaminated)
│   ├── exp_5_diffusion_limit.py  # Diffusion limit and K_S diagonal
│   ├── exp_6_mean_flow.py        # Mean-flow drift and fixed points
│   ├── exp_7_local_ou.py         # Local OU tracking error and variance ratios
│   ├── exp_12_sensitivity_grids.py    # GARCH/ν/frac/bandwidth sensitivity
│   ├── exp_13_robustness_adaptation.py  # Variance break and jump misspecification
│   └── exp_14_nile_location.py   # Gaussian location filter on Nile data
├── src/                          # Shared utilities
│   ├── score_drivers.py          # u_log, u_DP, u_CRPS, u_MMD, location variants
│   └── filters.py                # ScaleDrivenFilter, LocationFilter
├── tests/                        # pytest tests (143 items)
│   ├── test_exp_4.py
│   ├── test_exp_5.py
│   ├── test_exp_6.py
│   ├── test_exp_7.py
│   ├── test_exp_12.py
│   ├── test_exp_13.py
│   └── test_exp_14.py
├── results/                      # Figures (PNG), tables (CSV), RESULTS.md
├── data/                         # Raw market data (not committed)
├── AGENTS.md                     # Persistent memory for AI agents
├── README.md
└── requirements.txt
```

## Results

See [`results/RESULTS.md`](results/RESULTS.md) for a full summary of all
experiment outputs.

## Requirements

- Python ≥ 3.10
- See `requirements.txt` for package list (numpy, pandas, scipy, matplotlib,
  seaborn, statsmodels, arch, yfinance, numba, pytest)
