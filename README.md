# Equity Portfolio Risk Engine

Monte Carlo VaR/CVaR and drawdown analytics for an equal-weight
24-asset equity portfolio, computed on real Yahoo Finance price history.

## Results (2021-08 to 2026-08 window, $1M book)

Portfolio: CAGR 17.16%, annual vol 18.19%, Sharpe 0.78,
max drawdown -28.62% (2021-08-17 peak to 2022-10-11 trough).

Monte Carlo VaR/CVaR on the one-month horizon P&L (10,000 paths):

| Method | VaR 95 | CVaR 95 | VaR 99 | CVaR 99 |
|---|---|---|---|---|
| Multivariate normal | 7.60% | 9.88% | 11.39% | 13.10% |
| Student-t (df=5) | 7.60% | 9.83% | 11.45% | 13.09% |
| Historical rolling windows | 8.60% | 11.19% | 12.74% | 14.16% |

Stress tests apply each scenario's realized per-asset returns to the
current equal-weight book:

```
covid_crash_2020   2020-02-19 -> 2020-03-23  -30.2%  ($-301,776)
bear_market_2022   2022-01-03 -> 2022-10-12  -26.1%  ($-261,310)
q4_2018_selloff    2018-09-20 -> 2018-12-24  -19.9%  ($-198,581)
```

Worst single-asset drawdowns over the trailing window:
META -76.7%, NFLX -75.9%, TSLA -73.6%.

## Layout

```
run_analysis.py          entry point
src/config.py            universe, weights, scenario windows, MC settings
src/market_data.py       Yahoo Finance fetch, disk cache, offline fallback
src/monte_carlo.py       correlated normal/Student-t paths, VaR/CVaR
src/analytics.py         rolling Sharpe, drawdowns, correlation matrix
src/stress.py            historical shock scenarios
src/report.py            figures and summary.json
reports/                 generated figures and summary.json
```

## Setup

```
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Run

```
.venv/bin/python run_analysis.py              # live Yahoo Finance data
.venv/bin/python run_analysis.py --offline    # synthetic market, no network
```

The script downloads 10 years of adjusted closes so the stress windows
stay in range, runs all analytics on the trailing 5 years, simulates
10,000 correlated paths over a 21-day horizon, writes four figures and
`summary.json` into `reports/`, and prints the console report.

## Method notes

- Simulation draws daily asset returns with the empirical covariance
  (Cholesky factorization); the Student-t variant inflates tails at
  df=5 with a variance correction.
- Path max-drawdown distribution comes from the same simulated value
  paths (median path DD -4.1%, worst path DD -17.6%).
- When Yahoo Finance is unreachable, the engine falls back to a seeded
  synthetic market with the 2020 crash stamped onto it, so every code
  path stays testable offline.
