"""End-to-end portfolio risk analysis.

Usage:
    .venv/bin/python run_analysis.py                 # Yahoo Finance data
    .venv/bin/python run_analysis.py --offline       # synthetic market
"""
import argparse

import numpy as np
import pandas as pd

from src import analytics, config, market_data, monte_carlo, report, stress


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--offline", action="store_true", help="Skip the network")
    ap.add_argument("--paths", type=int, default=config.MC_PATHS)
    ap.add_argument("--horizon", type=int, default=config.MC_HORIZON_DAYS)
    args = ap.parse_args()

    if args.offline:
        closes_full, source = market_data._synthetic_prices(config.FETCH_PERIOD), "synthetic"
    else:
        closes_full, source = market_data.fetch_prices()

    tickers = [t for t in config.UNIVERSE if t in closes_full.columns]
    n_years_back = int(len(pd.bdate_range(end=closes_full.index[-1], periods=252 * config.ANALYSIS_YEARS)))
    closes_5y = closes_full.iloc[-n_years_back:]
    print(f"[data] source={source} rows={len(closes_5y):,} "
          f"({closes_5y.index[0].date()} to {closes_5y.index[-1].date()}), "
          f"assets={len(tickers)}")

    rets = closes_5y[tickers].pct_change().dropna()
    log_rets = np.log1p(rets)
    weights = np.full(len(tickers), 1.0 / len(tickers))

    port_ret = analytics.portfolio_daily_returns(closes_5y, tickers)
    port_value = (1 + port_ret).cumprod() * 100
    dd = analytics.drawdown_series(port_value)
    rs = analytics.rolling_sharpe(port_ret)
    corr = analytics.correlation_matrix(closes_5y[tickers])
    stats = analytics.summary_stats(port_ret, dd)
    worst_assets = analytics.per_asset_max_drawdown(closes_5y[tickers]).head(5)

    pnl_norm, value_paths = monte_carlo.simulate_horizon(
        log_rets, weights, args.paths, args.horizon,
        seed=config.RNG_SEED, t_df=None,
    )
    mc_norm = monte_carlo.var_cvar(pnl_norm, config.CONFIDENCE_LEVELS)

    pnl_t, _ = monte_carlo.simulate_horizon(
        log_rets, weights, args.paths, args.horizon,
        seed=config.RNG_SEED + 1, t_df=config.T_DIST_DF,
    )
    mc_t = monte_carlo.var_cvar(pnl_t, config.CONFIDENCE_LEVELS)

    hist_risk, hist_pnl = monte_carlo.historical_var(
        log_rets, weights, args.horizon, config.CONFIDENCE_LEVELS
    )
    dd_dist = monte_carlo.drawdown_distribution(value_paths)

    stress_df = stress.run_stress_tests(closes_full, weights, tickers)

    config.REPORTS_DIR.mkdir(exist_ok=True)
    report.plot_correlation_heatmap(corr, config.REPORTS_DIR / "correlation_heatmap.png")
    report.plot_drawdown(dd, config.REPORTS_DIR / "portfolio_drawdown.png")
    report.plot_rolling_sharpe(rs, config.REPORTS_DIR / "rolling_sharpe.png")
    report.plot_mc_distribution(
        pnl_t, mc_t, config.REPORTS_DIR / "mc_var_distribution.png",
        args.horizon, args.paths,
    )

    report.print_report(stats, mc_norm, mc_t, hist_risk, stress_df, source)
    print("\nWorst single-asset drawdowns over the window:")
    for ticker, mdd in worst_assets.items():
        print(f"  {ticker:5s} {mdd*100:.1f}%")

    print(f"\nMonte Carlo path max-drawdown distribution "
          f"(median={np.median(dd_dist)*100:.1f}%, "
          f"worst={dd_dist.min()*100:.1f}%)")

    payload = {
        "data_source": source,
        "window": [str(closes_5y.index[0].date()), str(closes_5y.index[-1].date())],
        "assets": tickers,
        "portfolio_stats": stats,
        "monte_carlo_normal": mc_norm,
        f"monte_carlo_student_t_df_{config.T_DIST_DF}": mc_t,
        "historical_var_cvar": hist_risk,
        "mc_path_drawdown_median": float(np.median(dd_dist)),
        "stress_tests": None if stress_df is None else stress_df.to_dict("records"),
        "worst_asset_drawdowns": worst_assets.to_dict(),
    }
    report.write_summary(payload)
    print(f"[done] figures and summary.json -> {config.REPORTS_DIR}/")


if __name__ == "__main__":
    main()
