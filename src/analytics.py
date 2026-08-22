"""Rolling Sharpe, drawdown series, and per-asset statistics."""
import numpy as np
import pandas as pd

from . import config


def portfolio_daily_returns(closes, tickers):
    weights = np.full(len(tickers), 1.0 / len(tickers))
    return closes[tickers].pct_change().dropna() @ weights


def rolling_sharpe(port_returns, window=config.ROLLING_WINDOW_DAYS, rf_annual=config.RF_ANNUAL):
    excess = port_returns - rf_annual / 252
    mean = excess.rolling(window).mean()
    std = excess.rolling(window).std()
    return (mean / std) * np.sqrt(252)


def drawdown_series(port_value):
    running_max = port_value.cummax()
    return port_value / running_max - 1.0


def max_drawdown_stats(dd):
    trough = dd.idxmin()
    peak = dd.loc[:trough].idxmax()
    return {
        "max_drawdown": float(dd.min()),
        "dd_peak": str(peak.date()),
        "dd_trough": str(trough.date()),
    }


def summary_stats(port_returns, dd):
    years = len(port_returns) / 252
    total_growth = (1 + port_returns).prod()
    cagr = float(total_growth ** (1 / years) - 1)
    vol = float(port_returns.std() * np.sqrt(252))
    sharpe = float((cagr - config.RF_ANNUAL) / vol)
    stats = {"cagr": cagr, "annual_vol": vol, "sharpe": sharpe}
    stats.update(max_drawdown_stats(dd))
    return stats


def per_asset_max_drawdown(closes):
    out = {}
    for ticker in closes.columns:
        dd = drawdown_series(closes[ticker].dropna())
        out[ticker] = float(dd.min())
    return pd.Series(out).sort_values()


def correlation_matrix(closes):
    return closes.pct_change().dropna().corr()
