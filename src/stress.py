"""Historical stress scenarios applied to the current portfolio."""
import numpy as np
import pandas as pd

from . import config


def run_stress_tests(full_closes, weights, tickers, portfolio_value=config.PORTFOLIO_VALUE):
    """Apply each scenario's cumulative per-asset returns to today's book.

    Returns a DataFrame sorted by loss.
    """
    rows = []
    for label, start, end in config.STRESS_WINDOWS:
        window = full_closes.loc[start:end]
        if len(window) < 5:
            continue
        cum = window.iloc[-1] / window.iloc[0] - 1.0     # simple per-asset shock
        cum = cum.reindex(tickers).fillna(0.0).values
        pnl = float(np.dot(weights, cum)) * portfolio_value
        rows.append({
            "scenario": label,
            "start": start,
            "end": end,
            "portfolio_return": float(np.dot(weights, cum)),
            "pnl_usd": pnl,
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("pnl_usd").reset_index(drop=True)
    return df


def covid_fallback_pnl(weights, portfolio_value=config.PORTFOLIO_VALUE):
    """Shock vector for the synthetic market when real history is absent."""
    shocks = np.array([config.FALLBACK_COVID_SHOCKS.get(t, -0.3) for t in config.UNIVERSE])
    return float(np.dot(weights, shocks)) * portfolio_value
