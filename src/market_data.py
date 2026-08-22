"""Price history from Yahoo Finance, with a cached and an offline path."""
import numpy as np
import pandas as pd

from . import config


def fetch_prices(period=config.FETCH_PERIOD):
    """Return a close-price frame indexed by trading day.

    Tries Yahoo Finance first and caches the result under data/.
    Falls back to a synthetic correlated market when the network is down.
    """
    cache = config.DATA_DIR / "prices.csv"
    config.DATA_DIR.mkdir(exist_ok=True)
    try:
        import yfinance as yf
        raw = yf.download(
            config.UNIVERSE, period=period, auto_adjust=True,
            progress=False, threads=False,
        )
        closes = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
        available = [t for t in config.UNIVERSE if t in closes.columns]
        if len(available) < 20:
            raise RuntimeError(f"only {len(available)} tickers returned")
        closes = closes[available].dropna(how="all").ffill().dropna()
        if len(closes) < 400:
            raise RuntimeError(f"only {len(closes)} rows returned")
        closes.to_csv(cache)
        return closes, "yfinance"
    except Exception as exc:
        if cache.exists():
            cached = pd.read_csv(cache, index_col=0, parse_dates=True)
            if len(cached) >= 400:
                return cached[config.UNIVERSE], "cache"
        print(f"[data] yahoo fetch failed ({type(exc).__name__}: {exc}); "
              "using the synthetic fallback")
        return _synthetic_prices(period), "synthetic"


def _synthetic_prices(period):
    """Correlated geometric random walk with per-asset drift and vol.

    This is a demo substitute, not market data. The 2020 crash is
    stamped onto the generated series so stress tests stay meaningful.
    """
    rng = np.random.default_rng(config.RNG_SEED)
    days = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=2520)
    n = len(days)
    k = len(config.UNIVERSE)

    vols = np.array([0.16, 0.20] + [0.28] * 8 + [0.22] * 14)[:k]
    vols = np.resize(vols, k)
    drifts = rng.uniform(0.04, 0.14, k)
    base_corr = 0.55
    corr = np.full((k, k), base_corr) + np.eye(k) * (1 - base_corr)
    chol = np.linalg.cholesky(corr)

    daily_vol = vols / np.sqrt(252)
    shocks = rng.standard_normal((n, k)) @ chol.T
    rets = drifts[np.newaxis, :] / 252 - 0.5 * daily_vol**2 + daily_vol * shocks

    # Stamp the 2020 crash window onto the synthetic series.
    mask = (days >= "2020-02-19") & (days <= "2020-03-23")
    crash_days = days[mask]
    if len(crash_days) > 5:
        target = np.array([config.FALLBACK_COVID_SHOCKS.get(t, -0.3) for t in config.UNIVERSE])
        idx_span = np.linspace(0, len(crash_days) - 1, len(crash_days))
        per_day = target / len(crash_days)
        rets[mask] = per_day + rng.normal(0, daily_vol.mean() * 0.5, (mask.sum(), k))

    prices = 100 * np.exp(np.cumsum(rets, axis=0))
    return pd.DataFrame(prices, index=days, columns=config.UNIVERSE)
