"""Monte Carlo simulation of the portfolio horizon P&L."""
import numpy as np


def simulate_horizon(daily_log_returns, weights, paths, horizon, seed=42, t_df=None):
    """Simulate horizon portfolio log returns.

    Draws multivariate normal or Student-t daily returns with the
    empirical covariance, compounds them over the horizon, and applies
    the weights. Returns (portfolio_horizon_log_returns, value_paths).
    """
    rng = np.random.default_rng(seed)
    mu = daily_log_returns.mean().values
    cov = daily_log_returns.cov().values
    k = len(mu)
    chol = np.linalg.cholesky(cov)

    if t_df:
        # Scale so the Student-t draw keeps the covariance of the data.
        scale = np.sqrt((t_df - 2) / t_df)
        normals = rng.standard_t(t_df, size=(paths, horizon, k)) * scale
    else:
        normals = rng.standard_normal((paths, horizon, k))
    daily = normals @ chol.T + mu

    asset_cum = daily.sum(axis=1)                      # (paths, k)
    port_cum = asset_cum @ weights                     # (paths,)

    port_daily = daily @ weights                       # (paths, horizon)
    value_paths = 100 * np.exp(np.cumsum(port_daily, axis=1))
    return port_cum, value_paths


def var_cvar(pnl, levels=(0.95, 0.99)):
    """Historical VaR and CVaR on the simulated P&L distribution."""
    out = {}
    losses = -pnl
    for level in levels:
        var = np.quantile(losses, level)
        cvar = losses[losses >= var].mean()
        out[f"var_{int(level*100)}"] = float(var)
        out[f"cvar_{int(level*100)}"] = float(cvar)
    return out


def historical_var(daily_log_returns, weights, horizon, levels=(0.95, 0.99)):
    """Rolling historical horizon P&L from the observed series."""
    port = daily_log_returns @ weights
    windows = len(port) - horizon + 1
    pnl = np.array([
        port.iloc[i : i + horizon].sum() for i in range(windows)
    ])
    return var_cvar(pnl, levels), pnl


def drawdown_distribution(value_paths):
    """Max drawdown per simulated path."""
    running_max = np.maximum.accumulate(value_paths, axis=1)
    dd = value_paths / running_max - 1.0
    return dd.min(axis=1)
