"""Figures and the console summary."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config


def plot_correlation_heatmap(corr, path):
    fig, ax = plt.subplots(figsize=(10, 8.5))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=90, fontsize=7)
    ax.set_yticks(range(len(corr)), corr.index, fontsize=7)
    fig.colorbar(im, shrink=0.75, label="correlation")
    ax.set_title("Daily-return correlation matrix (5y)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_drawdown(dd_series, path):
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.fill_between(dd_series.index, dd_series.values * 100, 0, color="#b03a2e", alpha=0.55)
    ax.set_ylabel("drawdown %")
    ax.set_title("Equal-weight portfolio drawdown (5y)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_rolling_sharpe(rs, path):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(rs.index, rs.values, color="#2b6cb0", lw=1.2)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_ylabel("Sharpe")
    ax.set_title(f"Rolling {63}-day Sharpe ratio")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_mc_distribution(pnl, risk_metrics, path, horizon_days, paths):
    fig, ax = plt.subplots(figsize=(9, 5))
    losses_pct = -np.asarray(pnl) * 100
    ax.hist(losses_pct, bins=120, color="#5d6d7e", alpha=0.8)
    for key, color in [("var_95", "#e67e22"), ("var_99", "#c0392b")]:
        v = risk_metrics[key] * 100
        ax.axvline(v, color=color, lw=2,
                   label=f"{key.upper()} = -{v:.2f}%")
    ax.set_xlabel(f"horizon loss over {horizon_days} trading days (%)")
    ax.set_ylabel("paths")
    ax.set_title(f"Monte Carlo horizon P&L ({paths:,} paths)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def write_summary(payload):
    config.REPORTS_DIR.mkdir(exist_ok=True)

    def default(o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, pd.Timestamp):
            return str(o.date())
        raise TypeError(type(o))

    with open(config.REPORTS_DIR / "summary.json", "w") as fh:
        json.dump(payload, fh, indent=2, default=default)


def print_report(stats, mc_norm, mc_t, hist_var, stress_df, source):
    print(f"\n=== Portfolio risk report (data source: {source}) ===")
    print(f"assets={len(config.UNIVERSE)}  weights=equal  "
          f"book=${config.PORTFOLIO_VALUE:,.0f}")
    print(f"CAGR={stats['cagr']:.2%}  vol={stats['annual_vol']:.2%}  "
          f"Sharpe={stats['sharpe']:.2f}  maxDD={stats['max_drawdown']:.2%} "
          f"({stats['dd_peak']} -> {stats['dd_trough']})")

    print("\nMonte Carlo VaR/CVaR on the horizon P&L:")
    for label, m in [
        ("normal draws", mc_norm),
        (f"student-t df={config.T_DIST_DF} draws", mc_t),
    ]:
        parts = []
        for k, v in m.items():
            name = "VaR" if k.startswith("var") else "CVaR"
            parts.append(f"{name}{k.split('_')[1]}={v*100:.2f}%")
        print(f"  {label}: " + "  ".join(parts))
    print(f"  historical rolling window: "
          + "  ".join(f"{k.upper()}={v*100:.2f}%" for k, v in hist_var.items()))

    print("\nStress tests:")
    if stress_df is not None and not stress_df.empty:
        for _, row in stress_df.iterrows():
            print(f"  {row['scenario']:20s} {row['start']} -> {row['end']}  "
                  f"{row['portfolio_return']*100:+.1f}%  (${row['pnl_usd']:+,.0f})")


def top_drawdown_assets(closes, path=None):
    from .analytics import per_asset_max_drawdown
    return per_asset_max_drawdown(closes)
