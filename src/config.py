"""Universe, weights, and simulation settings."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / "reports"

# Equal-weight universe of 24 liquid US equities and ETFs.
UNIVERSE = [
    "SPY", "QQQ", "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META",
    "TSLA", "JPM", "V", "JNJ", "PG", "XOM", "KO", "PEP",
    "WMT", "HD", "DIS", "NFLX", "AMD", "CSCO", "ADBE", "CVX",
]
PORTFOLIO_VALUE = 1_000_000.0

FETCH_PERIOD = "10y"      # longer window so stress scenarios stay in range
ANALYSIS_YEARS = 5        # analytics run on the trailing window
MC_PATHS = 10_000
MC_HORIZON_DAYS = 21      # one trading month
CONFIDENCE_LEVELS = (0.95, 0.99)
T_DIST_DF = 5             # fat-tailed alternative to the normal draw
ROLLING_WINDOW_DAYS = 63
RF_ANNUAL = 0.03
RNG_SEED = 42

# Historical stress windows: (label, start, end).
STRESS_WINDOWS = [
    ("covid_crash_2020", "2020-02-19", "2020-03-23"),
    ("q4_2018_selloff", "2018-09-20", "2018-12-24"),
    ("bear_market_2022", "2022-01-03", "2022-10-12"),
]

# Approximate per-asset shocks for the offline fallback generator,
# expressed as cumulative log returns over the 2020 crash window.
FALLBACK_COVID_SHOCKS = {
    "SPY": -0.31, "QQQ": -0.27, "AAPL": -0.24, "MSFT": -0.26, "GOOGL": -0.28,
    "AMZN": -0.12, "NVDA": -0.32, "META": -0.33, "TSLA": -0.50, "JPM": -0.37,
    "V": -0.32, "JNJ": -0.15, "PG": -0.11, "XOM": -0.55, "KO": -0.15,
    "PEP": -0.14, "WMT": -0.08, "HD": -0.24, "DIS": -0.38, "NFLX": -0.16,
    "AMD": -0.38, "CSCO": -0.24, "ADBE": -0.27, "CVX": -0.50,
}
