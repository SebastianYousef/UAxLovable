"""Price loading with an on-disk cache.

Hackathon wifi dies, Yahoo rate-limits, the demo must still run. Every
successful download is written to .cache/ and reused from then on.
"""
from __future__ import annotations

import os

import pandas as pd

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache")


def _cache_path(tickers: tuple[str, ...], start: str) -> str:
    key = "_".join(sorted(tickers)) + "_" + start
    safe = "".join(c if c.isalnum() or c in "._-" else "-" for c in key)
    return os.path.join(CACHE_DIR, safe[:120] + ".csv")


def fetch_prices(tickers, start: str = "2005-01-01") -> pd.DataFrame:
    """Adjusted close prices, one column per ticker. Cached to disk."""
    tickers = tuple(t.strip().upper() for t in tickers if t.strip())
    if not tickers:
        raise ValueError("No tickers given")

    path = _cache_path(tickers, start)
    if os.path.exists(path):
        return pd.read_csv(path, index_col=0, parse_dates=True)

    import yfinance as yf

    raw = yf.download(list(tickers), start=start, auto_adjust=True,
                      progress=False, group_by="column")
    if raw is None or raw.empty:
        raise RuntimeError(f"No data returned for {tickers}")

    if isinstance(raw.columns, pd.MultiIndex):
        close = raw.xs("Close", axis=1, level=0)
    else:
        close = raw[["Close"]].copy()
        close.columns = [tickers[0]]

    close = close.dropna(how="all").ffill()
    missing = [t for t in tickers if t not in close.columns]
    if missing:
        raise RuntimeError(f"Unknown ticker(s): {', '.join(missing)}")

    os.makedirs(CACHE_DIR, exist_ok=True)
    close.to_csv(path)
    return close
