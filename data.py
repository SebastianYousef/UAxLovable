"""Price loading with an on-disk cache.

Hackathon wifi dies, Yahoo rate-limits, the demo must still run. Every
successful download is written to .cache/ and reused from then on.
"""
from __future__ import annotations

import os

import numpy as np
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


def repair_scale_breaks(series: pd.Series) -> pd.Series:
    """Undo mid-history unit changes in a price or FX series.

    Yahoo quotes some FX pairs per unit for part of their history and per 100
    units for the rest -- JPYSEK=X jumps by exactly 100x on 2017-11-10. The
    jump is silent and turns any portfolio holding that currency into
    nonsense.

    Each observation is snapped back to the series' dominant order of
    magnitude rather than repaired cumulatively, so several breaks in either
    direction all resolve to the same scale. A value must sit more than 3.2x
    away from the median to move at all, which no exchange rate does over the
    horizons here, so genuine data is left alone.
    """
    clean = series.dropna()
    positive = clean[clean > 0]
    if len(positive) < 2:
        return series

    magnitude = np.log10(positive)
    offset = (magnitude - magnitude.median()).round()
    if not offset.any():
        return series

    return clean / 10 ** offset.reindex(clean.index).fillna(0)


def sanity_check_fx(fx: pd.DataFrame, max_daily_move: float = 0.25) -> pd.DataFrame:
    """Repair scale breaks, then refuse to return an implausible FX series."""
    if fx.empty:
        return fx

    repaired = fx.apply(repair_scale_breaks)

    worst = repaired.pct_change().abs().max()
    broken = worst[worst > max_daily_move]
    if not broken.empty:
        names = ", ".join(f"{p} ({worst[p]:.0%} in a day)" for p in broken.index)
        raise ValueError(f"Implausible exchange-rate history: {names}")

    return repaired
