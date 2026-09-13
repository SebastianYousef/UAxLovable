"""Codex-owned market-data adapter. Existing data.py remains Claude-owned."""
from __future__ import annotations

import pandas as pd

from data import fetch_prices


def convert_to_base(prices: pd.DataFrame, currencies: dict[str, str],
                    fx: pd.DataFrame, base: str) -> pd.DataFrame:
    """Convert adjusted prices; never infer listing currency from country."""
    output = prices.copy()
    minor_units = {"GBp": ("GBP", 0.01), "GBX": ("GBP", 0.01),
                   "ZAc": ("ZAR", 0.01), "ILA": ("ILS", 0.01)}
    for ticker in prices:
        currency = currencies.get(ticker)
        if not currency:
            raise ValueError(f"Cannot determine the listing currency for {ticker}.")
        currency, scale = minor_units.get(currency, (currency, 1.0))
        output[ticker] = prices[ticker] * scale
        if currency != base:
            pair = f"{currency}{base}=X"
            if pair not in fx or fx[pair].dropna().empty:
                raise ValueError(f"Missing FX history for {pair}; cannot compare in {base}.")
            # Use only past FX observations and tolerate a short holiday gap.
            rates = fx[pair].dropna().sort_index().reindex(
                prices.index, method="ffill", tolerance=pd.Timedelta(days=5))
            output[ticker] *= rates
    return output


def load_market(tickers: tuple[str, ...], start: str, base: str) -> tuple[pd.DataFrame, dict]:
    import yfinance as yf

    prices = fetch_prices(tickers, start).reindex(columns=list(tickers))
    currencies = {}
    for ticker in tickers:
        try:
            currency = yf.Ticker(ticker).fast_info["currency"]
        except Exception as exc:
            raise ValueError(f"Could not verify {ticker}'s listing currency. Retry the data request.") from exc
        if not currency:
            raise ValueError(f"No currency supplied for {ticker}.")
        currencies[ticker] = currency
    major = {"GBp": "GBP", "GBX": "GBP", "ZAc": "ZAR", "ILA": "ILS"}
    pairs = sorted({f"{major.get(c, c)}{base}=X" for c in currencies.values()
                    if major.get(c, c) != base})
    fx = fetch_prices(tuple(pairs), start) if pairs else pd.DataFrame()
    return convert_to_base(prices, currencies, fx, base), currencies
