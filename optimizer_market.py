"""Codex-owned market-data adapter. Existing data.py remains Claude-owned."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from data import fetch_prices, sanity_check_fx


def _listing_currencies() -> dict[str, str]:
    """Read listing metadata independently of the directory launching the app."""
    try:
        universe = pd.read_csv(Path(__file__).resolve().with_name("universe.csv"),
                               dtype=str, keep_default_na=False)
    except FileNotFoundError:
        return {}
    if not {"ticker", "currency"}.issubset(universe.columns):
        return {}
    return {ticker.strip().upper(): currency.strip()
            for ticker, currency in zip(universe["ticker"], universe["currency"])
            if ticker.strip() and currency.strip()}


def _validated_rates(fx: pd.DataFrame, pair: str, base: str) -> pd.Series:
    if pair not in fx or fx[pair].dropna().empty:
        raise ValueError(f"Missing FX history for {pair}; cannot compare in {base}.")
    rates = fx[pair].dropna()
    try:
        rates = pd.to_numeric(rates, errors="raise")
    except (ValueError, TypeError) as exc:
        raise ValueError(f"Invalid FX history for {pair}; rates must be positive finite numbers.") from exc
    if not np.isfinite(rates).all() or (rates <= 0).any():
        raise ValueError(f"Invalid FX history for {pair}; rates must be positive finite numbers.")
    return rates.sort_index()


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
            # Use only past FX observations and tolerate a short holiday gap.
            rates = _validated_rates(fx, pair, base).reindex(
                prices.index, method="ffill", tolerance=pd.Timedelta(days=5))
            if not (rates.notna() & prices[ticker].notna()).any():
                raise ValueError(f"Missing FX history for {pair} on the price dates; cannot compare in {base}.")
            output[ticker] *= rates
    return output


def load_market(tickers: tuple[str, ...], start: str, base: str) -> tuple[pd.DataFrame, dict]:
    """Load prices in one currency, using local listing metadata when available."""
    prices = fetch_prices(tickers, start).reindex(columns=list(tickers))
    listing = _listing_currencies()
    currencies = {}
    for ticker in tickers:
        currency = listing.get(ticker.strip().upper())
        if not currency:
            import yfinance as yf

            try:
                currency = yf.Ticker(ticker).fast_info["currency"]
            except Exception as exc:
                raise ValueError(f"Could not verify {ticker}'s listing currency. Retry the data request.") from exc
            currency = currency.strip() if isinstance(currency, str) else None
        if not currency:
            raise ValueError(f"No currency supplied for {ticker}.")
        currencies[ticker] = currency
    major = {"GBp": "GBP", "GBX": "GBP", "ZAc": "ZAR", "ILA": "ILS"}
    pairs = sorted({f"{major.get(c, c)}{base}=X" for c in currencies.values()
                    if major.get(c, c) != base})
    fx = fetch_prices(tuple(pairs), start) if pairs else pd.DataFrame()
    # Validate the raw observations too: scale repair must not conceal bad rates.
    for pair in pairs:
        _validated_rates(fx, pair, base)
    fx = sanity_check_fx(fx)
    return convert_to_base(prices, currencies, fx, base), currencies
