"""Check every ticker in universe.csv actually resolves, and report the dead ones.

Run after editing universe.csv:  .venv/bin/python validate_universe.py
Add --prune to drop failures from the file.
"""
from __future__ import annotations

import sys

import pandas as pd
import yfinance as yf

BATCH = 40


def check(tickers: list[str]) -> set[str]:
    """Return the subset that came back with usable recent price data."""
    alive: set[str] = set()
    for i in range(0, len(tickers), BATCH):
        chunk = tickers[i:i + BATCH]
        raw = yf.download(chunk, period="6mo", auto_adjust=True,
                          progress=False, group_by="column")
        if raw is None or raw.empty:
            continue
        close = raw.xs("Close", axis=1, level=0) if isinstance(
            raw.columns, pd.MultiIndex) else raw[["Close"]].set_axis(chunk, axis=1)
        for t in chunk:
            if t in close.columns and close[t].notna().sum() > 20:
                alive.add(t)
        print(f"  checked {min(i + BATCH, len(tickers))}/{len(tickers)}", flush=True)
    return alive


def main() -> int:
    universe = pd.read_csv("universe.csv")
    alive = check(universe["ticker"].tolist())
    dead = universe[~universe["ticker"].isin(alive)]

    print(f"\n{len(alive)}/{len(universe)} tickers OK")
    if not dead.empty:
        print("\nNo usable data:")
        for _, row in dead.iterrows():
            print(f"  {row['ticker']:<14} {row['name']} ({row['country']})")

    if "--prune" in sys.argv and not dead.empty:
        universe[universe["ticker"].isin(alive)].to_csv("universe.csv", index=False)
        print(f"\npruned {len(dead)} rows from universe.csv")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
