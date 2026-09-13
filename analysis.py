"""Second-layer analysis: exposure, tail risk, clustering, and what to add.

`xray.py` answers "what is this portfolio?". This module answers the follow-up
questions a user asks once they have seen the headline number.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from xray import (TRADING_DAYS, annualised_vol, effective_bets, max_drawdown,
                  normalise_weights, portfolio_series)


def exposure_breakdown(contributions: pd.DataFrame, meta: pd.DataFrame,
                       dimension: str) -> pd.DataFrame:
    """Aggregate money and risk by country, sector or asset class.

    The interesting column is the gap: 8% of the money in one country can be
    25% of the risk.
    """
    joined = contributions.join(meta.set_index("ticker")[[dimension]], how="left")
    joined[dimension] = joined[dimension].fillna("Unknown")

    grouped = (
        joined.groupby(dimension)[["weight", "risk_share"]]
        .sum()
        .sort_values("risk_share", ascending=False)
    )
    grouped["gap"] = grouped["risk_share"] - grouped["weight"]
    return grouped


def longest_underwater(returns: pd.Series) -> int:
    """Longest run of days spent below a previous peak -- the patience test."""
    curve = (1 + returns).cumprod()
    underwater = (curve / curve.cummax() - 1) < -1e-12

    longest = current = 0
    for flag in underwater.values:
        current = current + 1 if flag else 0
        longest = max(longest, current)
    return int(longest)


def tail_risk(returns: pd.Series) -> dict[str, float]:
    """What the bad days look like, measured rather than imagined.

    VaR 95 is the loss a normal-bad day does not exceed 95% of the time.
    CVaR is the average loss on the days that *do* exceed it -- the number
    that matters, because VaR tells you nothing about how bad the tail gets.
    """
    values = returns.dropna().values
    if len(values) == 0:
        return {}

    var95 = float(np.percentile(values, 5))
    var99 = float(np.percentile(values, 1))
    tail = values[values <= var95]

    return {
        "var_95": var95,
        "var_99": var99,
        "cvar_95": float(tail.mean()) if len(tail) else float("nan"),
        "worst_day": float(values.min()),
        "best_day": float(values.max()),
        "positive_days": float((values > 0).mean()),
        "current_drawdown": float(
            ((1 + returns).cumprod() / (1 + returns).cumprod().cummax() - 1).iloc[-1]
        ),
        "longest_underwater_days": float(longest_underwater(returns)),
    }


def correlation_clusters(returns: pd.DataFrame, threshold: float = 0.75) -> list[list[str]]:
    """Group holdings that move together into single-linkage clusters.

    Two holdings correlated above the threshold land in the same cluster, and
    clusters merge transitively. A cluster of five is five names but one bet.
    """
    columns = list(returns.columns)
    corr = returns.corr().values

    parent = list(range(len(columns)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            if corr[i, j] >= threshold:
                parent[find(i)] = find(j)

    groups: dict[int, list[str]] = {}
    for i, name in enumerate(columns):
        groups.setdefault(find(i), []).append(name)

    return sorted(groups.values(), key=len, reverse=True)


def rolling_risk(portfolio: pd.Series, benchmark: pd.Series,
                 window: int = 126) -> pd.DataFrame:
    """Rolling volatility and rolling correlation to the benchmark.

    Correlation is not a constant. It rises in crashes -- exactly when
    diversification was supposed to help.
    """
    frame = pd.concat([portfolio, benchmark], axis=1).dropna()
    frame.columns = ["portfolio", "benchmark"]

    return pd.DataFrame({
        "Volatility (annualised)": frame["portfolio"].rolling(window).std()
        * np.sqrt(TRADING_DAYS),
        "Correlation to benchmark": frame["portfolio"].rolling(window).corr(
            frame["benchmark"]
        ),
    }).dropna()


def diversifier_scan(holdings_returns: pd.DataFrame, weights: pd.Series,
                     candidate_returns: pd.DataFrame,
                     add_weight: float = 0.10) -> pd.DataFrame:
    """Which candidate, added at `add_weight`, buys the most diversification?

    Each candidate is scored on the overlapping history it shares with the
    portfolio, and the baseline is recomputed on that same window, so a short
    history cannot flatter a candidate.
    """
    rows = []
    scaled = weights * (1 - add_weight)

    for candidate in candidate_returns.columns:
        if candidate in weights.index:
            continue

        combined = pd.concat(
            [holdings_returns, candidate_returns[[candidate]]], axis=1
        ).dropna()
        if len(combined) < TRADING_DAYS:
            continue

        base_window = combined[list(weights.index)]
        base_bets = effective_bets(base_window, weights)
        base_vol = annualised_vol(portfolio_series(base_window, weights))

        new_weights = pd.concat([scaled, pd.Series({candidate: add_weight})])
        new_series = portfolio_series(combined, new_weights)

        rows.append({
            "candidate": candidate,
            "bets_after": effective_bets(combined, new_weights),
            "bets_gained": effective_bets(combined, new_weights) - base_bets,
            "vol_change": annualised_vol(new_series) - base_vol,
            "drawdown_after": max_drawdown(new_series),
            "years_of_history": len(combined) / TRADING_DAYS,
        })

    if not rows:
        return pd.DataFrame()

    return (
        pd.DataFrame(rows)
        .set_index("candidate")
        .sort_values("bets_gained", ascending=False)
    )


def top_holdings_share(weights: pd.Series, n: int = 3) -> float:
    """Share of the portfolio sitting in its largest n positions."""
    return float(weights.nlargest(n).sum())


def equal_weight_comparison(returns: pd.DataFrame, weights: pd.Series) -> dict[str, float]:
    """Does the chosen weighting beat simply splitting evenly across the same names?

    Equal weight is a genuinely hard benchmark, and losing to it is informative.
    """
    equal = normalise_weights(pd.Series(1.0, index=weights.index))

    chosen_series = portfolio_series(returns, weights)
    equal_series = portfolio_series(returns, equal)

    return {
        "chosen_vol": annualised_vol(chosen_series),
        "equal_vol": annualised_vol(equal_series),
        "chosen_bets": effective_bets(returns, weights),
        "equal_bets": effective_bets(returns, equal),
    }
