"""Codex-owned consumer recommendations, built on the existing analytics."""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from optimizer import (CURRENT, EQUAL, MC, MIN_VAR, aligned_returns,
                       bootstrap_scenarios, holdout_validation, optimize,
                       rebalance_plan)

GOALS = {
    "Balance growth & risk": MC,
    "A smoother ride": MIN_VAR,
    "Keep it simple": EQUAL,
}
GOAL_REASONS = {
    MC: "The search favors a mix with more estimated return for the amount of price movement you take on.",
    MIN_VAR: "The model favors a mix with smaller estimated price swings, using how your holdings move together.",
    EQUAL: "The model spreads your money evenly across your existing holdings, making the portfolio easier to understand and maintain.",
}
PRESETS = {
    "US stocks & bonds example": ["SPY", "QQQ", "VGT", "AAPL", "MSFT", "AGG"],
    "Swedish shares example": ["VOLV-B.ST", "ERIC-B.ST", "INVE-B.ST", "HM-B.ST", "SEB-A.ST", "EVO.ST"],
    "Global mix example": ["VT", "AGG", "GLD", "VNQ", "VWO", "TIP"],
    "Stocks & bonds with a tech tilt": ["SPY", "QQQ", "AGG", "TLT"],
    "Magnificent Seven example": ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA"],
}
DEFAULT_SETTINGS = {
    "samples": 10000, "cap": 0.5, "floor": 0.01, "risk_free": 0.03, "seed": 42,
    "mean_shrinkage": 0.5, "covariance_shrinkage": 0.1,
    "cost_bps": 10.0, "band": 0.05, "review_days": 63,
    "block_days": 21, "years": 5, "paths": 500,
}


def validate_holdings(values: pd.Series) -> pd.Series:
    values = values.astype(float)
    if not values.index.is_unique or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Give each holding one nonnegative amount or percentage.")
    values = values[values > 0]
    if not 2 <= len(values) <= 15:
        raise ValueError("Add a positive amount for between 2 and 15 holdings. Zero rows are not included.")
    return values / values.sum()


def demo_prices(tickers: tuple[str, ...]) -> pd.DataFrame:
    rng = np.random.default_rng(1701)
    dates = pd.bdate_range("2020-01-02", periods=1512)
    factor = rng.normal(0.00025, 0.009, (len(dates), 1))
    raw = factor * np.linspace(1.2, 0.2, len(tickers)) + rng.normal(0, 0.004, (len(dates), len(tickers))) + 0.0001
    return pd.DataFrame(100 * np.cumprod(1 + raw, axis=0), index=dates, columns=tickers)


def build_research(prices: pd.DataFrame, current: pd.Series, goal: str,
                   value: float, base: str, source: str, start: str,
                   currencies: dict, settings: dict) -> dict:
    """One coherent run shared by the landing page and every detail view."""
    if goal not in GOALS:
        raise ValueError("Choose one of the available goals.")
    current = validate_holdings(current)
    if not np.isfinite(value) or value <= 0:
        raise ValueError("Enter a portfolio value above zero.")
    returns = aligned_returns(prices.loc[:, current.index])
    params = {key: settings[key] for key in (
        "samples", "cap", "floor", "risk_free", "seed", "mean_shrinkage", "covariance_shrinkage")}
    result = optimize(returns, current, **params)
    validation = holdout_validation(returns, current, **params,
        cost_bps=settings["cost_bps"], band=settings["band"], review_days=settings["review_days"])
    method = GOALS[goal]
    target = result.weights[method]
    fan, outcomes = bootstrap_scenarios(returns, target, current, value,
        years=settings["years"], paths=settings["paths"], seed=settings["seed"],
        block_days=settings["block_days"], review_days=settings["review_days"],
        band=settings["band"], cost_bps=settings["cost_bps"])
    plan, trades = rebalance_plan(current, target, value, settings["band"], settings["cost_bps"])
    plan["Why"] = [trade_reason(ticker, row, method, settings["cap"], settings["floor"])
                   for ticker, row in plan.iterrows()]
    identifier = hashlib.sha256(repr((current.to_dict(), goal, value, base, source,
        start, settings, str(returns.index[-1]), result.weights.to_dict())).encode()).hexdigest()[:20]
    return dict(id=identifier, result=result, validation=validation, returns=returns,
        current=current, value=value, base=base, source=source, start=start,
        currencies=currencies, review={21: "Monthly", 63: "Quarterly", 252: "Yearly"}[settings["review_days"]],
        band=settings["band"], cost_bps=settings["cost_bps"], block_days=settings["block_days"],
        **params, goal=goal, method=method, target=target, fan=fan, outcomes=outcomes,
        plan=plan, trades=trades, years=settings["years"], paths=settings["paths"])


def trade_reason(ticker: str, row: pd.Series, method: str, cap: float,
                 floor: float = 0.0) -> str:
    if row["Action"] == "Hold":
        return "No material trade is needed in this holding for the current plan. Keep it as it is for now."
    if row["Action"] == "Sell" and row["Target weight"] >= row["Current weight"] - 1e-10:
        return "This small sale helps fund trading costs while keeping the intended share of the smaller after-cost portfolio."
    if floor > 0 and row["Target weight"] <= 1e-9 and row["Action"] == "Sell":
        return (f"Sell this holding in full. The model wanted less than the {floor:.1%} minimum "
                "worth holding, so the money goes to your other holdings instead of leaving a sliver here.")
    if row["Current weight"] > cap + 1e-8 and row["Action"] == "Sell":
        return f"Reduce its share from {row['Current weight']:.1%}; it exceeds the {cap:.0%} target limit for one holding."
    if method == EQUAL:
        return f"Bring this holding closer to the same share of your money as the others ({row['Target weight']:.1%})."
    verb = "Increase" if row["Action"] == "Buy" else "Reduce"
    reason = ("The model chose this share to give your whole mix smaller estimated price swings."
              if method == MIN_VAR else
              "The search chose this share to balance your mix's estimated growth against price swings.")
    return f"{verb} its share to {row['Target weight']:.1%}. {reason}"
