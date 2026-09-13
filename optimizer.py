"""Codex-owned portfolio research: constrained allocation, validation and scenarios.

All returns are simple, base-currency daily returns. No brokerage integration.
Only numpy/pandas are required; importing this module never downloads data.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

DAYS = 252
MC = "Monte Carlo · highest Sharpe"
MIN_VAR = "Minimum variance"
INV_VOL = "Inverse volatility"
EQUAL = "Equal weight"
CURRENT = "Current portfolio"


def aligned_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Use common history, without filling missing returns with zero."""
    if prices.shape[1] < 2 or not prices.columns.is_unique:
        raise ValueError("Choose at least two distinct holdings.")
    if not isinstance(prices.index, pd.DatetimeIndex) or prices.index.has_duplicates:
        raise ValueError("Prices need a unique date index.")
    prices = prices.sort_index().astype(float)
    missing = prices.columns[prices.notna().sum() < 2].tolist()
    if missing:
        raise ValueError(f"Missing price history: {', '.join(missing)}.")
    if np.isinf(prices.to_numpy()).any() or (prices <= 0).any().any():
        raise ValueError("Prices must be finite and positive where available.")
    returns = prices.pct_change(fill_method=None).dropna(how="any")
    if len(returns) < DAYS:
        raise ValueError(
            f"Only {len(returns)} shared daily returns; at least {DAYS} are needed. "
            "Choose an earlier start date or holdings with longer shared history."
        )
    return returns


def _weights(weights: pd.Series, columns: pd.Index) -> pd.Series:
    if not weights.index.is_unique or set(weights.index) != set(columns):
        raise ValueError("Weights must name every holding exactly once.")
    weights = weights.reindex(columns).astype(float)
    if not np.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
        raise ValueError("Weights must be finite, nonnegative and sum above zero.")
    return weights / weights.sum()


def project_weights(values: np.ndarray, cap: float) -> np.ndarray:
    """Euclidean projection onto sum(w)=1, 0<=w<=cap, row by row."""
    values = np.asarray(values, dtype=float)
    one_row = values.ndim == 1
    matrix = np.atleast_2d(values)
    n = matrix.shape[1]
    if not np.isfinite(matrix).all() or not np.isfinite(cap) or not 1 / n - 1e-12 <= cap <= 1:
        raise ValueError(f"Maximum weight must be at least {100 / n:.2f}% and at most 100%.")
    if abs(cap * n - 1) < 1e-10:
        result = np.full_like(matrix, 1 / n)
    else:
        lower = matrix.min(axis=1, keepdims=True) - cap
        upper = matrix.max(axis=1, keepdims=True)
        for _ in range(48):
            middle = (lower + upper) / 2
            too_much = np.clip(matrix - middle, 0, cap).sum(axis=1, keepdims=True) > 1
            lower = np.where(too_much, middle, lower)
            upper = np.where(too_much, upper, middle)
        result = np.clip(matrix - (lower + upper) / 2, 0, cap)
    return result[0] if one_row else result


def estimate_moments(returns: pd.DataFrame, mean_shrinkage: float = 0.5,
                     covariance_shrinkage: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    if len(returns) < 2 or returns.shape[1] < 2 or not np.isfinite(returns.to_numpy()).all():
        raise ValueError("At least two complete observations and holdings are required.")
    if (returns <= -1).any().any():
        raise ValueError("Simple returns must exceed -100%.")
    if not 0 <= mean_shrinkage <= 1 or not 0 <= covariance_shrinkage <= 1:
        raise ValueError("Shrinkage must be between zero and one.")
    means = returns.mean().to_numpy() * DAYS
    means = (1 - mean_shrinkage) * means + mean_shrinkage * means.mean()
    covariance = returns.cov().to_numpy() * DAYS
    covariance = ((1 - covariance_shrinkage) * covariance
                  + covariance_shrinkage * np.diag(np.diag(covariance)))
    if np.diag(covariance).min() < 1e-12:
        raise ValueError("A holding has no measurable variation; remove flat or stale series.")
    return means, covariance


def minimum_variance(covariance: np.ndarray, cap: float) -> np.ndarray:
    """Projected gradient on the convex long-only minimum-variance problem."""
    w = np.full(len(covariance), 1 / len(covariance))
    largest = float(np.linalg.eigvalsh(covariance).max())
    if largest <= 0:
        raise ValueError("Covariance must have positive variance.")
    for _ in range(5000):
        new = project_weights(w - covariance @ w / largest, cap)
        if np.max(np.abs(new - w)) < 1e-9:
            return new
        w = new
    raise ValueError("Minimum-variance solver did not converge; increase covariance shrinkage.")


def _inverse_volatility(covariance: np.ndarray, cap: float) -> np.ndarray:
    # Redistribute capped weight proportionally, preserving inverse-vol ratios
    # among all holdings that have not hit the cap.
    score = 1 / np.sqrt(np.diag(covariance))
    lower, upper = 0.0, 1 / score.min()
    for _ in range(60):
        scale = (lower + upper) / 2
        if np.minimum(scale * score, cap).sum() > 1:
            upper = scale
        else:
            lower = scale
    return np.minimum((lower + upper) / 2 * score, cap)


def model_metrics(weights: np.ndarray, means: np.ndarray, covariance: np.ndarray,
                  risk_free: float) -> pd.DataFrame:
    w = np.atleast_2d(weights)
    expected = w @ means
    volatility = np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", w, covariance, w), 0))
    sharpe = np.divide(expected - risk_free, volatility,
                       out=np.full(len(w), np.nan), where=volatility > 1e-12)
    return pd.DataFrame({"Expected return": expected, "Volatility": volatility,
                         "Sharpe": sharpe})


@dataclass
class AllocationResult:
    weights: pd.DataFrame  # ticker rows, method columns
    metrics: pd.DataFrame
    cloud: pd.DataFrame
    means: np.ndarray
    covariance: np.ndarray


def optimize(returns: pd.DataFrame, current: pd.Series, samples: int = 10000,
             cap: float = 0.5, risk_free: float = 0.03, seed: int = 42,
             mean_shrinkage: float = 0.5,
             covariance_shrinkage: float = 0.1) -> AllocationResult:
    if not isinstance(samples, (int, np.integer)) or not 100 <= samples <= 100000:
        raise ValueError("Use between 100 and 100,000 candidate portfolios.")
    if not np.isfinite(risk_free) or not -0.1 <= risk_free <= 1:
        raise ValueError("Invalid annual risk-free rate.")
    current = _weights(current, returns.columns)
    means, covariance = estimate_moments(returns, mean_shrinkage, covariance_shrinkage)
    n = len(means)
    project_weights(np.ones(n) / n, cap)  # fail early for infeasible constraints
    rng = np.random.default_rng(seed)
    # Mix concentrated and diffuse Dirichlet draws. Projection respects the cap;
    # this is a reproducible search, not uniform sampling of the feasible region.
    concentrations = (0.25, 1.0, 4.0)
    counts = [samples // 3, samples // 3, samples - 2 * (samples // 3)]
    candidates = np.vstack([rng.dirichlet(np.full(n, a), count)
                            for a, count in zip(concentrations, counts)])
    candidates = project_weights(candidates, cap)
    candidates[0] = np.full(n, 1 / n)
    cloud = model_metrics(candidates, means, covariance, risk_free)
    chosen = int(cloud["Sharpe"].idxmax())
    allocations = pd.DataFrame({
        MC: candidates[chosen],
        MIN_VAR: minimum_variance(covariance, cap),
        INV_VOL: _inverse_volatility(covariance, cap),
        EQUAL: np.full(n, 1 / n),
        CURRENT: current,
    }, index=returns.columns)
    metrics = model_metrics(allocations.to_numpy().T, means, covariance, risk_free)
    metrics.index = allocations.columns
    metrics["One-way turnover"] = allocations.sub(current, axis=0).abs().sum() / 2
    metrics["Largest weight"] = allocations.max()
    return AllocationResult(allocations, metrics, cloud, means, covariance)


def rebalance_plan(current: pd.Series, target: pd.Series, value: float,
                   band: float = 0.05, cost_bps: float = 10) -> tuple[pd.DataFrame, dict]:
    """Whole-portfolio reset if any weight breaches its band.

    Trades solve V_after = V_before - fee * sum(abs(target*V_after - holdings)).
    This funds buys and costs from sells with no implied borrowing or extra cash.
    """
    current = _weights(current, target.index)
    target = _weights(target, current.index)
    if not np.isfinite([value, band, cost_bps]).all() or value <= 0 or not 0 <= band < 1 or not 0 <= cost_bps <= 1000:
        raise ValueError("Invalid portfolio value, band or trading cost.")
    before = current * value
    drift = current - target
    triggered = bool((drift.abs() > band + 1e-10).any())
    fee = cost_bps / 10000
    after_value = value
    if triggered:
        for _ in range(100):
            updated = value - fee * (target * after_value - before).abs().sum()
            if abs(updated - after_value) < 1e-8:
                after_value = updated
                break
            after_value = updated
        trades = target * after_value - before
    else:
        trades = before * 0
    lower = (target - band).clip(lower=0)
    upper = (target + band).clip(upper=1)
    plan = pd.DataFrame({
        "Current weight": current, "Target weight": target, "Drift": drift,
        "Buy below weight": lower, "Sell above weight": upper,
        "Current amount": before, "Target amount after costs": target * after_value,
        "Trade amount": trades,
        "Estimated cost": trades.abs() * fee,
        "Action": np.where(trades > 0.005, "Buy", np.where(trades < -0.005, "Sell", "Hold")),
    })
    # Relative price triggers assume every other holding's value is unchanged.
    # No finite trigger exists for an unowned position or an unattainable boundary.
    for column, threshold in (("Buy price move", lower), ("Sell price move", upper)):
        valid = (current > 0) & (current < 1) & (threshold > 0) & (threshold < 1)
        moves = pd.Series(np.nan, index=current.index)
        moves.loc[valid] = (threshold[valid] * (1 - current[valid])
                            / (current[valid] * (1 - threshold[valid]))) - 1
        plan[column] = moves
    costs = float(plan["Estimated cost"].sum())
    return plan, {"triggered": triggered, "cost": costs,
                  "buys": float(trades.clip(lower=0).sum()),
                  "sells": float(-trades.clip(upper=0).sum()),
                  "after_value": after_value}


def backtest(returns: pd.DataFrame, target: pd.Series, current: pd.Series,
             cost_bps: float = 10, band: float = 0.05, review_days: int = 21,
             risk_free: float = 0.03) -> tuple[pd.Series, dict]:
    """Invest at fixed training weights, drift daily, review at period ends.

    Initial allocation is made before the first test return, with transaction
    costs. Review-period trades happen AFTER that day's return (no look-ahead).
    """
    if review_days < 1:
        raise ValueError("Review interval must be positive.")
    target = _weights(target, returns.columns)
    current = _weights(current, returns.columns)
    _, initial = rebalance_plan(current, target, 1.0, band=0, cost_bps=cost_bps)
    holdings = target.to_numpy() * initial["after_value"]
    previous = 1.0
    result = []
    total_cost = initial["cost"]
    rebalance_count = int(initial["triggered"])
    for day, row in enumerate(returns.to_numpy(), 1):
        holdings *= 1 + row
        total = float(holdings.sum())
        if day % review_days == 0:
            drifted = pd.Series(holdings / total, index=returns.columns)
            _, plan = rebalance_plan(drifted, target, total, band, cost_bps)
            if plan["triggered"]:
                holdings = target.to_numpy() * plan["after_value"]
                total_cost += plan["cost"]
                rebalance_count += 1
                total = float(holdings.sum())
        result.append(total / previous - 1)
        previous = total
    series = pd.Series(result, index=returns.index)
    growth = np.r_[1.0, (1 + series).cumprod().to_numpy()]
    volatility = float(series.std(ddof=1) * np.sqrt(DAYS))
    metrics = {
        "Return (CAGR)": float(growth[-1] ** (DAYS / len(series)) - 1),
        "Volatility": volatility,
        "Sharpe": float((series.mean() * DAYS - risk_free) / volatility) if volatility > 1e-12 else np.nan,
        "Worst drawdown": float((growth / np.maximum.accumulate(growth) - 1).min()),
        "Costs / starting value": total_cost,
        "Rebalances": rebalance_count,
    }
    return series, metrics


def holdout_validation(returns: pd.DataFrame, current: pd.Series, *,
                       samples: int = 10000, cap: float = 0.5,
                       risk_free: float = 0.03, seed: int = 42,
                       mean_shrinkage: float = 0.5, covariance_shrinkage: float = 0.1,
                       cost_bps: float = 10, band: float = 0.05,
                       review_days: int = 21) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    split = int(len(returns) * 0.8)
    if split < 126 or len(returns) - split < 50:
        raise ValueError("Not enough history for an 80/20 chronological holdout.")
    training, test = returns.iloc[:split], returns.iloc[split:]
    fitted = optimize(training, current, samples, cap, risk_free, seed,
                      mean_shrinkage, covariance_shrinkage)
    metrics, curves = {}, {}
    for method in fitted.weights:
        series, report = backtest(test, fitted.weights[method], current,
                                  cost_bps, band, review_days, risk_free)
        metrics[method] = report
        curves[method] = (1 + series).cumprod() * 100
    return (pd.DataFrame(metrics).T, pd.DataFrame(curves), fitted.weights,
            training.index[-1])


def bootstrap_scenarios(returns: pd.DataFrame, target: pd.Series, current: pd.Series,
                        value: float = 100000, years: int = 5, paths: int = 1000,
                        seed: int = 42, block_days: int = 21, review_days: int = 21,
                        band: float = 0.05, cost_bps: float = 10) -> tuple[pd.DataFrame, dict]:
    """Joint moving-block bootstrap with weight drift and fee-funded rebalancing.

    Whole historical rows are resampled, keeping cross-asset dependence; each
    consecutive block keeps short-term serial dependence. No fitted mean/vol
    forecast: these scenarios replay the empirical historical distribution.
    """
    target = _weights(target, returns.columns)
    current = _weights(current, returns.columns)
    if not 1 <= years <= 10 or not 100 <= paths <= 5000 or not 1 <= block_days <= len(returns) or review_days < 1:
        raise ValueError("Invalid scenario horizon, path count or block length.")
    if not np.isfinite(returns.to_numpy()).all() or (returns <= -1).any().any():
        raise ValueError("Scenarios need complete simple returns above -100%.")
    _, initial = rebalance_plan(current, target, value, band=0, cost_bps=cost_bps)
    rng = np.random.default_rng(seed)
    days = years * DAYS
    raw = returns.to_numpy()
    target_array = target.to_numpy()
    holdings = np.tile(target_array * initial["after_value"], (paths, 1))
    fee = cost_bps / 10000
    snapshots = [np.full(paths, value)]
    times = [0]
    for day in range(days):
        if day % block_days == 0:
            starts = rng.integers(0, len(raw) - block_days + 1, size=paths)
        sampled = raw[starts + day % block_days]
        holdings *= 1 + sampled
        total = holdings.sum(axis=1)
        if (day + 1) % review_days == 0:
            trigger = (np.abs(holdings / total[:, None] - target_array) > band + 1e-10).any(axis=1)
            active = holdings[trigger]
            old = total[trigger]
            after = old.copy()
            for _ in range(16):
                after = old - fee * np.abs(after[:, None] * target_array - active).sum(axis=1)
            holdings[trigger] = after[:, None] * target_array
        if (day + 1) % 21 == 0 or day + 1 == days:
            snapshots.append(holdings.sum(axis=1))
            times.append((day + 1) / DAYS)
    values = np.asarray(snapshots)
    quantiles = np.percentile(values, [5, 25, 50, 75, 95], axis=1).T
    fan = pd.DataFrame(quantiles, columns=["5th", "25th", "Median", "75th", "95th"])
    fan.index = pd.Index(times, name="Years")
    terminal = values[-1]
    cutoff = np.percentile(terminal, 5)
    return fan, {"median": float(np.median(terminal)), "p05": float(cutoff),
                 "p95": float(np.percentile(terminal, 95)),
                 "loss_probability": float(np.mean(terminal < value)),
                 "worst_5_mean": float(terminal[terminal <= cutoff].mean())}
