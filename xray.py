"""Portfolio risk analytics.

Everything here is plain numpy on a covariance matrix -- no model fitting,
no magic. Each function maps to one claim the app makes on screen.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252

# Windows chosen as peak -> trough of well-known market stress events.
SHOCKS: dict[str, tuple[str, str]] = {
    "Global Financial Crisis": ("2007-10-09", "2009-03-09"),
    "COVID crash": ("2020-02-19", "2020-03-23"),
    "2022 rate shock": ("2022-01-03", "2022-10-12"),
    "Q4 2018 selloff": ("2018-09-20", "2018-12-24"),
}


def normalise_weights(weights: pd.Series) -> pd.Series:
    weights = weights.astype(float)
    total = weights.sum()
    if total <= 0:
        raise ValueError("Weights must sum to something positive")
    return weights / total


def daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change().dropna(how="all")


def portfolio_series(returns: pd.DataFrame, weights: pd.Series) -> pd.Series:
    cols = [c for c in weights.index if c in returns.columns]
    return returns[cols].fillna(0.0).dot(weights[cols])


def annualised_vol(r: pd.Series) -> float:
    return float(r.std(ddof=1) * np.sqrt(TRADING_DAYS))


def annualised_return(r: pd.Series) -> float:
    if len(r) == 0:
        return float("nan")
    return float((1 + r).prod() ** (TRADING_DAYS / len(r)) - 1)


def max_drawdown(r: pd.Series) -> float:
    """Worst peak-to-trough fall. The number people actually feel."""
    curve = (1 + r).cumprod()
    return float((curve / curve.cummax() - 1).min())


def sharpe(r: pd.Series, risk_free: float = 0.0) -> float:
    vol = annualised_vol(r)
    if vol == 0:
        return float("nan")
    return float((annualised_return(r) - risk_free) / vol)


def _annual_cov(returns: pd.DataFrame, weights: pd.Series) -> np.ndarray:
    cols = list(weights.index)
    return returns[cols].cov().values * TRADING_DAYS


def risk_contributions(returns: pd.DataFrame, weights: pd.Series) -> pd.DataFrame:
    """How much of total portfolio risk each position is responsible for.

    Marginal contribution to risk: d(sigma_p)/d(w_i) = (Sigma w)_i / sigma_p.
    Weighting that by w_i gives contributions that sum to sigma_p, so the
    normalised column sums to 1 -- directly comparable to the weight column.
    """
    cols = list(weights.index)
    cov = _annual_cov(returns, weights)
    w = weights.values

    port_vol = float(np.sqrt(w @ cov @ w))
    if port_vol == 0:
        raise ValueError("Portfolio has zero volatility")

    contrib = w * (cov @ w) / port_vol

    return pd.DataFrame(
        {
            "weight": w,
            "risk_share": contrib / port_vol,
            "standalone_vol": np.sqrt(np.diag(cov)),
        },
        index=cols,
    )


def effective_bets(returns: pd.DataFrame, weights: pd.Series) -> float:
    """Effective number of independent bets (Meucci's diversification entropy).

    Rotate the portfolio into the uncorrelated principal components of the
    covariance matrix, see how risk is spread across them, and take the
    exponential of the entropy of that spread. Ten holdings that all move
    together score ~1; ten genuinely unrelated holdings score ~10.
    """
    cov = _annual_cov(returns, weights)
    w = weights.values

    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    eigenvalues = np.clip(eigenvalues, 1e-18, None)

    rotated = eigenvectors.T @ w
    contrib = (rotated ** 2) * eigenvalues
    p = contrib / contrib.sum()
    p = np.clip(p, 1e-18, None)

    return float(np.exp(-(p * np.log(p)).sum()))


def effective_positions(weights: pd.Series) -> float:
    """Inverse Herfindahl -- diversification by size alone, ignoring correlation."""
    return float(1.0 / (weights.values ** 2).sum())


def average_correlation(returns: pd.DataFrame, weights: pd.Series) -> float:
    corr = returns[list(weights.index)].corr().values
    off_diagonal = corr[~np.eye(len(corr), dtype=bool)]
    return float(off_diagonal.mean())


def benchmark_fit(portfolio: pd.Series, benchmark: pd.Series) -> dict[str, float]:
    """Regress the portfolio on a single index.

    r2 near 1 means the portfolio is, statistically, just that index with
    extra steps. beta is how amplified it is.
    """
    joined = pd.concat([portfolio, benchmark], axis=1).dropna()
    joined.columns = ["portfolio", "benchmark"]
    if len(joined) < 30:
        return {"beta": float("nan"), "alpha": float("nan"), "r2": float("nan")}

    y = joined["portfolio"].values
    x = joined["benchmark"].values

    beta = float(np.cov(y, x, ddof=1)[0, 1] / np.var(x, ddof=1))
    alpha = float((y.mean() - beta * x.mean()) * TRADING_DAYS)
    r2 = float(np.corrcoef(y, x)[0, 1] ** 2)

    return {"beta": beta, "alpha": alpha, "r2": r2}


def shock_report(returns: pd.DataFrame, weights: pd.Series) -> pd.DataFrame:
    """Replay historical crashes against this portfolio.

    Holdings that did not exist yet are dropped and the rest reweighted, with
    the covered share of the portfolio reported so the number stays honest.
    """
    rows = []
    for name, (start, end) in SHOCKS.items():
        window = returns.loc[start:end]

        available = [
            c for c in weights.index
            if c in window.columns and window[c].notna().mean() > 0.5
        ]
        covered = float(weights[available].sum()) if available else 0.0

        if window.empty or covered <= 0:
            rows.append({
                "shock": name, "from": start, "to": end,
                "return": float("nan"), "max_drawdown": float("nan"),
                "coverage": 0.0,
            })
            continue

        sub = portfolio_series(window, normalise_weights(weights[available]))
        rows.append({
            "shock": name,
            "from": start,
            "to": end,
            "return": float((1 + sub).prod() - 1),
            "max_drawdown": max_drawdown(sub),
            "coverage": covered,
        })

    return pd.DataFrame(rows).set_index("shock")
