"""Sanity checks on synthetic data -- no network needed.

Run with: .venv/bin/python test_xray.py
"""
import numpy as np
import pandas as pd

from xray import (average_correlation, effective_bets, effective_positions,
                  normalise_weights, portfolio_series, risk_contributions)


def build_fixture():
    """Three clones of one market factor, plus one independent asset."""
    rng = np.random.default_rng(0)
    n = 2000
    index = pd.bdate_range("2015-01-01", periods=n)
    market = rng.normal(0.0003, 0.01, n)
    return pd.DataFrame(
        {
            "A": market + rng.normal(0, 0.002, n),
            "B": market + rng.normal(0, 0.002, n),
            "C": market + rng.normal(0, 0.002, n),
            "D": rng.normal(0.0002, 0.006, n),
        },
        index=index,
    )


def main():
    returns = build_fixture()
    weights = normalise_weights(pd.Series(dict.fromkeys("ABCD", 25.0)))

    assert abs(effective_positions(weights) - 4.0) < 1e-9, "equal weights = 4 positions"

    bets = effective_bets(returns, weights)
    assert bets < 2.0, f"three clones should collapse to ~1 bet, got {bets:.2f}"

    contributions = risk_contributions(returns, weights)
    assert abs(contributions["risk_share"].sum() - 1.0) < 1e-9, "risk shares must sum to 1"
    assert contributions.loc["D", "risk_share"] < 0.10, (
        "the uncorrelated asset should carry far less risk than its 25% weight"
    )
    assert contributions.loc["A", "risk_share"] > 0.25, "clones carry more risk than weight"

    single = normalise_weights(pd.Series({"A": 1.0}))
    assert abs(effective_bets(returns, single) - 1.0) < 1e-6, "one holding = one bet"

    independent = returns[["A", "D"]].copy()
    spread = effective_bets(independent, normalise_weights(pd.Series({"A": 50.0, "D": 50.0})))
    assert spread > 1.5, f"two unrelated assets should score near 2, got {spread:.2f}"

    assert 0.0 < average_correlation(returns, weights) < 1.0
    assert len(portfolio_series(returns, weights)) == len(returns)

    print(f"all checks passed  (4 positions -> {bets:.2f} effective bets)")


if __name__ == "__main__":
    main()
