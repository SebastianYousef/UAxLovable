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

    test_analysis_layer()
    test_fx_repair()

    print(f"all checks passed  (4 positions -> {bets:.2f} effective bets)")




def test_analysis_layer():
    """Checks for the second-layer analytics in analysis.py."""
    from analysis import (correlation_clusters, diversifier_scan,
                          exposure_breakdown, longest_underwater, tail_risk,
                          top_holdings_share)
    from xray import risk_contributions

    returns = build_fixture()
    weights = normalise_weights(pd.Series(dict.fromkeys("ABCD", 25.0)))

    clusters = correlation_clusters(returns, threshold=0.75)
    clone_cluster = max(clusters, key=len)
    assert set(clone_cluster) == {"A", "B", "C"}, (
        f"the three clones should form one cluster, got {clusters}"
    )
    assert ["D"] in clusters, "the independent asset should stand alone"

    meta = pd.DataFrame({
        "ticker": list("ABCD"),
        "sector": ["Tech", "Tech", "Tech", "Bonds"],
    })
    breakdown = exposure_breakdown(risk_contributions(returns, weights), meta, "sector")
    assert abs(breakdown["weight"].sum() - 1.0) < 1e-9
    assert abs(breakdown["risk_share"].sum() - 1.0) < 1e-9
    assert breakdown.loc["Tech", "gap"] > 0, "the correlated sector should over-carry risk"

    tails = tail_risk(portfolio_series(returns, weights))
    assert tails["var_95"] < 0 and tails["cvar_95"] <= tails["var_95"], (
        "CVaR must be at least as bad as VaR"
    )
    assert 0.0 < tails["positive_days"] < 1.0

    flat = pd.Series([0.01, -0.02, 0.005, 0.004, 0.05])
    assert longest_underwater(flat) == 3, "should count the run below the peak"

    assert abs(top_holdings_share(weights, 4) - 1.0) < 1e-9

    scan = diversifier_scan(returns[["A", "B", "C"]],
                            normalise_weights(pd.Series(dict.fromkeys("ABC", 1.0))),
                            returns[["D"]])
    assert not scan.empty and scan.loc["D", "bets_gained"] > 0, (
        "adding an uncorrelated asset must increase effective bets"
    )


def test_fx_repair():
    """Checks for the FX scale-break repair in data.py."""
    from data import repair_scale_breaks, sanity_check_fx

    dates = pd.bdate_range("2015-01-01", periods=300)
    rng = np.random.default_rng(7)
    honest = pd.Series(0.078 * np.cumprod(1 + rng.normal(0, 0.004, 300)), index=dates)

    assert (repair_scale_breaks(honest) - honest).abs().max() < 1e-12, (
        "a clean series must pass through untouched"
    )

    # The real JPYSEK=X failure: a stretch quoted per 100 units.
    broken = honest.copy()
    broken.iloc[100:160] *= 100
    repaired = repair_scale_breaks(broken)
    assert (repaired - honest).abs().max() < 1e-12, "the 100x stretch should be undone"

    # Breaks in both directions, and a series whose bad stretch is at the end.
    tail_break = honest.copy()
    tail_break.iloc[200:] /= 100
    assert (repair_scale_breaks(tail_break) - honest).abs().max() < 1e-12

    frame = pd.DataFrame({"GOODSEK=X": honest, "JPYSEK=X": broken})
    checked = sanity_check_fx(frame)
    assert checked["JPYSEK=X"].pct_change().abs().max() < 0.25

    wild = pd.DataFrame({"BADSEK=X": pd.Series([1.0, 1.6, 1.0, 1.7, 1.0] * 20,
                                               index=dates[:100])})
    try:
        sanity_check_fx(wild)
    except ValueError as exc:
        assert "Implausible" in str(exc)
    else:
        raise AssertionError("an implausible FX series must be rejected")


if __name__ == "__main__":
    main()
