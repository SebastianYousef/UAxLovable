"""Codex-owned numerical regression tests; no downloads. Run with unittest."""
import unittest

import numpy as np
import pandas as pd

from optimizer import (CURRENT, EQUAL, INV_VOL, MC, MIN_VAR, aligned_returns,
                       backtest, bootstrap_scenarios, holdout_validation,
                       minimum_variance, optimize, project_weights, rebalance_plan)
from optimizer_market import convert_to_base


def fixture(n=630):
    rng = np.random.default_rng(11)
    factor = rng.normal(0.0005, 0.009, n)
    return pd.DataFrame({"A": factor + rng.normal(0, 0.003, n),
                         "B": factor * 0.8 + rng.normal(0, 0.004, n),
                         "C": rng.normal(0.0002, 0.004, n)},
                        index=pd.bdate_range("2020-01-01", periods=n))


class OptimizerTests(unittest.TestCase):
    def setUp(self):
        self.returns = fixture()
        self.current = pd.Series({"A": 0.7, "B": 0.2, "C": 0.1})

    def test_projection_and_infeasible_cap(self):
        weights = project_weights(np.array([[4, -2, 3], [0.1, 0.3, 0.6]]), 0.4)
        np.testing.assert_allclose(weights.sum(axis=1), 1, atol=1e-12)
        self.assertTrue(((weights >= 0) & (weights <= 0.4)).all())
        with self.assertRaisesRegex(ValueError, "Maximum weight"):
            project_weights(np.ones(3), 0.3)
        np.testing.assert_allclose(project_weights(np.ones(3), 1 / 3), 1 / 3)

    def test_min_variance_matches_analytical_independent_solution(self):
        covariance = np.diag([0.04, 0.09, 0.01])
        expected = (1 / np.diag(covariance)) / (1 / np.diag(covariance)).sum()
        np.testing.assert_allclose(minimum_variance(covariance, 1), expected, atol=1e-7)
        capped = minimum_variance(covariance, 0.5)
        self.assertAlmostEqual(capped[2], 0.5, places=7)
        self.assertAlmostEqual(capped.sum(), 1)

    def test_reproducibility_constraints_and_objective(self):
        result = optimize(self.returns, self.current, samples=600, cap=0.45)
        again = optimize(self.returns, self.current, samples=600, cap=0.45)
        pd.testing.assert_frame_equal(result.weights, again.weights)
        np.testing.assert_allclose(result.weights.sum(), 1, atol=1e-10)
        chosen = result.weights.drop(columns=CURRENT)
        self.assertTrue((chosen >= 0).all().all())
        self.assertTrue((chosen <= 0.45 + 1e-10).all().all())
        self.assertAlmostEqual(result.metrics.loc[MC, "Sharpe"], result.cloud["Sharpe"].max())
        self.assertLessEqual(result.metrics.loc[MIN_VAR, "Volatility"], result.cloud["Volatility"].min() + 1e-7)
        np.testing.assert_allclose(result.weights[CURRENT], self.current)

    def test_inverse_volatility_uses_proportions(self):
        result = optimize(self.returns, self.current, samples=100, cap=1)
        scores = 1 / np.sqrt(np.diag(result.covariance))
        np.testing.assert_allclose(result.weights[INV_VOL], scores / scores.sum())

    def test_alignment_and_missing_series(self):
        prices = 100 * (1 + self.returns).cumprod()
        prices.loc[prices.index[:100], "C"] = np.nan
        result = aligned_returns(prices)
        self.assertEqual(result.index[0], prices.index[101])
        prices["C"] = np.nan
        with self.assertRaisesRegex(ValueError, "Missing price history: C"):
            aligned_returns(prices)
        with self.assertRaisesRegex(ValueError, "at least 252"):
            aligned_returns((1 + self.returns.iloc[:100]).cumprod())

    def test_invalid_weights_and_flat_data(self):
        for invalid in (pd.Series({"A": -1, "B": 2, "C": 1}),
                        pd.Series({"A": np.nan, "B": 2, "C": 1}),
                        pd.Series({"A": 0, "B": 0, "C": 0})):
            with self.assertRaises(ValueError):
                optimize(self.returns, invalid, samples=100)
        flat = self.returns.copy()
        flat["C"] = 0
        with self.assertRaisesRegex(ValueError, "no measurable variation"):
            optimize(flat, self.current, samples=100)

    def test_trades_are_self_financing_after_fees(self):
        target = pd.Series({"A": 0.2, "B": 0.3, "C": 0.5})
        plan, report = rebalance_plan(self.current, target, 100000, cost_bps=100)
        self.assertTrue(report["triggered"])
        self.assertAlmostEqual(report["sells"], report["buys"] + report["cost"], places=6)
        resulting = plan["Current amount"] + plan["Trade amount"]
        np.testing.assert_allclose(resulting / resulting.sum(), target, atol=1e-10)
        self.assertAlmostEqual(resulting.sum(), 100000 - report["cost"], places=6)
        self.assertEqual(plan.loc["A", "Action"], "Sell")
        self.assertEqual(plan.loc["C", "Action"], "Buy")

    def test_hold_inside_band_including_exact_boundary(self):
        target = pd.Series({"A": 0.65, "B": 0.2, "C": 0.15})
        plan, report = rebalance_plan(self.current, target, 100000, band=0.05)
        self.assertFalse(report["triggered"])
        self.assertEqual(report["cost"], 0)
        self.assertTrue((plan["Action"] == "Hold").all())

    def test_price_trigger_accounts_for_changing_denominator(self):
        current = pd.Series({"A": 0.4, "B": 0.6})
        target = pd.Series({"A": 0.4, "B": 0.6})
        plan, _ = rebalance_plan(current, target, 100000)
        ratio = 1 + plan.loc["A", "Sell price move"]
        self.assertAlmostEqual(0.4 * ratio / (0.6 + 0.4 * ratio), 0.45)
        zero = pd.Series({"A": 0.0, "B": 1.0})
        plan, _ = rebalance_plan(zero, target, 100000)
        self.assertTrue(np.isnan(plan.loc["A", "Buy price move"]))

    def test_backtest_counts_loss_on_first_day(self):
        returns = pd.DataFrame({"A": [-0.1, 0, 0], "B": [-0.1, 0, 0]},
                               index=pd.bdate_range("2020-01-01", periods=3))
        weight = pd.Series({"A": 0.5, "B": 0.5})
        series, metrics = backtest(returns, weight, weight, cost_bps=0)
        self.assertAlmostEqual(metrics["Worst drawdown"], -0.1)
        self.assertAlmostEqual((1 + series).prod(), 0.9)

    def test_holdout_weights_do_not_see_future_returns(self):
        original = holdout_validation(self.returns, self.current, samples=300)
        changed = self.returns.copy()
        changed.iloc[int(len(changed) * 0.8):, 0] += 0.025
        perturbed = holdout_validation(changed, self.current, samples=300)
        pd.testing.assert_frame_equal(original[2], perturbed[2])
        self.assertFalse(original[1].equals(perturbed[1]))
        self.assertLess(original[3], original[1].index[0])

    def test_bootstrap_reproducibility_quantiles_and_costs(self):
        target = pd.Series({"A": 0.3, "B": 0.3, "C": 0.4})
        kwargs = dict(years=1, paths=100, seed=5)
        fan, summary = bootstrap_scenarios(self.returns, target, self.current, **kwargs)
        again, _ = bootstrap_scenarios(self.returns, target, self.current, **kwargs)
        pd.testing.assert_frame_equal(fan, again)
        self.assertTrue((np.diff(fan.to_numpy(), axis=1) >= 0).all())
        self.assertTrue((fan > 0).all().all())
        self.assertEqual(fan.index[-1], 1)
        self.assertTrue((fan.iloc[0] == 100000).all())
        self.assertTrue(0 <= summary["loss_probability"] <= 1)
        flat = pd.DataFrame(0.0, index=self.returns.index, columns=self.returns.columns)
        fan, _ = bootstrap_scenarios(flat, target, self.current, cost_bps=100, **kwargs)
        _, plan = rebalance_plan(self.current, target, 100000, band=0, cost_bps=100)
        np.testing.assert_allclose(fan.iloc[-1], plan["after_value"])

    def test_joint_bootstrap_preserves_cross_asset_dependence(self):
        # Perfectly opposite returns cancel with daily exact rebalancing. An
        # incorrect independent per-asset bootstrap would create random P&L.
        returns = self.returns[["A", "B"]].copy()
        returns["B"] = -returns["A"]
        weight = pd.Series({"A": 0.5, "B": 0.5})
        fan, _ = bootstrap_scenarios(returns, weight, weight, years=1, paths=100,
                                     review_days=1, band=0, cost_bps=0)
        np.testing.assert_allclose(fan, 100000, atol=1e-6)

    def test_fx_conversion_handles_pence_and_no_lookahead(self):
        dates = pd.bdate_range("2025-01-01", periods=4)
        prices = pd.DataFrame({"UK.L": [100, 110, 120, 130], "US": [10, 10, 10, 10]}, index=dates)
        fx = pd.DataFrame({"GBPUSD=X": [1.2, 1.4]}, index=[dates[1], dates[3]])
        result = convert_to_base(prices, {"UK.L": "GBp", "US": "USD"}, fx, "USD")
        self.assertTrue(np.isnan(result.iloc[0, 0]))
        self.assertAlmostEqual(result.iloc[1, 0], 1.1 * 1.2)
        self.assertAlmostEqual(result.iloc[2, 0], 1.2 * 1.2)
        self.assertAlmostEqual(result.iloc[3, 0], 1.3 * 1.4)
        np.testing.assert_allclose(result["US"], 10)
        with self.assertRaisesRegex(ValueError, "Missing FX"):
            convert_to_base(prices, {"UK.L": "GBP", "US": "USD"}, pd.DataFrame(), "USD")


if __name__ == "__main__":
    unittest.main()
