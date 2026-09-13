"""Consumer explanations must describe the actual fee-aware trading plan."""
import unittest

import pandas as pd

from optimizer import EQUAL, MC, MIN_VAR, rebalance_plan
from optimizer_ui import _trade_explanation


class TradingExplanationTests(unittest.TestCase):
    def test_no_breach_explains_holding_even_when_target_differs(self):
        current = pd.Series({"A": .48, "B": .52})
        target = pd.Series({"A": .50, "B": .50})
        plan, summary = rebalance_plan(current, target, 100_000, band=.05)
        for _, row in plan.iterrows():
            explanation = _trade_explanation(row, EQUAL, summary["triggered"])
            self.assertIn("Keep this holding", explanation)
            self.assertNotIn("Increase", explanation)
            self.assertNotIn("Reduce", explanation)

    def test_buy_sell_reasons_follow_weight_direction_and_objective(self):
        current = pd.Series({"A": .70, "B": .30})
        target = pd.Series({"A": .40, "B": .60})
        plan, summary = rebalance_plan(current, target, 100_000)
        sell = _trade_explanation(plan.loc["A"], MIN_VAR, summary["triggered"])
        buy = _trade_explanation(plan.loc["B"], MIN_VAR, summary["triggered"])
        self.assertIn("Reduce its share from 70.0% toward 40.0%", sell)
        self.assertIn("Increase its share from 30.0% toward 60.0%", buy)
        self.assertIn("portfolio's estimated price swings", buy)
        self.assertNotIn("expected return for B", buy)

    def test_cost_funded_sale_does_not_claim_target_share_falls(self):
        current = pd.Series({"A": .30, "B": .60, "C": .10})
        # A's target share rises slightly, but total wealth falls enough after
        # costs that its target cash amount requires a small sale.
        target = pd.Series({"A": .301, "B": .399, "C": .30})
        plan, summary = rebalance_plan(current, target, 100_000, cost_bps=100)
        self.assertEqual(plan.loc["A", "Action"], "Sell")
        explanation = _trade_explanation(plan.loc["A"], MC, summary["triggered"])
        self.assertIn("sale funds trading costs", explanation)
        self.assertNotIn("Reduce its share", explanation)
        self.assertNotIn("Increase its share", explanation)


if __name__ == "__main__":
    unittest.main()
