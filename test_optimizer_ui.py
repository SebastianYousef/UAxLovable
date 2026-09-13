"""Codex-owned Streamlit integration checks. No network requests."""
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from streamlit.testing.v1 import AppTest

from optimizer import MIN_VAR
from portfolio_service import DEFAULT_SETTINGS, demo_prices

ROOT = Path(__file__).parent
PAGE = ROOT / "optimizer_standalone.py"


def button(app, label):
    return next(item for item in app.button if item.label == label)


class OptimizerUITests(unittest.TestCase):
    def test_offline_allocation_scenarios_and_invalid_input(self):
        app = AppTest.from_file(str(PAGE), default_timeout=30).run()
        self.assertFalse(app.exception)
        self.assertTrue(any("Set your current weights" in i.value for i in app.info))
        app.radio(key="optimizer_source").set_value("Synthetic demo (offline)").run()
        button(app, "Run optimization").click().run()
        self.assertFalse(app.exception, [e.message for e in app.exception])
        self.assertFalse(app.error, [e.value for e in app.error])
        self.assertEqual(len(app.tabs), 5)
        self.assertTrue(any("SYNTHETIC DEMO" in w.value for w in app.warning))
        app.selectbox(key="optimizer_method").set_value(MIN_VAR).run()
        app.slider(key="optimizer_years").set_value(1)
        button(app, "Simulate future paths").click().run()
        self.assertFalse(app.exception, [e.message for e in app.exception])
        self.assertTrue(any(m.label == "Median ending value" for m in app.metric))
        # Changing the target must not leave a stale fan chart visible.
        app.selectbox(key="optimizer_method").set_value("Equal weight").run()
        self.assertFalse(any(m.label == "Median ending value" for m in app.metric))
        cap = next(s for s in app.slider if s.label == "Maximum weight per holding (%)")
        cap.set_value(5)
        button(app, "Run optimization").click().run()
        self.assertTrue(any("cap is infeasible" in e.value for e in app.error))
        self.assertEqual(len(app.tabs), 0)

    def test_network_error_is_actionable(self):
        app = AppTest.from_file(str(PAGE), default_timeout=30).run()
        with patch("optimizer_ui._market", side_effect=RuntimeError("Provider unavailable")):
            button(app, "Run optimization").click().run()
        self.assertFalse(app.exception)
        self.assertTrue(any("Provider unavailable" in e.value for e in app.error))
        self.assertEqual(len(app.tabs), 0)

    def test_xray_navigation_shares_the_saved_portfolio_with_optimizer(self):
        weights = {"AAPL": .5, "MSFT": .3, "AGG": .2}

        def prices(tickers, start, base):
            return demo_prices(tuple(tickers)), dict.fromkeys(tickers, base)

        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30)
        # Existing X-Ray session holdings migrate to the consumer profile.
        app.session_state["holdings"] = list(weights)
        app.session_state["weight_table"] = weights.copy()
        app.session_state["_consumer_goal"] = "A smoother ride"
        app.session_state["_consumer_settings"] = {
            **DEFAULT_SETTINGS, "samples": 2000, "years": 1,
        }
        with patch("consumer_ui.market_prices", side_effect=prices) as market:
            app.run()
            self.assertFalse(app.exception, [e.message for e in app.exception])
            self.assertFalse(app.error, [e.value for e in app.error])
            self.assertEqual(app.title[0].value, "A clearer plan for your money.")
            self.assertTrue(any(m.label == "The middle outcome" for m in app.metric))
            self.assertEqual(app.session_state["_consumer_profile"]["weights"], weights)

            app.switch_page("views/monte_carlo.py").run()
            self.assertFalse(app.exception, [e.message for e in app.exception])
            self.assertFalse(app.error, [e.value for e in app.error])
            self.assertEqual(app.title[0].value, "Monte Carlo, explained")
            self.assertEqual([tab.label for tab in app.tabs],
                             ["Compare mixes", "Reality check", "Possible futures", "Buy & sell details"])
            self.assertEqual(app.selectbox(key="optimizer_method").value, MIN_VAR)
            self.assertTrue(any(m.label == "Median ending value" for m in app.metric))
            # The shared renderer receives the saved weights, and the detailed
            # page has no second holdings picker that could diverge from them.
            self.assertEqual(len(app.multiselect), 0)
            allocation = app.dataframe[0].value
            pd.testing.assert_series_equal(allocation["Current"],
                                           pd.Series(weights, name="Current"))

            app.switch_page("views/your_portfolio.py").run()
            self.assertEqual(app.multiselect(key="consumer_edit_holdings").value, list(weights))
            self.assertTrue(market.called)
            self.assertTrue(all(call.args[0] == tuple(weights) for call in market.call_args_list))


if __name__ == "__main__":
    unittest.main()
