"""Consumer journey regression checks. Synthetic data; no network required."""
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

from optimizer import MC, rebalance_plan
from portfolio_service import (DEFAULT_SETTINGS, GOALS, PRESETS, build_research,
                               demo_prices, trade_reason, validate_holdings)

ROOT = Path(__file__).parent


def button(app, label):
    return next(item for item in app.button if item.label == label)


def app_fixture(weights=None, settings=None):
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30)
    app.session_state["_consumer_profile"] = {
        "weights": weights if weights is not None else {"SPY": 0.6, "AGG": 0.3, "GLD": 0.1},
        "value": 100000.0, "base": "USD", "source": "Synthetic demo (offline)",
        "start": "2015-01-01", "is_example": True,
    }
    app.session_state["_consumer_settings"] = {
        **DEFAULT_SETTINGS, "samples": 2000, "years": 1, **(settings or {}),
    }
    return app.run()


class ConsumerJourneyTests(unittest.TestCase):
    def test_landing_automatically_builds_plan_with_simple_controls(self):
        app = app_fixture()
        self.assertFalse(app.exception, [e.message for e in app.exception])
        self.assertFalse(app.error, [e.value for e in app.error])
        self.assertEqual(app.title[0].value, "A clearer plan for your money.")
        self.assertTrue(any(m.label == "The middle outcome" for m in app.metric))
        self.assertTrue(any("buy, sell & keep" in item.value for item in app.subheader))
        self.assertEqual(len(app.slider), 0)
        self.assertEqual(len(app.number_input), 0)
        self.assertFalse(any("Run" in item.label for item in app.button))
        self.assertTrue(any("Offline example" in item.value for item in app.warning))
        self.assertTrue(all(item.proto.help for item in app.metric))
        additions = next(m for m in app.metric if m.label == "Investments to add money to")
        self.assertRegex(additions.value, r"^[1-9]\d* held now$")
        sidebar_markup = "\n".join(item.value for item in app.sidebar.markdown)
        self.assertIn("Ali Saleh", sidebar_markup)
        self.assertIn("linkedin.com/in/ali-saleh2004", sidebar_markup)
        self.assertIn("Sebastian Yousef", sidebar_markup)
        self.assertIn("linkedin.com/in/sebastian-yousef", sidebar_markup)

    def test_currency_change_updates_every_analysis_page_immediately(self):
        app = app_fixture()
        app.switch_page("views/your_portfolio.py").run()
        app.selectbox(key="consumer_base").set_value("SEK").run()
        self.assertFalse(app.exception, [e.message for e in app.exception])
        self.assertEqual(app.session_state["_consumer_profile"]["base"], "SEK")
        self.assertTrue(any("Currency updated to SEK across the app" in item.value
                            for item in app.success))

        app.switch_page("views/your_plan.py").run()
        middle = next(m for m in app.metric if m.label == "The middle outcome")
        self.assertTrue(middle.value.endswith(" SEK"), middle.value)
        self.assertTrue(any("Values in SEK" in item.value for item in app.caption))

        app.switch_page("views/monte_carlo.py").run()
        ending = next(m for m in app.metric if m.label == "Median ending value")
        self.assertTrue(ending.value.endswith(" SEK"), ending.value)
        self.assertTrue(any("Values in SEK" in item.value for item in app.caption))

        app.switch_page("views/portfolio_details.py").run()
        self.assertTrue(any("Measured in SEK" in item.value for item in app.caption))

    def test_portfolio_edit_saves_across_pages(self):
        app = app_fixture()
        button(app, "Edit portfolio").click().run()
        self.assertEqual(app.title[0].value, "Your portfolio")
        # AppTest needs its script cursor moved explicitly after st.switch_page
        # called from a button; real browsers keep that cursor themselves.
        app.switch_page("views/your_portfolio.py").run()
        next(item for item in app.number_input if item.label == "How much is your portfolio worth?").set_value(200000.0)
        button(app, "Save portfolio & see my plan").click().run()
        self.assertFalse(app.exception, [e.message for e in app.exception])
        self.assertEqual(app.session_state["_consumer_profile"]["value"], 200000.0)
        self.assertEqual(app.title[0].value, "A clearer plan for your money.")
        app.switch_page("views/help_me.py").run()
        self.assertFalse(app.exception, [e.message for e in app.exception])
        app.switch_page("views/your_plan.py").run()
        self.assertEqual(app.session_state["_consumer_profile"]["value"], 200000.0)

    def test_detailed_pages_keep_same_holdings(self):
        app = app_fixture()
        for page in ("views/portfolio_details.py", "views/monte_carlo.py", "views/diversify.py"):
            app.switch_page(page).run()
            self.assertFalse(app.exception, [e.message for e in app.exception])
            self.assertEqual(set(app.session_state["_consumer_profile"]["weights"]), {"SPY", "AGG", "GLD"})
        self.assertTrue(any("actual market history" in item.value for item in app.info))

    def test_data_failure_does_not_show_stale_recommendations(self):
        app = app_fixture()
        saved = dict(app.session_state["_consumer_profile"])
        saved["source"] = "Yahoo Finance"
        app.session_state["_consumer_profile"] = saved
        with patch("consumer_ui.market_prices", side_effect=ValueError("Missing FX")):
            app.run()
        self.assertFalse(app.exception)
        self.assertTrue(app.error)
        self.assertFalse(any(m.label == "The middle outcome" for m in app.metric))
        self.assertFalse(any("buy, sell & keep" in item.value for item in app.subheader))

    def test_reducing_holdings_adjusts_saved_cap_and_builds_a_fresh_plan(self):
        app = app_fixture(settings={"cap": .35})
        self.assertFalse(app.error, [e.value for e in app.error])
        app.switch_page("views/your_portfolio.py").run()
        app.multiselect(key="consumer_edit_holdings").set_value(["SPY", "AGG"]).run()
        button(app, "Save portfolio & see my plan").click().run()
        self.assertFalse(app.exception, [e.message for e in app.exception])
        self.assertFalse(app.error, [e.value for e in app.error])
        self.assertEqual(app.title[0].value, "A clearer plan for your money.")
        self.assertEqual(set(app.session_state["_consumer_profile"]["weights"]), {"SPY", "AGG"})
        self.assertEqual(app.session_state["_consumer_settings"]["cap"], .5)
        self.assertEqual(app.session_state["_consumer_settings"]["samples"], 2000)
        self.assertTrue(any("raised the maximum share" in item.value for item in app.info))
        self.assertTrue(any(m.label == "The middle outcome" for m in app.metric))
        app.switch_page("views/monte_carlo.py").run()
        self.assertFalse(app.exception, [e.message for e in app.exception])
        cap = next(item for item in app.slider if item.label == "Maximum share in one holding (%)")
        self.assertEqual(cap.value, 50)
        self.assertEqual(set(app.dataframe[0].value.index), {"SPY", "AGG"})

    def test_minimum_share_setting_drops_slivers(self):
        app = app_fixture(weights=dict.fromkeys(PRESETS["Magnificent Seven example"], 1 / 7))
        app.switch_page("views/monte_carlo.py").run()
        self.assertFalse(app.exception, [e.message for e in app.exception])
        floor = next(item for item in app.slider if item.label == "Smallest share worth holding (%)")
        self.assertEqual(floor.value, DEFAULT_SETTINGS["floor"] * 100)

        floor.set_value(10.0)
        button(app, "Update analysis").click().run()
        self.assertFalse(app.exception, [e.message for e in app.exception])
        self.assertFalse(app.error, [e.value for e in app.error])
        self.assertEqual(app.session_state["_consumer_settings"]["floor"], 0.1)
        targets = app.dataframe[0].value["Target"]
        self.assertTrue(((targets <= 1e-9) | (targets >= 0.1 - 1e-9)).all(), targets.to_dict())
        self.assertAlmostEqual(targets.sum(), 1)

    def test_minimum_share_above_what_the_maximum_allows_is_rejected(self):
        app = app_fixture(weights=dict.fromkeys(PRESETS["Magnificent Seven example"], 1 / 7))
        app.switch_page("views/monte_carlo.py").run()
        next(item for item in app.slider if item.label == "Maximum share in one holding (%)").set_value(15)
        next(item for item in app.slider if item.label == "Smallest share worth holding (%)").set_value(20.0)
        button(app, "Update analysis").click().run()
        self.assertTrue(any("smallest share can be at most 14.3%" in item.value for item in app.error),
                        [item.value for item in app.error])
        self.assertEqual(app.session_state["_consumer_settings"]["floor"], DEFAULT_SETTINGS["floor"])
        self.assertEqual(app.session_state["_consumer_settings"]["cap"], DEFAULT_SETTINGS["cap"])

    def test_smaller_preset_adjusts_saved_cap_before_portfolio_save(self):
        original = PRESETS["Magnificent Seven example"]
        app = app_fixture(weights=dict.fromkeys(original, 1 / len(original)), settings={"cap": .15})
        self.assertFalse(app.error, [e.value for e in app.error])
        app.switch_page("views/your_portfolio.py").run()
        preset = "Stocks & bonds with a tech tilt"
        next(item for item in app.selectbox if item.label == "Example portfolio").set_value(preset)
        button(app, "Use this example").click().run()
        self.assertFalse(app.exception, [e.message for e in app.exception])
        self.assertEqual(list(app.session_state["_consumer_profile"]["weights"]), PRESETS[preset])
        self.assertEqual(app.multiselect(key="consumer_edit_holdings").value, PRESETS[preset])
        self.assertEqual(app.session_state["_consumer_settings"]["cap"], .25)
        self.assertEqual(app.session_state["_consumer_settings"]["years"], 1)
        button(app, "Save portfolio & see my plan").click().run()
        self.assertFalse(app.exception, [e.message for e in app.exception])
        self.assertFalse(app.error, [e.value for e in app.error])
        self.assertTrue(any(m.label == "The middle outcome" for m in app.metric))


class ConsumerPlanTests(unittest.TestCase):
    def test_current_holdings_only_and_funded_trades_for_every_goal(self):
        current = pd.Series({"SPY": 0.6, "AGG": 0.3, "GLD": 0.1})
        prices = demo_prices(tuple(current.index))
        settings = {**DEFAULT_SETTINGS, "samples": 300, "years": 1, "paths": 100}
        for goal in GOALS:
            run = build_research(prices, current, goal, 100000, "USD", "Synthetic demo (offline)",
                                 "2015-01-01", dict.fromkeys(current.index, "USD"), settings)
            self.assertEqual(set(run["target"].index), set(current.index))
            self.assertAlmostEqual(run["target"].sum(), 1)
            self.assertTrue((run["target"] <= settings["cap"] + 1e-10).all())
            self.assertAlmostEqual(run["trades"]["sells"], run["trades"]["buys"] + run["trades"]["cost"], places=5)
            self.assertTrue(run["plan"]["Why"].str.len().gt(20).all())
            self.assertEqual(run["method"], GOALS[goal])

    def test_zero_amount_is_not_treated_as_an_owned_investment(self):
        result = validate_holdings(pd.Series({"SPY": 60, "AGG": 40, "NEW": 0}))
        self.assertEqual(list(result.index), ["SPY", "AGG"])
        np.testing.assert_allclose(result, [0.6, 0.4])
        with self.assertRaises(ValueError):
            validate_holdings(pd.Series({"SPY": 100, "AGG": 0}))

    def test_fee_funded_sale_is_explained_when_target_share_does_not_fall(self):
        current = pd.Series({"A": .30, "B": .60, "C": .10})
        for a_target in (.30, .301):
            with self.subTest(target_share=a_target):
                target = pd.Series({"A": a_target, "B": .7 - a_target, "C": .30})
                plan, summary = rebalance_plan(current, target, 100_000, cost_bps=100)
                row = plan.loc["A"]
                self.assertTrue(summary["triggered"])
                self.assertEqual(row["Action"], "Sell")
                self.assertGreaterEqual(row["Target weight"] + 1e-10, row["Current weight"])
                explanation = trade_reason("A", row, MC, .5)
                self.assertIn("fund trading costs", explanation)
                self.assertNotIn("Trimming", explanation)
                self.assertNotIn("exceeds", explanation)

    def test_hold_explanation_does_not_claim_the_whole_portfolio_is_inside_band(self):
        current = pd.Series({"A": .3, "B": .6, "C": .1})
        target = pd.Series({"A": .3, "B": .3, "C": .4})
        plan, summary = rebalance_plan(current, target, 100_000, cost_bps=0)
        self.assertTrue(summary["triggered"])
        self.assertEqual(plan.loc["A", "Action"], "Hold")
        explanation = trade_reason("A", plan.loc["A"], MC, .5)
        self.assertIn("No material trade", explanation)
        self.assertIn("this holding", explanation)
        self.assertNotIn("within", explanation)


if __name__ == "__main__":
    unittest.main()
