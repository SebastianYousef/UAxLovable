"""Checks for the 100%-locked weight sliders on the portfolio page.

Run with: .venv/bin/python test_weight_split.py
"""
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

from portfolio_service import PRESETS

ROOT = Path(__file__).parent


def fake_prices(tickers, start):
    rng = np.random.default_rng(11)
    tickers = list(dict.fromkeys(tickers))
    dates = pd.bdate_range("2015-01-02", "2026-09-11")
    factor = rng.normal(0.0003, 0.009, (len(dates), 1))
    moves = factor + rng.normal(0, 0.005, (len(dates), len(tickers)))
    return pd.DataFrame(100 * np.cumprod(1 + moves, axis=0), index=dates,
                        columns=tickers)


def portfolio_page():
    with patch("data.fetch_prices", side_effect=fake_prices):
        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120).run()
        app.switch_page("views/your_portfolio.py").run()
    return app


def slider_total(app):
    return sum(s.value for s in app.slider
               if s.key and s.key.startswith("consumer_split_")
               and s.key != "consumer_split_mode")


class WeightSplitTests(unittest.TestCase):
    def test_sliders_start_at_one_hundred(self):
        app = portfolio_page()
        self.assertFalse(app.exception, [e.message for e in app.exception])
        self.assertEqual(slider_total(app), 100)

    def test_dragging_one_holding_keeps_the_total_at_one_hundred(self):
        app = portfolio_page()
        keys = [s.key for s in app.slider if s.key.startswith("consumer_split_")
                and s.key != "consumer_split_mode"]
        self.assertGreaterEqual(len(keys), 2)

        for target in (80, 0, 45, 100):
            with patch("data.fetch_prices", side_effect=fake_prices):
                app.slider(key=keys[0]).set_value(target).run()
            self.assertFalse(app.exception, [e.message for e in app.exception])
            self.assertEqual(app.slider(key=keys[0]).value, target)
            self.assertEqual(slider_total(app), 100,
                             f"total drifted after setting {target}")

    def test_others_keep_their_relative_sizes(self):
        app = portfolio_page()
        keys = [s.key for s in app.slider if s.key.startswith("consumer_split_")
                and s.key != "consumer_split_mode"]
        before = {k: app.slider(key=k).value for k in keys[1:]}

        with patch("data.fetch_prices", side_effect=fake_prices):
            app.slider(key=keys[0]).set_value(50).run()
        after = {k: app.slider(key=k).value for k in keys[1:]}

        # Everything that was larger should still be larger.
        ordered_before = sorted(before, key=before.get)
        ordered_after = sorted(after, key=after.get)
        self.assertEqual(ordered_before, ordered_after)
        self.assertEqual(sum(after.values()), 50)

    def test_split_survives_a_trip_to_another_page(self):
        app = portfolio_page()
        before = slider_total(app)
        self.assertEqual(before, 100)

        with patch("data.fetch_prices", side_effect=fake_prices):
            app.switch_page("views/your_plan.py").run()
            app.switch_page("views/your_portfolio.py").run()
        self.assertFalse(app.exception, [e.message for e in app.exception])
        self.assertEqual(slider_total(app), 100, "the split reset on the way back")

    def test_choosing_an_example_saves_it(self):
        app = portfolio_page()
        example = next(s for s in app.selectbox if s.label == "Example portfolio")
        target = "Magnificent Seven example"

        with patch("data.fetch_prices", side_effect=fake_prices):
            example.set_value(target).run()
            next(b for b in app.button if b.label == "Use this example").click().run()
        self.assertEqual(slider_total(app), 100)

        with patch("data.fetch_prices", side_effect=fake_prices):
            next(b for b in app.button
                 if b.label == "Save portfolio & see my plan").click().run()
        self.assertFalse(app.error, [e.value for e in app.error])
        self.assertEqual(list(app.session_state["_consumer_profile"]["weights"]),
                         PRESETS[target])

    def test_typed_amounts_still_work(self):
        app = portfolio_page()
        with patch("data.fetch_prices", side_effect=fake_prices):
            app.radio(key="consumer_split_mode").set_value("Type exact amounts").run()
        self.assertFalse(app.exception, [e.message for e in app.exception])
        self.assertEqual(slider_total(app), 0, "sliders should be gone in table mode")


if __name__ == "__main__":
    unittest.main()
