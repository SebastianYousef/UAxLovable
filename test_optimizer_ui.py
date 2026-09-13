"""Codex-owned Streamlit integration checks. No network requests."""
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

from optimizer import MIN_VAR

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

    def test_xray_app_exposes_the_merged_optimiser(self):
        def prices(tickers, start):
            rng = np.random.default_rng(4)
            tickers = list(dict.fromkeys(tickers))
            dates = pd.bdate_range("2005-01-03", "2026-09-11")
            factor = rng.normal(0.0003, 0.009, (len(dates), 1))
            values = factor + rng.normal(0, 0.005, (len(dates), len(tickers)))
            return pd.DataFrame(100 * np.cumprod(1 + values, axis=0), index=dates, columns=tickers)

        with patch("data.fetch_prices", side_effect=prices):
            app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
        self.assertFalse(app.exception, [e.message for e in app.exception])
        self.assertEqual(app.title[0].value, "🔬 Portfolio X-Ray")
        # The optimizer is a tab of the X-Ray app, driven by the holdings
        # already chosen in its sidebar -- there is no second picker.
        self.assertIn("Optimise", [tab.label for tab in app.tabs])
        self.assertEqual(app.multiselect(key="holdings").value,
                         ["SPY", "QQQ", "VGT", "AAPL", "MSFT", "AGG"])


if __name__ == "__main__":
    unittest.main()
