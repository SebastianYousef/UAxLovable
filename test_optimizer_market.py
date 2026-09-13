"""Offline regression tests for Codex's market-data and FX adapter."""
from pathlib import Path
import unittest
from unittest.mock import call, patch

import numpy as np
import pandas as pd

import optimizer_market as market


class MarketDataTests(unittest.TestCase):
    def setUp(self):
        self.dates = pd.bdate_range("2025-01-01", periods=5)
        self.start = "2025-01-01"

    def prices(self, tickers):
        return pd.DataFrame({ticker: np.arange(100.0, 105.0) for ticker in tickers},
                            index=self.dates)

    def test_known_listing_currencies_skip_yahoo_metadata_and_unneeded_fx(self):
        prices = self.prices(("AAPL", "MSFT"))
        with patch.object(market, "fetch_prices", return_value=prices) as fetch, \
                patch("yfinance.Ticker") as ticker:
            result, currencies = market.load_market(("AAPL", "MSFT"), self.start, "USD")
        ticker.assert_not_called()
        fetch.assert_called_once_with(("AAPL", "MSFT"), self.start)
        self.assertEqual(currencies, {"AAPL": "USD", "MSFT": "USD"})
        pd.testing.assert_frame_equal(result, prices)

    def test_universe_path_is_relative_to_module(self):
        metadata = pd.DataFrame({"ticker": ["EXAMPLE", "EMPTY"], "currency": ["GBp", " "]})
        with patch.object(market.pd, "read_csv", return_value=metadata) as read:
            self.assertEqual(market._listing_currencies(), {"EXAMPLE": "GBp"})
        self.assertEqual(read.call_args.args[0],
                         Path(market.__file__).resolve().with_name("universe.csv"))

    def test_pence_listing_converts_once_using_pounds_fx(self):
        prices = self.prices(("AZN.L",))
        fx = pd.DataFrame({"GBPUSD=X": [1.2] * 5}, index=self.dates)
        with patch.object(market, "fetch_prices", side_effect=[prices, fx]) as fetch, \
                patch("yfinance.Ticker") as ticker:
            result, currencies = market.load_market(("AZN.L",), self.start, "USD")
        ticker.assert_not_called()
        self.assertEqual(currencies, {"AZN.L": "GBp"})
        self.assertEqual(fetch.call_args_list,
                         [call(("AZN.L",), self.start), call(("GBPUSD=X",), self.start)])
        np.testing.assert_allclose(result["AZN.L"], prices["AZN.L"] * 0.01 * 1.2)

    def test_fx_scale_break_repaired_before_conversion(self):
        prices = self.prices(("7203.T",))
        fx = pd.DataFrame({"JPYSEK=X": [0.07, 0.0701, 0.0702, 7.03, 7.04]},
                          index=self.dates)
        with patch.object(market, "fetch_prices", side_effect=[prices, fx]), \
                patch("yfinance.Ticker") as ticker:
            result, _ = market.load_market(("7203.T",), self.start, "SEK")
        ticker.assert_not_called()
        expected_fx = np.array([0.07, 0.0701, 0.0702, 0.0703, 0.0704])
        np.testing.assert_allclose(result["7203.T"], prices["7203.T"] * expected_fx)
        self.assertLess(result["7203.T"].pct_change().abs().max(), 0.03)

    def test_only_unknown_ticker_requests_yahoo_metadata(self):
        prices = self.prices(("AAPL", "NEW"))
        fx = pd.DataFrame({"EURUSD=X": [1.1] * 5}, index=self.dates)
        with patch.object(market, "fetch_prices", side_effect=[prices, fx]) as fetch, \
                patch("yfinance.Ticker") as ticker:
            ticker.return_value.fast_info = {"currency": "EUR"}
            result, currencies = market.load_market(("AAPL", "NEW"), self.start, "USD")
        ticker.assert_called_once_with("NEW")
        self.assertEqual(fetch.call_args_list[-1], call(("EURUSD=X",), self.start))
        self.assertEqual(currencies, {"AAPL": "USD", "NEW": "EUR"})
        np.testing.assert_allclose(result["NEW"], prices["NEW"] * 1.1)

    def test_missing_listing_metadata_falls_back_to_yahoo(self):
        metadata = pd.DataFrame({"ticker": ["AAPL"], "currency": [""]})
        with patch.object(market, "fetch_prices", return_value=self.prices(("AAPL",))), \
                patch.object(market.pd, "read_csv", return_value=metadata), \
                patch("yfinance.Ticker") as ticker:
            ticker.return_value.fast_info = {"currency": "USD"}
            _, currencies = market.load_market(("AAPL",), self.start, "USD")
        ticker.assert_called_once_with("AAPL")
        self.assertEqual(currencies, {"AAPL": "USD"})

    def test_unknown_currency_failure_does_not_guess_or_convert(self):
        with patch.object(market, "fetch_prices", return_value=self.prices(("NEW",))) as fetch, \
                patch("yfinance.Ticker", side_effect=RuntimeError("offline")), \
                patch.object(market, "convert_to_base") as convert:
            with self.assertRaisesRegex(ValueError, "Could not verify NEW's listing currency"):
                market.load_market(("NEW",), self.start, "USD")
        fetch.assert_called_once()
        convert.assert_not_called()

    def test_empty_unknown_currency_does_not_guess(self):
        for currency in (None, "", " ", np.nan):
            with self.subTest(currency=currency), \
                    patch.object(market, "fetch_prices", return_value=self.prices(("NEW",))), \
                    patch("yfinance.Ticker") as ticker:
                ticker.return_value.fast_info = {"currency": currency}
                with self.assertRaisesRegex(ValueError, "No currency supplied for NEW"):
                    market.load_market(("NEW",), self.start, "USD")

    def test_missing_fx_fails_before_conversion(self):
        for fx in (pd.DataFrame(), pd.DataFrame({"EURUSD=X": [np.nan] * 5}, index=self.dates)):
            with self.subTest(fx=fx.shape), \
                    patch.object(market, "fetch_prices", side_effect=[self.prices(("SAP.DE",)), fx]), \
                    patch.object(market, "convert_to_base") as convert:
                with self.assertRaisesRegex(ValueError, "Missing FX history for EURUSD=X"):
                    market.load_market(("SAP.DE",), self.start, "USD")
            convert.assert_not_called()

    def test_invalid_fx_rates_are_rejected_before_repair(self):
        for bad in (0.0, -1.0, np.inf, -np.inf, "invalid"):
            fx = pd.DataFrame({"EURUSD=X": [1.1, 1.1, bad, 1.1, 1.1]}, index=self.dates)
            with self.subTest(rate=bad), \
                    patch.object(market, "fetch_prices", side_effect=[self.prices(("SAP.DE",)), fx]), \
                    patch.object(market, "sanity_check_fx") as repair:
                with self.assertRaisesRegex(ValueError, "Invalid FX history for EURUSD=X"):
                    market.load_market(("SAP.DE",), self.start, "USD")
            repair.assert_not_called()

    def test_implausible_fx_move_propagates_sanity_check_error(self):
        fx = pd.DataFrame({"EURUSD=X": [1.1, 1.1, 1.7, 1.1, 1.1]}, index=self.dates)
        with patch.object(market, "fetch_prices", side_effect=[self.prices(("SAP.DE",)), fx]):
            with self.assertRaisesRegex(ValueError, "Implausible exchange-rate history"):
                market.load_market(("SAP.DE",), self.start, "USD")

    def test_fx_without_usable_price_date_overlap_fails(self):
        fx = pd.DataFrame({"EURUSD=X": [1.1]}, index=[pd.Timestamp("2024-01-01")])
        with patch.object(market, "fetch_prices", side_effect=[self.prices(("SAP.DE",)), fx]):
            with self.assertRaisesRegex(ValueError, "on the price dates"):
                market.load_market(("SAP.DE",), self.start, "USD")


if __name__ == "__main__":
    unittest.main()
