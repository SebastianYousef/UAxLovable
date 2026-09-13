# 🔬 Portfolio X-Ray

**You think you own 6 things. You own 1.2.**

Paste your holdings — stocks, bonds, funds — and this tells you three things your
broker never will:

1. **How many independent bets you actually hold.** Six positions that all rise and
   fall together are one bet wearing six hats.
2. **Where your risk really sits**, as opposed to where your money sits. They are
   rarely the same place.
3. **What would have happened to you** in 2008, in March 2020, and through 2022 —
   using real prices, not a questionnaire.

## The demo

The default portfolio looks sensibly diversified: an S&P 500 fund, a Nasdaq fund, a
tech fund, two big stocks and a bond fund.

```
6 positions  →  1.16 independent bets
Global Financial Crisis:  -46.8%
COVID crash:              -27.5%
2022 rate shock:          -27.5%

AGG (bonds) is 10% of the money and 0.1% of the risk.
```

Six holdings, one bet. That is the entire pitch.

## Run it

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/streamlit run app.py
```

Enter holdings as `TICKER, weight` — one per line. Weights are normalised, so they
can be percentages, kronor, or number of shares' worth. Nordic tickers work with the
Yahoo suffix (`VOLV-B.ST`, `ERIC-B.ST`).

Prices are cached to `.cache/` on first fetch, so the demo survives bad wifi.

## How it works

| Claim on screen | What computes it |
|---|---|
| Independent bets | Entropy of risk spread across the covariance matrix's principal components (Meucci). Ten clones score ~1; ten unrelated assets score ~10. |
| Share of risk | Marginal contribution to risk, `w_i · (Σw)_i / σ_p`. Sums to 100%, so it sits next to the weight column and the gap is visible. |
| "Explained by SPY" | R² of the portfolio's daily returns regressed on the benchmark. Above 90% means you are paying for stock picking and receiving the index. |
| Crash replay | Peak-to-trough windows of real crises. Holdings that did not exist yet are dropped and the rest reweighted — `coverage` reports how much of the portfolio was actually around. |

No fitted models, no black box. Everything is numpy on a covariance matrix, in
`xray.py`, and each function maps to exactly one claim the app makes.

## Layout

```
app.py     Streamlit UI
xray.py    the analytics — all of the actual thinking lives here
data.py    price loading + on-disk cache
test_xray.py   sanity checks on synthetic data
```

## Things to extend

- **Cheaper twin.** Search for a portfolio with the same risk profile and a lower
  total fee, then show the 30-year cost of the difference in kronor.
- **Marginal trade.** "What does adding 5% of X do?" — recompute effective bets and
  show whether it genuinely diversifies or is just another hat on the same bet.
- **Fund look-through.** Resolve funds to their real holdings so overlap shows up at
  the company level, not just in the correlations.
- **Custom shock.** Let the user draw their own scenario — rates +2%, tech −30%.
