# 🔬 Portfolio X-Ray

**You think you own 6 things. You own 1.2.**

Pick your holdings from a searchable universe of 244 instruments across 20
countries — stocks, bonds, funds, commodities — and this tells you what your
broker never will:

1. **How many independent bets you actually hold.** Six positions that all rise and
   fall together are one bet wearing six hats.
2. **Where your risk really sits**, as opposed to where your money sits. They are
   rarely the same place.
3. **What would have happened to you** in 2008, in March 2020, and through 2022 —
   using real prices, not a questionnaire.
4. **Which holdings are secretly the same bet**, clustered by correlation.
5. **What you should add** to actually diversify — ranked, measured, not guessed.

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

Filter the universe by country, sector or asset class, then search it by ticker or
company name. Start from a preset, weight equally or by hand. Prices are cached to
`.cache/` on first fetch, so the demo survives bad wifi.

To extend the instrument list, append to `universe.csv` and run:

```bash
.venv/bin/python validate_universe.py --prune
```

## How it works

| Claim on screen | What computes it |
|---|---|
| Independent bets | Entropy of risk spread across the covariance matrix's principal components (Meucci). Ten clones score ~1; ten unrelated assets score ~10. |
| Share of risk | Marginal contribution to risk, `w_i · (Σw)_i / σ_p`. Sums to 100%, so it sits next to the weight column and the gap is visible. |
| "Explained by SPY" | R² of the portfolio's daily returns regressed on the benchmark. Above 90% means you are paying for stock picking and receiving the index. |
| Crash replay | Peak-to-trough windows of real crises. Holdings that did not exist yet are dropped and the rest reweighted — `coverage` reports how much of the portfolio was actually around. |
| Same-bet clusters | Single-linkage union-find over the correlation matrix. Five names above the threshold are one cluster, and one bet. |
| Bad days (VaR / CVaR) | Historical percentiles of daily returns. CVaR is the average of the tail beyond VaR — the number that says how bad "bad" gets. |
| What to add | Each candidate is added at 10%, the portfolio scaled to make room, and effective bets recomputed on the history the two actually share, so a short track record cannot flatter a candidate. |

No fitted models, no black box. Everything is numpy on a covariance matrix, in
`xray.py`, and each function maps to exactly one claim the app makes.

## Layout

```
app.py               Streamlit UI, five tabs
xray.py              core risk maths
analysis.py          exposure, tail risk, clustering, diversifier scan
data.py              price loading + on-disk cache
universe.csv         244 instruments, 20 countries
validate_universe.py checks every ticker still resolves
test_xray.py         sanity checks on synthetic data, no network needed
COORDINATION.md      file ownership for working two agents in parallel
```

## Things to extend

- **Cheaper twin.** Search for a portfolio with the same risk profile and a lower
  total fee, then show the 30-year cost of the difference in kronor.
- **Fund look-through.** Resolve funds to their real holdings so overlap shows up at
  the company level, not just in the correlations.
- **Custom shock.** Let the user draw their own scenario — rates +2%, tech −30%.
