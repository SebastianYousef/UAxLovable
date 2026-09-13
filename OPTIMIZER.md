# Monte Carlo Optimizer — Codex handoff

## Run

Start the existing app as usual: `streamlit run app.py`. Streamlit automatically
adds **Monte Carlo Optimizer** to its sidebar from
`pages/1_Monte_Carlo_Optimizer.py`; there is no edit to `app.py`.

On Windows, from the repository root:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

For a quick offline check choose **Synthetic demo (offline)** and **Run
optimization**. The demo uses invented data and labels it on every results
screen. For actual research choose **Yahoo Finance**, enter current position
weights and portfolio value, and select the portfolio currency. New allocations
are research outputs; nothing connects to a broker or executes orders.

## File ownership and integration

Following `COORDINATION.md`, Codex owns these newly created files:

- `optimizer.py`: allocation math, holdout backtest, scenarios, rebalance amounts.
- `optimizer_market.py`: currency verification and historical FX conversion,
  calling Claude's existing `data.fetch_prices` without modifying it.
- `optimizer_ui.py`: the new, separate screen.
- `pages/1_Monte_Carlo_Optimizer.py`: Streamlit page entry point.
- `test_optimizer.py`, `test_optimizer_ui.py`: offline regression tests.
- `OPTIMIZER.md`: this handoff.

Claude's `app.py`, `xray.py`, `analysis.py`, `data.py`, and existing tests are
unchanged. No dependency changes were needed; all math uses numpy and pandas.
This user-requested optimizer supersedes the older optional Codex task list
(universe expansion, fees, API) for this turn.

If Claude later introduces `st.navigation` in the main app, explicitly register
this page there: Streamlit ignores the `pages/` directory once `st.navigation`
is used. Do not replace the optimizer files as part of that routing change.

Inputs use `optimizer_` widget keys and `_optimizer_` persistent result keys.
Initial tickers are copied from X-Ray's `holdings` session key when present;
weights are entered independently so a stale X-Ray editor cannot silently
change the optimizer's baseline. The optimizer never writes X-Ray state.

## Methods

All target allocations are long-only, sum to one, and respect the selected
per-holding cap. The current portfolio is an unconstrained comparison baseline.

1. **Monte Carlo highest Sharpe:** sample a mixture of Dirichlet allocations,
   project onto the capped simplex, and choose the highest estimated Sharpe
   among those draws. This is an approximate search, not a global-optimum proof
   or uniform sample of all feasible allocations.
2. **Minimum variance:** solve the convex covariance objective using projected
   gradient descent. Independent of return estimates.
3. **Inverse volatility:** allocate inversely to estimated standalone volatility,
   redistributing capped weight proportionally. This is not risk parity.
4. **Equal weight:** transparent, feasible baseline.

Model metrics use annualized arithmetic means, a user-selected shrinkage of
means toward their cross-sectional average, and covariance shrinkage toward its
diagonal. Estimated returns and Sharpe are not CAGR. Portfolio volatility uses
the full shrunk covariance, including correlations.

## Validation and scenarios

- The chronological 80/20 holdout fits allocations on training data only. The
  test period includes entry costs, daily drifting weights, and after-close
  periodic band reviews. The return series includes first-day losses in maximum
  drawdown. The allocation screen separately refits on the full data, explicitly
  distinguishing those targets from the historical validation weights.
- Future scenarios resample whole joint historical return blocks. This retains
  cross-asset dependence and within-block serial dependence. They simulate
  drifting holdings and fee-funded rebalancing, not a cost-free daily constant
  weight portfolio. They reuse raw historical returns, not shrunk means.
- All comparisons use complete common returns from adjusted prices converted to
  the portfolio currency using historical FX. Listing currency comes from Yahoo
  metadata, not issuer country. Pence quotations are converted into pounds
  before FX. Missing currencies or FX fail visibly; no implicit 1:1 rates.
- Price and FX observations use the existing disk cache; the latest common date
  is displayed and observations older than 10 days are flagged. Market holidays
  may have been forward-filled by `data.py`. Annualization uses 252 observations
  per year; mixed calendars remain an approximation.

## Trading rule

At the selected review interval, if any weight differs from its target by more
than the absolute percentage-point band, rebalance the entire portfolio. The
plan solves for portfolio value after transaction costs so that sells fund buys
and costs without requiring extra cash. Costs apply to every amount bought and
sold. Zero-band initialization buys the target immediately in the backtest and
scenarios; the conditional trading plan can instead remain on hold.

Price-move examples solve `threshold = w*r / (1-w+w*r)`, assuming all other
position values stay fixed. They are conditional relative movements from the
valuation behind the input weights, not fair values, live quotes, executable
orders or market-timing predictions. Tax lots, actual spreads, minimum orders,
contributions, inflation, and separate fund fees are outside the model.

## Verification

```powershell
.\.venv\Scripts\python.exe -m unittest test_optimizer test_optimizer_ui -v
.\.venv\Scripts\python.exe test_xray.py
```

Numerical checks include analytical minimum-variance solutions, cap feasibility,
reproducible sampling, preserved cross-asset dependence, no future data in
training weights, fee-funded trades, first-day drawdowns, missing data, and FX
conversion without look-ahead. UI checks cover navigation from the existing
app, offline allocation and scenarios, changing targets, invalid inputs and
network failure.

## References

- [Streamlit pages navigation](https://docs.streamlit.io/develop/concepts/multipage-apps/overview)
- [MOSEK: estimation error and shrinkage](https://docs.mosek.com/portfolio-cookbook/estimationerror.html)
- [Investor.gov: allocation and rebalancing](https://www.investor.gov/additional-resources/general-resources/publications-research/info-sheets/beginners-guide-asset)
