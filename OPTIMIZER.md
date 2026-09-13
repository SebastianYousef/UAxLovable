# Consumer portfolio app and Monte Carlo optimizer — Codex handoff

## Run

Start the shared app with `streamlit run app.py`. Its **Your plan** landing page
automatically analyzes the saved portfolio, runs future scenarios and explains
conditional buy/sell/keep amounts. The initial portfolio is visibly labeled as
an example. A goal selector is the main analysis control on the landing page;
holdings and advanced assumptions have their own pages.

On Windows, from the repository root:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

The consumer UI requires **Streamlit 1.63 or newer**. Reinstall requirements in
an existing environment after pulling; the verified native help, navigation
and controls use this version. Theme defaults are in `.streamlit/config.toml`.

For an offline check, open **Your portfolio → Data options**, choose
**Synthetic demo (offline)** and select **Save portfolio & see my plan**.
Analysis runs automatically. If market loading fails first, **Explore offline
example** provides the same fallback. Invented prices are labeled on result
pages and are not suitable for deciding trades.

For market research, save **Yahoo Finance** as the source and enter your current
holdings, amounts or percentages, portfolio value and currency. The shared
profile lasts for the Streamlit session; it is not a connected brokerage
account or a permanent account record. No orders are submitted.

The original form-based optimizer remains available for isolated development:

```powershell
.\.venv\Scripts\python.exe -m streamlit run optimizer_standalone.py
```

There, select **Synthetic demo (offline)** and **Run optimization**. Its form and
saved results are independent of the main consumer app.

## File ownership and integration

Following `COORDINATION.md`, Codex owns these newly created files:

- `optimizer.py`: allocation math, holdout backtest, scenarios, rebalance amounts.
- `optimizer_market.py`: listing-currency lookup, FX validation and conversion,
  using Claude's `data.fetch_prices` and `data.sanity_check_fx`.
- `optimizer_ui.py`: reusable detailed optimizer renderers and standalone form.
- `consumer_ui.py`: page registration, shared portfolio editor, automatic
  landing page, optional diversification and detailed optimizer integration.
- `portfolio_service.py`: goal mapping, one shared research run, trade reasons.
- `portfolio_details.py`: readable views of the existing X-Ray analytics.
- `consumer_help.py`, `consumer_style.css`: glossary, Help me and styling.
- `.streamlit/config.toml`, `requirements.txt`: theme defaults and runtime
  requirements for the consumer UI.
- `views/*.py`: six thin page entry points; no legacy `pages/` routing.
- `test_consumer.py`, `test_optimizer*.py`: offline regression checks.
- `OPTIMIZER.md`: this handoff.

The user requested this routing migration: `app.py` now calls
`consumer_ui.render_app()`. It remains Claude-owned for future changes and should
stay thin. Claude's math/data modules (`xray.py`, `analysis.py`, `data.py`) retain
their ownership and contracts. Coordinate future routing or consumer UI changes
using `COORDINATION.md` rather than editing another agent's page module.

`st.navigation` explicitly registers **Your plan**, **Your portfolio**,
**Portfolio details**, **Monte Carlo**, **Diversify** and **Help me**.
`_consumer_profile` stores the single portfolio; `_consumer_goal` and
`_consumer_settings` store its goal and assumptions. All consumer analysis pages
read those values. **Help me** works without loading market data.

Keep these shared `optimizer_ui.py` interfaces stable:

```python
_allocation_tab(run) -> tuple[str, pd.Series]
_validation_tab(run) -> None
_scenarios_tab(run, method, target) -> None
_trading_tab(run, method, target) -> None
```

`portfolio_service.build_research()` builds the consumer `run`. The standalone
`optimizer_ui.render()` builds the compatible `_optimizer_run` dict. New required
fields must be supplied by both builders. The shared renderers retain their
`optimizer_` widget and `_optimizer_` result keys. Selecting another detailed
method is an exploration; the landing page still follows its saved goal.

The landing page only reweights existing positive holdings. **Diversify** is a
separate, optional search that tests adding a candidate at 10% and scaling the
current mix to 90%; it never edits the portfolio automatically. The main plan
does not add a new ticker, assume extra cash, or allocate to cash outside the
selected holdings. Native question-mark help uses the shared
`consumer_help.help_text` and `section_heading` functions.

## Methods

All target allocations are long-only, sum to one, and respect the selected
per-holding cap. The current portfolio is an unconstrained comparison baseline.

The landing goals map to methods as follows: **Balance growth & risk** uses
Monte Carlo highest Sharpe, **A smoother ride** uses minimum variance, and
**Keep it simple** uses equal weight. Inverse volatility is available in the
detailed comparison. Goals select the objective, not whichever method happens
to win the historical holdout.

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
  the portfolio currency using historical FX. Listing currency comes from
  `universe.csv`, with a Yahoo metadata fallback for unknown listings, never
  issuer country. Pence quotations are converted into pounds before FX. Raw FX
  rates are validated and scale breaks checked before conversion. Missing
  currencies or FX fail visibly; no implicit 1:1 rates.
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

The landing page states that scenarios assume adopting the target first, even
when the conditional plan currently says to hold. A holding can require no
material trade while others rebalance, and a small sale can fund costs even
when its target percentage is unchanged. Card amounts are rounded for reading;
the downloaded plan retains the underlying values and reasons. Historical
Portfolio details charts use fixed daily weights and do not deduct the separate
trading-cost assumption; their assumptions differ from the optimizer's drifting
weights and scheduled reviews.

## Verification

```powershell
.\.venv\Scripts\python.exe -m unittest test_consumer test_optimizer test_optimizer_ui test_optimizer_clarity test_optimizer_market -v
.\.venv\Scripts\python.exe test_xray.py
```

Numerical checks include analytical minimum-variance solutions, cap feasibility,
reproducible sampling, preserved cross-asset dependence, no future data in
training weights, fee-funded trades, first-day drawdowns, missing data, and FX
conversion without look-ahead. Consumer checks cover automatic landing results,
saved holdings across pages, current-holdings-only plans, funded trades and data
failure without stale recommendations. Standalone checks cover offline
allocation and scenarios, changing targets, invalid inputs and network failure.

Consumer migration verification on 13 September 2026: 42 discovered unit tests
passed, as did the separate `test_xray.py` checks, Python compilation and
`pip check`. Browser checks covered automatic market-data results, question-mark
help, changing goals, saved portfolio value across pages, glossary search and
a 390-pixel-wide layout without horizontal overflow.

## References

- [Streamlit pages navigation](https://docs.streamlit.io/develop/concepts/multipage-apps/overview)
- [MOSEK: estimation error and shrinkage](https://docs.mosek.com/portfolio-cookbook/estimationerror.html)
- [Investor.gov: allocation and rebalancing](https://www.investor.gov/additional-resources/general-resources/publications-research/info-sheets/beginners-guide-asset)
