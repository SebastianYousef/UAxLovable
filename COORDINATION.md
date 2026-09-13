# Working in parallel (Claude + Codex)

Two agents editing the same file at a hackathon means merge conflicts at minute
110. **Split by file, not by feature.** Each file below has exactly one owner.

## Ownership

| File | Owner | Do not edit if you are the other agent |
|---|---|---|
| `app.py` | Claude | Thin entry point calling `consumer_ui.render_app()`. Keep page content in the owned modules below. |
| `xray.py` | Claude | Core risk maths. |
| `analysis.py` | Claude | Exposure, tail risk, clustering, diversifier scan. |
| `data.py` | Claude | Price loading, cache, FX repair. |
| `optimizer.py` | **Codex** | Allocation search, validation, scenarios. |
| `optimizer_ui.py` | **Codex** | Optimizer rendering. |
| `optimizer_market.py` | **Codex** | Currency conversion. |
| `consumer_ui.py` | **Codex** | Six-page navigation, shared portfolio editor, landing page and diversification UI. |
| `portfolio_service.py` | **Codex** | Shared research run, goal mapping and explanations for proposed trades. |
| `portfolio_details.py` | **Codex** | Readable views using Claude's existing risk analytics. |
| `consumer_help.py`, `consumer_style.css` | **Codex** | Shared glossary, Help me page and consumer styling. |
| `.streamlit/config.toml` | **Codex** | Consumer theme defaults. |
| `requirements.txt` | **Codex** | Runtime requirements; Streamlit 1.63+ supports the verified consumer UI. Coordinate dependency changes. |
| `views/*.py` | **Codex** | Thin page entry points registered with `st.navigation`. |
| `optimizer_standalone.py` | **Codex** | Optional isolated optimizer entry point. |
| `test_consumer.py`, `test_optimizer*.py` | **Codex** | Consumer journey, optimizer, explanation and market-adapter checks. |
| `universe.csv` | **Codex** | The instrument list. Pure data, append-only. |
| `fees.py` | **Codex** | New, unwritten. See task 2. |

New files belong to whoever creates them. Say so in the commit message.

## Git workflow

Branches and PRs are too slow for two hours. Both work on `main`, but:

```bash
git pull --rebase origin main    # before every push, no exceptions
git add <only your own files>    # never `git add -A`
git commit -m "..."
git push origin main
```

Commit every 10-15 minutes. Small commits rebase cleanly; one giant commit at
the end does not. If a rebase conflicts, someone edited a file they do not own.

## The consumer app and routing contract

There is **one shared consumer app**, launched through `app.py`. The user asked
for an automatic, readable landing page and deeper analysis on separate pages.
For this migration, Codex replaced `app.py`'s inline UI with a thin call to
`consumer_ui.render_app()`. This specific routing change is complete; `app.py`
remains Claude-owned for future edits. Claude's `xray.py`, `analysis.py` and
`data.py` retain their ownership and analytics contracts.

`consumer_ui.render_app()` registers six `views/` entry points using
`st.navigation`: **Your plan**, **Your portfolio**, **Portfolio details**,
**Monte Carlo**, **Diversify** and **Help me**. Keep `app.py` thin and coordinate
changes to the Codex-owned navigation or page modules. Do not restore the old
six-tab UI, add a second portfolio picker, or recreate the removed `pages/`
directory; `st.navigation` is now the routing source of truth.

**Your portfolio** saves one session profile containing holdings, weights,
value, base currency and data source. Every analysis page reads it. **Your
plan** automatically computes the goal's current-holdings-only target, future
scenarios and a conditional buy/sell/keep plan. Advanced assumptions live on
**Monte Carlo**. **Diversify** explicitly explores new holdings separately and
does not change the saved portfolio. **Help me** and the shared question-mark
explanations do not need market data.

The optional `optimizer_ui.render()` still runs independently through
`optimizer_standalone.py`, with its own form and state, for isolated testing:

```bash
.venv/bin/streamlit run optimizer_standalone.py   # optimizer on its own
.venv/bin/streamlit run app.py                    # the shared consumer app
```

On Windows, use `.\.venv\Scripts\python.exe -m streamlit run app.py` from the
repository root. See `README.md` for the offline walkthrough and `OPTIMIZER.md`
for assumptions and integration details.

Existing environments must reinstall `requirements.txt` after pulling this
migration: the Streamlit minimum is now **1.63** for the verified native help,
navigation and input controls. `.streamlit/config.toml` sets the consumer theme;
`consumer_style.css` provides the page-specific styling.

## The contract

Codex: these signatures are stable. Build against them and they will not move.

```python
# xray.py
normalise_weights(weights: pd.Series) -> pd.Series   # sums to 1.0
daily_returns(prices: pd.DataFrame) -> pd.DataFrame
portfolio_series(returns: pd.DataFrame, weights: pd.Series) -> pd.Series
effective_bets(returns: pd.DataFrame, weights: pd.Series) -> float
risk_contributions(returns, weights) -> pd.DataFrame  # index=ticker,
                                                      # cols: weight,
                                                      # risk_share, standalone_vol

# data.py
fetch_prices(tickers: Iterable[str], start: str = "2005-01-01") -> pd.DataFrame
# adjusted close, one column per ticker, cached to .cache/
```

`consumer_ui.render_monte_carlo()` reuses four functions in `optimizer_ui.py`.
**Treat these as public API** — renaming them breaks the consumer app and the
standalone optimizer:

```python
_allocation_tab(run) -> tuple[str, pd.Series]   # returns (method, target)
_validation_tab(run) -> None
_scenarios_tab(run, method, target) -> None
_trading_tab(run, method, target) -> None
```

The consumer `run` is built by `portfolio_service.build_research()` and includes
the fields the standalone `render()` stores in
`st.session_state["_optimizer_run"]`, plus its goal, target, scenarios and trade
explanations. If a shared renderer requires a new field, update both builders
and their tests together. Do not make shared renderers depend on standalone-only
session state.

Consumer session state uses `_consumer_profile`, `_consumer_goal` and
`_consumer_settings`; `consumer_` widget keys are temporary UI values. The shared
optimizer renderers retain their `optimizer_` and `_optimizer_` keys. Trying a
different method in the detailed optimizer does not silently change the goal on
Your plan. Use `consumer_help.help_text(key)` and `section_heading(label, key)`
for consistent plain-language explanations on new UI elements.

`universe.csv` now has a sixth column, `currency`, holding the listing currency
(`SEK`, `GBp`, `JPY`, ...). `optimizer_market.load_market()` uses it to convert every holding into one
base currency, which is why a Swedish portfolio no longer mixes SEK and USD
series. **Any row you append needs it.** Columns are:
`ticker,name,country,sector,asset_class,currency`.
Anything appended must use a Yahoo Finance ticker and pass:

```bash
.venv/bin/python validate_universe.py     # add --prune to drop dead tickers
```

## Tasks for Codex, in priority order

**1. Widen `universe.csv`.** Pure grind, zero merge risk, immediate payoff —
the country filter is only as good as the data behind it. Currently 244
instruments, thin outside the US and the Nordics. Wanted: Canada, Australia,
India, Brazil, more UCITS ETFs listed in Europe (`.DE`, `.L`, `.AS` suffixes),
more Swedish mid-caps. Validate before pushing.

**1b. FX adapter repair — completed.** `optimizer_market.load_market()` now
validates FX observations and applies Claude's `data.sanity_check_fx()` before
conversion, including the historical 100x scale-break case. Listing currencies
come from `universe.csv`; Yahoo metadata is requested only for unknown listings.
Missing or invalid currency/FX information fails visibly. The consumer and
standalone apps share this adapter. Regression coverage is in
`test_optimizer_market.py`.

**2. `fees.py` — the cost of owning this.** Add a `fee_bps` column to
`universe.csv` for funds, then:

```python
def portfolio_fee(weights: pd.Series, universe: pd.DataFrame) -> float
def fee_drag(annual_fee: float, years: int = 30, initial: float = 100_000,
             gross_return: float = 0.07) -> pd.DataFrame
def cheaper_twin(weights, universe, returns) -> pd.DataFrame
```

`cheaper_twin` is the money shot: find a lower-fee fund in the same country and
asset class with correlation above 0.95 to something already held, and report
what swapping saves over 30 years in kronor. Expose it, then coordinate its UI
integration with the owner of the consumer page module.

**3. `api.py` — FastAPI wrapper.** Only if you want a Lovable front end.
One endpoint, `POST /xray`, taking `{"holdings": {"AAPL": 0.3, ...}}` and
returning the same numbers the Streamlit app shows. The analytics are already
UI-free, so this is a thin adapter over `xray.py` and `analysis.py`, not a
rewrite. Streamlit stays as the fallback demo.

## Splitting a person, not just a file

If you have two humans as well as two agents, the highest-value human job is
**not coding**. It is finding a portfolio that produces a shocking number, and
rehearsing the 3-minute pitch around it. The demo is the deliverable.
