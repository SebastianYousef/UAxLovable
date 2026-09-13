# Working in parallel (Claude + Codex)

Two agents editing the same file at a hackathon means merge conflicts at minute
110. **Split by file, not by feature.** Each file below has exactly one owner.

## Ownership

| File | Owner | Do not edit if you are the other agent |
|---|---|---|
| `app.py` | Claude | The single entry point. Every screen element lives here. |
| `xray.py` | Claude | Core risk maths. |
| `analysis.py` | Claude | Exposure, tail risk, clustering, diversifier scan. |
| `data.py` | Claude | Price loading, cache, FX repair. |
| `optimizer.py` | **Codex** | Allocation search, validation, scenarios. |
| `optimizer_ui.py` | **Codex** | Optimizer rendering. |
| `optimizer_market.py` | **Codex** | Currency conversion. |
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

## The apps are merged

There is **one** Streamlit app now: `app.py`, with the optimizer as its sixth
tab. The portfolio is chosen once, in the X-Ray sidebar, and flows into every
tab including the optimizer — there is no second holdings picker.

`pages/1_Monte_Carlo_Optimizer.py` is gone, because a `pages/` directory is what
creates a second nav entry. `optimizer_ui.render()` is untouched and still runs
standalone via `optimizer_standalone.py`, which is where the UI tests now point:

```bash
.venv/bin/streamlit run optimizer_standalone.py   # optimizer on its own
.venv/bin/streamlit run app.py                    # the merged demo
```

Claude edited two Codex-owned files to do this, both minimally:
`test_optimizer_ui.py` (repointed at `optimizer_standalone.py`, and the
navigation test now asserts the merged tab exists) and `universe.csv` (see
below). Nothing in `optimizer.py`, `optimizer_ui.py` or `optimizer_market.py`
was changed.

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

`app.py` renders the optimizer by calling four functions in `optimizer_ui.py`
rather than duplicating them. **Treat these as public API** — rename them and
the merged app breaks:

```python
_allocation_tab(run) -> tuple[str, pd.Series]   # returns (method, target)
_validation_tab(run) -> None
_scenarios_tab(run, method, target) -> None
_trading_tab(run, method, target) -> None
```

`run` is the same dict `render()` builds in `st.session_state["_optimizer_run"]`.
If you add a key to it, `app.py` must set it too — say so.

`universe.csv` now has a sixth column, `currency`, holding the listing currency
(`SEK`, `GBp`, `JPY`, ...). `app.py` uses it to convert every holding into one
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

**1b. `optimizer_market.load_market` has a live data bug.** Yahoo quotes some
FX pairs per 100 units for part of their history — `JPYSEK=X` jumps by exactly
100x on 2017-11-10 — which silently wrecks any portfolio holding that currency.
`data.sanity_check_fx()` repairs it and rejects implausible series; `app.py`
already routes through it. Please apply it to `load_market`'s `fx` frame too.
`load_market` also calls `yf.Ticker().fast_info` once per ticker; the
`currency` column in `universe.csv` now makes that unnecessary.

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
what swapping saves over 30 years in kronor. Expose it and Claude will wire it
into a new tab.

**3. `api.py` — FastAPI wrapper.** Only if you want a Lovable front end.
One endpoint, `POST /xray`, taking `{"holdings": {"AAPL": 0.3, ...}}` and
returning the same numbers the Streamlit app shows. The analytics are already
UI-free, so this is a thin adapter over `xray.py` and `analysis.py`, not a
rewrite. Streamlit stays as the fallback demo.

## Splitting a person, not just a file

If you have two humans as well as two agents, the highest-value human job is
**not coding**. It is finding a portfolio that produces a shocking number, and
rehearsing the 3-minute pitch around it. The demo is the deliverable.
