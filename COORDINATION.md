# Working in parallel (Claude + Codex)

Two agents editing the same file at a hackathon means merge conflicts at minute
110. **Split by file, not by feature.** Each file below has exactly one owner.

## Ownership

| File | Owner | Do not edit if you are the other agent |
|---|---|---|
| `app.py` | Claude | The UI. Every screen element lives here. |
| `xray.py` | Claude | Core risk maths. |
| `analysis.py` | Claude | Exposure, tail risk, clustering, diversifier scan. |
| `data.py` | Claude | Price loading + cache. |
| `universe.csv` | **Codex** | The instrument list. Pure data, append-only. |
| `fees.py` | **Codex** | New, unwritten. See task 2. |
| `api.py` | **Codex** | New, unwritten. See task 3. |

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

`universe.csv` columns are fixed: `ticker,name,country,sector,asset_class`.
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
