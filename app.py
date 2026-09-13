"""Portfolio X-Ray -- what you actually own, and what it actually risks."""
from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from analysis import (correlation_clusters, diversifier_scan,
                      equal_weight_comparison, exposure_breakdown,
                      rolling_risk, tail_risk, top_holdings_share)
from data import fetch_prices, sanity_check_fx
from optimizer import aligned_returns, holdout_validation, optimize
from optimizer_ui import (REVIEWS, _allocation_tab, _scenarios_tab,
                          _trading_tab, _validation_tab)
from xray import (annualised_return, annualised_vol, average_correlation,
                  benchmark_fit, daily_returns, effective_bets,
                  effective_positions, max_drawdown, normalise_weights,
                  portfolio_series, risk_contributions, sharpe, shock_report)

st.set_page_config(page_title="Portfolio X-Ray", page_icon="🔬", layout="wide")

PRESETS: dict[str, list[str]] = {
    "Classic 'I'm diversified'": ["SPY", "QQQ", "VGT", "AAPL", "MSFT", "AGG"],
    "Swedish retail favourites": ["VOLV-B.ST", "ERIC-B.ST", "INVE-B.ST", "HM-B.ST",
                                  "SEB-A.ST", "EVO.ST"],
    "Nordic blue chips": ["NOVO-B.CO", "EQNR.OL", "NOKIA.HE", "ATCO-A.ST",
                          "DSV.CO", "SAMPO.HE"],
    "60/40 with a tech tilt": ["SPY", "QQQ", "AGG", "TLT"],
    "Genuinely spread out": ["VT", "AGG", "GLD", "VNQ", "VWO", "TIP"],
    "Magnificent Seven": ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA"],
}

# Broad, liquid instruments spanning asset classes -- the pool searched when
# answering "what should I add?".
DIVERSIFIER_POOL = ["AGG", "TLT", "SHY", "TIP", "LQD", "HYG", "EMB", "GLD", "SLV",
                    "DBC", "VNQ", "VNQI", "VWO", "VEA", "EFA", "EWD", "EWJ",
                    "USMV", "XLU", "XLP", "BTC-USD"]

BENCHMARKS = ["SPY", "VT", "QQQ", "EWD", "IEUR", "AGG", "EFA"]


@st.cache_data
def load_universe() -> pd.DataFrame:
    return pd.read_csv("universe.csv")


@st.cache_data(show_spinner="Fetching prices…")
def load_prices(tickers: tuple[str, ...], start: str) -> pd.DataFrame:
    return fetch_prices(tickers, start)


@st.cache_data(show_spinner="Fetching prices and exchange rates…")
def load_in_currency(tickers: tuple[str, ...], start: str, base: str,
                     currencies: tuple[tuple[str, str], ...]) -> pd.DataFrame:
    """Prices converted to one currency, so cross-country risk is real risk.

    A Swedish investor holding US stocks carries the dollar as well as the
    stock; comparing raw local-currency series silently ignores that. Listing
    currencies come from universe.csv rather than a per-ticker lookup, so this
    costs one download per FX pair instead of one per holding.
    """
    from optimizer_market import convert_to_base

    currencies = dict(currencies)
    prices = fetch_prices(tickers, start).reindex(columns=list(tickers))

    major = {"GBp": "GBP", "GBX": "GBP"}
    pairs = tuple(sorted({f"{major.get(c, c)}{base}=X" for c in currencies.values()
                          if major.get(c, c) != base}))
    fx = sanity_check_fx(fetch_prices(pairs, start)) if pairs else pd.DataFrame()

    return convert_to_base(prices, currencies, fx, base)


universe = load_universe()
labels = {
    row.ticker: f"{row.ticker} · {row.name} · {row.country}"
    for row in universe.itertuples()
}

# ------------------------------------------------------------------ sidebar

st.sidebar.title("Build a portfolio")

if "holdings" not in st.session_state:
    st.session_state.holdings = PRESETS["Classic 'I'm diversified'"]


def apply_preset() -> None:
    st.session_state.holdings = list(PRESETS[st.session_state.preset_choice])


st.sidebar.selectbox("Start from a preset", list(PRESETS), key="preset_choice",
                     on_change=apply_preset)

with st.sidebar.expander("Filter what you can pick from", expanded=False):
    countries = st.multiselect("Country", sorted(universe["country"].unique()))
    asset_classes = st.multiselect("Asset class",
                                   sorted(universe["asset_class"].unique()))
    sectors = st.multiselect("Sector", sorted(universe["sector"].unique()))

filtered = universe
if countries:
    filtered = filtered[filtered["country"].isin(countries)]
if asset_classes:
    filtered = filtered[filtered["asset_class"].isin(asset_classes)]
if sectors:
    filtered = filtered[filtered["sector"].isin(sectors)]

st.sidebar.caption(
    f"{len(filtered)} of {len(universe)} instruments match — "
    "type in the box below to search them."
)

# Streamlit requires every selected value to also be an option.
options = sorted(set(filtered["ticker"]) | set(st.session_state.holdings))
selected = st.sidebar.multiselect(
    "Holdings", options, key="holdings",
    format_func=lambda t: labels.get(t, t),
    placeholder="Search by ticker, company or country…",
)

if len(selected) < 2:
    st.title("🔬 Portfolio X-Ray")
    st.info("Pick at least two holdings in the sidebar to x-ray a portfolio.")
    st.stop()

equal_weighted = st.sidebar.toggle("Equal weight everything", value=True)

if equal_weighted:
    weights = normalise_weights(pd.Series(1.0, index=selected))
else:
    previous = st.session_state.get("weight_table", {})
    editable = pd.DataFrame({
        "ticker": selected,
        "weight": [float(previous.get(t, round(100 / len(selected), 1)))
                   for t in selected],
    })
    edited = st.sidebar.data_editor(
        editable, hide_index=True, width="stretch",
        disabled=["ticker"],
        column_config={"weight": st.column_config.NumberColumn(
            "Weight", min_value=0.0, step=1.0, format="%.1f")},
    )
    st.session_state.weight_table = dict(zip(edited["ticker"], edited["weight"]))
    if edited["weight"].sum() <= 0:
        st.sidebar.error("Weights must add up to more than zero.")
        st.stop()
    weights = normalise_weights(edited.set_index("ticker")["weight"])

benchmark_ticker = st.sidebar.selectbox("Compare against", BENCHMARKS, index=0)
start_date = st.sidebar.selectbox(
    "History from", ["2005-01-01", "2010-01-01", "2015-01-01", "2020-01-01"],
    index=0,
)
base_currency = st.sidebar.selectbox("Value everything in",
                                     ["USD", "SEK", "EUR", "GBP"], index=0)

# -------------------------------------------------------------------- data

# dict.fromkeys dedupes while keeping order: the benchmark is often also a
# holding, and a repeated ticker would produce duplicate price columns.
tickers = tuple(dict.fromkeys(tuple(weights.index) + (benchmark_ticker,)))
currency_warning = None
listing_currency = dict(zip(universe["ticker"], universe["currency"]))
currencies = {t: listing_currency.get(t) for t in tickers}

try:
    prices = load_in_currency(tickers, start_date, base_currency,
                              tuple(sorted(currencies.items())))
except Exception as exc:  # noqa: BLE001 -- fall back rather than lose the demo
    try:
        prices = load_prices(tickers, start_date)
    except Exception as inner:  # noqa: BLE001 -- surface the real reason
        st.error(f"Could not load prices: {inner}")
        st.stop()
    currency_warning = (
        f"Could not convert to {base_currency} ({exc}). Showing each holding "
        "in its own listing currency, so cross-country numbers ignore "
        "exchange-rate moves."
    )

returns = daily_returns(prices)
holdings_returns = returns[list(weights.index)].dropna(how="all")
portfolio = portfolio_series(holdings_returns, weights)
benchmark = returns[benchmark_ticker]
contributions = risk_contributions(holdings_returns, weights)

# ---------------------------------------------------------------- headline

st.title("🔬 Portfolio X-Ray")

if currency_warning:
    st.warning(currency_warning)

n_bets = effective_bets(holdings_returns, weights)
st.subheader(
    f"You hold **{len(weights)} positions** — but you own "
    f"**{n_bets:.2f} independent bets**."
)
st.caption(
    "Holdings that move together are one bet wearing several hats. By size "
    f"alone this looks like {effective_positions(weights):.1f} positions; once "
    f"correlation is accounted for, only {n_bets:.2f} do separate work."
)

fit = benchmark_fit(portfolio, benchmark)
metrics = st.columns(6)
metrics[0].metric("Return p.a.", f"{annualised_return(portfolio):.1%}")
metrics[1].metric("Volatility p.a.", f"{annualised_vol(portfolio):.1%}")
metrics[2].metric("Worst drawdown", f"{max_drawdown(portfolio):.1%}")
metrics[3].metric("Sharpe", f"{sharpe(portfolio):.2f}")
metrics[4].metric("Beta", f"{fit['beta']:.2f}")
metrics[5].metric(f"Explained by {benchmark_ticker}", f"{fit['r2']:.0%}")

if fit["r2"] > 0.9:
    st.warning(
        f"{fit['r2']:.0%} of this portfolio's movement is explained by "
        f"{benchmark_ticker} alone. You are paying for stock picking and "
        "receiving the index."
    )

overview, exposure, risk, stress, additions, optimise = st.tabs(
    ["Overview", "Exposure", "Risk", "Stress tests", "What to add", "Optimise"]
)

# ---------------------------------------------------------------- overview

with overview:
    left, right = st.columns([3, 2])

    with left:
        st.subheader("Where your money is vs. where your risk is")
        chart_data = (
            contributions[["weight", "risk_share"]]
            .rename(columns={"weight": "Share of money",
                             "risk_share": "Share of risk"})
            .reset_index(names="ticker")
            .melt(id_vars="ticker", var_name="measure", value_name="share")
        )
        st.altair_chart(
            alt.Chart(chart_data).mark_bar().encode(
                x=alt.X("share:Q", axis=alt.Axis(format="%"), title=None),
                y=alt.Y("ticker:N", sort="-x", title=None),
                yOffset="measure:N",
                color=alt.Color("measure:N", title=None,
                                scale=alt.Scale(range=["#7c8ca8", "#e0643a"])),
                tooltip=["ticker", "measure", alt.Tooltip("share:Q", format=".1%")],
            ).properties(height=38 * len(contributions)),
            width="stretch",
        )
        worst = (contributions["risk_share"] - contributions["weight"]).idxmax()
        row = contributions.loc[worst]
        st.caption(
            f"**{worst}** is {row['weight']:.0%} of your money but "
            f"{row['risk_share']:.0%} of your risk."
        )

    with right:
        st.subheader("Concentration")
        st.metric("In the top 3 positions", f"{top_holdings_share(weights, 3):.0%}")

        comparison = equal_weight_comparison(holdings_returns, weights)
        st.metric(
            "Bets vs. equal weighting",
            f"{comparison['chosen_bets']:.2f}",
            delta=f"{comparison['chosen_bets'] - comparison['equal_bets']:+.2f} "
                  "vs equal",
        )
        st.caption(
            "Equal weighting across the same names is a hard benchmark. A "
            "negative number means your sizing is concentrating risk, not "
            "spreading it."
        )

    st.subheader(f"Growth of 100 vs {benchmark_ticker}")
    growth = pd.DataFrame({
        "Portfolio": (1 + portfolio).cumprod() * 100,
        benchmark_ticker: (1 + benchmark).cumprod() * 100,
    }).dropna()
    st.line_chart(growth)

# ---------------------------------------------------------------- exposure

with exposure:
    st.subheader("Money and risk, broken down")
    st.caption(
        "The gap column is the point: a country or sector can be a small part "
        "of the money and a large part of the risk."
    )

    for dimension, title in [("country", "By country"),
                             ("sector", "By sector"),
                             ("asset_class", "By asset class")]:
        breakdown = exposure_breakdown(contributions, universe, dimension)

        st.markdown(f"**{title}**")
        chart_data = (
            breakdown[["weight", "risk_share"]]
            .rename(columns={"weight": "Share of money",
                             "risk_share": "Share of risk"})
            .reset_index(names=dimension)
            .melt(id_vars=dimension, var_name="measure", value_name="share")
        )
        st.altair_chart(
            alt.Chart(chart_data).mark_bar().encode(
                x=alt.X("share:Q", axis=alt.Axis(format="%"), title=None),
                y=alt.Y(f"{dimension}:N", sort="-x", title=None),
                yOffset="measure:N",
                color=alt.Color("measure:N", title=None,
                                scale=alt.Scale(range=["#7c8ca8", "#e0643a"])),
                tooltip=[dimension, "measure",
                         alt.Tooltip("share:Q", format=".1%")],
            ).properties(height=max(120, 38 * len(breakdown))),
            width="stretch",
        )

        biggest = breakdown["gap"].idxmax()
        if breakdown.loc[biggest, "gap"] > 0.02:
            st.caption(
                f"**{biggest}** carries {breakdown.loc[biggest, 'gap']:+.0%} more "
                "risk than its share of the money."
            )

# -------------------------------------------------------------------- risk

with risk:
    st.subheader("What the bad days look like")

    tails = tail_risk(portfolio)
    cols = st.columns(4)
    cols[0].metric("Bad day (VaR 95%)", f"{tails['var_95']:.1%}",
                   help="5% of days are worse than this.")
    cols[1].metric("When it's worse (CVaR)", f"{tails['cvar_95']:.1%}",
                   help="Average loss on the days that break through VaR.")
    cols[2].metric("Worst single day", f"{tails['worst_day']:.1%}")
    cols[3].metric("Longest time underwater",
                   f"{tails['longest_underwater_days']:.0f} days",
                   help="Longest stretch spent below a previous peak.")

    st.divider()
    st.subheader("Which holdings are secretly the same bet?")

    threshold = st.slider("Count as 'the same bet' above correlation", 0.5, 0.95,
                          0.75, 0.05)
    clusters = correlation_clusters(holdings_returns, threshold)
    grouped = [c for c in clusters if len(c) > 1]

    st.markdown(
        f"Your {len(weights)} holdings form **{len(clusters)} clusters** at this "
        "threshold."
    )
    if grouped:
        for cluster in grouped:
            share = weights[cluster].sum()
            st.markdown(
                f"- {' + '.join(cluster)} — **{share:.0%}** of the portfolio "
                "moving as one"
            )
    else:
        st.markdown("No two holdings are that tightly linked. Genuinely spread out.")

    st.divider()
    left, right = st.columns(2)

    with left:
        st.subheader("Correlation")
        corr_long = (
            holdings_returns.corr()
            .reset_index(names="a")
            .melt(id_vars="a", var_name="b", value_name="correlation")
        )
        st.altair_chart(
            alt.Chart(corr_long).mark_rect().encode(
                x=alt.X("a:N", title=None),
                y=alt.Y("b:N", title=None),
                color=alt.Color("correlation:Q", title=None,
                                scale=alt.Scale(scheme="redyellowblue",
                                                reverse=True, domain=[-1, 1])),
                tooltip=["a", "b", alt.Tooltip("correlation:Q", format=".2f")],
            ).properties(height=36 * len(holdings_returns.columns)),
            width="stretch",
        )
        st.caption(
            "Average pairwise correlation: "
            f"**{average_correlation(holdings_returns, weights):.2f}**"
        )

    with right:
        st.subheader("Risk over time")
        st.caption(
            "Correlation is not a constant. It rises in crashes — exactly when "
            "diversification was supposed to help."
        )
        st.line_chart(rolling_risk(portfolio, benchmark))

# ------------------------------------------------------------------ stress

with stress:
    st.subheader("What this portfolio would have done in past crashes")

    shocks = shock_report(holdings_returns, weights)
    st.dataframe(
        shocks.style.format({"return": "{:.1%}", "max_drawdown": "{:.1%}",
                             "coverage": "{:.0%}"}),
        width="stretch",
    )
    st.caption(
        "`coverage` is how much of the portfolio actually existed back then. "
        "The rest is excluded and the remainder reweighted, rather than "
        "quietly assumed to be flat."
    )

    worst_shock = shocks["return"].idxmin()
    if pd.notna(shocks.loc[worst_shock, "return"]):
        st.markdown(
            f"Worst episode: **{worst_shock}**, "
            f"**{shocks.loc[worst_shock, 'return']:.1%}**. On 100 000 kr that is "
            f"**{abs(shocks.loc[worst_shock, 'return']) * 100_000:,.0f} kr** gone."
        )

# --------------------------------------------------------------- what to add

with additions:
    st.subheader("What would actually diversify this?")
    st.caption(
        "Each candidate is added at 10% and the portfolio scaled down to make "
        "room. Ranked by how many independent bets it buys you — measured on "
        "the history the candidate and portfolio share, so a short track "
        "record cannot flatter it."
    )

    if st.button("Scan for diversifiers", type="primary"):
        pool = tuple(t for t in DIVERSIFIER_POOL if t not in weights.index)
        try:
            candidate_prices = load_prices(pool, start_date)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not load candidates: {exc}")
        else:
            scan = diversifier_scan(holdings_returns, weights,
                                    daily_returns(candidate_prices))
            if scan.empty:
                st.info("Not enough overlapping history to compare candidates.")
            else:
                display = scan.join(universe.set_index("ticker")[["name",
                                                                 "asset_class"]])
                st.dataframe(
                    display[["name", "asset_class", "bets_gained", "bets_after",
                             "vol_change", "drawdown_after", "years_of_history"]]
                    .style.format({
                        "bets_gained": "{:+.2f}", "bets_after": "{:.2f}",
                        "vol_change": "{:+.1%}", "drawdown_after": "{:.1%}",
                        "years_of_history": "{:.0f}y",
                    }),
                    width="stretch",
                )

                best = scan.index[0]
                st.success(
                    f"**{labels.get(best, best)}** buys the most diversification: "
                    f"{scan.loc[best, 'bets_gained']:+.2f} bets, with a "
                    f"{scan.loc[best, 'vol_change']:+.1%} change in volatility."
                )

# ---------------------------------------------------------------- optimise

with optimise:
    st.subheader("Keep these holdings. Find better weights.")
    st.caption(
        "The tabs so far diagnose the portfolio you have. This one searches "
        "for weightings of the *same* holdings that carry less risk for the "
        "return — then checks the answer on history it was not fitted to. "
        "'Best' means best under one objective and past data, not a promise."
    )

    if len(weights) > 15:
        st.info(
            f"{len(weights)} holdings selected. The search still runs, but it "
            "is slower and the estimates get noisier the more holdings you add."
        )

    with st.form("xray_optimiser"):
        row = st.columns(4)
        value = row[0].number_input(f"Portfolio value ({base_currency})",
                                    min_value=100.0, max_value=1e10,
                                    value=100_000.0, step=1000.0)
        cap_percent = row[1].slider("Max weight per holding (%)", 5, 100, 50, 5)
        samples = row[2].select_slider("Portfolios to search",
                                       [2000, 5000, 10000, 25000, 50000],
                                       value=10000)
        risk_free = row[3].number_input("Risk-free rate (% / year)",
                                        min_value=-5.0, max_value=25.0,
                                        value=3.0, step=0.25)

        with st.expander("Estimation and trading assumptions"):
            fine = st.columns(3)
            mean_shrink = fine[0].slider(
                "Return shrinkage (%)", 0, 100, 50, 10,
                help="Pull each historical mean toward the average. Historical "
                     "means are a famously bad forecast, so shrinking them hard "
                     "is the honest default.")
            covariance_shrink = fine[1].slider("Covariance shrinkage (%)", 0, 100, 10, 5)
            seed = fine[2].number_input("Random seed", 0, 2147483647, 42, 1)

            trade = st.columns(4)
            review = trade[0].selectbox("Review schedule", list(REVIEWS), index=1)
            band = trade[1].slider("Rebalance band (points)", 1, 20, 5)
            costs = trade[2].number_input("Trading cost (bps)", 0.0, 1000.0, 10.0, 5.0)
            block_days = trade[3].selectbox("Scenario block (days)", [5, 21, 63], index=1)

        submitted = st.form_submit_button("Run optimisation", type="primary")

    if submitted:
        st.session_state.pop("_optimizer_run", None)
        st.session_state.pop("_optimizer_scenario", None)
        try:
            if len(weights) * cap_percent < 100:
                raise ValueError(
                    f"The cap is infeasible: {len(weights)} × {cap_percent}% is "
                    f"below 100%. Raise it to at least {100 / len(weights):.1f}%."
                )
            with st.spinner("Searching allocations and testing them out of sample…"):
                optimiser_returns = aligned_returns(prices[list(weights.index)])
                current = normalise_weights(
                    weights.reindex(optimiser_returns.columns))
                params = dict(samples=int(samples), cap=cap_percent / 100,
                              risk_free=risk_free / 100, seed=int(seed),
                              mean_shrinkage=mean_shrink / 100,
                              covariance_shrinkage=covariance_shrink / 100)
                st.session_state["_optimizer_run"] = dict(
                    id=pd.Timestamp.now().isoformat(),
                    result=optimize(optimiser_returns, current, **params),
                    validation=holdout_validation(
                        optimiser_returns, current, **params, cost_bps=costs,
                        band=band / 100, review_days=REVIEWS[review]),
                    returns=optimiser_returns, current=current, value=value,
                    base=base_currency, source="Yahoo Finance", start=start_date,
                    currencies=currencies, review=review, band=band / 100,
                    cost_bps=costs, block_days=block_days, **params)
        except Exception as exc:  # noqa: BLE001 -- show the real constraint
            st.error(f"Optimisation could not run: {exc}")

    run = st.session_state.get("_optimizer_run")

    if not run:
        st.info("Set your assumptions above, then run the optimisation.")
    elif list(run["current"].index) != list(weights.index):
        st.info(
            "Your holdings changed since the last run. Run the optimisation "
            "again to refresh these results."
        )
    else:
        dates = run["returns"].index
        st.caption(
            f"{len(dates):,} shared daily returns · {dates[0]:%d %b %Y} – "
            f"{dates[-1]:%d %b %Y} · {run['base']} · {run['cap']:.0%} holding "
            f"cap · seed {run['seed']}."
        )
        allocation, validation, scenarios, trading = st.tabs(
            ["Allocation", "Out-of-sample check", "Future scenarios",
             "When to trade"]
        )
        with allocation:
            method, target = _allocation_tab(run)
        with validation:
            _validation_tab(run)
        with scenarios:
            _scenarios_tab(run, method, target)
        with trading:
            _trading_tab(run, method, target)
