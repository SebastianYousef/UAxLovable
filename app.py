"""Portfolio X-Ray -- what you actually own, and what it actually risks."""
from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from analysis import (correlation_clusters, diversifier_scan,
                      equal_weight_comparison, exposure_breakdown,
                      rolling_risk, tail_risk, top_holdings_share)
from data import fetch_prices
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

# -------------------------------------------------------------------- data

try:
    prices = load_prices(tuple(weights.index) + (benchmark_ticker,), start_date)
except Exception as exc:  # noqa: BLE001 -- surface the real reason to the user
    st.error(f"Could not load prices: {exc}")
    st.stop()

returns = daily_returns(prices)
holdings_returns = returns[list(weights.index)].dropna(how="all")
portfolio = portfolio_series(holdings_returns, weights)
benchmark = returns[benchmark_ticker]
contributions = risk_contributions(holdings_returns, weights)

# ---------------------------------------------------------------- headline

st.title("🔬 Portfolio X-Ray")

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

overview, exposure, risk, stress, additions = st.tabs(
    ["Overview", "Exposure", "Risk", "Stress tests", "What to add"]
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
