"""Portfolio X-Ray -- what you actually own, and what it actually risks."""
from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from data import fetch_prices
from xray import (
    annualised_return,
    annualised_vol,
    average_correlation,
    benchmark_fit,
    daily_returns,
    effective_bets,
    effective_positions,
    max_drawdown,
    normalise_weights,
    portfolio_series,
    risk_contributions,
    sharpe,
    shock_report,
)

st.set_page_config(page_title="Portfolio X-Ray", page_icon="🔬", layout="wide")

PRESETS = {
    "Classic 'I'm diversified'": "SPY, 30\nQQQ, 25\nVGT, 15\nAAPL, 10\nMSFT, 10\nAGG, 10",
    "Swedish retail": "VOLV-B.ST, 20\nERIC-B.ST, 20\nINVE-B.ST, 20\nHM-B.ST, 15\nSEB-A.ST, 15\nAZN.ST, 10",
    "60/40 with a tech tilt": "SPY, 45\nQQQ, 15\nAGG, 30\nTLT, 10",
}


def parse_portfolio(text: str) -> pd.Series:
    """Read 'TICKER, weight' lines. Blank lines and # comments ignored."""
    rows: dict[str, float] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.replace("\t", ",").split(",")]
        if len(parts) < 2:
            continue
        try:
            rows[parts[0].upper()] = float(parts[1].replace("%", ""))
        except ValueError:
            continue
    return pd.Series(rows, dtype=float)


@st.cache_data(show_spinner="Fetching prices…")
def load_prices(tickers: tuple[str, ...], start: str) -> pd.DataFrame:
    return fetch_prices(tickers, start)


# ---------------------------------------------------------------- sidebar

st.sidebar.header("Your portfolio")
preset = st.sidebar.selectbox("Start from", list(PRESETS))
holdings_text = st.sidebar.text_area(
    "One holding per line: TICKER, weight", PRESETS[preset], height=200
)
benchmark_ticker = st.sidebar.text_input("Benchmark", "SPY").strip().upper()
start_date = st.sidebar.text_input("History from", "2005-01-01").strip()

weights_raw = parse_portfolio(holdings_text)
if weights_raw.empty:
    st.info("Add some holdings in the sidebar to get started.")
    st.stop()

weights = normalise_weights(weights_raw)

try:
    prices = load_prices(tuple(weights.index) + (benchmark_ticker,), start_date)
except Exception as exc:  # noqa: BLE001 -- show the user what broke
    st.error(f"Could not load prices: {exc}")
    st.stop()

returns = daily_returns(prices)
holdings_returns = returns[list(weights.index)]
portfolio = portfolio_series(holdings_returns, weights)
benchmark = returns[benchmark_ticker]

# ---------------------------------------------------------------- headline

st.title("🔬 Portfolio X-Ray")

n_holdings = len(weights)
n_bets = effective_bets(holdings_returns, weights)
n_positions = effective_positions(weights)

st.subheader(
    f"You hold **{n_holdings} positions** — but you own "
    f"**{n_bets:.1f} independent bets**."
)
st.caption(
    "Holdings that move together are one bet wearing several hats. "
    f"By size alone you look like {n_positions:.1f} positions; once correlation "
    f"is accounted for, only {n_bets:.1f} of them are really doing separate work."
)

fit = benchmark_fit(portfolio, benchmark)

cols = st.columns(5)
cols[0].metric("Return p.a.", f"{annualised_return(portfolio):.1%}")
cols[1].metric("Volatility p.a.", f"{annualised_vol(portfolio):.1%}")
cols[2].metric("Worst drawdown", f"{max_drawdown(portfolio):.1%}")
cols[3].metric("Sharpe", f"{sharpe(portfolio):.2f}")
cols[4].metric(f"Explained by {benchmark_ticker}", f"{fit['r2']:.0%}")

if fit["r2"] > 0.9:
    st.warning(
        f"{fit['r2']:.0%} of this portfolio's movement is explained by "
        f"{benchmark_ticker} alone (beta {fit['beta']:.2f}). You are paying "
        f"for stock picking and receiving the index."
    )

st.divider()

# ------------------------------------------------- weight vs risk contribution

left, right = st.columns([3, 2])

contributions = risk_contributions(holdings_returns, weights)

with left:
    st.subheader("Where your money is vs. where your risk is")

    chart_data = (
        contributions[["weight", "risk_share"]]
        .rename(columns={"weight": "Share of money", "risk_share": "Share of risk"})
        .reset_index(names="ticker")
        .melt(id_vars="ticker", var_name="measure", value_name="share")
    )

    st.altair_chart(
        alt.Chart(chart_data)
        .mark_bar()
        .encode(
            x=alt.X("share:Q", axis=alt.Axis(format="%"), title=None),
            y=alt.Y("ticker:N", sort="-x", title=None),
            yOffset="measure:N",
            color=alt.Color("measure:N", title=None,
                            scale=alt.Scale(range=["#7c8ca8", "#e0643a"])),
            tooltip=[
                "ticker",
                "measure",
                alt.Tooltip("share:Q", format=".1%"),
            ],
        )
        .properties(height=40 * len(contributions)),
        use_container_width=True,
    )

    gap = (contributions["risk_share"] - contributions["weight"]).idxmax()
    gap_row = contributions.loc[gap]
    st.caption(
        f"**{gap}** is {gap_row['weight']:.0%} of your money but "
        f"{gap_row['risk_share']:.0%} of your risk."
    )

with right:
    st.subheader("How alike is everything?")
    corr = holdings_returns.corr()

    corr_long = corr.reset_index(names="a").melt(id_vars="a", var_name="b",
                                                 value_name="correlation")
    st.altair_chart(
        alt.Chart(corr_long)
        .mark_rect()
        .encode(
            x=alt.X("a:N", title=None),
            y=alt.Y("b:N", title=None),
            color=alt.Color(
                "correlation:Q",
                scale=alt.Scale(scheme="redyellowblue", reverse=True,
                                domain=[-1, 1]),
                title=None,
            ),
            tooltip=["a", "b", alt.Tooltip("correlation:Q", format=".2f")],
        )
        .properties(height=40 * len(corr)),
        use_container_width=True,
    )
    st.caption(
        f"Average pairwise correlation: "
        f"**{average_correlation(holdings_returns, weights):.2f}** "
        "(1.00 means everything falls on the same day)."
    )

st.divider()

# ---------------------------------------------------------------- stress test

st.subheader("What this portfolio would have done in past crashes")

shocks = shock_report(holdings_returns, weights)
st.dataframe(
    shocks.style.format({
        "return": "{:.1%}",
        "max_drawdown": "{:.1%}",
        "coverage": "{:.0%}",
    }),
    use_container_width=True,
)
st.caption(
    "`coverage` is how much of the portfolio actually existed back then — "
    "the rest is excluded and the remainder reweighted, rather than quietly "
    "assumed to be flat."
)

# ---------------------------------------------------------------- growth

st.subheader(f"Growth of 100 vs {benchmark_ticker}")

growth = pd.DataFrame({
    "Portfolio": (1 + portfolio).cumprod() * 100,
    benchmark_ticker: (1 + benchmark).cumprod() * 100,
}).dropna()

st.line_chart(growth)
