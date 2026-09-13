"""Codex-owned detailed views of the current, shared portfolio."""
from __future__ import annotations

from collections.abc import Callable

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from analysis import (correlation_clusters, equal_weight_comparison,
                      exposure_breakdown, rolling_risk, tail_risk,
                      top_holdings_share)
from consumer_help import help_text, section_heading
from xray import (SHOCKS, annualised_return, annualised_vol,
                  average_correlation, benchmark_fit, effective_bets,
                  effective_positions, max_drawdown, normalise_weights,
                  portfolio_series, risk_contributions, sharpe, shock_report)


BENCHMARKS = {
    "SPY": "SPY · Large US companies",
    "VT": "VT · Global shares",
    "QQQ": "QQQ · Nasdaq's largest nonfinancial companies",
    "EWD": "EWD · Swedish shares",
    "IEUR": "IEUR · European shares",
    "AGG": "AGG · US bonds",
    "EFA": "EFA · Developed markets outside North America",
}


def _number(value: float, format_spec: str = ".1%") -> str:
    return format(value, format_spec) if np.isfinite(value) else "Unavailable"


def _share_chart(frame: pd.DataFrame, label: str) -> None:
    chart_data = (frame[["weight", "risk_share"]]
                  .rename(columns={"weight": "Share of money", "risk_share": "Share of risk"})
                  .reset_index(names=label)
                  .melt(id_vars=label, var_name="Measure", value_name="Share"))
    st.altair_chart(
        alt.Chart(chart_data).mark_bar(cornerRadiusEnd=3).encode(
            x=alt.X("Share:Q", axis=alt.Axis(format="%"), title="Share of the portfolio"),
            y=alt.Y(f"{label}:N", sort="-x", title=None),
            yOffset="Measure:N",
            color=alt.Color("Measure:N", title=None,
                            scale=alt.Scale(domain=["Share of money", "Share of risk"],
                                            range=["#16806A", "#CB794A"])),
            tooltip=[alt.Tooltip(f"{label}:N"), "Measure:N", alt.Tooltip("Share:Q", format=".1%")],
        ).properties(height=max(150, 44 * len(frame))), width="stretch")
    st.caption("Compare the two bars for each row. A longer risk bar means that holding or group "
               "contributed more to price swings than its share of your money. A negative risk share "
               "means it offset some swings in this history.")


def _benchmark(run: dict, load_prices_fn: Callable) -> tuple[str, pd.Series | None]:
    returns = run["returns"]
    synthetic = "synthetic" in str(run.get("source", "")).lower()
    available = list(returns.columns)
    choices = available if synthetic else list(dict.fromkeys([*BENCHMARKS, *available]))
    preferred = next((ticker for ticker in BENCHMARKS if ticker in available), choices[0])
    # A stored choice can disappear when the portfolio or data source changes.
    if st.session_state.get("details_benchmark") not in choices:
        st.session_state["details_benchmark"] = preferred
    ticker = st.selectbox(
        "Compare with", choices, key="details_benchmark",
        format_func=lambda code: BENCHMARKS.get(code, code), help=help_text("benchmark"))
    if synthetic:
        st.caption("This comparison uses the same synthetic example data as your portfolio, "
                   "not the investment's actual market history.")
    if ticker in returns:
        return ticker, returns[ticker]
    try:
        prices, _ = load_prices_fn((ticker,), run["start"], run["base"])
        if ticker not in prices:
            raise ValueError("The data provider did not return this investment.")
        benchmark = prices[ticker].pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan).dropna()
        if benchmark.empty:
            raise ValueError("There are no usable daily changes for this investment.")
        return ticker, benchmark
    except Exception as exc:
        st.info(f"The comparison investment could not be loaded: {exc} "
                "Your portfolio's other details are still available.")
        return ticker, None


def _overview(run: dict, returns: pd.DataFrame, weights: pd.Series,
              portfolio: pd.Series, contributions: pd.DataFrame | None,
              load_prices_fn: Callable) -> tuple[str, pd.Series | None]:
    section_heading("What is driving your portfolio?", "effective_bets")
    first = st.columns(3)
    first[0].metric("Separate patterns of risk", _number(effective_bets(returns, weights), ".2f"),
                    help=help_text("effective_bets"), border=True)
    first[1].metric("Spread by investment size", _number(effective_positions(weights), ".2f"),
                    help=help_text("effective_positions"), border=True)
    first[2].metric("Money in your three largest holdings", _number(top_holdings_share(weights, 3)),
                    help=help_text("concentration"), border=True)
    st.caption("More investment names do not always mean more independence. These scores "
               "look at both how much you hold and how the investments moved together.")

    second = st.columns(4)
    for col, label, value, key, fmt in (
        (second[0], "Yearly growth in this history", annualised_return(portfolio), "returns", ".1%"),
        (second[1], "Size of yearly price swings", annualised_vol(portfolio), "volatility", ".1%"),
        (second[2], "Largest fall from a high", max_drawdown(portfolio), "drawdown", ".1%"),
        (second[3], "Return for the risk taken", sharpe(portfolio), "sharpe", ".2f"),
    ):
        col.metric(label, _number(value, fmt), help=help_text(key), border=True)
    st.caption("Yearly growth compounds the daily returns onto a one-year scale. The return-for-risk "
               "score here uses a 0% cash reference; the optimizer uses its own selected reference rate.")

    with st.container(border=True):
        section_heading("Where your money sits — and where the swings come from", "risk_share")
        if contributions is None:
            st.info("Risk shares are unavailable because this history has no measurable portfolio swings.")
        else:
            _share_chart(contributions, "Holding")
            valid_gap = (contributions["risk_share"] - contributions["weight"]).dropna()
            if not valid_gap.empty:
                ticker = valid_gap.idxmax()
                row = contributions.loc[ticker]
                st.write(f"**{ticker}** holds **{row['weight']:.1%}** of your money and accounts for "
                         f"**{row['risk_share']:.1%}** of the measured price swings.")

    with st.container(border=True):
        section_heading("What if you split the money evenly?", "equal_weight")
        comparison = equal_weight_comparison(returns, weights)
        table = pd.DataFrame({
            "Mix": ["Your current mix", "Same holdings, equal amounts"],
            "Yearly price swings (%)": [comparison["chosen_vol"] * 100, comparison["equal_vol"] * 100],
            "Separate patterns of risk": [comparison["chosen_bets"], comparison["equal_bets"]],
        })
        st.dataframe(table, hide_index=True, width="stretch", column_config={
            "Mix": st.column_config.TextColumn(help=help_text("equal_weight")),
            "Yearly price swings (%)": st.column_config.NumberColumn(format="%.1f%%", help=help_text("volatility")),
            "Separate patterns of risk": st.column_config.NumberColumn(format="%.2f", help=help_text("effective_bets")),
        })
        st.caption("The holdings and dates stay the same. Only their shares change. Lower price swings "
                   "mean a smoother historical ride; more separate patterns suggest less dependence on one source of risk.")

    with st.container(border=True):
        section_heading("How your mix grew beside a comparison investment", "growth")
        ticker, benchmark = _benchmark(run, load_prices_fn)
        if benchmark is None:
            return ticker, None
        joined = pd.concat([portfolio.rename("Your current mix"), benchmark.rename(f"Comparison · {ticker}")], axis=1).dropna()
        if len(joined) < 2:
            st.info("There are not enough shared dates to compare growth. Try another comparison investment.")
            return ticker, None
        growth = (1 + joined).cumprod() * 100
        # The starting point is before the first return, so the first day's loss is visible.
        origin = pd.DataFrame(100.0, columns=growth.columns,
                              index=[growth.index[0] - pd.Timedelta(days=1)])
        st.line_chart(pd.concat([origin, growth]), y_label=f"Value of 100 {run['base']}", width="stretch")
        st.caption(f"Both lines start with 100 {run['base']} on the same dates. Higher means more accumulated "
                   f"growth. Comparison period: {joined.index[0]:%d %b %Y}–{joined.index[-1]:%d %b %Y}. "
                   "Trading costs and taxes are not deducted in this historical chart.")
        if len(joined) < 30 or joined.iloc[:, 1].std() == 0:
            st.info("Market sensitivity needs at least 30 shared days and a comparison investment whose price changes.")
        else:
            fit = benchmark_fit(joined.iloc[:, 0], joined.iloc[:, 1])
            cols = st.columns(2)
            cols[0].metric("Response to the comparison market", _number(fit["beta"], ".2f"),
                           help=help_text("beta"), border=True)
            cols[1].metric("Movements linked to this comparison", _number(fit["r2"], ".0%"),
                           help=help_text("r_squared"), border=True)
            st.caption("A response near 1 means similar-sized moves in the fitted relationship. "
                       "A high linked-movements percentage means the two often moved together; "
                       "it does not mean they hold the same investments.")
        return ticker, benchmark


def _exposure(contributions: pd.DataFrame | None, universe: pd.DataFrame) -> None:
    section_heading("The areas your money depends on", "exposure")
    st.caption("These are the listed investments' labels. A fund's country or sector label is not "
               "a full breakdown of the companies it owns inside.")
    if contributions is None:
        st.info("The money-and-risk breakdown needs measurable historical portfolio swings.")
        return
    for dimension, label, key in (("country", "By country", "exposure"),
                                  ("sector", "By type of business", "sector"),
                                  ("asset_class", "By investment type", "asset_class")):
        with st.container(border=True):
            section_heading(label, key)
            meta = universe.copy()
            if dimension not in meta:
                meta[dimension] = "Unknown"
            breakdown = exposure_breakdown(contributions, meta, dimension)
            _share_chart(breakdown, "Group")
            display = breakdown.rename(columns={"weight": "Share of money (%)",
                                                 "risk_share": "Share of risk (%)",
                                                 "gap": "Risk minus money (points)"}) * 100
            st.dataframe(display.reset_index(names="Group"), hide_index=True, width="stretch", column_config={
                "Group": st.column_config.TextColumn(help=help_text(key)),
                "Share of money (%)": st.column_config.NumberColumn(format="%.1f%%", help=help_text("weights")),
                "Share of risk (%)": st.column_config.NumberColumn(format="%.1f%%", help=help_text("risk_share")),
                "Risk minus money (points)": st.column_config.NumberColumn(format="%+.1f", help=
                    "Risk share minus money share, in percentage points. If 10% of the money contributes 25% of the risk, the gap is +15 points."),
            })


def _risk(returns: pd.DataFrame, weights: pd.Series, portfolio: pd.Series,
          benchmark: pd.Series | None, base: str, value: float) -> None:
    section_heading("What difficult days looked like", "cvar")
    tails = tail_risk(portfolio)
    first = st.columns(4)
    for col, label, key, tooltip in (
        (first[0], "Bottom 5% boundary", "var_95", "var"),
        (first[1], "Average of the worst 5% of days", "cvar_95", "cvar"),
        (first[2], "Largest one-day loss", "worst_day", "worst_day"),
        (first[3], "Longest wait below a previous high", "longest_underwater_days", "underwater"),
    ):
        amount = tails.get(key, float("nan"))
        label_value = f"{amount:,.0f} trading days" if key == "longest_underwater_days" else _number(amount)
        col.metric(label, label_value, help=help_text(tooltip), border=True)
    if np.isfinite(tails.get("worst_day", np.nan)):
        change = tails["worst_day"] * value
        st.caption(f"Applying that worst day's return to {value:,.0f} {base} would change its value by "
                   f"{change:+,.0f} {base}. It is a historical example, not a maximum possible loss.")
    second = st.columns(4)
    for col, label, key, tooltip in (
        (second[0], "Bottom 1% boundary", "var_99", "var"),
        (second[1], "Largest one-day gain", "best_day", "best_day"),
        (second[2], "Days with a gain", "positive_days", "positive_days"),
        (second[3], "Below the high at the end", "current_drawdown", "drawdown"),
    ):
        col.metric(label, _number(tails.get(key, float("nan"))), help=help_text(tooltip), border=True)

    with st.container(border=True):
        section_heading("Holdings that tended to move together", "cluster")
        threshold = st.slider("How strong must the link be?", 0.5, 0.95, 0.75, 0.05,
                              key="details_cluster_threshold", help=help_text("correlation"))
        clusters = correlation_clusters(returns, threshold)
        st.write(f"Your **{len(weights)} holdings** form **{len(clusters)} groups** at this threshold.")
        linked = [group for group in clusters if len(group) > 1]
        if linked:
            for group in linked:
                st.write(f"**{' + '.join(group)}** · {weights[group].sum():.1%} of your money")
        else:
            st.info("No pairs reached this link threshold in the selected history. They may still share risks.")
        st.caption("A link means the historical correlation met the chosen threshold. Groups can connect "
                   "through other holdings; not every pair within a group must have a strong direct link.")

        section_heading("How closely each pair moved", "correlation")
        long = returns.corr().reset_index(names="Holding").melt(
            id_vars="Holding", var_name="Compared with", value_name="Link")
        st.altair_chart(alt.Chart(long).mark_rect().encode(
            x=alt.X("Holding:N", title=None), y=alt.Y("Compared with:N", title=None),
            color=alt.Color("Link:Q", scale=alt.Scale(scheme="redblue", domain=[-1, 1]), title="Link"),
            tooltip=["Holding:N", "Compared with:N", alt.Tooltip("Link:Q", format=".2f")],
        ).properties(height=max(220, len(weights) * 35)), width="stretch")
        st.caption("Each square compares two holdings. Read the color scale: +1 means similar-direction "
                   "moves, −1 means opposite-direction moves, and 0 means no strong linear link. "
                   "The diagonal is always 1 because each holding matches itself. Hover for the pair and score.")
        if len(weights) > 1:
            st.metric("Average link between different holdings", _number(average_correlation(returns, weights), ".2f"),
                      help=help_text("correlation"))

    with st.container(border=True):
        section_heading("How the ride changed over time", "rolling")
        st.caption("Every point uses the most recent 126 trading days, approximately six months. "
                   "Higher price swings mean a bumpier recent period.")
        if benchmark is None:
            volatility = portfolio.rolling(126).std().dropna() * np.sqrt(252)
            if volatility.empty:
                st.info("At least 126 trading days are needed for this rolling view.")
            else:
                st.line_chart((volatility * 100).rename("Yearly price swings (%)"), y_label="Yearly price swings (%)")
                st.caption("The comparison link is unavailable because its history could not be matched.")
        else:
            rolling = rolling_risk(portfolio, benchmark)
            if rolling.empty:
                st.info("At least 126 shared trading days with the comparison investment are needed for this rolling view.")
            else:
                st.line_chart((rolling["Volatility (annualised)"] * 100).rename("Yearly price swings (%)"),
                              y_label="Yearly price swings (%)")
                section_heading("How the market link changed", "correlation")
                st.line_chart(rolling["Correlation to benchmark"].rename("Link to comparison investment"),
                              y_label="Link (−1 to +1)")
                st.caption("A line closer to +1 means the portfolio and comparison moved more similarly "
                           "during that recent window. This relationship can change.")


def _stress(run: dict, returns: pd.DataFrame, weights: pd.Series) -> None:
    section_heading("How the current mix behaved during past market shocks", "coverage")
    if "synthetic" in str(run.get("source", "")).lower():
        st.info("This example uses synthetic prices. Switch to market data to see actual historical shock periods.")
        return
    shocks = shock_report(returns, weights)
    spans = []
    partial = []
    for event, (start, end) in SHOCKS.items():
        window = returns.loc[start:end].dropna(how="all")
        spans.append("No shared data" if window.empty else
                     f"{window.index[0]:%d %b %Y} – {window.index[-1]:%d %b %Y}")
        partial.append(not window.empty and
                       (returns.index.min() > pd.Timestamp(start) or returns.index.max() < pd.Timestamp(end)))
    display = pd.DataFrame({
        "Event": shocks.index,
        "Event starts": shocks["from"].to_numpy(),
        "Event ends": shocks["to"].to_numpy(),
        "Available dates": spans,
        "Period coverage": ["Partial event" if part else ("No data" if pd.isna(ret) else "Full event window")
                            for part, ret in zip(partial, shocks["return"])],
        "Change over available dates (%)": shocks["return"].to_numpy() * 100,
        "Largest fall from a high (%)": shocks["max_drawdown"].to_numpy() * 100,
        "Holdings covered (%)": shocks["coverage"].to_numpy() * 100,
    })
    st.dataframe(display, hide_index=True, width="stretch", column_config={
        "Event": st.column_config.TextColumn(help="A named past market shock, measured over the dates shown."),
        "Event starts": st.column_config.TextColumn(help="The beginning of the historical event window."),
        "Event ends": st.column_config.TextColumn(help="The end of the historical event window."),
        "Available dates": st.column_config.TextColumn(help=help_text("history")),
        "Period coverage": st.column_config.TextColumn(help="Partial event means your shared data starts after the event begins or finishes before it ends. The result covers only the displayed available dates."),
        "Change over available dates (%)": st.column_config.NumberColumn(format="%+.1f%%", help=help_text("returns")),
        "Largest fall from a high (%)": st.column_config.NumberColumn(format="%.1f%%", help=help_text("drawdown")),
        "Holdings covered (%)": st.column_config.NumberColumn(format="%.0f%%", help=help_text("coverage")),
    })
    st.caption("Read available dates and coverage before comparing losses. These calculations use the "
               "same shared history as the overview. A partial event can miss the worst part of a crash. "
               "If some holdings lack data, their share is excluded and the covered holdings are rescaled.")
    valid = shocks["return"].dropna()
    if valid.empty:
        st.info("Your holdings' shared history does not cover these events. Choose an earlier start date "
                "or holdings with longer histories to make more events available.")
        return
    worst = valid.idxmin()
    change = float(valid[worst]) * float(run["value"])
    st.write(f"Lowest return among the available episodes: **{worst}**, **{valid[worst]:+.1%}**. "
             f"Applied to **{run['value']:,.0f} {run['base']}**, that is a change of "
             f"**{change:+,.0f} {run['base']}** before trading costs and taxes.")
    if any(partial):
        st.info("At least one replay covers only part of its event. Its result is not a full-crash estimate.")


def render_details(run: dict, universe: pd.DataFrame, load_prices_fn: Callable) -> None:
    """Render the original X-Ray analytics without changing the shared portfolio."""
    st.title("Portfolio details")
    st.write("Understand the mix you currently own. Open a question mark for the meaning behind a number.")
    weights = normalise_weights(run["current"])
    returns = run["returns"].loc[:, weights.index].dropna()
    if len(returns) < 2:
        st.info("At least two shared days of returns are needed to show portfolio details.")
        return
    portfolio = portfolio_series(returns, weights)
    st.caption(f"{len(weights)} holdings · {returns.index[0]:%d %b %Y}–{returns.index[-1]:%d %b %Y} "
               f"· Measured in {run['base']}. These historical views keep today's weights fixed each day "
               "and do not deduct separate trading costs or taxes. They describe your current mix, "
               "not the proposed target weights.")
    try:
        contributions = risk_contributions(returns, weights)
    except ValueError:
        contributions = None
    overview, exposure, risk, stress = st.tabs(["Overview", "Exposure", "Risk", "Stress tests"])
    with overview:
        _, benchmark = _overview(run, returns, weights, portfolio, contributions, load_prices_fn)
    with exposure:
        _exposure(contributions, universe)
    with risk:
        _risk(returns, weights, portfolio, benchmark, run["base"], float(run["value"]))
    with stress:
        _stress(run, returns, weights)
