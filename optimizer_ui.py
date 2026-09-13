"""Codex-owned Streamlit interface for the separate Monte Carlo menu."""
from __future__ import annotations

from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from optimizer import (CURRENT, EQUAL, INV_VOL, MC, MIN_VAR, aligned_returns,
                       bootstrap_scenarios, holdout_validation, optimize,
                       rebalance_plan)
from optimizer_market import load_market

DEFAULTS = ["SPY", "QQQ", "VGT", "AAPL", "MSFT", "AGG"]
REVIEWS = {"Monthly": 21, "Quarterly": 63, "Yearly": 252}
DESCRIPTIONS = {
    MC: "Best estimated return per unit of risk among the sampled portfolios. Sensitive to return estimates; more draws do not guarantee a global optimum.",
    MIN_VAR: "Numerically minimizes estimated variance using correlations and the same holding cap. It does not use expected returns.",
    INV_VOL: "Gives more weight to less volatile holdings, subject to the cap. It does not account for correlations and is not equal-risk-contribution investing.",
    EQUAL: "Splits the portfolio evenly. A transparent baseline with no return forecasting.",
}


@st.cache_data(ttl=3600, show_spinner=False)
def _market(tickers: tuple[str, ...], start: str, base: str):
    return load_market(tickers, start, base)


def _demo(tickers: tuple[str, ...]) -> pd.DataFrame:
    """Clearly labeled synthetic data for an offline, reproducible demonstration."""
    rng = np.random.default_rng(1701)
    n = 1512
    factor = rng.normal(0.00025, 0.009, (n, 1))
    exposure = np.linspace(1.2, 0.2, len(tickers))
    residual = rng.normal(0, 0.004, (n, len(tickers)))
    returns = factor * exposure + residual + 0.0001
    return pd.DataFrame(100 * np.cumprod(1 + returns, axis=0),
                        columns=tickers, index=pd.bdate_range("2020-01-02", periods=n))


def _percent_table(frame: pd.DataFrame, percent: list[str], precision: dict | None = None):
    formats = {column: "{:.1%}" for column in percent if column in frame}
    formats.update(precision or {})
    st.dataframe(frame.style.format(formats, na_rep="—"), width="stretch")


def _allocation_tab(run: dict):
    result = run["result"]
    st.subheader("Choose what you want to optimize")
    method = st.selectbox("Target allocation", list(DESCRIPTIONS), key="optimizer_method")
    target = result.weights[method]
    st.caption(DESCRIPTIONS[method])
    scores = result.metrics.loc[method]
    cols = st.columns(4)
    cols[0].metric("Estimated return / year", f"{scores['Expected return']:.1%}")
    cols[1].metric("Estimated volatility / year", f"{scores['Volatility']:.1%}")
    cols[2].metric("Estimated Sharpe", f"{scores['Sharpe']:.2f}")
    cols[3].metric("One-way turnover", f"{scores['One-way turnover']:.1%}")
    if scores["Sharpe"] <= 0:
        st.warning("This allocation's estimated return does not exceed your assumed risk-free rate. The optimizer is fully invested; it does not allocate to cash.")

    left, right = st.columns([3, 2])
    with left:
        st.markdown("**The portfolios we explored**")
        cloud = result.cloud.iloc[::max(1, len(result.cloud) // 4000)]
        dots = alt.Chart(cloud).mark_circle(size=15, opacity=0.25).encode(
            x=alt.X("Volatility:Q", axis=alt.Axis(format="%"), title="Estimated annual volatility"),
            y=alt.Y("Expected return:Q", axis=alt.Axis(format="%"), title="Estimated annual return"),
            color=alt.Color("Sharpe:Q", scale=alt.Scale(scheme="viridis")),
            tooltip=[alt.Tooltip("Volatility:Q", format=".1%"),
                     alt.Tooltip("Expected return:Q", format=".1%"),
                     alt.Tooltip("Sharpe:Q", format=".2f")])
        marked = result.metrics.reset_index(names="Method")
        marks = alt.Chart(marked).mark_point(size=160, filled=True, stroke="white", strokeWidth=1.5).encode(
            x="Volatility:Q", y="Expected return:Q", shape="Method:N",
            color=alt.value("#ec743d"), tooltip=["Method:N",
                alt.Tooltip("Volatility:Q", format=".1%"),
                alt.Tooltip("Expected return:Q", format=".1%")])
        st.altair_chart((dots + marks).properties(height=340), width="stretch")
        st.caption(f"{run['samples']:,} candidates searched. The cloud is sampled portfolios, not an exact efficient frontier; at most 4,000 points are displayed.")
    with right:
        st.markdown("**From your weights to the target**")
        weights = pd.DataFrame({"Current": run["current"], "Target": target})
        chart = weights.reset_index(names="Ticker").melt(id_vars="Ticker", var_name="Allocation", value_name="Weight")
        st.altair_chart(alt.Chart(chart).mark_bar().encode(
            x=alt.X("Weight:Q", axis=alt.Axis(format="%")), y="Ticker:N",
            yOffset="Allocation:N", color=alt.Color("Allocation:N", scale=alt.Scale(range=["#8b9bb1", "#ec743d"])),
            tooltip=["Ticker", "Allocation", alt.Tooltip("Weight:Q", format=".1%")]
        ).properties(height=340), width="stretch")
        _percent_table(weights, ["Current", "Target"])
    st.markdown("**Compare every method on the same data and constraints**")
    _percent_table(result.metrics, ["Expected return", "Volatility", "One-way turnover", "Largest weight"], {"Sharpe": "{:.2f}"})
    st.caption("Estimates use the full common history and shrinkage settings. Return is annualized arithmetic mean, not CAGR or a guaranteed gain. Current weights are a baseline and may exceed the target cap. Turnover excludes fees.")
    download = result.weights.copy()
    st.download_button("Download all target weights (CSV)", download.to_csv(),
                       "optimizer_weights.csv", "text/csv")
    return method, target


def _validation_tab(run: dict):
    table, curves, training_weights, training_end = run["validation"]
    test_start = curves.index[0]
    st.subheader("Did the method survive data it had not seen?")
    st.write(f"Fit on the first 80% of history, ending **{training_end:%d %b %Y}**. Freeze those targets, then test on **{test_start:%d %b %Y} – {curves.index[-1]:%d %b %Y}**.")
    # Include the starting capital before the first day's return and entry costs.
    anchor = pd.DataFrame(100.0, index=[training_end], columns=curves.columns)
    st.line_chart(pd.concat([anchor, curves]), y_label="Growth of 100 after modeled costs")
    _percent_table(table, ["Return (CAGR)", "Volatility", "Worst drawdown", "Costs / starting value"],
                   {"Sharpe": "{:.2f}", "Rebalances": "{:.0f}"})
    st.caption(f"Includes entry costs and {run['review'].lower()} reviews with a {run['band']:.0%}-point drift band. Each method buys its training target at the start; the current-portfolio baseline assumes today's weights at that date. Dividends and splits are represented by adjusted prices. Taxes, spreads beyond the cost assumption, and fund fees are not separately modeled.")
    st.info("This checks an earlier fit of each method. The targets in Allocation are refitted on all available history, so this chart is not an unseen test of those exact targets. Repeatedly tuning settings after seeing this chart can overfit the holdout.")
    with st.expander("Inspect the weights actually used in this test"):
        _percent_table(training_weights, list(training_weights))


def _scenarios_tab(run: dict, method: str, target: pd.Series):
    st.subheader("Many possible paths, not one forecast")
    st.write(f"Explore **{method}** by resampling entire {run['block_days']}-day blocks of historical returns across all holdings together.")
    years = st.slider("Scenario horizon (years)", 1, 10, 5, key="optimizer_years")
    path_count = st.select_slider("Simulated paths", options=[500, 1000, 2000], value=1000, key="optimizer_paths")
    scenario_key = (run["id"], method, years, path_count)
    if st.button("Simulate future paths", type="primary"):
        with st.spinner("Simulating price paths, weight drift and trading costs…"):
            fan, summary = bootstrap_scenarios(
                run["returns"], target, run["current"], run["value"], years, path_count,
                run["seed"], run["block_days"], REVIEWS[run["review"]], run["band"], run["cost_bps"])
        st.session_state["_optimizer_scenario"] = (scenario_key, fan, summary)
    stored = st.session_state.get("_optimizer_scenario")
    if stored and stored[0] == scenario_key:
        _, fan, summary = stored
        chart_data = fan.reset_index()
        outer = alt.Chart(chart_data).mark_area(color="#5e89b5", opacity=0.15).encode(x="Years:Q", y=alt.Y("5th:Q", title=f"Portfolio value ({run['base']})"), y2="95th:Q")
        inner = alt.Chart(chart_data).mark_area(color="#5e89b5", opacity=0.25).encode(x="Years:Q", y="25th:Q", y2="75th:Q")
        median = alt.Chart(chart_data).mark_line(color="#ec743d", strokeWidth=3).encode(
            x="Years:Q", y="Median:Q", tooltip=["Years:Q", alt.Tooltip("Median:Q", format=",.0f"), alt.Tooltip("5th:Q", format=",.0f"), alt.Tooltip("95th:Q", format=",.0f")])
        st.altair_chart((outer + inner + median).properties(height=380), width="stretch")
        cols = st.columns(4)
        cols[0].metric("Median ending value", f"{summary['median']:,.0f} {run['base']}")
        cols[1].metric("5th percentile value", f"{summary['p05']:,.0f} {run['base']}")
        cols[2].metric("95th percentile value", f"{summary['p95']:,.0f} {run['base']}")
        cols[3].metric("Simulated chance of loss", f"{summary['loss_probability']:.1%}")
        st.caption(f"Average ending value in the worst 5% of simulated outcomes: {summary['worst_5_mean']:,.0f} {run['base']}. Orange: median. Dark band: 25th–75th percentiles. Light band: 5th–95th. Each band is a pointwise range, not an individual path.")
        st.download_button("Download scenario percentiles (CSV)", fan.to_csv(), "optimizer_scenarios.csv", "text/csv")
    else:
        st.info("Run the simulation for the selected allocation and horizon.")
    st.caption("Scenarios reuse the raw historical return distribution, including its historical mean, rather than the shrinkage-adjusted optimization estimates. They include initial allocation costs, weight drift and the chosen review rule. No contributions, withdrawals or inflation are included. These conditional frequencies are not calibrated probabilities of future events; unseen regimes and longer dependencies may be missed.")


def _trading_tab(run: dict, method: str, target: pd.Series):
    st.subheader("When to buy, when to sell, and how much")
    st.write(f"Policy for **{method}**: review **{run['review'].lower()}**. If any holding drifts more than **{run['band'] * 100:g} percentage points** from its target, reset the whole portfolio to target weights.")
    plan, summary = rebalance_plan(run["current"], target, run["value"], run["band"], run["cost_bps"])
    if summary["triggered"]:
        st.success(f"The entered weights breach the band. At the next review, this model would sell {summary['sells']:,.2f} {run['base']}, buy {summary['buys']:,.2f} {run['base']}, and reserve {summary['cost']:,.2f} {run['base']} for estimated costs.")
    else:
        st.info("All entered weights are inside the band: hold until a scheduled review finds a breach. Target amounts below are reference amounts, not orders to place now.")
    display = plan[["Action", "Current weight", "Target weight", "Trade amount", "Estimated cost", "Target amount after costs"]]
    _percent_table(display, ["Current weight", "Target weight"],
                   {"Trade amount": "{:+,.2f}", "Estimated cost": "{:,.2f}", "Target amount after costs": "{:,.2f}"})
    st.caption(f"All amounts are {run['base']}. Positive trades are buys; negative trades are sells. Trades cover the entire portfolio when any band is breached, so some holdings inside their own bands may also trade. Buys plus estimated costs equal sells. No extra cash is assumed.")

    st.markdown("**Your conditional trading rules**")
    for ticker, row in plan.iterrows():
        lower, upper = row["Buy below weight"], row["Sell above weight"]
        conditions = []
        if lower > 0:
            conditions.append(f"buy when its portfolio weight falls below **{lower:.1%}**")
        if upper < 1:
            conditions.append(f"sell when its portfolio weight rises above **{upper:.1%}**")
        rule = "; ".join(conditions) or "no individual band can be breached"
        st.markdown(f"- **{ticker}** — {rule}. Restore to **{row['Target weight']:.1%}**, about **{row['Target amount after costs']:,.0f} {run['base']}** at the entered portfolio value.")

    with st.expander("Translate the bands into illustrative price moves"):
        st.write("Assuming the values of every other holding stay fixed, these relative price moves from the valuation underlying your entered weights would reach each boundary. A move beyond the boundary triggers the rule. They are not fair-value targets, live quotes, stop losses or limit orders. FX changes also change base-currency values.")
        _percent_table(plan[["Buy price move", "Sell price move"]],
                       ["Buy price move", "Sell price move"])
        st.caption("A dash means no finite price trigger exists, including an unowned position. If already beyond a boundary, follow the current band decision rather than wait for the displayed boundary. Recalculate all weights as prices change; there is no unconditional 'sell at this price' prediction.")
    st.write("At each review, update position values and rerun the analysis. Check whether the allocation still fits your horizon and risk tolerance. Use available new contributions for underweight positions when practical; taxes, minimum order sizes and actual broker costs can change which trades make sense.")
    st.caption("The scenario and holdout experiments buy the selected target immediately, then apply the review rule. This conditional plan waits for a band breach. The chosen review interval and band are policy assumptions, not optimized market-timing signals.")
    st.download_button("Download rebalance plan (CSV)", plan.to_csv(), "optimizer_rebalance.csv", "text/csv")


def render():
    st.set_page_config(page_title="Monte Carlo Optimizer", page_icon="🎲", layout="wide")
    st.title("🎲 Monte Carlo Optimizer")
    st.write("Keep the holdings. Explore better weights. See the trade-offs before you rebalance.")
    st.caption("A research tool: 'best' means best under a chosen objective and historical assumptions, not a guaranteed future return.")
    universe = pd.read_csv(Path(__file__).parent / "universe.csv")
    labels = dict(zip(universe["ticker"], universe["name"]))
    saved = st.session_state.get("_optimizer_run")
    initial = (list(saved["current"].index) if saved else
               list(st.session_state.get("holdings", DEFAULTS))[:15])
    st.sidebar.header("Optimization inputs")
    source = st.sidebar.radio("Data source", ["Yahoo Finance", "Synthetic demo (offline)"], key="optimizer_source")
    selected = st.sidebar.multiselect("Holdings to optimize", sorted(set(labels) | set(initial)),
                                     default=initial, max_selections=15,
                                     format_func=lambda t: f"{t} · {labels.get(t, t)}",
                                     key="optimizer_holdings")
    st.sidebar.caption("Starts with X-Ray's tickers when available. Enter current weights here; changes stay in this menu.")
    if len(selected) < 2:
        st.info("Choose between 2 and 15 holdings to optimize.")
        return
    with st.sidebar.form("optimizer_inputs"):
        base = st.selectbox("Portfolio currency", ["USD", "SEK", "EUR", "GBP", "CAD", "AUD"], key="optimizer_base")
        start = st.selectbox("History from", ["2010-01-01", "2015-01-01", "2020-01-01"], index=1)
        previous = saved["current"].mul(100).to_dict() if saved else {}
        current_frame = pd.DataFrame({"Ticker": selected,
            "Weight (%)": [previous.get(t, 100 / len(selected)) for t in selected]})
        edited = st.data_editor(current_frame, hide_index=True, disabled=["Ticker"],
            column_config={"Weight (%)": st.column_config.NumberColumn(min_value=0.0, max_value=100.0, step=1.0, format="%.2f")},
            key="optimizer_weights_editor", width="stretch")
        st.caption("Weights are normalized to 100%. Enter position-value proportions, not share counts.")
        value = st.number_input("Current portfolio value", min_value=100.0, max_value=1e10, value=100000.0, step=1000.0)
        cap_percent = st.slider("Maximum weight per holding (%)", 5, 100, 50, 5)
        samples = st.select_slider("Monte Carlo portfolios", [2000, 5000, 10000, 25000, 50000], value=10000)
        risk_free = st.number_input("Assumed risk-free rate (% / year)", min_value=-5.0, max_value=25.0, value=3.0, step=0.25)
        with st.expander("Estimation and trading assumptions"):
            mean_shrink = st.slider("Return estimate shrinkage (%)", 0, 100, 50, 10,
                help="Pull each historical mean toward the cross-holding average. 100% gives every holding the same estimated return.")
            covariance_shrink = st.slider("Covariance shrinkage (%)", 0, 100, 10, 5,
                help="Pull the sample covariance toward a diagonal matrix to reduce sensitivity to estimated correlations.")
            review = st.selectbox("Review schedule", list(REVIEWS))
            band = st.slider("Rebalance band (percentage points)", 1, 20, 5)
            costs = st.number_input("Trading cost (bps per amount bought or sold)", min_value=0.0, max_value=1000.0, value=10.0, step=5.0,
                help="10 bps = 0.1% of each buy and each sell. Include your estimated commissions and spread; taxes are excluded.")
            block_days = st.selectbox("Scenario block length (trading days)", [5, 21, 63], index=1)
            seed = st.number_input("Random seed", min_value=0, max_value=2147483647, value=42, step=1)
        submitted = st.form_submit_button("Run optimization", type="primary", width="stretch")

    if submitted:
        # Discard old results first: a failed request must never display stale orders.
        st.session_state.pop("_optimizer_run", None)
        st.session_state.pop("_optimizer_scenario", None)
        try:
            current = edited.set_index("Ticker")["Weight (%)"].astype(float)
            if not np.isfinite(current).all() or (current < 0).any() or current.sum() <= 0:
                raise ValueError("Enter valid nonnegative weights with a positive total.")
            current = current / current.sum()
            if len(selected) * cap_percent < 100:
                raise ValueError(f"The cap is infeasible: {len(selected)} × {cap_percent}% is below 100%. Increase the cap to at least {100 / len(selected):.1f}%.")
            with st.spinner("Loading common history and comparing allocations…"):
                if source == "Synthetic demo (offline)":
                    prices, currencies = _demo(tuple(selected)), {t: base for t in selected}
                else:
                    prices, currencies = _market(tuple(selected), start, base)
                returns = aligned_returns(prices)
                params = dict(samples=int(samples), cap=cap_percent / 100,
                              risk_free=risk_free / 100, seed=int(seed),
                              mean_shrinkage=mean_shrink / 100,
                              covariance_shrinkage=covariance_shrink / 100)
                result = optimize(returns, current, **params)
                validation = holdout_validation(returns, current, **params,
                    cost_bps=costs, band=band / 100, review_days=REVIEWS[review])
                st.session_state["_optimizer_run"] = dict(
                    id=pd.Timestamp.now().isoformat(), result=result,
                    validation=validation, returns=returns, current=current,
                    value=value, base=base, source=source, start=start,
                    currencies=currencies, review=review, band=band / 100,
                    cost_bps=costs, block_days=block_days, **params)
        except Exception as exc:
            st.error(f"Optimization could not run: {exc}")
            st.info("Check the inputs or retry Yahoo Finance. The synthetic demo works without downloads and is explicitly labeled.")
            return

    run = st.session_state.get("_optimizer_run")
    if not run:
        st.info("Set your current weights in the sidebar, then select Run optimization. Use Synthetic demo to explore the menu without downloading prices.")
        columns = st.columns(3)
        columns[0].markdown("**Explore allocations**\n\nSearch thousands of weight combinations with a holding limit.")
        columns[1].markdown("**Challenge the result**\n\nCompare simple methods on a later, unseen slice of history.")
        columns[2].markdown("**Plan the rebalance**\n\nSee conditional buy/sell amounts and modeled trading costs.")
        return
    if list(run["current"].index) != selected or run["source"] != source:
        st.info("Your inputs changed. Select Run optimization to replace the previous results.")
        return
    if run["source"] != "Yahoo Finance":
        st.warning("SYNTHETIC DEMO — invented prices, not actual performance of these tickers. Do not use these allocations or trading amounts for investment decisions.")
    dates = run["returns"].index
    st.caption(f"Last completed run · {len(dates):,} shared daily returns · {dates[0]:%d %b %Y} – {dates[-1]:%d %b %Y} · {run['base']} · {run['value']:,.0f} portfolio value · {run['cap']:.0%} holding cap · seed {run['seed']}. Submit the form to apply changed assumptions.")
    if run["source"] == "Yahoo Finance":
        st.caption("Prices and FX use the existing disk cache, which may be older than today. Trading amounts use your entered current weights and value, not live broker positions or executable quotes.")
        if (pd.Timestamp.now().normalize() - dates[-1].tz_localize(None)).days > 10:
            st.warning("The latest common observation is more than 10 calendar days old. Treat this as historical research and obtain fresh prices before considering trades.")
    allocation, validation_tab, scenarios, trading, methodology = st.tabs([
        "Allocation", "Out-of-sample check", "Future scenarios", "When to trade", "How it works"])
    with allocation:
        method, target = _allocation_tab(run)
    with validation_tab:
        _validation_tab(run)
    with scenarios:
        _scenarios_tab(run, method, target)
    with trading:
        _trading_tab(run, method, target)
    with methodology:
        st.subheader("What the model knows — and what it assumes")
        st.markdown(f"""
- **Allocation constraints:** long-only, fully invested, no borrowing, a {run['cap']:.0%} limit per holding. Current weights are an unconstrained reference.
- **Monte Carlo search:** reproducible Dirichlet draws with several concentration levels, projected onto the weight constraints. Pick the highest estimated Sharpe among those samples; the search is not uniform and does not prove a global optimum.
- **Estimates:** daily arithmetic means × 252, with {run['mean_shrinkage']:.0%} shrinkage toward the average mean; annualized covariance with {run['covariance_shrinkage']:.0%} shrinkage toward its diagonal. Sharpe = (estimated annual return − {run['risk_free']:.1%} assumed cash rate) / annual volatility.
- **Historical data:** adjusted closes, converted from verified listing currencies to {run['base']} with historical FX; only complete common daily returns are analyzed. The original loader forward-fills market holidays. Annualization assumes 252 observations per year, including for mixed-market calendars.
- **Validation:** fit only on the first 80%, test on the final 20%, with no future returns used to choose those training weights. Today's weights are separately refitted using all history.
- **Scenarios:** joint historical block resampling with drifting holdings, periodic band reviews and modeled trading costs. The model cannot simulate a crisis pattern absent from its history or guarantee future probabilities.
- **Trading:** amount-based, fee-funded rebalancing; no tax-lot selection, live quotes, order submission, or claim to know an optimal entry/exit price. Scenario initialization buys targets immediately; the conditional plan may wait.
""")
        st.markdown("Sources: [MOSEK on estimation error](https://docs.mosek.com/portfolio-cookbook/estimationerror.html) · [Investor.gov on allocation and rebalancing](https://www.investor.gov/additional-resources/general-resources/publications-research/info-sheets/beginners-guide-asset).")
        st.dataframe(pd.DataFrame.from_dict(run["currencies"], orient="index", columns=["Listing currency"]), width="stretch")
