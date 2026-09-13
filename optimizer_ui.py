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
from consumer_help import help_text, section_heading

DEFAULTS = ["SPY", "QQQ", "VGT", "AAPL", "MSFT", "AGG"]
REVIEWS = {"Monthly": 21, "Quarterly": 63, "Yearly": 252}
DESCRIPTIONS = {
    MC: "Balance growth and bumps: try thousands of ways to split your money, then choose the one with the strongest estimated return for its price swings. This depends on past data; it cannot find a guaranteed best future mix.",
    MIN_VAR: "Aim for a steadier ride: choose the mix with the smallest estimated price swings, taking account of how your holdings move together. This does not aim for the highest return.",
    INV_VOL: "Give steadier holdings more room: put a larger share into holdings whose prices moved less. This simple method does not consider how holdings move together.",
    EQUAL: "Keep it simple: give every holding the same share of your money. This provides a useful comparison without predicting returns.",
}
METHOD_HELP = {MC: "monte_carlo", MIN_VAR: "minimum_variance",
               INV_VOL: "inverse_volatility", EQUAL: "equal_weight", CURRENT: "weights"}
METHOD_REASONS = {
    MC: "This mix seeks more estimated return for the size of its price swings across the whole portfolio.",
    MIN_VAR: "This mix aims to reduce the portfolio's estimated price swings, considering how holdings move together.",
    INV_VOL: "This mix assigns more money to historically steadier holdings, within the holding limit.",
    EQUAL: "This mix aims to give every holding an equal share of your money.",
}
TABLE_HELP = {
    "Current": ("Current share", "weights"), "Target": ("Suggested share", "weights"),
    "Expected return": ("Estimated yearly return", "returns"),
    "Volatility": ("Price swings / year", "volatility"),
    "Sharpe": ("Return for the risk (Sharpe)", "sharpe"),
    "One-way turnover": ("Share of money to move", "turnover"),
    "Largest weight": ("Largest holding's share", "concentration"),
    "Holdings": ("Holdings kept", "floor"),
    "Return (CAGR)": ("Yearly growth (CAGR)", "growth"),
    "Worst drawdown": ("Largest fall from a peak", "drawdown"),
    "Costs / starting value": ("Costs as share of starting money", "costs"),
    "Rebalances": ("Number of adjustments", "rebalance"),
    "Action": ("Suggested action", "rebalance"),
    "Current weight": ("Current share", "weights"),
    "Target weight": ("Suggested share", "weights"),
    "Trade amount": ("Amount to buy (+) or sell (−)", "rebalance"),
    "Estimated cost": ("Estimated trading cost", "costs"),
    "Target amount after costs": ("Suggested holding value after costs", "costs"),
    "Buy price move": ("Price change reaching the buy boundary", "band"),
    "Sell price move": ("Price change reaching the sell boundary", "band"),
    "Why": ("Why this action?", "rebalance"),
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
    config = {}
    for column in frame.columns:
        label, key = TABLE_HELP.get(column, (column, METHOD_HELP.get(column, "weights")))
        explanation = help_text(key)
        if column == "Return (CAGR)":
            explanation = "The steady yearly growth rate that would turn the starting value into the ending value over this test period, allowing gains and losses to build on earlier changes. Actual yearly results varied; this is a summary of the modeled past after its trading costs."
        config[column] = st.column_config.Column(label=label, help=explanation)
    st.dataframe(frame.style.format(formats, na_rep="—"), column_config=config, width="stretch")


def _trade_explanation(row: pd.Series, method: str, triggered: bool) -> str:
    """Explain a modeled action without inventing forecasts for a holding."""
    current, target = row["Current weight"], row["Target weight"]
    if not triggered:
        return "Keep this holding for now: every holding is within the allowed distance from its suggested share. Check again at the next review."
    if row["Action"] == "Hold":
        return "This holding is already close enough to its suggested cash amount; no meaningful trade is needed."
    if row["Action"] == "Buy":
        reason = f"Increase its share from {current:.1%} toward {target:.1%}."
    elif target < current:
        reason = f"Reduce its share from {current:.1%} toward {target:.1%}."
    else:
        # Fees can require a small sale even where the target share is unchanged
        # or slightly higher. Explain the actual cash trade, not a false direction.
        reason = f"A small sale funds trading costs while bringing this holding toward its {target:.1%} share of the remaining portfolio."
    return f"{reason} {METHOD_REASONS.get(method, 'This follows the selected portfolio mix.')}"


def _allocation_tab(run: dict):
    result = run["result"]
    section_heading("Find a mix that fits your goal", "goal")
    method = st.selectbox("Target allocation", list(DESCRIPTIONS), key="optimizer_method",
                          help=help_text("weights") + " Choose a method below to compare different ways of splitting the same holdings.")
    target = result.weights[method]
    st.caption(DESCRIPTIONS[method])
    scores = result.metrics.loc[method]
    cols = st.columns(4)
    cols[0].metric("Estimated return / year", f"{scores['Expected return']:.1%}", help=help_text("returns"))
    cols[1].metric("Estimated volatility / year", f"{scores['Volatility']:.1%}", help=help_text("volatility"))
    cols[2].metric("Estimated Sharpe", f"{scores['Sharpe']:.2f}", help=help_text("sharpe"))
    cols[3].metric("One-way turnover", f"{scores['One-way turnover']:.1%}", help=help_text("turnover"))
    if scores["Sharpe"] <= 0:
        st.warning("This allocation's estimated return does not exceed your assumed risk-free rate. The optimizer is fully invested; it does not allocate to cash.")

    left, right = st.columns([3, 2])
    with left:
        section_heading("The mixes we explored", "monte_carlo")
        st.caption("Each dot is a different split of the same holdings. Higher means more estimated growth; farther left means smaller price swings.")
        cloud = result.cloud.iloc[::max(1, len(result.cloud) // 4000)]
        dots = alt.Chart(cloud).mark_circle(size=15, opacity=0.25).encode(
            x=alt.X("Volatility:Q", axis=alt.Axis(format="%"), title="Price swings / year → more bumpy"),
            y=alt.Y("Expected return:Q", axis=alt.Axis(format="%"), title="Estimated growth / year"),
            color=alt.Color("Sharpe:Q", title="Return for risk", scale=alt.Scale(scheme="viridis")),
            tooltip=[alt.Tooltip("Volatility:Q", title="Yearly price swings", format=".1%"),
                     alt.Tooltip("Expected return:Q", title="Estimated yearly return", format=".1%"),
                     alt.Tooltip("Sharpe:Q", title="Return for risk (Sharpe)", format=".2f")])
        marked = result.metrics.reset_index(names="Method")
        marks = alt.Chart(marked).mark_point(size=160, filled=True, stroke="white", strokeWidth=1.5).encode(
            x=alt.X("Volatility:Q", title="Price swings / year → more bumpy", axis=alt.Axis(format="%")),
            y=alt.Y("Expected return:Q", title="Estimated growth / year", axis=alt.Axis(format="%")), shape="Method:N",
            color=alt.value("#527b62"), tooltip=["Method:N",
                alt.Tooltip("Volatility:Q", format=".1%"),
                alt.Tooltip("Expected return:Q", format=".1%")])
        st.altair_chart((dots + marks).properties(height=340), width="stretch")
        st.caption(f"We tried {run['samples']:,} mixes and show up to 4,000. Large markers show the named methods. More dots do not guarantee a better future result.")
    with right:
        section_heading("How your money would be split", "weights")
        st.caption("Compare each holding's share today with the suggested share. A longer suggested bar means the model gives that holding a larger share.")
        weights = pd.DataFrame({"Current": run["current"], "Target": target})
        chart = weights.reset_index(names="Ticker").melt(id_vars="Ticker", var_name="Allocation", value_name="Weight")
        st.altair_chart(alt.Chart(chart).mark_bar().encode(
            x=alt.X("Weight:Q", title="Share of your portfolio", axis=alt.Axis(format="%")), y=alt.Y("Ticker:N", title="Holding"),
            yOffset="Allocation:N", color=alt.Color("Allocation:N", title="Mix", scale=alt.Scale(range=["#b5beb4", "#527b62"])),
            tooltip=["Ticker", "Allocation", alt.Tooltip("Weight:Q", format=".1%")]
        ).properties(height=340), width="stretch")
        _percent_table(weights, ["Current", "Target"])
    section_heading("Compare the ways to divide your money", "goal")
    _percent_table(result.metrics, ["Expected return", "Volatility", "One-way turnover", "Largest weight"], {"Sharpe": "{:.2f}"})
    caption = "All methods use the same past data and holding limit. Hover over a column heading to understand it. These are estimates before trading costs, not promises. Your current mix is shown for comparison even if it exceeds the suggested holding limit."
    if run.get("floor", 0) > 0:
        caption += f" Suggested mixes hold either nothing or at least {run['floor']:.1%} of a holding; equal weight and your current mix are shown as they are."
    st.caption(caption)
    download = result.weights.copy()
    st.download_button("Download all target weights (CSV)", download.to_csv(),
                       "optimizer_weights.csv", "text/csv")
    return method, target


def _validation_tab(run: dict):
    table, curves, training_weights, training_end = run["validation"]
    test_start = curves.index[0]
    section_heading("How did each approach do on a later period?", "holdout")
    st.write(f"We chose each mix using the older 80% of the data, ending **{training_end:%d %b %Y}**. We then tried it on the later period, **{test_start:%d %b %Y} – {curves.index[-1]:%d %b %Y}**, which it had not used to choose its mix.")
    section_heading("What happened to 100 invested?", "growth")
    st.caption("Every line starts with 100. A line at 120 means a 20% gain after the modeled trading costs. Look for both the ending value and the size of the dips along the way.")
    # Include the starting capital before the first day's return and entry costs.
    anchor = pd.DataFrame(100.0, index=[training_end], columns=curves.columns)
    st.line_chart(pd.concat([anchor, curves]), x_label="Date", y_label="Value of 100 invested, after modeled costs")
    section_heading("Results from this historical test", "holdout")
    _percent_table(table, ["Return (CAGR)", "Volatility", "Worst drawdown", "Costs / starting value"],
                   {"Sharpe": "{:.2f}", "Rebalances": "{:.0f}"})
    st.caption(f"The test starts by buying each historical mix, then reviews it {run['review'].lower()}. It adjusts if any holding moves more than {run['band'] * 100:g} percentage points from its chosen share. The current-portfolio line starts with today's shares of each holding. Prices reflect dividends and stock splits; tax and fund fees are not separately included.")
    st.info("Use this as a check on the approach. Today's suggested mix uses all the data, so this chart does not test those exact suggestions. Repeatedly changing settings to improve this chart can make past results look better without improving future results.")
    with st.expander("Inspect the weights actually used in this test"):
        section_heading("The mix chosen before the test period", "training")
        _percent_table(training_weights, list(training_weights))


def _scenarios_tab(run: dict, method: str, target: pd.Series):
    section_heading("What could the journey look like?", "scenarios")
    st.write(f"Explore **{method}** with many possible journeys made by rearranging {run['block_days']}-day stretches of past market moves. Holdings are kept together in each stretch so their shared ups and downs stay connected.")
    years = st.slider("Scenario horizon (years)", 1, 10,
        None if "optimizer_years" in st.session_state else run.get("years", 5),
        key="optimizer_years", help="How many years each simulated journey lasts. A longer period gives returns more time to build on each other and allows a wider range of results. It is not a recommended holding period.")
    path_count = st.select_slider("Simulated paths", options=[500, 1000, 2000],
        value=None if "optimizer_paths" in st.session_state else run.get("paths", 1000),
        key="optimizer_paths", help=help_text("scenarios"))
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
        section_heading("A range of possible portfolio values", "percentile")
        st.caption("The line shows the middle outcome. The wider shaded area contains the middle 90% of simulated values at each date; values outside it are still possible.")
        outer = alt.Chart(chart_data).mark_area(color="#527b62", opacity=0.12).encode(x=alt.X("Years:Q", title="Years from now"), y=alt.Y("5th:Q", title=f"Your money ({run['base']})", axis=alt.Axis(format=",.0f")), y2="95th:Q")
        inner = alt.Chart(chart_data).mark_area(color="#527b62", opacity=0.25).encode(x="Years:Q", y="25th:Q", y2="75th:Q")
        median = alt.Chart(chart_data).mark_line(color="#375f49", strokeWidth=3).encode(
            x="Years:Q", y="Median:Q", tooltip=["Years:Q", alt.Tooltip("Median:Q", format=",.0f"), alt.Tooltip("5th:Q", format=",.0f"), alt.Tooltip("95th:Q", format=",.0f")])
        st.altair_chart((outer + inner + median).properties(height=380), width="stretch")
        cols = st.columns(4)
        cols[0].metric("Median ending value", f"{summary['median']:,.0f} {run['base']}", help=help_text("median"))
        cols[1].metric("5th percentile value", f"{summary['p05']:,.0f} {run['base']}", help=help_text("percentile") + " About 5 in 100 simulated journeys ended below this amount. It is not a worst-case limit.")
        cols[2].metric("95th percentile value", f"{summary['p95']:,.0f} {run['base']}", help=help_text("percentile") + " About 95 in 100 simulated journeys ended below this amount, and 5 above it.")
        cols[3].metric("Simulated chance of loss", f"{summary['loss_probability']:.1%}", help=help_text("loss_probability"))
        st.caption(f"The worst 5% of simulated journeys ended with {summary['worst_5_mean']:,.0f} {run['base']} on average. The darker band contains the middle half of simulated values. The shaded edges do not represent one person's journey.")
        st.download_button("Download scenario percentiles (CSV)", fan.to_csv(), "optimizer_scenarios.csv", "text/csv")
    else:
        st.info("Run the simulation for the selected allocation and horizon.")
    st.caption("These journeys reuse past market moves, including their average growth, and include modeled trading costs. They start by moving to the suggested mix immediately, then follow your review rule. They do not add savings, withdrawals or inflation. The displayed chance of loss describes this simulation, not the real-world odds; a new kind of crisis may be missing from the past data.")


def _trading_tab(run: dict, method: str, target: pd.Series):
    section_heading("What to buy or sell, and why", "rebalance")
    st.write(f"For **{method}**, check your portfolio **{run['review'].lower()}**. If any holding moves more than **{run['band'] * 100:g} percentage points** from its suggested share, the model moves the whole portfolio back to the suggested mix.")
    st.caption(f"For example, a 20% suggested share with a {run['band'] * 100:g}-point allowance is checked against {max(0, 20 - run['band'] * 100):g}% and {min(100, 20 + run['band'] * 100):g}%. These rules respond to how your money is split, not a prediction of tomorrow's price.")
    plan, summary = rebalance_plan(run["current"], target, run["value"], run["band"], run["cost_bps"])
    if summary["triggered"]:
        st.success(f"The entered weights breach the band. At the next review, this model would sell {summary['sells']:,.2f} {run['base']}, buy {summary['buys']:,.2f} {run['base']}, and reserve {summary['cost']:,.2f} {run['base']} for estimated costs.")
    else:
        st.info("All entered weights are inside the band: hold until a scheduled review finds a breach. Target amounts below are reference amounts, not orders to place now.")
    plan = plan.copy()
    plan["Why"] = plan.apply(lambda row: _trade_explanation(row, method, summary["triggered"]), axis=1)
    section_heading("Your suggested adjustments", "rebalance")
    st.caption(METHOD_REASONS.get(method, "The amounts below follow the selected portfolio mix."))
    display = plan[["Action", "Current weight", "Target weight", "Trade amount", "Estimated cost", "Target amount after costs", "Why"]]
    _percent_table(display, ["Current weight", "Target weight"],
                   {"Trade amount": "{:+,.2f}", "Estimated cost": "{:,.2f}", "Target amount after costs": "{:,.2f}"})
    st.caption(f"All amounts are {run['base']}. Positive trades are buys; negative trades are sells. Trades cover the entire portfolio when any band is breached, so some holdings inside their own bands may also trade. Buys plus estimated costs equal sells. No extra cash is assumed.")

    section_heading("At your next review", "band")
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
        section_heading("An example if only one holding's price changes", "band")
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
    st.sidebar.header("Your portfolio and assumptions", help=help_text("weights"))
    source = st.sidebar.radio("Data source", ["Yahoo Finance", "Synthetic demo (offline)"], key="optimizer_source",
                              help="Yahoo Finance uses past market prices. Synthetic demo uses invented prices so you can try the app without downloading data.")
    selected = st.sidebar.multiselect("Holdings to optimize", sorted(set(labels) | set(initial)),
                                     default=initial, max_selections=15,
                                     format_func=lambda t: f"{t} · {labels.get(t, t)}",
                                     key="optimizer_holdings", help="Choose holdings already in your portfolio. The optimizer only changes how your money is split among these holdings.")
    st.sidebar.caption("Starts with X-Ray's tickers when available. Enter current weights here; changes stay in this menu.")
    if len(selected) < 2:
        st.info("Choose between 2 and 15 holdings to optimize.")
        return
    with st.sidebar.form("optimizer_inputs"):
        base = st.selectbox("Portfolio currency", ["USD", "SEK", "EUR", "GBP", "CAD", "AUD"], key="optimizer_base", help=help_text("currency"))
        start = st.selectbox("History from", ["2010-01-01", "2015-01-01", "2020-01-01"], index=1, help=help_text("history"))
        previous = saved["current"].mul(100).to_dict() if saved else {}
        current_frame = pd.DataFrame({"Ticker": selected,
            "Weight (%)": [previous.get(t, 100 / len(selected)) for t in selected]})
        edited = st.data_editor(current_frame, hide_index=True, disabled=["Ticker"],
            column_config={"Ticker": st.column_config.TextColumn(help="The market symbol used to identify a stock or fund."),
                           "Weight (%)": st.column_config.NumberColumn(min_value=0.0, max_value=100.0, step=1.0, format="%.2f", help=help_text("weights"))},
            key="optimizer_weights_editor", width="stretch")
        st.caption("Weights are normalized to 100%. Enter position-value proportions, not share counts.")
        value = st.number_input("Current portfolio value", min_value=100.0, max_value=1e10, value=100000.0, step=1000.0, help=help_text("current_value"))
        cap_percent = st.slider("Maximum weight per holding (%)", 5, 100, 50, 5, help=help_text("cap"))
        floor_percent = st.slider("Minimum weight per holding (%)", 0.0, 20.0, 1.0, 0.5,
            help=help_text("floor") + " Weights below it are dropped to zero, not rounded up.")
        samples = st.select_slider("Monte Carlo portfolios", [2000, 5000, 10000, 25000, 50000], value=10000, help=help_text("monte_carlo"))
        risk_free = st.number_input("Assumed risk-free rate (% / year)", min_value=-5.0, max_value=25.0, value=3.0, step=0.25, help=help_text("risk_free"))
        with st.expander("Estimation and trading assumptions"):
            section_heading("How the model makes its estimates", "shrinkage")
            mean_shrink = st.slider("Return estimate shrinkage (%)", 0, 100, 50, 10,
                help=help_text("shrinkage"))
            covariance_shrink = st.slider("Covariance shrinkage (%)", 0, 100, 10, 5,
                help=help_text("covariance_shrinkage"))
            review = st.selectbox("Review schedule", list(REVIEWS), help=help_text("review"))
            band = st.slider("Rebalance band (percentage points)", 1, 20, 5, help=help_text("band"))
            costs = st.number_input("Trading cost (bps per amount bought or sold)", min_value=0.0, max_value=1000.0, value=10.0, step=5.0,
                help=help_text("costs") + " 10 bps (basis points) means 0.1% of each buy and each sell.")
            block_days = st.selectbox("Scenario block length (trading days)", [5, 21, 63], index=1, help=help_text("scenario_blocks"))
            seed = st.number_input("Random seed", min_value=0, max_value=2147483647, value=42, step=1, help=help_text("seed"))
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
            largest_floor = 100 / np.ceil(100 / cap_percent - 1e-9)
            if floor_percent > largest_floor + 1e-9:
                raise ValueError(f"The minimum weight is infeasible: with a {cap_percent}% cap, at least {np.ceil(100 / cap_percent - 1e-9):.0f} holdings must share 100%, so the minimum cannot exceed {largest_floor:.1f}%.")
            with st.spinner("Loading common history and comparing allocations…"):
                if source == "Synthetic demo (offline)":
                    prices, currencies = _demo(tuple(selected)), {t: base for t in selected}
                else:
                    prices, currencies = _market(tuple(selected), start, base)
                returns = aligned_returns(prices)
                params = dict(samples=int(samples), cap=cap_percent / 100,
                              floor=floor_percent / 100,
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
        with columns[0]:
            section_heading("Find your mix", "weights")
            st.write("Try thousands of ways to split your money across the holdings you already own.")
        with columns[1]:
            section_heading("Check the approach", "holdout")
            st.write("See how each method would have done in a later period of history.")
        with columns[2]:
            section_heading("Plan your next step", "rebalance")
            st.write("Get buy and sell amounts, the reason for each change, and an estimate of trading costs.")
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
        section_heading("How this analysis works", "monte_carlo")
        st.write("The model compares different ways to divide the same portfolio. It learns from past prices, checks the approaches on later history, and shows what its assumptions mean for your money.")
        section_heading("Only change the mix of what you own", "cap")
        st.write(f"All your money stays invested in your selected holdings, with no borrowing or short selling. Each suggested holding is limited to {run['cap']:.0%} of the total. Your current mix is included as a comparison, even if it exceeds that limit.")
        if run.get("floor", 0) > 0:
            st.write(f"A suggested holding is also either dropped or kept at {run['floor']:.1%} or more, so no plan asks you to keep a sliver too small to be worth trading and following. Raising that minimum drops more holdings and concentrates the mix.")
        section_heading("Try thousands of possible mixes", "monte_carlo")
        st.write("The Monte Carlo search tries many random splits and keeps the mix with the best estimated return for its price swings. Other methods aim for steadier prices or a simpler split. Trying more mixes does not prove that the best possible one has been found.")
        section_heading("Reduce how much we trust historical estimates", "shrinkage")
        st.write(f"We pull each holding's past average return {run['mean_shrinkage']:.0%} toward the average across holdings, and reduce the estimated relationships between holdings by {run['covariance_shrinkage']:.0%}. This reduces sensitivity to noisy historical estimates. Return for risk (Sharpe) compares growth above the assumed {run['risk_free']:.1%} cash rate with the size of price swings.")
        section_heading("Use a common price history", "adjusted_prices")
        st.write(f"Prices account for dividends and stock splits and are converted into {run['base']}. We only analyze dates shared by all holdings and use 252 observations as a year. Market holidays can carry forward the latest price. Currency conversion uses historical exchange rates.")
        section_heading("Check an approach before trusting its result", "holdout")
        st.write("Each method first uses the older 80% of the data to choose a mix. We test that mix on the last 20%. Today's suggestions then use the full history. The later-period test evaluates the method, not today's exact mix.")
        section_heading("Treat the future as a range", "scenarios")
        st.write("We build simulated journeys from stretches of past market moves. Prices change the shares of each holding; scheduled checks may trigger adjustments and trading costs. These journeys may miss crises and market conditions that have never appeared in the data.")
        section_heading("Turn the mix into an understandable plan", "rebalance")
        st.write("Selling funds both the suggested purchases and estimated trading costs. No trades are placed by this app. The simulation buys the target mix immediately; the trading plan waits for a review and a drift beyond the allowed band. Taxes and your broker's actual order rules need a separate check.")
        st.markdown("Sources: [MOSEK on estimation error](https://docs.mosek.com/portfolio-cookbook/estimationerror.html) · [Investor.gov on allocation and rebalancing](https://www.investor.gov/additional-resources/general-resources/publications-research/info-sheets/beginners-guide-asset).")
        section_heading("The currency each holding trades in", "currency")
        st.dataframe(pd.DataFrame.from_dict(run["currencies"], orient="index", columns=["Listing currency"]),
                     column_config={"Listing currency": st.column_config.TextColumn(help=help_text("currency"))}, width="stretch")
