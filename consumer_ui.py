"""Codex-owned consumer journey. One saved portfolio powers every page."""
from __future__ import annotations

from html import escape
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from analysis import diversifier_scan
from consumer_help import help_text, render_help, section_heading
from optimizer import CURRENT, MC, MIN_VAR
from optimizer_market import load_market
from portfolio_service import (DEFAULT_SETTINGS, GOALS, GOAL_REASONS, PRESETS,
                               build_research, demo_prices, validate_holdings)

ROOT = Path(__file__).parent
PAGE_PATHS = {
    "home": "views/your_plan.py", "portfolio": "views/your_portfolio.py",
    "details": "views/portfolio_details.py", "monte_carlo": "views/monte_carlo.py",
    "diversify": "views/diversify.py", "help": "views/help_me.py",
}
DIVERSIFIERS = ["AGG", "TLT", "SHY", "TIP", "LQD", "HYG", "GLD", "VNQ", "VWO", "VEA", "EFA", "USMV", "XLU", "XLP"]
CURRENCIES = ["USD", "SEK", "EUR", "GBP", "CAD", "AUD"]
LINKEDIN_PROFILES = (
    ("Ali Saleh", "https://www.linkedin.com/in/ali-saleh2004"),
    ("Sebastian Yousef", "https://www.linkedin.com/in/sebastian-yousef"),
)


@st.cache_data
def universe_data():
    return pd.read_csv(ROOT / "universe.csv")


@st.cache_data(ttl=3600, show_spinner=False)
def market_prices(tickers: tuple[str, ...], start: str, base: str):
    return load_market(tickers, start, base)


@st.cache_data(show_spinner=False, max_entries=24)
def research(prices, current, goal, value, base, source, start, currencies, settings):
    return build_research(prices, current, goal, value, base, source, start, currencies, settings)


def profile():
    if "_consumer_profile" not in st.session_state:
        existing = list(st.session_state.get("holdings", PRESETS["US stocks & bonds example"]))
        previous = st.session_state.get("weight_table", {})
        weights = pd.Series({t: previous.get(t, 1.0) for t in existing})
        try:
            weights = validate_holdings(weights)
        except ValueError:
            weights = pd.Series(1 / 6, index=PRESETS["US stocks & bonds example"])
        st.session_state["_consumer_profile"] = {
            "weights": weights.to_dict(), "value": 100000.0, "base": "USD",
            "start": "2015-01-01", "source": "Yahoo Finance", "is_example": True,
        }
    st.session_state.setdefault("_consumer_goal", next(iter(GOALS)))
    # Merge, not setdefault: a session saved before a new option still needs its key.
    st.session_state["_consumer_settings"] = {
        **DEFAULT_SETTINGS, **st.session_state.get("_consumer_settings", {})}
    return st.session_state["_consumer_profile"]


def _go(page):
    st.switch_page(PAGE_PATHS[page])


def _save_goal():
    st.session_state["_consumer_goal"] = st.session_state["consumer_goal"]


def _save_currency():
    """Apply a currency change immediately and discard results in the old unit."""
    saved = profile()
    base = st.session_state["consumer_base"]
    if saved["base"] == base:
        return
    saved["base"] = base
    # These objects contain values expressed in the previous currency. The
    # research and market caches already include base in their cache keys.
    st.session_state.pop("_optimizer_scenario", None)
    st.session_state.pop("_consumer_diversify", None)
    st.session_state["_consumer_currency_notice"] = (
        f"Currency updated to {base} across the app. The portfolio value stays "
        "the same number until you edit it below.")


def _fit_cap_to_holdings(count):
    settings = st.session_state["_consumer_settings"].copy()
    if settings["cap"] * count < 1:
        settings["cap"] = float(np.ceil(100 / count / 5) * 5 / 100)
        st.session_state["_consumer_settings"] = settings
        st.session_state["_consumer_notice"] = (
            f"We raised the maximum share per holding to {settings['cap']:.0%} so your "
            f"{count} holdings can add up to 100%. You can change this in Monte Carlo → Advanced assumptions.")


def _theme():
    st.html((ROOT / "consumer_style.css").read_text(encoding="utf-8"))


def _team_links():
    """Show the two project creators as compact, accessible LinkedIn links."""
    links = "".join(
        f'''<a class="linkedin-profile" href="{escape(url)}" target="_blank"
                rel="noopener noreferrer" aria-label="Open {escape(name)} on LinkedIn">
                <span class="linkedin-mark" aria-hidden="true">in</span>
                <span class="linkedin-person"><strong>{escape(name)}</strong><small>LinkedIn</small></span>
            </a>'''
        for name, url in LINKEDIN_PROFILES
    )
    st.sidebar.markdown(
        f'<div class="team-links"><p class="team-label">Meet the creators</p>{links}</div>',
        unsafe_allow_html=True,
    )


def render_app():
    st.set_page_config(page_title="Portfolio X-Ray · Your plan", page_icon="🌿", layout="wide")
    _theme()
    saved = profile()
    st.sidebar.markdown('<div class="brand-mark"><span>↗</span> Portfolio X-Ray</div><p class="brand-note">Make sense of your investments.</p>', unsafe_allow_html=True)
    st.sidebar.caption(f"Plan currency · {saved['base']}")
    pages = [
        st.Page(PAGE_PATHS["home"], title="Your plan", icon=":material/auto_awesome:", default=True),
        st.Page(PAGE_PATHS["portfolio"], title="Your portfolio", icon=":material/account_balance_wallet:"),
        st.Page(PAGE_PATHS["details"], title="Portfolio details", icon=":material/monitoring:"),
        st.Page(PAGE_PATHS["monte_carlo"], title="Monte Carlo", icon=":material/science:"),
        st.Page(PAGE_PATHS["diversify"], title="Diversify", icon=":material/park:"),
        st.Page(PAGE_PATHS["help"], title="Help me", icon=":material/help:"),
    ]
    selected = st.navigation(pages, expanded=True)
    st.sidebar.divider()
    st.sidebar.caption("A little help, wherever you need it")
    st.sidebar.markdown("Hover or focus a **?** beside a term for a plain-language explanation. Find the full guide in **Help me**.")
    _team_links()
    selected.run()


def _load_run():
    saved = profile()
    current = pd.Series(saved["weights"], dtype=float)
    source = saved["source"]
    try:
        with st.spinner("Finding a clearer plan: checking your mix and exploring possible outcomes…"):
            if source == "Synthetic demo (offline)":
                prices = demo_prices(tuple(current.index))
                currencies = {t: saved["base"] for t in current.index}
            else:
                prices, currencies = market_prices(tuple(current.index), saved["start"], saved["base"])
            run = research(prices, current, st.session_state["_consumer_goal"],
                saved["value"], saved["base"], source, saved["start"], currencies,
                st.session_state["_consumer_settings"])
        return run
    except Exception as exc:
        st.error("We couldn't build a reliable plan from this data yet.")
        st.write("Check your holdings or try again. If you just want to explore the app, the offline example works without market downloads.")
        with st.expander("What happened?"):
            st.write(str(exc))
        a, b = st.columns(2)
        if a.button("Edit portfolio", key="error_edit"):
            _go("portfolio")
        if b.button("Explore offline example", key="error_demo"):
            saved["source"] = "Synthetic demo (offline)"
            saved["is_example"] = True
            st.rerun()
        return None


def _data_note(run):
    if run["source"] == "Synthetic demo (offline)":
        st.warning("Offline example · These are invented prices, not actual performance. Explore the explanations; do not use these amounts to trade.")
    else:
        last = run["returns"].index[-1].tz_localize(None)
        if (pd.Timestamp.now().normalize() - last).days > 10:
            st.warning(f"The latest shared prices are from {last:%d %b %Y}. This plan uses older data; refresh your price source before considering trades.")
    first, last = run["returns"].index[[0, -1]]
    st.caption(f"History checked: {first:%b %Y}–{last:%b %Y} · Values in {run['base']} · Historical data and estimates, not live quotes.")


def _money(value, base):
    return f"{value:,.0f} {base}"


def _hero():
    st.markdown('<div class="eyebrow">YOUR MONEY. A LITTLE CLEARER.</div>', unsafe_allow_html=True)
    st.title("A clearer plan for your money.")
    st.write("Understand your mix, see what could change, and take the next step with a reason behind it.")


def render_home():
    saved = profile()
    _hero()
    with st.container(border=True, key="portfolio_summary"):
        left, right = st.columns([4, 1], vertical_alignment="center")
        with left:
            label = "Example portfolio" if saved["is_example"] else "Your saved portfolio"
            st.markdown(f"**{label}** · {len(saved['weights'])} holdings · {_money(saved['value'], saved['base'])}", help=help_text("current_value"))
            st.caption(" · ".join(saved["weights"]))
        if right.button("Edit portfolio", width="stretch"):
            _go("portfolio")
    if saved["is_example"]:
        st.caption("Start with this example, or add your own holdings in Edit portfolio. The figures below describe the example until you save yours.")
    st.session_state.setdefault("consumer_goal", st.session_state["_consumer_goal"])
    st.segmented_control("What matters most to you?", list(GOALS),
        key="consumer_goal", required=True, on_change=_save_goal,
        help=help_text("goal"), width="stretch")
    if st.session_state.get("_consumer_notice"):
        st.info(st.session_state.pop("_consumer_notice"))
    run = _load_run()
    if run is None:
        return
    _data_note(run)

    scores = run["result"].metrics
    method = run["method"]
    vol_before = scores.loc[CURRENT, "Volatility"]
    vol_after = scores.loc[method, "Volatility"]
    reduction = (vol_before - vol_after) / vol_before if vol_before > 0 else 0
    changed = run["trades"]["triggered"]
    with st.container(border=True, key="recommendation_hero"):
        st.markdown('<div class="eyebrow">THE SUGGESTED NEXT STEP</div>', unsafe_allow_html=True)
        title = "Same holdings. A different balance." if changed else "You're close to the suggested balance."
        st.subheader(title, help=help_text("rebalance"))
        st.write(GOAL_REASONS[method])
        if changed:
            st.write("The amounts below shift money between investments you already hold. Your total available money also covers the estimated trading costs.")
        else:
            st.write("Your current mix is within the review band. There is no suggested trade now; check again at your next review.")
        left, middle, right = st.columns(3)
        left.metric("Price swings", f"{abs(reduction):.0%} {'smaller' if reduction >= 0 else 'larger'}", help=help_text("volatility"))
        buys = int((run["plan"]["Action"] == "Buy").sum()) if changed else 0
        middle.metric("Investments to add money to", f"{buys} held now" if buys else "None now",
                      help="This live count shows how many investments you already own have a suggested purchase. The main plan never adds a new holding. New holdings are explored separately under Diversify.")
        right.metric("When to review", run["review"], help=help_text("review"))
        st.caption("Price swings compare estimated annual volatility of the current and suggested mix. Smaller swings can mean lower potential returns; this is not a measure of every kind of risk.")

    section_heading(f"What could {_money(run['value'], run['base'])} look like in {run['years']} years?", "scenarios")
    st.write(f"We automatically explored {run['paths']:,} possible paths for the suggested mix. These three figures show a range, not a promise.")
    outcomes = run["outcomes"]
    for column, heading, key, text in zip(st.columns(3),
        ["A difficult outcome", "The middle outcome", "A strong outcome"],
        ["p05", "median", "p95"],
        ["5 in 100 simulated outcomes ended below this amount.",
         "Half the simulated outcomes ended below this; half above.",
         "5 in 100 simulated outcomes ended above this amount."]):
        with column, st.container(border=True):
            st.metric(heading, _money(outcomes[key], run["base"]), help=help_text("median" if key == "median" else "percentile"))
            st.caption(text)
    st.markdown(f"**{outcomes['loss_probability']:.0%} of the simulations finished with less money than you started with.**", help=help_text("loss_probability"))
    st.caption("These paths assume you adopt the suggested mix first and then follow the review rule. They include estimated trading costs, but exclude tax, inflation and new deposits. They may miss events not present in the historical data.")
    with st.expander("See the paths, explained"):
        chart = run["fan"].reset_index()
        area = alt.Chart(chart).mark_area(color="#95b9a9", opacity=0.35).encode(
            x=alt.X("Years:Q", title="Years from now"),
            y=alt.Y("5th:Q", title=f"Portfolio value ({run['base']})", scale=alt.Scale(zero=False)), y2="95th:Q")
        line = alt.Chart(chart).mark_line(color="#245c4c", strokeWidth=3).encode(x="Years:Q", y="Median:Q",
            tooltip=[alt.Tooltip("Years:Q", format=".1f"), alt.Tooltip("Median:Q", format=",.0f")])
        st.markdown("**How to read this**", help=help_text("scenarios"))
        st.write("Move from left (today) to right (later). The dark line is the middle simulated value. The shaded area covers the middle 90% of values at each point; individual paths can go outside it.")
        st.altair_chart((area + line).properties(height=280), width="stretch")

    _home_trades(run)
    with st.expander("How much should I trust this suggestion?"):
        section_heading("Checked against a later slice of history", "holdout")
        table, _, _, training_end = run["validation"]
        goal_score = table.loc[method, "Return (CAGR)"]
        original_score = table.loc[CURRENT, "Return (CAGR)"]
        st.write(f"An earlier version of this method was fitted using data through {training_end:%b %Y}, then tested on later data it had not seen. Its annualized return was **{goal_score:.1%}**, versus **{original_score:.1%}** for the entered mix, after the modeled trading costs.")
        st.write("Today's suggested weights use all the available history, so they are different from those tested earlier. A method can look good in the past and disappoint later. Your goal selects the method; we do not choose it by whichever wins this test.")
        if st.button("Explore the full comparison"):
            _go("monte_carlo")

    st.divider()
    with st.container(border=True, key="diversify_invitation"):
        st.markdown('<div class="eyebrow">OPTIONAL NEXT STEP</div>', unsafe_allow_html=True)
        section_heading("Could a new investment spread your risk?", "diversification")
        st.write("Once your current mix makes sense, explore investments that behaved differently from what you own. This is separate from the plan above and will not change your holdings.")
        if st.button("Explore diversification →", type="primary"):
            _go("diversify")
    st.caption("Need a term translated? Every ? is there to help. The Help me page walks through the whole process.")


def _home_trades(run):
    section_heading("Your buy, sell & keep plan", "rebalance")
    st.write(f"At your next {run['review'].lower()} review, use these amounts if your weights and total value still match what you entered. Update them first if prices or your holdings have changed.")
    names = universe_data().set_index("ticker")["name"].to_dict()
    actions = run["plan"].copy()
    actions["_order"] = actions["Action"].map({"Sell": 0, "Buy": 1, "Hold": 2})
    actions = actions.sort_values(["_order", "Trade amount"])
    for ticker, row in actions.iterrows():
        label = "Keep" if row["Action"] == "Hold" else row["Action"]
        color = {"Sell": "sell", "Buy": "buy", "Keep": "keep"}[label]
        amount = "No trade now" if label == "Keep" else _money(abs(row["Trade amount"]), run["base"])
        with st.container(border=True, key=f"trade_{ticker}"):
            left, right = st.columns([3, 2], vertical_alignment="center")
            with left:
                st.markdown(f'<span class="action-pill {color}">{label}</span> <strong>{escape(names.get(ticker, ticker))}</strong> <span class="ticker-note">{escape(ticker)}</span>', unsafe_allow_html=True)
                st.caption(row["Why"])
            right.metric(f"Share: {row['Current weight']:.1%} → {row['Target weight']:.1%}", amount,
                help="Left is the share of your money held here now; right is the suggested target. The amount is an illustrative trade in your portfolio currency, after allowing for estimated costs. 'Keep' means no material trade is needed in this holding for the current plan.")
    costs = run["trades"]
    st.markdown(f"**Estimated trading costs: {_money(costs['cost'], run['base'])}.**", help=help_text("costs"))
    if costs["triggered"]:
        st.caption(f"Sells of {costs['sells']:,.2f} = buys of {costs['buys']:,.2f} + costs of {costs['cost']:,.2f} {run['base']}. Displayed cards round amounts; the download keeps the underlying numbers. No extra cash is assumed.")
    else:
        st.caption("The arrows show reference targets. They do not mean you need to place a trade while you are within the review band.")
    st.markdown(f"**The rule:** check {run['review'].lower()}. If any holding moves more than {run['band'] * 100:g} percentage points away from its target, bring the whole mix back to the target weights.", help=help_text("band"))
    st.caption("This is a rebalancing rule, not a prediction of the best day or price to trade. Consider tax and actual broker fees before acting. No orders are sent.")
    st.download_button("Save my plan (CSV)", run["plan"].to_csv(), "my_portfolio_plan.csv", "text/csv",
                       help="Download all existing holdings, current and target weights, illustrative buy/sell amounts, estimated costs and reasons.")


def _split_key(ticker):
    return f"consumer_split_{ticker}"


def _write_split(shares: dict, target: int) -> None:
    """Round to whole percent without losing or inventing a point.

    Proportional rescaling almost never lands on integers, so the rounding
    drift is absorbed by the largest holding, where it is least visible.
    """
    rounded = {t: int(round(v)) for t, v in shares.items()}
    if not rounded:
        return
    largest = max(rounded, key=rounded.get)
    rounded[largest] = max(0, rounded[largest] + target - sum(rounded.values()))
    for ticker, value in rounded.items():
        st.session_state[_split_key(ticker)] = value


def _rebalance_split(moved: str, tickers: list) -> None:
    """Keep every slider summing to 100%: the others absorb the change pro rata.

    Dragging one holding up takes the difference from the rest in proportion
    to what they already hold, so their relative sizes survive the move.
    """
    others = [t for t in tickers if t != moved]
    if not others:
        st.session_state[_split_key(moved)] = 100
        return

    remaining = 100 - int(st.session_state[_split_key(moved)])
    current = {t: float(st.session_state.get(_split_key(t), 0)) for t in others}
    total = sum(current.values())

    if total > 0:
        shares = {t: current[t] / total * remaining for t in others}
    else:
        shares = dict.fromkeys(others, remaining / len(others))
    _write_split(shares, remaining)


def _sync_split(selected: list, saved: dict) -> None:
    """Seed the sliders, keeping what the user already set for kept holdings."""
    previous = st.session_state.get("consumer_split_tickers")
    # Streamlit drops slider state while another page is showing, so the list of
    # tickers matching is not enough: without the positions themselves the
    # sliders would come back at 0% and the saved split would be lost.
    if previous == selected and all(_split_key(t) in st.session_state for t in selected):
        return

    base = {}
    for ticker in selected:
        key = _split_key(ticker)
        if previous and ticker in previous and key in st.session_state:
            base[ticker] = float(st.session_state[key])
        else:
            base[ticker] = float(saved["weights"].get(ticker, 0)) * 100

    total = sum(base.values())
    if total <= 0:
        base = dict.fromkeys(selected, 100 / len(selected))
        total = 100

    _write_split({t: v / total * 100 for t, v in base.items()}, 100)
    st.session_state["consumer_split_tickers"] = list(selected)


def _clear_split():
    """Forget slider positions, but keep the user's choice of editor."""
    for key in list(st.session_state):
        if key.startswith("consumer_split_") and key != "consumer_split_mode":
            del st.session_state[key]


def _split_sliders(selected: list, saved: dict, names: dict) -> pd.Series:
    """One slider per holding, always totalling 100%."""
    _sync_split(selected, saved)
    st.caption("Drag any holding and the rest adjust to keep the total at 100%.")

    for ticker in selected:
        st.slider(f"{ticker} · {names.get(ticker, ticker)}", 0, 100,
                  key=_split_key(ticker), format="%d%%",
                  on_change=_rebalance_split, args=(ticker, selected),
                  help=help_text("weights"))

    values = pd.Series({t: float(st.session_state[_split_key(t)]) for t in selected})
    st.markdown(f"**Total: {values.sum():.0f}%**")
    return values


def _split_table(selected: list, saved: dict) -> pd.Series:
    """Free-form amounts, converted into shares of the total on save."""
    st.caption("Enter relative amounts or percentages. We convert them into "
               "shares of the total. A zero removes that investment from the "
               "analysis.")
    frame = pd.DataFrame({"Ticker": selected,
                          "Amount or %": [saved["weights"].get(t, 0) * 100
                                          for t in selected]})
    edited = st.data_editor(
        frame, disabled=["Ticker"], hide_index=True,
        key="consumer_edit_weights", width="stretch", column_config={
            "Ticker": st.column_config.TextColumn("Investment", help=help_text("ticker")),
            "Amount or %": st.column_config.NumberColumn(
                "Amount or %", min_value=0.0, step=0.1, format="%.2f",
                help=help_text("weights"))})
    return edited.set_index("Ticker")["Amount or %"]


def render_portfolio():
    saved = profile()
    st.title("Your portfolio")
    st.write("Tell us what you already own. Your plan will adjust only these holdings.")
    universe = universe_data()
    names = universe.set_index("ticker")["name"].to_dict()
    with st.expander("Start with an example instead"):
        preset = st.selectbox("Example portfolio", list(PRESETS), help=help_text("weights"))
        if st.button("Use this example"):
            chosen = PRESETS[preset]
            saved["weights"] = dict.fromkeys(chosen, 1 / len(chosen))
            saved["is_example"] = True
            _fit_cap_to_holdings(len(chosen))
            # Assign rather than delete: dropping the key only clears the server
            # copy, and the browser keeps showing the old list because the
            # multiselect is still mounted and ignores a changed default. Writing
            # the value is what marks the widget for the frontend to pick up.
            st.session_state["consumer_edit_holdings"] = list(chosen)
            st.session_state.pop("consumer_edit_weights", None)
            _clear_split()
            st.rerun()
    selected = st.multiselect("What do you own?", sorted(set(names) | set(saved["weights"])),
        default=list(saved["weights"]), format_func=lambda t: f"{t} · {names.get(t, t)}",
        max_selections=15, key="consumer_edit_holdings", help=help_text("weights"))
    st.markdown("**How is your money split today?**", help=help_text("weights"))
    if not selected:
        st.info("Pick what you own above, then set how your money is split.")
        amounts = pd.Series(dtype=float)
    else:
        # Sliders live outside the form on purpose: inside one, Streamlit defers
        # every callback until submit, so they could not rebalance as you drag.
        mode = st.radio("How would you like to set it?",
                        ["Sliders — always adds up to 100%", "Type exact amounts"],
                        horizontal=True, key="consumer_split_mode",
                        label_visibility="collapsed")
        if mode.startswith("Sliders"):
            amounts = _split_sliders(selected, saved, names)
        else:
            amounts = _split_table(selected, saved)

    st.session_state.setdefault("consumer_base", saved["base"])
    st.selectbox("Currency used across the app", CURRENCIES, key="consumer_base",
                 on_change=_save_currency, help=help_text("currency"))
    st.caption("Changing this updates currency conversion and labels on every analysis page immediately. "
               "It does not convert the portfolio-value number for you.")
    if st.session_state.get("_consumer_currency_notice"):
        st.success(st.session_state.pop("_consumer_currency_notice"))

    with st.form("consumer_portfolio_form"):
        value = st.number_input("How much is your portfolio worth?", 100.0, 1e10, float(saved["value"]), 1000.0, help=help_text("current_value"))
        own = st.checkbox("These are my holdings (remove the example label)", value=not saved["is_example"])
        with st.expander("Data options"):
            source = st.selectbox("Price data", ["Yahoo Finance", "Synthetic demo (offline)"],
                index=int(saved["source"] != "Yahoo Finance"), help=help_text("adjusted_prices"))
            starts = ["2005-01-01", "2010-01-01", "2015-01-01", "2020-01-01"]
            start = st.selectbox("Use history from", starts, index=starts.index(saved["start"]), help=help_text("history"))
        submit = st.form_submit_button("Save portfolio & see my plan", type="primary")
    if submit:
        try:
            weights = validate_holdings(amounts)
        except ValueError as exc:
            st.error(str(exc))
        else:
            st.session_state["_consumer_profile"] = dict(weights=weights.to_dict(), value=value,
                base=st.session_state["consumer_base"],
                source=source, start=start, is_example=not own or source != "Yahoo Finance")
            _fit_cap_to_holdings(len(weights))
            _go("home")


def render_monte_carlo():
    from optimizer_ui import (_allocation_tab, _scenarios_tab, _trading_tab, _validation_tab)
    st.title("Monte Carlo, explained")
    st.write("Go deeper into the plan. Your holdings are shared across all pages; the controls here are optional.")
    section_heading("Choose how to explore your mix", "monte_carlo")
    settings = {**DEFAULT_SETTINGS, **st.session_state.get("_consumer_settings", {})}
    with st.expander("Advanced assumptions"):
        with st.form("consumer_advanced"):
            a, b, c = st.columns(3)
            cap = a.slider("Maximum share in one holding (%)", 10, 100, round(settings["cap"] * 100), 5, help=help_text("cap"))
            floor = b.slider("Smallest share worth holding (%)", 0.0, 20.0, settings["floor"] * 100, 0.5, help=help_text("floor"))
            samples = c.select_slider("Portfolios to explore", [2000, 5000, 10000, 25000, 50000], value=settings["samples"], help=help_text("monte_carlo"))
            a, b, c = st.columns(3)
            rate = a.number_input("Assumed cash return (% / year)", -5.0, 25.0, settings["risk_free"] * 100, 0.25, help=help_text("risk_free"))
            shrink = b.slider("Return estimate shrinkage (%)", 0, 100, round(settings["mean_shrinkage"] * 100), 10, help=help_text("shrinkage"))
            cov = c.slider("Covariance shrinkage (%)", 0, 100, round(settings["covariance_shrinkage"] * 100), 5, help=help_text("covariance_shrinkage"))
            a, b, c = st.columns(3)
            seed = a.number_input("Random seed", 0, 2147483647, settings["seed"], 1, help=help_text("seed"))
            reviews = {"Monthly": 21, "Quarterly": 63, "Yearly": 252}
            review = b.selectbox("Review schedule", list(reviews), index=list(reviews.values()).index(settings["review_days"]), help=help_text("review"))
            band = c.slider("Rebalance band (percentage points)", 1, 20, round(settings["band"] * 100), help=help_text("band"))
            a, b, c = st.columns(3)
            costs = a.number_input("Trading costs (basis points)", 0.0, 1000.0, float(settings["cost_bps"]), 5.0, help=help_text("costs"))
            years = b.slider("Plan horizon (years)", 1, 10, settings["years"], help=help_text("scenarios"))
            blocks = c.selectbox("Historical block (trading days)", [5, 21, 63], index=[5, 21, 63].index(settings["block_days"]), help=help_text("scenario_blocks"))
            paths = st.select_slider("Simulated paths", [500, 1000, 2000], value=settings["paths"], help=help_text("scenarios"))
            st.caption("A holding below the smallest share is sold in full instead of kept as a sliver. Your current mix and the even split are always shown as they are.")
            if st.form_submit_button("Update analysis", type="primary"):
                largest = 100 / np.ceil(100 / cap - 1e-9)
                if cap / 100 * len(profile()["weights"]) < 1:
                    st.error("The maximum shares must be able to add up to 100%. Increase the limit.")
                elif floor > largest + 1e-9:
                    st.error(f"With a {cap}% maximum, the smallest share can be at most {largest:.1f}%. "
                             "Lower it, or raise the maximum share.")
                else:
                    st.session_state["_consumer_settings"] = dict(samples=samples, cap=cap / 100,
                        floor=floor / 100, risk_free=rate / 100, seed=seed, mean_shrinkage=shrink / 100,
                        covariance_shrinkage=cov / 100, cost_bps=costs, band=band / 100,
                        review_days=reviews[review], block_days=blocks, years=years, paths=paths)
                    st.session_state.pop("_optimizer_scenario", None)
    run = _load_run()
    if run is None:
        return
    _data_note(run)
    st.session_state.setdefault("optimizer_method", run["method"])
    allocation, validation, scenarios, trading = st.tabs(["Compare mixes", "Reality check", "Possible futures", "Buy & sell details"])
    with allocation:
        method, target = _allocation_tab(run)
        st.caption("Trying another method here explores it. Your plan page still follows the goal you selected there.")
    with validation:
        _validation_tab(run)
    with scenarios:
        # The goal's scenario is already computed on page load; expose it to the
        # shared detailed renderer when its controls match, no second run needed.
        if method == run["method"] and "_optimizer_scenario" not in st.session_state:
            st.session_state["optimizer_years"] = run["years"]
            st.session_state["optimizer_paths"] = run["paths"]
            st.session_state["_optimizer_scenario"] = ((run["id"], method, run["years"], run["paths"]), run["fan"], run["outcomes"])
        _scenarios_tab(run, method, target)
    with trading:
        _trading_tab(run, method, target)


def render_details():
    from portfolio_details import render_details as details
    run = _load_run()
    if run is not None:
        _data_note(run)
        details(run, universe_data(), market_prices)


@st.cache_data(show_spinner=False, max_entries=12)
def _diversification(returns, weights, start, base):
    tickers = tuple(t for t in DIVERSIFIERS if t not in weights.index)
    if not tickers:
        return pd.DataFrame()
    prices, _ = market_prices(tickers, start, base)
    candidates = prices.pct_change(fill_method=None).dropna(how="all")
    return diversifier_scan(returns, weights, candidates)


def render_diversify():
    st.title("A little more variety?")
    section_heading("Explore investments that move differently", "diversification")
    st.write("This optional step looks beyond what you already own. Your original plan stays as it is.")
    run = _load_run()
    if run is None:
        return
    _data_note(run)
    st.markdown("**What we compare**", help=help_text("diversification"))
    st.write("We try putting 10% into each candidate and keeping 90% in your existing mix. We compare how independently the holdings moved, using only the history they share.")
    if run["source"] != "Yahoo Finance":
        st.info("The diversification search needs actual market history. Save Yahoo Finance as your data source on Your portfolio to use it.")
        return
    if st.button("Find ways to diversify", type="primary"):
        try:
            with st.spinner("Comparing possible additions…"):
                result = _diversification(run["returns"], run["current"], run["start"], run["base"])
            st.session_state["_consumer_diversify"] = (run["id"], result)
        except Exception as exc:
            st.error(f"We couldn't compare additions: {exc}")
    saved = st.session_state.get("_consumer_diversify")
    if not saved or saved[0] != run["id"]:
        return
    result = saved[1]
    if result.empty or result.iloc[0]["bets_gained"] <= 0:
        st.info("None of the candidates with sufficient history improved this measure of diversification. You do not need to add something just to add something.")
        return
    names = universe_data().set_index("ticker")["name"].to_dict()
    for ticker, row in result[result["bets_gained"] > 0].head(3).iterrows():
        with st.container(border=True):
            st.subheader(names.get(ticker, ticker), help=help_text("diversification"))
            st.caption(ticker)
            st.write(f"In the shared history, a 10% allocation added **{row['bets_gained']:.2f} independent sources of movement** to your portfolio.")
            st.metric("Change in estimated price swings", f"{row['vol_change'] * 100:+.1f} percentage points", help=help_text("volatility"))
            st.caption(f"Compared over {row['years_of_history']:.1f} years. A negative change means smaller historical swings. This is a candidate to research, not a buy order.")
    with st.expander("See all candidates and measures"):
        st.dataframe(result, width="stretch", column_config={
            "bets_gained": st.column_config.NumberColumn("Added independent movement", format="%.2f", help=help_text("effective_bets")),
            "bets_after": st.column_config.NumberColumn("Independent movement after", format="%.2f", help=help_text("effective_bets")),
            "vol_change": st.column_config.NumberColumn("Price swing change (fraction)", format="%.3f", help=help_text("volatility")),
            "drawdown_after": st.column_config.NumberColumn("Worst fall (fraction)", format="%.3f", help=help_text("drawdown")),
            "years_of_history": st.column_config.NumberColumn("Shared years", format="%.1f", help=help_text("history"))})
    st.caption("Check fund fees, what the investment actually owns, tax treatment and currency exposure before adding it. Diversification can reduce concentration, but cannot remove the possibility of loss.")


def render_help_page():
    render_help()
