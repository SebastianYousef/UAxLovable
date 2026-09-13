"""Codex-owned plain-language explanations shared by the consumer screens.

Keep this module independent of price loading: Help me must work offline.
"""
from __future__ import annotations

import re

import streamlit as st


# Each explanation says what the number means and how to read it. The analytics
# modules, rather than general investing conventions, define model-specific terms.
GLOSSARY: dict[str, tuple[str, str]] = {
    "weights": ("Weight / allocation", "The share of your portfolio's money in each investment. A 20% weight means 20 out of every 100 is invested there. Current is the mix you entered; target is the mix the model suggests. Changing weights changes how much you own, without adding new investments."),
    "returns": ("Return", "How much an investment gained or lost, shown as a percentage. A 10% gain turns 100 into 110. Historical return describes the past; estimated return is a model input based on that history. Neither promises what you will earn next."),
    "volatility": ("Volatility / size of price swings", "How widely returns move up and down. A higher number means a bumpier ride. The app usually expresses this on a yearly scale so portfolios are comparable. It is not the biggest possible loss or a prediction of next year's change."),
    "drawdown": ("Drawdown / fall from a peak", "How far the portfolio fell below its previous high. If 100 falls to 80, the drawdown is 20%. Worst drawdown is the deepest fall in the history shown; current drawdown is the gap from the high at the end of that history. A future fall could be larger."),
    "sharpe": ("Sharpe / return for the risk taken", "A score comparing return above an assumed cash-like rate with the size of price swings. Higher means more return for each unit of these swings in this calculation. A negative score means return was below that assumed rate. Compare scores from the same model and period; a high score cannot guarantee a good outcome."),
    "beta": ("Beta / sensitivity to the comparison market", "How strongly this portfolio tended to respond when the chosen market benchmark moved. Around 1 means similar-sized responses; above 1 means larger responses on average. For example, 1.2 suggests roughly a 1.2% response to a 1% market move in the fitted relationship. It is not a rule for every day."),
    "r_squared": ("R-squared / how much the market explains", "How much of the portfolio's day-to-day variation is explained by a simple relationship with the chosen benchmark. A value near 100% means their movements were closely linked in this history. It does not mean a 100% chance of profit or identical holdings."),
    "alpha": ("Alpha / return beyond the market relationship", "The annualized return left over after the model accounts for the portfolio's relationship with the chosen benchmark. Positive means it did better than that simple relationship would suggest in this period. It is a historical estimate, not proof of investment skill."),
    "effective_bets": ("Effective bets / different sources of risk", "An estimate of how many separate patterns of movement drive the portfolio's risk. Many holdings can still depend on the same pattern. A low score is a prompt to inspect overlap; it is not a literal count of safe investments or a guarantee that a higher score will prevent losses."),
    "effective_positions": ("Effective positions / spread by size", "How evenly your money is spread, expressed as an equivalent number of equally sized holdings. Four equal holdings score 4; one dominant holding lowers the score. This measure only checks sizes, so it can miss investments that move together."),
    "risk_share": ("Risk share / contribution to price swings", "The portion of the portfolio's overall price swings attributed to a holding, taking its size and relationship with the others into account. A 10% money weight can contribute much more than 10% of the risk. A negative contribution means it offset some swings in this history."),
    "concentration": ("Concentration / where most of the money sits", "How much of your portfolio sits in a few investments or one area. For example, a top-three share of 70% means 70 out of every 100 is in just three holdings. This helps you spot dependence on a small group; different names can still share similar risks."),
    "covariance": ("Covariance / how investments move together", "A calculation combining the size of each investment's swings with how they move together. The optimizer uses it to estimate the risk of the whole mix. You do not need to interpret the raw numbers: their purpose is to avoid treating every holding as independent."),
    "correlation": ("Correlation / how similar the movements are", "A score from -1 to +1 describing how two investments moved together. Near +1 means similar direction; near -1 means opposite direction; near 0 means no strong straight-line relationship. It helps reveal overlap, but these relationships can change when markets change."),
    "var": ("VaR / bad-day threshold", "In this app, the 95% VaR is the daily return at the bottom 5% boundary of past days. If it shows -2%, roughly 5 in 100 historical days were around that bad or worse. The 99% version uses the bottom 1%. This is a threshold, not a limit on losses."),
    "cvar": ("CVaR / average of the worst days", "The average daily return among the worst 5% of days in this history. If it shows -3%, those especially bad days lost 3% on average. This describes what happened beyond the bad-day threshold, but future bad days can be worse."),
    "underwater": ("Time underwater / time below a previous high", "The longest stretch of trading days the historical portfolio spent below a previous peak. It helps show how long you might have had to wait for a recovery in that period. Trading days exclude most weekends and holidays, and an unfinished recovery is not a forecast of when it will end."),
    "monte_carlo": ("Monte Carlo / trying many possible mixes", "The allocation search tries thousands of different weights for your existing holdings, then selects the sampled mix with the best estimated return relative to price swings. More trials explore more mixes; they do not prove that the winner is the best possible mix or will win in the future."),
    "minimum_variance": ("Minimum variance / aiming for a smoother ride", "This method looks for a mix with the smallest estimated price swings, using how holdings moved individually and together. It respects the maximum holding size and does not use expected returns to choose weights. Smoother in this model still allows losses and does not mean the best return."),
    "inverse_volatility": ("Inverse volatility / more weight to steadier holdings", "This method gives larger shares to holdings that had smaller price swings, subject to the size limit. It is a simple way to reduce emphasis on the bumpiest holdings. It does not consider how holdings move together, so it does not give every holding an equal share of portfolio risk."),
    "equal_weight": ("Equal weight / split evenly", "The same share of money goes into every holding: four holdings get 25% each. This is an easy reference for judging more complex methods. Equal money does not mean equal risk, because some holdings move more sharply or overlap with others."),
    "turnover": ("Turnover / how much the mix changes", "One-way turnover is half the total percentage-point changes between current and target weights. Moving 10% of the portfolio from one holding to another gives 10% turnover. It helps compare the amount of trading required, before costs; it is not an extra return or the final cash trade amount."),
    "risk_free": ("Risk-free rate / cash-like comparison rate", "The annual rate used as a cash-like reference when calculating the Sharpe score. It is an assumption you can change, not a live savings quote or promised return. The optimizer uses it to compare investments but does not add a cash holding to your portfolio."),
    "shrinkage": ("Return shrinkage / tempering historical winners", "Moves each holding's estimated return toward the average across your holdings. At 0%, past differences are kept; at 100%, all estimated returns are the same. This reduces reliance on extreme historical winners, but it cannot remove uncertainty about future returns."),
    "covariance_shrinkage": ("Covariance shrinkage / simplifying risk estimates", "Reduces estimated links between different holdings while keeping each holding's estimated swings. At 0%, historical links are kept; at 100%, the model treats the holdings as uncorrelated. It can make estimates less sensitive to noisy data, but real-world links do not disappear."),
    "cap": ("Holding cap / maximum share in one investment", "The largest target weight allowed for any single holding. A 40% cap means no proposed holding gets more than 40 out of every 100. The limit must allow all weights to add to 100%. Today's weights may exceed it; the cap constrains proposed mixes."),
    "seed": ("Random seed / repeatable simulation", "A number that makes the random search repeatable. Keep the same data, settings and seed to reproduce a result. A different seed explores different random draws; choosing a lucky seed after seeing the result does not make a strategy more reliable."),
    "holdout": ("Holdout / checking on a later period", "The model learns from the first 80% of the available history, freezes its targets, then checks them on the remaining 20%. This helps reveal a method that only fitted its earlier data. It tests earlier targets, not the exact targets now fitted on all the history."),
    "training": ("Training period / history used to choose the mix", "The earlier part of the data used to estimate returns and risk and select target weights for the later-period test. The test's later prices are left out of this step. This separation makes the comparison more useful, although repeatedly tuning after seeing test results weakens it."),
    "scenarios": ("Scenarios / possible portfolio paths", "Many example paths created by reusing blocks of historical returns. They show how different sequences of gains and losses could affect the selected mix under the model's rules. The shaded chart shows ranges across these examples, not a promised range for your real portfolio."),
    "scenario_blocks": ("Scenario blocks / keeping groups of days together", "The simulation samples consecutive groups of historical trading days and joins them into example futures. All holdings use the same sampled dates, preserving their relationships within each group. Longer patterns and events missing from the history may still be missed."),
    "median": ("Median / the middle simulation result", "Half of the simulated results finish above this amount and half below. It is a useful middle example, not an average, a most-likely forecast or an amount you can count on receiving."),
    "percentile": ("Percentile / a point in the range of outcomes", "A way to place a result among all the simulations. At the 5th percentile, about 5% of simulated values are lower; at the 95th, about 95% are lower. These are boundaries in the model's examples, not best- and worst-case limits for real life."),
    "loss_probability": ("Simulated chance of loss", "The percentage of simulated paths that finish below the starting portfolio value after modeled trading costs. A result of 20% means 20 out of 100 model paths lost money. It is a frequency in these examples, not a calibrated probability of your future loss; inflation is not included."),
    "rebalance": ("Rebalance / move back toward the chosen mix", "Sell some of the holdings above their target shares and use the proceeds to buy those below target. Here, one holding outside the allowed drift can trigger resetting the whole portfolio at a scheduled review. The plan uses only current holdings and reserves estimated trading costs from the proceeds."),
    "band": ("Drift band / how far weights may move", "The allowed distance from a target, measured in percentage points. With a 20% target and a 5-point band, a review triggers a rebalance if the weight is below 15% or above 25%. This is a portfolio-share rule, not a prediction of the right stock price."),
    "review": ("Review schedule / when to check the mix", "How often the model checks whether any holding has moved outside its target band: monthly, quarterly or yearly. A review does not always mean a trade. The app shows a plan; it does not watch your brokerage account or place orders."),
    "costs": ("Trading costs / money spent making changes", "The plan reserves the chosen cost estimate on the value of purchases and sales. Ten basis points means 0.10%, or 1 per 1,000 traded. This is an assumption, not your broker's quote. Taxes, actual dealing charges, currency-conversion fees and differences in execution prices can change the real total."),
    "currency": ("Base currency / one unit for all amounts", "The currency used to compare holdings and show portfolio values and trade amounts. Converting foreign holdings includes exchange-rate movements in their measured returns. A currency warning means that conversion did not succeed; cross-currency comparisons then need extra care."),
    "coverage": ("Coverage / how much history is available", "The share of today's portfolio with enough data for the historical event being shown. If coverage is 60%, the result represents that 60% rescaled to a whole portfolio, not all of today's holdings. Low coverage makes the event less representative of your actual mix."),
    "diversification": ("Diversification / spreading what you depend on", "Spreading money among investments with different sources of risk. Several funds or stocks can still move together, so more names alone may not help. The optional diversification page explores adding a new holding; it does not change the recommendation for your existing holdings automatically. Diversification cannot prevent every loss."),
    "goal": ("Your goal / what the model prioritizes", "Choose the trade-off you want the analysis to explore, such as smaller price swings, a simple even split, or estimated return for the risk taken. It changes how target weights are selected. The app does not know your debts, emergency savings, taxes or when you need the money."),
    "current_value": ("Portfolio value / money used for the example", "The total value entered for your holdings in the chosen currency. It converts target percentages into money amounts. For example, a 20% target in a 10,000 portfolio means about 2,000 before costs. A sample value is an illustration, not a live balance from your account."),
    "adjusted_prices": ("Adjusted prices / allowing for dividends and splits", "Historical prices adjusted to account for events such as share splits and dividends. This helps avoid mistaking a split for a crash and represents the effect of dividends in return calculations. These are historical data-provider values, not executable prices for placing trades."),
    "history": ("Historical data / the period behind the results", "The past prices available for the selected holdings. The optimizer uses dates with returns for every holding, so a newer investment can shorten the shared period. A different period can produce different targets. Historical gains, relationships and crashes do not cover every possible future."),
    "benchmark": ("Benchmark / a comparison investment", "A reference such as a broad-market fund, used to put your portfolio's results in context. It helps answer how the mix behaved compared with that reference over the same period. It may carry different risks, and choosing it does not add it to your portfolio."),
    "growth": ("Growth / how a starting amount changed", "The line shows what a starting amount would become as the displayed returns build on one another. A line from 100 to 120 represents a 20% gain over that period. Historical charts describe a modeled past, while scenario charts describe simulated examples; read the chart's cost assumptions."),
    "exposure": ("Exposure / which areas you depend on", "Groups your holdings by country, industry or investment type to show where money and estimated risk sit. The risk share can be larger than the money share. These labels describe the listed instruments; a fund label may not describe every company or country it holds inside."),
    "cluster": ("Cluster / holdings with linked movements", "A group joined by strong historical relationships between holdings. It flags investments that may react to similar events. Groups can be connected through intermediate holdings, so not every pair inside a group must have the same strong link."),
    "rolling": ("Rolling window / how risk changed over time", "Each point recalculates a measure using a recent slice of history, then moves that slice forward. It helps show that price swings and relationships are not constant. The window describes the past at that date; it is not a forecast for the next period."),
    "ticker": ("Ticker / an investment's short code", "The code used to identify a listed investment, such as AAPL for Apple. A suffix can identify the exchange. The code helps the app load the correct price history; check the full name and listing before acting on a plan."),
    "asset_class": ("Asset class / type of investment", "A broad category such as shares, bonds or property. The app uses these labels to summarize the kind of investments you hold. Different categories can react differently to the same events, but a category name alone does not tell you all its risks."),
    "sector": ("Sector / type of business", "A group of businesses such as technology, healthcare or energy. The breakdown helps show when much of your money depends on one area of the economy. Broad funds may be labeled broadly rather than broken down into every underlying business."),
    "best_day": ("Best day / largest daily gain", "The largest one-day percentage gain in the history shown. It gives context for the good days this mix experienced. It is not the usual daily return or a gain the portfolio is expected to repeat."),
    "worst_day": ("Worst day / largest daily loss", "The largest one-day percentage loss in the history shown. It helps translate the idea of risk into a fall that actually occurred in this modeled history. It is not a limit on how bad a future day could be."),
    "positive_days": ("Positive days / how often it gained", "The percentage of historical days with a return above zero. More winning days do not necessarily mean an overall profit: a few large losses can outweigh many small gains. Read it alongside total return and the size of the losses."),
}

ALIASES = {
    "weight": "weights", "allocation": "weights", "target": "weights",
    "return": "returns", "expected_return": "returns", "cagr": "returns",
    "max_drawdown": "drawdown", "worst_drawdown": "drawdown",
    "current_drawdown": "drawdown", "r2": "r_squared", "r_square": "r_squared",
    "risk_contributions": "risk_share", "risk_contribution": "risk_share",
    "top_holdings": "concentration", "var_95": "var", "var_99": "var",
    "cvar_95": "cvar", "tail_risk": "cvar", "longest_underwater_days": "underwater",
    "min_var": "minimum_variance", "inv_vol": "inverse_volatility",
    "mc": "monte_carlo", "samples": "monte_carlo", "candidate_portfolios": "monte_carlo",
    "mean_shrinkage": "shrinkage", "maximum_weight": "cap", "random_seed": "seed",
    "validation": "holdout", "paths": "scenarios", "simulation": "scenarios",
    "block_days": "scenario_blocks", "p05": "percentile", "p95": "percentile",
    "rebalance_band": "band", "cost_bps": "costs", "basis_points": "costs",
    "fees": "costs", "base_currency": "currency", "portfolio_value": "current_value",
    "prices": "adjusted_prices", "date_range": "history", "diversifier": "diversification",
    "correlation_clusters": "cluster", "rolling_risk": "rolling",
}


def help_text(key: str) -> str:
    """Return concise tooltip copy without inventing a meaning for unknown keys."""
    normalized = re.sub(r"[\s\-]+", "_", str(key).strip().lower())
    term = GLOSSARY.get(ALIASES.get(normalized, normalized))
    if term:
        return term[1]
    return ("This label does not have a glossary entry yet. Read the explanation "
            "beside its chart or table, or open Help me to find a related term.")


def section_heading(label: str, key: str) -> None:
    """Use Streamlit's native help control, including its keyboard support."""
    st.subheader(label, help=help_text(key))


def render_help() -> None:
    """Render an offline, mobile-friendly guide without a second portfolio form."""
    st.title("Help me")
    st.write("Understand your portfolio, one plain-English explanation at a time.")
    st.caption("Hover over a question mark, or focus it with your keyboard, for a quick explanation. You can also search the same explanations below.")

    section_heading("Start with these four steps", "goal")
    for title, description in (
        ("1 · Check the portfolio", "Make sure the holdings, their shares and total value match the portfolio you want to explore. If a sample portfolio is shown, its results are examples. A weight of 25% means one quarter of your money."),
        ("2 · Pick what matters to you", "Choose your goal on the overview. The app compares different ways to divide money among the investments already in your portfolio and runs example future paths."),
        ("3 · Read the suggested changes", "Look at the current share, suggested share and explanation for each holding. The buy and sell amounts show how that change could be funded within the portfolio, after estimated trading costs."),
        ("4 · Go deeper only when you want to", "Use the analysis pages to inspect historical risk, compare methods or change the simulation settings. The optional diversification section explores adding investments separately."),
    ):
        with st.container(border=True):
            st.markdown(f"**{title}**")
            st.write(description)

    section_heading("What a buy or sell suggestion means", "rebalance")
    st.write("The main recommendation changes the amounts in your current holdings. It does not select a new stock, add borrowing or assume extra money. A suggested sale usually reduces a holding toward its target share; small sales can also help fund trading costs. A suggested purchase moves a holding toward its target. It is a model-based allocation idea, not an order placed for you.")
    with st.expander("A simple example: the target is a share, not a stock price"):
        st.write("Suppose the target for a holding is 20% and the allowed drift is 5 percentage points. At a scheduled review, a weight above 25% or below 15% triggers the rule. The model then brings all holdings back toward their targets, reserving money for the estimated cost of buying and selling.")
        st.write("If every holding stays within its band, the rule says to hold. It cannot tell you that a particular share price is the perfect time to trade. The app does not monitor an account, place orders, or promise that its suggested mix will earn more.")
    st.write("A suitable real-world mix also depends on when you need your money and how much loss you can tolerate. Diversification and rebalancing can help manage risk, but cannot remove it. [Read the Investor.gov guide to allocation and diversification](https://www.investor.gov/introduction-investing/getting-started/asset-allocation).")

    section_heading("How to read the simulations", "scenarios")
    st.write("The app uses Monte Carlo in two different ways: it tries many mixes of your holdings to find a model candidate, and it creates many example future paths by rearranging blocks of historical returns. The first selects weights; the second shows a range of outcomes under those assumptions.")
    st.write("The middle result is called the median. The shaded bands show how widely the examples spread. A displayed chance of loss counts losses in the simulations; it is not a reliable promise about the odds of a future market event. New crises, different market relationships and patterns missing from the data may not appear.")
    with st.expander("How the four allocation methods differ"):
        for key in ("monte_carlo", "minimum_variance", "inverse_volatility", "equal_weight"):
            st.markdown(f"**{GLOSSARY[key][0]}**")
            st.write(help_text(key))
        st.caption("All proposed mixes use the same holdings and size limit. They remain fully invested; a lower-risk goal does not introduce cash or guarantee protection of your starting money.")
    with st.expander("Why the past-performance check matters"):
        st.write(help_text("holdout"))
        st.write("The allocation page refits its targets using all available history. The later-period test describes how an earlier fit behaved. Changing settings again and again after seeing that test can make a result look stronger than it really is.")
        st.write("Historical X-Ray charts use fixed portfolio weights across daily returns. The optimizer's later-period test and future paths allow weights to drift and apply the selected review rule. Their assumptions differ, so results can differ even for the same holdings.")

    section_heading("Which costs are included?", "costs")
    with st.expander("Included in the trading plan, later-period test and scenarios"):
        st.write("The chosen trading-cost percentage applies to both purchases and sales. The plan reserves those estimated costs so proposed buys plus costs are funded by sales. The optimizer's later-period test and future paths include initial allocation costs and subsequent trades under the selected review rule.")
        st.write("Historical X-Ray summaries and the allocation search's estimated return and risk figures do not deduct this separate trading-cost assumption. The one-way turnover score describes the change in weights before costs.")
    with st.expander("What your real account may add or change"):
        st.write("Taxes, your broker's exact charges, currency-conversion charges, share rounding and trading restrictions are not individually calculated. Actual spreads and execution prices may differ from the cost estimate. Ongoing fund charges are not separately added; charges already reflected in a fund's historical price are part of those observed returns.")
        st.write("Future paths do not include new deposits, withdrawals or inflation. Check the holdings, latest values and actual costs before using any amounts outside this example. [Learn how fees affect a portfolio at Investor.gov](https://www.investor.gov/introduction-investing/general-resources/news-alerts/alerts-bulletins/investor-bulletins/updated).")

    section_heading("Find a word or number", "weights")
    query = st.text_input("Search the glossary", placeholder="Try: risk, weight, Monte Carlo or buy", key="consumer_help_search", help="Search the names and explanations below. Clear the search to see every term.").strip().casefold()
    matching = [
        (key, label, explanation)
        for key, (label, explanation) in GLOSSARY.items()
        if not query or query in f"{key} {label} {explanation}".casefold()
        or any(query in alias.replace("_", " ") for alias, target in ALIASES.items() if target == key)
    ]
    if not matching:
        st.info("No matching explanation yet. Try a shorter word, such as 'risk', 'cost' or 'return'.")
    else:
        st.caption(f"{len(matching)} explanation{'s' if len(matching) != 1 else ''}")
        for _, label, explanation in sorted(matching, key=lambda item: item[1].casefold()):
            with st.expander(label, expanded=bool(query)):
                st.write(explanation)
