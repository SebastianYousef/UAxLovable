# Portfolio X-Ray

**Understand your investments and explore a clearer balance.**

Portfolio X-Ray turns historical portfolio analysis into plain-language
explanations. Its landing page automatically compares ways to divide money
among your current holdings, shows possible outcomes and explains what the
model would buy, sell or keep. Question marks explain the financial terms
without requiring a maths background.

## Run it

From the repository root on Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

On macOS or Linux:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/streamlit run app.py
```

Open the local URL printed by Streamlit, normally
[http://localhost:8501](http://localhost:8501).

**Updating an existing checkout?** Run the install command again after pulling.
The consumer interface requires **Streamlit 1.63 or newer** for its native
question-mark help, navigation and controls.

For a quick offline test, open **Your portfolio → Data options**, choose
**Synthetic demo (offline)** and select **Save portfolio & see my plan**.
The plan runs automatically using clearly labeled invented prices. If the first
market download fails, **Explore offline example** provides the same fallback.

For market data, choose **Yahoo Finance** and save your holdings, amounts or
percentages, total value and currency. Prices are cached in `.cache/`. The
latest shared data period appears with the results; the app does not use live
broker positions or execute orders.

## Start with your plan

The first page starts with a labeled example portfolio. Choose **Edit
portfolio** to enter your own, or explore the example first. Your saved inputs
are shared across pages for the current Streamlit session.

1. Choose a goal: **Balance growth & risk**, **A smoother ride**, or **Keep it
   simple**. The app chooses the corresponding method and runs the analysis.
2. Read the suggested balance and the difficult, middle and strong simulation
   outcomes. These describe examples under the model's assumptions, not promised
   future amounts.
3. Review the **buy, sell & keep plan**. Every change stays within your existing
   holdings. Estimated trading costs are funded from the portfolio, and each
   holding has a reason for its proposed change.
4. Explore **Diversify** at the bottom only if you want to research adding
   something new. This search does not alter your saved holdings or plan.

The trading rule checks portfolio weights at a chosen review interval. A target
of 20% with a five-percentage-point band means the rule triggers below 15% or
above 25%. It does not predict a perfect stock price or trading day. Real taxes,
broker charges and trading restrictions are not individually calculated.

## Pages

| Page | What you can do |
|---|---|
| Your plan | Read the automatic recommendation, possible outcomes and explained trade amounts. The goal is the main analysis control. |
| Your portfolio | Save 2–15 holdings, their relative amounts, portfolio value, currency and data source. Zero amounts are excluded. |
| Portfolio details | Inspect current holdings' risk, country/sector exposure, linked movements and historical stress periods. |
| Monte Carlo | Compare allocation methods, inspect the later-period check and scenarios, or change advanced assumptions. |
| Diversify | Optionally compare adding a new investment at 10%, with the current mix scaled to 90%. |
| Help me | Read the startup guide and search the plain-language glossary without loading market data. |

Hover over a question mark or focus it with the keyboard for a short
explanation. The glossary provides the same information in expandable text.

## How it works

The allocation search tries thousands of mixes of the same holdings and chooses
the sampled mix with the best estimated return relative to price swings. It
also compares minimum variance, inverse volatility and equal weights. Suggested
weights sum to 100%, have no short positions, and respect the maximum holding
size. These are model candidates, not proof of a globally optimal portfolio.

An 80/20 historical check chooses weights using the earlier period and tests
them on the later period. Today's targets are then fitted on all available
history, so they are not the exact targets tested earlier. Future scenarios
reuse blocks of historical returns and include estimated trading costs and
scheduled rebalancing. They can miss new events and relationships absent from
that history. Simulated loss frequencies are not calibrated future odds.

The current-portfolio detail charts keep today's weights fixed each day and do
not deduct separate trading costs or taxes. Listing currencies and historical
exchange rates put holdings into one base currency; the FX adapter validates
rates and applies scale-break repair before conversion. Missing data fails
visibly. See [OPTIMIZER.md](OPTIMIZER.md) for the complete assumptions and cost
treatment.

## Layout

```
app.py               thin entry point calling consumer_ui.render_app()
consumer_ui.py       navigation, landing page, shared editor, diversification
portfolio_service.py shared research run, goal mapping, trade reasons
portfolio_details.py readable views of the existing risk analytics
consumer_help.py     shared tooltips and searchable Help me docs
consumer_style.css   consumer styling
.streamlit/config.toml  theme defaults
requirements.txt     runtime dependencies, including Streamlit 1.63+
views/*.py           six thin st.navigation page entry points
xray.py              core risk maths
analysis.py          exposure, tail risk, clustering, diversifier scan
data.py              price loading + on-disk cache
universe.csv         instrument metadata, including listing currency
optimizer.py         allocation search, holdout validation, scenarios (Codex)
optimizer_ui.py      shared detailed renderers and standalone form (Codex)
optimizer_market.py  currency conversion (Codex)
optimizer_standalone.py  runs the optimizer UI on its own
validate_universe.py checks every ticker still resolves
test_consumer.py     automatic plan, shared profile and consumer journey checks
test_optimizer*.py   optimizer, UI, explanation and FX adapter checks
test_xray.py         sanity checks on synthetic data, no network needed
COORDINATION.md      file ownership for working two agents in parallel
```

`consumer_ui.py` registers the pages using `st.navigation`. Keep `app.py` thin;
do not add a legacy `pages/` directory or duplicate the portfolio form on an
analysis page. The form-based optimizer can also run independently with
`streamlit run optimizer_standalone.py`; its inputs are separate from the main
app and are useful for isolated development.

Run checks on Windows:

```powershell
.\.venv\Scripts\python.exe -m unittest test_consumer test_optimizer test_optimizer_ui test_optimizer_clarity test_optimizer_market -v
.\.venv\Scripts\python.exe test_xray.py
```

Before editing alongside another agent, read [COORDINATION.md](COORDINATION.md)
for file ownership, stable interfaces and the shared Git workflow. Claude owns
the thin `app.py` entry point and existing math/data modules; Codex owns the
consumer presentation modules and optimizer. Append instrument metadata to
`universe.csv` with its listing currency and run `validate_universe.py` before
pushing additions.

## Things to extend

- **Cheaper twin.** Search for a portfolio with the same risk profile and a lower
  total fee, then show the 30-year cost of the difference in kronor.
- **Fund look-through.** Resolve funds to their real holdings so overlap shows up at
  the company level, not just in the correlations.
- **Custom shock.** Let the user draw their own scenario — rates +2%, tech −30%.
