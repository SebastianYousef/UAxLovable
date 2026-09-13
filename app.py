"""Portfolio X-Ray entry point.

Consumer pages share one saved portfolio. Presentation lives in consumer_ui.py
and portfolio_details.py; existing financial analytics remain unchanged.
"""
from consumer_ui import render_app

render_app()
