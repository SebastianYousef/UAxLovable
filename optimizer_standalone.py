"""Standalone entry point for the optimizer UI.

The optimizer is merged into app.py's "Optimise" tab, which is the demo path.
This file keeps Codex's full `render()` runnable on its own for development and
tests. It deliberately lives outside pages/ so Streamlit does not show it as a
second page.

    .venv/bin/streamlit run optimizer_standalone.py
"""
from optimizer_ui import render

render()
