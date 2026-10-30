"""Entry point of the dashboard (dashboard/run.sh).

st.navigation with a single hidden page turns off Streamlit's automatic discovery of dashboard/pages/, so the
older multi-page app (kept as a backup, dashboard/run_old.sh) is not reachable from this one.
"""
import streamlit as st

# st.Page resolves a relative path against this file's folder
st.navigation([st.Page("app.py", title="YieldFormer-DE dashboard", default=True)], position="hidden").run()
