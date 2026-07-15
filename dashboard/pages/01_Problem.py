"""Why This Benchmark — Chapter 1 opener.

Frames the model-selection gap the benchmark closes: matching model
architecture to crop biology from first principles, rather than shopping
the leaderboard after the fact.
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import page_header

st.set_page_config(page_title="Why This Benchmark — Crop Yield Benchmark",
                   layout="wide")

page_header("Why This Benchmark",
            "The architecture-selection gap in crop-yield ML — and how a "
            "first-principles taxonomy closes it.")
st.caption("_Crop-yield ML has a dozen candidate architectures and no principled way to pick among them — this page frames the selection gap and the testable-hypothesis approach the rest of the dashboard defends._")

st.markdown("""
Crop-yield forecasting steers food-security policy, farmer decisions, and
climate-adaptation planning — yet the ML literature offers a dozen candidate
architectures (gradient-boosted trees, LSTMs, patch-based Transformers,
variate-first Transformers, TFTs, ...) with **no principled way to pick among
them**. Papers typically train one or two architectures on one or two crops
and declare a winner, leaving practitioners to shop the leaderboard after
the fact.

### The gap this benchmark closes

- Single unified dataset: 10 German crops, 401 NUTS-3 districts, 21 years
  (2003-2023). Same features, same splits, same anomaly framing — no more
  apples-to-oranges comparisons across studies.
- Six architectures head-to-head: XGBoost, DLinear, LSTM, PatchTST,
  iTransformer, and the Temporal Fusion Transformer.
- **A testable hypothesis, not a leaderboard.** The paper's headline
  contribution is a first-principles Crop-Model Compatibility Taxonomy
  (Chapter 5) that predicts which architecture should win which crop from
  crop biology and model inductive bias — and hits top-1 on 9 of 10 crops.
- Three independent stress tests (Chapter 6) — cross-model fusion, CMIP6
  climate counterfactuals, and comparison against published EU benchmarks —
  challenge the taxonomy story from three orthogonal angles.

### How to read the dashboard

The sidebar is ordered as a research paper: motivation (this page), data,
methods (per-model pages), empirical results (leaderboard, in-season,
spatial, interpretability), headline claim (taxonomy), robustness (stress
tests), and reproducibility (training launcher + geographic drilldown).
Read top-to-bottom for the full argument, or jump straight to the taxonomy
if you want the payoff first.
""")
