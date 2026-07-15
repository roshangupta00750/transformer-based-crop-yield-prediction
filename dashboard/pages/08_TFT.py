"""TFT page."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _model_template import render_model_page

render_model_page(
    "tft",
    narrative_intro="The Temporal Fusion Transformer — the only model here with native quantile forecasts (P10-P90) and gated variable selection, and the deepest inductive bias for irregularly-informative covariates.",
)
