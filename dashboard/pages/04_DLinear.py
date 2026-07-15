"""DLinear page."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _model_template import render_model_page

render_model_page(
    "dlinear",
    tldr="DLinear — a simple trend+seasonal linear decomposition — is the reality-check baseline whose best R² and taxonomy hit rate reveal whether deeper Transformers are actually earning their complexity on this bench.",
    narrative_intro="A deceptively simple trend-plus-seasonal linear decomposition that often matches or beats deeper Transformers on modest-length time series — the reality check of the six-model bench.",
)
