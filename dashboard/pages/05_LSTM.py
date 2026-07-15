"""LSTM page."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _model_template import render_model_page

render_model_page(
    "lstm",
    narrative_intro="The classic recurrent workhorse — sequential inductive bias, temporal dependencies learned step by step, and the yardstick every attention-based model has to justify.",
)
