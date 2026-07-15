"""PatchTST page."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _model_template import render_model_page

render_model_page(
    "patchtst",
    narrative_intro="Chunks the growing season into patches and applies attention across them — capturing phenology segments as tokens rather than raw daily steps.",
)
