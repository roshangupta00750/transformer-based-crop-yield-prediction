"""iTransformer page."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _model_template import render_model_page

render_model_page(
    "itransformer",
    narrative_intro="Inverts the token axis so each variable becomes a token and attention runs across variates — surfacing cross-feature interactions the temporal Transformers do not.",
)
