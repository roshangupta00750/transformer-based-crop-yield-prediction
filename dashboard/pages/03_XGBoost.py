"""XGBoost page."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _model_template import render_model_page

render_model_page(
    "xgboost",
    tldr="XGBoost's best R² and RMSE on the German NUTS-3 test set, and which crops it is the taxonomy-predicted primary for versus which crops it actually won.",
    narrative_intro="The gradient-boosted tabular baseline — strong out of the box, native handling of missing values, and the reference every deep-learning entry has to beat.",
)
