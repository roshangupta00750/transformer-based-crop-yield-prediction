"""How it works: the method and a glossary, for readers without a farming or AI background."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _results import AVG5, MODEL, header, plain_layout  # noqa: E402

header("How it works",
       "Germany's farmers grow wheat, barley, maize, potatoes and more in about 400 districts. "
       "How much each district harvests per hectare depends on its soil, its long-term trend and, above "
       "all, on the weather in that particular season. Our model learns this last part, how a season's "
       "weather pushes the harvest above or below normal, from 25 years of data.")

st.markdown("#### 1 · The question")
st.markdown("For every district, crop and year: **how many decitonnes per hectare will be harvested?** "
            "(1 dt/ha = 100 kg per hectare; a good wheat harvest is about 80 dt/ha, i.e. 8 t/ha.) "
            "Official numbers (Destatis) arrive months after harvest; the model gives an estimate right "
            "after harvest, for all districts at once.")

st.markdown("#### 2 · The data")
data = pd.DataFrame([
    ("Yields", "Destatis regional statistics 41241-01-03-4", "per district & crop, yearly", "2000–2025",
     "What we predict; matched to districts by official code"),
    ("Weather", "German Weather Service (DWD) stations; ERA5-Land where none", "daily", "2000–2025",
     "Max/min temperature, rain, humidity, wind"),
    ("Plant greenness", "NASA MODIS satellites (NDVI, EVI, LAI, fPAR)", "every 8–16 days, 250–500 m",
     "2000–2025", "How green and dense the crops are, over farmland only"),
    ("Growth stages", "DWD phenology network", "per season", "2000–2025",
     "Sowing, flowering, harvest dates: lines the weather up with the crop's life"),
    ("Soil moisture & water balance", "ERA5-Land (Copernicus)", "daily, 9 km", "2000–2025",
     "Soil water in three layers, evaporation demand, snow, frost on bare ground"),
    ("Soil & terrain", "SoilGrids 2.0 (ISRIC); elevation model", "static", "–",
     "Sand, clay, organic carbon, pH; altitude and slope"),
], columns=["Input", "Source", "Resolution", "Years", "Why it matters"])
st.table(data.set_index("Input"))

st.markdown("#### 3 · The model")
c = st.columns([3, 2])
c[0].markdown(
    "We use an **iTransformer**, a member of the *transformer* family behind modern AI language models. "
    "Instead of words, it reads about 40 daily time series per season (temperature, rain, soil moisture, greenness, …) plus fixed facts about the district (soil, terrain). "
    "Its *attention* mechanism learns which series matter together, for example heat during flowering "
    "combined with little rain.\n\n"
    "The model does not predict the yield from scratch. It starts from the district's **normal yield** "
    "(its long-term average with trend) and predicts the **deviation** caused by this season. One shared "
    "model learns from all 10 crops, then a copy is fine-tuned for each crop.\n\n"
    "The final model (v18) runs **three transformers** on exactly the same inputs and averages them: the "
    "iTransformer, the **Temporal Fusion Transformer (TFT)**, which also shows which inputs it relies on, and "
    "**PatchTST**, which reads the season in short chunks. They make partly different mistakes, so the average "
    "is steadier than any one of them. Finally, each district's estimate is corrected by part of its own errors "
    "in the previous 1–3 years (the **error memory**).")
fig = go.Figure()
yrs = list(range(2016, 2026))
normal = [72 + 0.4 * i for i in range(10)]
dev = [3, -2, -12, -5, 1, 2, -6, -1, 4, 2]
fig.add_trace(go.Scatter(x=yrs, y=normal, name="normal yield (trend)", line=dict(color=AVG5, dash="dot", width=2)))
fig.add_trace(go.Bar(x=yrs, y=dev, base=normal, name="season effect (what the model learns)",
                     marker_color=[MODEL if d >= 0 else "#e34948" for d in dev], opacity=0.75))
plain_layout(fig, 300, yaxis_title="dt/ha (illustration)")
c[1].plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
c[1].caption("Illustration, not real data: forecast = normal yield + season effect.")

st.markdown("#### 4 · How we test it honestly")
st.markdown(
    "- **Unseen years only.** To estimate, say, 2018, the model learns from the years up to 2016, uses 2017 "
    "to check when to stop learning, and only then sees 2018's weather. Every year from 2010 to 2025 is "
    "tested this way, as if it were the future.\n"
    "- **A set-aside test.** Ideas were developed on 2016–2025. The years 2010–2015 were kept aside and used "
    "only for the yes/no decision on each version (from v6 on). 14 such decisions were made on them (12 up to the final model v18), so "
    "they are the cleanest test we have, though not a perfectly fresh one.\n"
    "- **Written plans first.** Each version (v1–v20) has a pre-registration: what changes, how it is "
    "scored and when it is adopted, written down *before* running it. This prevents tuning on the test.\n"
    "- **Strict adoption rule.** A change is kept only if it is better with at least 95% certainty "
    "(checked by resampling whole years 10,000 times), with lower error, and no single crop worse in all six "
    "test years.\n"
    "- **Fair comparisons.** To compare with published papers we retrain on *their* training years and "
    "test on *their* test years.")

st.markdown("#### 5 · Glossary")
gl = pd.DataFrame([
    ("Accuracy", "100% minus the average error in percent. 88% ≈ a typical forecast is 12% off."),
    ("5-year average", "Mean yield of the previous five years in the district: the simplest honest forecast."),
    ("Long-term average / normal", "The district's average yield with its upward trend."),
    ("District (Landkreis)", "German county-level administrative area (NUTS-3); about 400 in total."),
    ("dt/ha", "Decitonnes per hectare = 100 kg per hectare."),
    ("Bad year", "A crop's national yield at least 10% below normal."),
    ("Transformer", "A type of neural network that uses 'attention' to weigh which inputs matter together."),
    ("NDVI / EVI", "Satellite greenness indices: higher = more, healthier green leaves."),
    ("LAI / fPAR", "Leaf area index / fraction of sunlight absorbed by plants, from satellites."),
    ("Phenology", "The timing of growth stages: sowing, emergence, flowering, ripening, harvest."),
    ("Hindcast", "Forecasting past years as if they were the future, to test the model."),
    ("Pre-registration", "A written plan, fixed before an experiment, of what will be measured and decided."),
    ("R² / RMSE / MAE / MAPE", "Error measures used by the papers: R² higher is better; RMSE, MAE (dt/ha) and "
                               "MAPE (%) lower is better."),
    ("ERA5-Land / HYRAS", "Gridded weather datasets: ERA5-Land is a global model reconstruction (9 km); HYRAS is "
                          "DWD's 1 km grid interpolated from its stations."),
], columns=["Term", "Meaning"])
st.table(gl.set_index("Term"))
