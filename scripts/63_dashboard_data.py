"""Build the small tables the dashboard reads (results/dashboard/).

Every number on the results pages comes from here, so definitions live in one place:

  accuracy      = 100 - mean absolute percentage error (MAPE), per district-crop-year
  5-year average = mean reported yield of the five previous years (B5)
  long-term average = the district's trend-adjusted historical average (y_hist)
  national      = plain mean over the districts reported that crop-year
  bad year      = national actual yield >= 10% below the long-term average
  good year     = national actual yield >= 10% above it

Sources: results/metrics/v18_predictions.parquet (main model v18 = ensemble of three transformers + error memory, each year predicted
by a model that never saw it), compare_equal_protocol_*.csv (papers' protocols),
results/forecasts/forecast_2026_issued_2026-09-27.csv, and the pre-registration
change logs (versions table, transcribed below).

Usage:  python scripts/63_dashboard_data.py
"""
from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MET = ROOT / "results/metrics"
OUT = ROOT / "results/dashboard"
BAD, GOOD = -10.0, 10.0

# Pre-registration change logs (paper/preregistration_*.md), in plain words.
# hindcast = accuracy on the clean test years 2010-2015 vs the main model on the same rows
# (comparable from v7 on, when the v6 scoring rules started).
VERSIONS = [
    # version, planned (pre-registration written), result (registered outcome), title, change,
    # hindcast_new, hindcast_main, papers_won, decision, lesson
    ("v1", "2026-09-26", "2026-09-27", "First transformer",
     "iTransformer on daily weather + satellite data over each crop's full season (incl. the autumn before "
     "for winter crops), with wet/frost indicators and satellite data averaged over farmland only",
     None, None, 4, "Was main",
     "Averaging satellite data over farmland only helped (67 of 100 crop-years, confirmed); the full "
     "season and the extra indicators were not confirmed but were kept with it by the old count rule."),
    ("v2", "2026-09-28", "2026-09-28", "Longer history, observed growth stages, national signal",
     "Training data from 2000 instead of 2003, DWD-observed growth-stage dates, a national-level component",
     None, None, None, "Not adopted",
     "None of the three ideas was confirmed. The 2000-2025 data stayed as the basis for later versions."),
    ("v3", "2026-09-29", "2026-09-29", "Harvest date, crop-stage weather, district anomalies",
     "Observed harvest date; weather summarised per growth stage; each district's deviation from its normal",
     None, None, 5, "Was main",
     "No single change was confirmed; all three were kept together by the old count rule (65 of 100 "
     "crop-years), which was later found to be a flawed way to choose."),
    ("v4", "2026-09-29", "2026-09-30", "Trend rule + one fine-tuned model per crop",
     "A fairer long-term trend for the starting point; the shared model is fine-tuned separately for each crop",
     None, None, 12, "Was main",
     "Neither change was confirmed; both were kept by the old count rule (68 of 100). Won 12 of 18 paper "
     "comparisons. Main model until v12."),
    ("v5", "2026-09-30", "2026-10-01", "National gate, separate autumn/spring models",
     "Switch to a national-level forecast where it looked safer; split autumn- and spring-sown crops",
     None, None, None, "Not adopted", "Neither confirmed; district forecasts got worse (MAPE 12.59 vs 12.24)."),
    ("v6", "2026-10-01", "2026-10-01", "Stricter scoring rules (no model)",
     "One fixed accuracy score, 2010-2015 kept as an untouched test, adoption only with >= 95% certainty",
     None, None, None, "Rules", "From here on every version is judged the same way."),
    ("v7", "2026-10-01", "2026-10-02", "Recent yields as input",
     "The model sees the last few years' yields of the district",
     88.74, 88.32, 8, "Not adopted", "Better on average, but only 87% certain (rule needs 95%)."),
    ("v8", "2026-10-02", "2026-10-02", "Rapeseed flowering index",
     "Satellite 'yellowness' during rapeseed flowering",
     88.17, 88.32, None, "Not adopted",
     "Rapeseed better on average in the clean test, but only 83% certain; all crops together slightly worse."),
    ("v9", "2026-10-03", "2026-10-03", "Blended baseline",
     "Mix of the 5-year and long-term average as the starting point",
     88.63, 88.32, 8, "Not adopted", "Better on average, but only 84% certain."),
    ("v10", "2026-10-03", "2026-10-04", "Corrected data foundation",
     "Four data changes at once: yields matched by official district code (fixes 18 city/county mix-ups), "
     "ERA5 weather for all districts, district-own trend reference, corrected satellite dates",
     87.67, 88.27, 11, "Not adopted", "Worse; with four changes at once the cause was unclear, so v11 tested each."),
    ("v11", "2026-10-04", "2026-10-04", "Which v10 change hurt? (diagnostic)",
     "Undo each v10 change one at a time",
     None, None, None, "Diagnostic",
     "ERA5 weather everywhere (-0.042 skill) and the added/corrected district series (-0.030) hurt; "
     "the trend reference and satellite dates made no difference."),
    ("v12", "2026-10-04", "2026-10-05", "Corrected yields + station weather + reliable series",
     "Correct district matching, DWD stations first (ERA5 only for gaps), train only on well-reported series",
     88.27, 88.27, 11, "Was base",
     "Exact tie with v4, so the 95% rule was not met; used as the base from v13 on because its data is "
     "correct and it covers ~80 more districts."),
    ("v13", "2026-10-05", "2026-10-06", "Crop-specific satellite averaging",
     "Satellite data averaged over each crop's own fields (EU crop map) instead of all farmland",
     88.25, 88.32, 10, "Not adopted", "Satellite pixels (250 m) are bigger than fields: adds noise, not information."),
    ("v14", "2026-10-06", "2026-10-07", "DWD HYRAS 1 km gridded weather",
     "Official DWD 1 km weather grid (built from ~2,000 stations) averaged over each district's farmland",
     88.22, 88.32, 11, "Not adopted",
     "Better weather data (rain r 0.94 vs 0.68 for ERA5) but no better forecasts, not even in districts "
     "without their own station (88.44 vs 88.47); 24% certain."),
    ("v15b", "2026-10-08", "2026-10-08", "Area-weighted training",
     "Training gives districts that grow more of a crop more weight (sqrt of census crop area), so noisy "
     "small-area statistics count less",
     88.45, 88.48, 12, "Not adopted",
     "No net gain: better for wheat and winter barley, worse for oats and summer barley; but 12/18 papers."),
    ("v15", "2026-10-08", "2026-10-08", "YieldFormer-DE-i: iTransformer + district error memory",
     "Each forecast is corrected by part of the district's own forecast error in the previous 1-3 years "
     "(the size of the correction is learned from earlier years only); no retraining",
     88.48, 88.32, None, "Was main",
     "First change to pass the strict rule (98.4% certain). Fixes the model's lagging district normals: "
     "2016-2025 accuracy 88.38% vs 87.90%."),
    ("v16", "2026-10-09", "2026-10-09", "Checks and forecasts",
     "Is the transformer needed (versions without the network), doubtful official figures, 90% ranges, the 2025 "
     "wheat miss; frozen forecasts for 2026 and 2027",
     None, None, None, "Diagnostic",
     "Without the network accuracy drops 1-2 points; doubtful figures change < 0.1 point; the 2025 wheat miss came "
     "from the network over-reading a dry spring."),
    ("v17", "2026-10-10", "2026-10-11", "YieldFormer-DE-T and -P: TFT and PatchTST",
     "The other two transformers (Temporal Fusion Transformer, PatchTST) on exactly the same data, years and "
     "fine-tuning, each with the error memory",
     88.76, 88.48, None, "Not adopted",
     "TFT more accurate (88.76% vs 88.48%; 88.91% on 2016-2025) but only 66% certain on 2010-2015; PatchTST weaker."),
    ("v18", "2026-10-11", "2026-10-11", "YieldFormer-DE: ensemble of -i, -T and -P",
     "Average of the three transformers (iTransformer, TFT, PatchTST), equal weights fixed in advance, plus the "
     "district error memory",
     88.80, 88.48, None, "Final model",
     "Passes the rule (96.7% certain) and wins 13 of 16 years; steadier than any single network. Not past the "
     "stricter multiple-testing bar (99.6%)."),
    ("v19", "2026-10-12", "2026-10-12", "Swing and range calibration",
     "Scale up the forecast's national good/bad-year swing; recalibrate the 80% ranges",
     88.79, 88.80, None, "Not adopted",
     "Scaling the swing also scales wrong signals (64% certain, worse on 2016-2025); recalibrated ranges "
     "overshot (85% coverage instead of 80%)."),
    ("v20", "2026-10-12", "2026-10-13", "Crop stress inputs; forecasts during the season",
     "Heat, drought and frost indicators at the critical crop stages for all three networks; forecasts with "
     "data cut at 30 June and 31 July",
     88.82, 88.80, None, "Not adopted",
     "Only 56% certain: helps TFT and PatchTST, hurts the iTransformer. In season: no better than the 5-year "
     "average at 30 June, +1.1 points at 31 July."),
]
PAPER_FILES = {"v1": "compare_equal_protocol.csv", "v3": "compare_equal_protocol_v3A.csv",
               "v4": "compare_equal_protocol_v4F.csv", "v7": "compare_equal_protocol_v7RF.csv",
               "v9": "compare_equal_protocol_v9BF.csv", "v10": "compare_equal_protocol_v10DF.csv",
               "v12": "compare_equal_protocol_v12QF.csv", "v13": "compare_equal_protocol_v13CF.csv",
               "v14": "compare_equal_protocol_v14HF.csv",
               "v15b": "compare_equal_protocol_v15WF.csv",
               "v15": "compare_equal_protocol_v15EF.csv",     # v15: Srivastava rows only (single-year tests)
               "v18": "compare_equal_protocol_v18XF.csv"}     # v18: ensemble without memory on the papers' splits
STATES = {"DE1": "Baden-Württemberg", "DE2": "Bavaria", "DE3": "Berlin", "DE4": "Brandenburg", "DE5": "Bremen",
          "DE6": "Hamburg", "DE7": "Hesse", "DE8": "Mecklenburg-Vorpommern", "DE9": "Lower Saxony",
          "DEA": "North Rhine-Westphalia", "DEB": "Rhineland-Palatinate", "DEC": "Saarland", "DED": "Saxony",
          "DEE": "Saxony-Anhalt", "DEF": "Schleswig-Holstein", "DEG": "Thuringia"}
EVENTS = {(2018,): "2018 drought", (2019,): "2019 drought", (2022,): "2022 drought",
          (2023,): "2023 dry spring", (2011,): "2011 spring drought", (2013,): "2013 late, wet spring"}


def acc(pred, true) -> float:
    return float(100 - 100 * np.mean(np.abs(np.asarray(pred) - np.asarray(true)) / np.asarray(true)))



DUDEN_VAR = {"ww": "winter_wheat", "wb": "winter_barley", "sb": "summer_barley", "rye": "rye", "oats": "oats",
             "triticale": "triticale", "silage_maize": "silage_maize", "potat_tot": "potatoes",
             "sugarbeet": "sugar_beet", "wrape": "winter_rapeseed"}


def analytics(P: pd.DataFrame, N: pd.DataFrame, names: pd.Series) -> dict:
    """Tables and numbers for the Insights, Where-grown, Error-analysis and Weather pages (descriptive)."""
    ins = {}
    # --- where crops are grown: latest census crop area, arable land, district area (Duden et al. 2024)
    f = ROOT / "data/raw/duden2024/Final_data.csv"
    if f.exists():
        Dd = pd.read_csv(f, dtype={"district_no": str})
        A = Dd[(Dd.measure == "area") & Dd.value.notna()]
        land = A[A["var"].isin(["ArabLand", "district"]) & (A.year == A.year.max())].pivot_table(
            index="nuts_id", columns="var", values="value")
        cr = A[A["var"].isin(DUDEN_VAR)].assign(crop=lambda d: d["var"].map(DUDEN_VAR))
        cr = cr[cr.year <= 2020].sort_values("year").groupby(["nuts_id", "crop"]).agg(
            area_ha=("value", "last"), census_year=("year", "last")).reset_index()
        cr = cr.merge(land, left_on="nuts_id", right_index=True, how="left").rename(
            columns={"nuts_id": "landkreis", "ArabLand": "arable_ha", "district": "district_ha"})
        yl = P[P.year >= 2016].groupby(["landkreis", "crop"]).actual.mean().rename("yield_2016_25")
        cr = cr.merge(yl, on=["landkreis", "crop"], how="left")
        cr["share_of_farmland"] = 100 * cr.area_ha / cr.arable_ha
        cr["production_t"] = cr.area_ha * cr.yield_2016_25 / 10           # dt/ha x ha / 10 = t
        cr["district"] = cr.landkreis.map(names)
        cr["state"] = cr.landkreis.str[:3].map(STATES)
        cr.to_csv(OUT / "crop_area.csv", index=False)

    # --- error analysis on log errors of the main model
    E = P.assign(e=np.log(P.predicted / P.actual), a_true=np.log(P.actual / P.avg_long),
                 a_pred=np.log(P.predicted / P.avg_long))
    E["e_nat"] = E.groupby(["crop", "year"]).e.transform("mean")
    E["e_state"] = E.groupby(["crop", "year", "state"]).e.transform("mean") - E.e_nat
    E["e_dist"] = E.e - E.e_nat - E.e_state
    tot = (E.e ** 2).mean()
    ins["error_share"] = {k: float(100 * (E[c] ** 2).mean() / tot) for k, c in
                          (("Germany-wide year effect", "e_nat"), ("State year effect", "e_state"),
                           ("Single district", "e_dist"))}
    Q = E.sort_values("year")
    Q["prev"] = Q.groupby(["landkreis", "crop"]).e_dist.shift(1)
    q = Q.dropna(subset=["prev"])
    ins["persistence_r"] = float(np.corrcoef(q.e_dist, q.prev)[0, 1])
    ins["calibration_slope"] = float(np.polyfit(E.a_pred, E.a_true, 1)[0])
    ins["sd_ratio"] = float(E.a_pred.std() / E.a_true.std())
    Q[["landkreis", "crop", "year", "e_dist", "prev"]].dropna().sample(min(4000, len(q)), random_state=0).to_csv(
        OUT / "error_persistence_sample.csv", index=False)
    E[["crop", "year", "a_true", "a_pred"]].sample(min(6000, len(E)), random_state=0).to_csv(
        OUT / "calibration_sample.csv", index=False)
    pd.DataFrame([{"component": k, "share": v} for k, v in ins["error_share"].items()]).to_csv(
        OUT / "error_decomposition.csv", index=False)
    st_ = P.groupby("state").agg(error_model=("err_predicted", lambda e: e.abs().mean()),
                                 error_avg5=("err_avg5", lambda e: e.abs().mean()), n=("actual", "size")).reset_index()
    st_.to_csv(OUT / "error_by_state.csv", index=False)
    if (OUT / "crop_area.csv").exists():
        ar = P.merge(pd.read_csv(OUT / "crop_area.csv")[["landkreis", "crop", "share_of_farmland"]],
                     on=["landkreis", "crop"], how="left").dropna(subset=["share_of_farmland"])
        ar["bin"] = pd.qcut(ar.share_of_farmland, 5, labels=["smallest 20%", "2nd 20%", "middle 20%", "4th 20%", "largest 20%"])
        ab = ar.groupby("bin", observed=True).agg(share_median=("share_of_farmland", "median"),
                                                  error_model=("err_predicted", lambda e: e.abs().mean()),
                                                  error_avg5=("err_avg5", lambda e: e.abs().mean()),
                                                  n=("actual", "size")).reset_index()
        ab.to_csv(OUT / "error_by_area.csv", index=False)
        ins["error_small_vs_large"] = [float(ab.error_model.iloc[0]), float(ab.error_model.iloc[-1])]
    by = P.groupby("year").agg(bias=("err_predicted", "mean"), error=("err_predicted", lambda e: e.abs().mean()),
                               bias_avg5=("err_avg5", "mean")).reset_index()
    by.to_csv(OUT / "bias_by_year.csv", index=False)
    worst = P.assign(abs_err=P.err_predicted.abs()).nlargest(30, "abs_err")[
        ["year", "crop", "district", "state", "actual", "predicted", "avg5", "err_predicted"]]
    worst.to_csv(OUT / "largest_misses.csv", index=False)
    ins["worst10_share"] = float(100 * P.err_predicted.abs().nlargest(len(P) // 10).sum() / P.err_predicted.abs().sum())
    # --- yield trend per crop (national mean, dt/ha per decade)
    # within-district trend (district fixed effects), so changes in which districts report do not bias it
    ins["trend_per_decade"] = {}
    for c, g in P.groupby("crop"):
        y = g.actual - g.groupby("landkreis").actual.transform("mean")
        t = g.year - g.groupby("landkreis").year.transform("mean")
        ins["trend_per_decade"][c] = float(10 * (y * t).sum() / (t * t).sum() / g.actual.mean() * 100)
    # --- weather drivers (national, 2003-2025): correlation of yield anomaly with seasonal weather
    wf, af = MET / "rca_national_weather.csv", MET / "rca_national_anomaly.csv"
    if wf.exists() and af.exists():
        W_ = pd.read_csv(wf).set_index("year"); A_ = pd.read_csv(af).set_index("year")
        corr = pd.DataFrame({c: {w: W_[w].corr(A_[c]) for w in W_.columns} for c in A_.columns})
        corr.to_csv(OUT / "weather_correlation.csv")
        ins["top_driver"] = {c: [corr[c].abs().idxmax(), float(corr[c][corr[c].abs().idxmax()])] for c in corr.columns}
    (OUT / "insights.json").write_text(json.dumps(ins, indent=2))
    return ins


MODEL_COLUMNS = {"YieldFormer-DE (v18)": ("v18_predictions.parquet", "XE18"),
                 "YieldFormer-DE-i (v15)": ("v15_predictions.parquet", "E15"),          # iTransformer + memory
                 "YieldFormer-DE-T (v17)": ("v18_predictions.parquet", "TE17"),          # TFT + memory
                 "YieldFormer-DE-P (v17)": ("v18_predictions.parquet", "PE17"),          # PatchTST + memory
                 "v12": ("v15_predictions.parquet", "Q12"),
                 "v15b": ("v15W_predictions.parquet", "WE15"), "v14": ("v14_predictions.parquet", "H14"),
                 "v13": ("v13_predictions.parquet", "C13"), "v10": ("v10_predictions.parquet", "D10"),
                 "Trend + memory": ("v16_predictions.parquet", "A16a"),   # v16 §2 baseline (no network)
                 "XGBoost + memory (reference)": ("v16_predictions.parquet", "GE16"),   # the one non-transformer reference
                 "5-year average": ("v15_predictions.parquet", "b5"),
                 "Long-term average": ("v15_predictions.parquet", "y_hist")}


def model_rows(names: pd.Series) -> None:
    """One row per district-crop-year with every model's estimate (same rows, same yields) for the BI page."""
    K = ["year", "crop", "landkreis"]
    base = pd.read_parquet(MET / "v15_predictions.parquet")[K + ["y_true"]]
    for label, (f, col) in MODEL_COLUMNS.items():
        if not (MET / f).exists():
            continue
        d = pd.read_parquet(MET / f)
        d = d[K + ["y_true", col]].rename(columns={col: label, "y_true": "yt"})
        base = base.merge(d, on=K, how="left")
        base.loc[(base.yt - base.y_true).abs() >= 0.05, label] = np.nan
        base = base.drop(columns="yt")
    base["district"] = base.landkreis.map(names)
    base["state"] = base.landkreis.str[:3].map(STATES)
    base.rename(columns={"y_true": "actual"}).to_parquet(OUT / "model_rows.parquet", index=False)

def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    g = gpd.read_file(ROOT / "data/raw/geo/NUTS_RG_01M_2021_4326_LEVL_3.shp")
    names = g[g.CNTR_CODE == "DE"].set_index("NUTS_ID")["NUTS_NAME"]

    # main model = v18 XE18 (transformer ensemble + memory); else v15 E15; else v12 Q12
    if (MET / "v18_predictions.parquet").exists():
        P = pd.read_parquet(MET / "v18_predictions.parquet")[["year", "crop", "landkreis", "y_true", "y_hist", "b5", "XE18"]]
        P = P.rename(columns={"XE18": "predicted"})
    elif (MET / "v15_predictions.parquet").exists():
        P = pd.read_parquet(MET / "v15_predictions.parquet")[["year", "crop", "landkreis", "y_true", "y_hist", "b5", "E15"]]
        P = P.rename(columns={"E15": "predicted"})
    else:
        P = pd.read_csv(MET / "v12_vs_actual_2010_2025.csv").rename(columns={"v12": "predicted"})
    P = P.rename(columns={"y_true": "actual", "b5": "avg5", "y_hist": "avg_long"})
    P["district"] = P["landkreis"].map(names)
    P["state"] = P["landkreis"].str[:3].map(STATES)
    for c in ("predicted", "avg5", "avg_long"):
        P[f"err_{c}"] = 100 * (P[c] - P["actual"]) / P["actual"]
    P[["year", "crop", "landkreis", "district", "state", "actual", "predicted", "avg5", "avg_long",
       "err_predicted", "err_avg5", "err_avg_long"]].to_parquet(OUT / "rows.parquet", index=False)

    N = P.groupby(["crop", "year"], as_index=False)[["actual", "predicted", "avg5", "avg_long"]].mean()
    N["n_districts"] = P.groupby(["crop", "year"]).size().values
    N["actual_vs_normal"] = 100 * (N.actual / N.avg_long - 1)
    N["predicted_vs_normal"] = 100 * (N.predicted / N.avg_long - 1)
    N["kind"] = np.where(N.actual_vs_normal <= BAD, "bad", np.where(N.actual_vs_normal >= GOOD, "good", "normal"))
    N["event"] = [next((v for k, v in EVENTS.items() if y in k), "") if kd == "bad" else ""
                  for y, kd in zip(N.year, N.kind)]
    N["detected"] = np.where(N.kind == "bad", N.predicted_vs_normal < 0,
                             np.where(N.kind == "good", N.predicted_vs_normal > 0,
                                      N.predicted_vs_normal.abs() < 10))
    N["share_captured"] = np.where(N.kind != "normal", 100 * N.predicted_vs_normal / N.actual_vs_normal, np.nan)
    N.to_csv(OUT / "national.csv", index=False)

    rows = [{"crop": c, "accuracy_model": acc(d.predicted, d.actual), "accuracy_avg5": acc(d.avg5, d.actual),
             "accuracy_avg_long": acc(d.avg_long, d.actual),
             "within10_model": 100 * (d.err_predicted.abs() <= 10).mean(),
             "within10_avg5": 100 * (d.err_avg5.abs() <= 10).mean(), "n": len(d)}
            for c, d in P.groupby("crop")]
    pd.DataFrame(rows).to_csv(OUT / "accuracy_by_crop.csv", index=False)
    rows = [{"year": y, "accuracy_model": acc(d.predicted, d.actual), "accuracy_avg5": acc(d.avg5, d.actual),
             "accuracy_avg_long": acc(d.avg_long, d.actual), "n": len(d)} for y, d in P.groupby("year")]
    pd.DataFrame(rows).to_csv(OUT / "accuracy_by_year.csv", index=False)
    D = pd.DataFrame([{"landkreis": k, "district": names.get(k, k), "accuracy_model": acc(d.predicted, d.actual),
                       "accuracy_avg5": acc(d.avg5, d.actual), "n": len(d)} for k, d in P.groupby("landkreis")])
    D["gain_vs_avg5"] = D.accuracy_model - D.accuracy_avg5
    D.to_csv(OUT / "accuracy_by_district.csv", index=False)

    bad = N[N.kind == "bad"]
    summary = {
        "accuracy_district": acc(P.predicted, P.actual), "accuracy_district_avg5": acc(P.avg5, P.actual),
        "accuracy_district_avg_long": acc(P.avg_long, P.actual),
        "accuracy_national": acc(N.predicted, N.actual), "accuracy_national_avg5": acc(N.avg5, N.actual),
        "accuracy_hindcast": acc(P[P.year <= 2015].predicted, P[P.year <= 2015].actual),
        "within10": float(100 * (P.err_predicted.abs() <= 10).mean()),
        "within10_avg5": float(100 * (P.err_avg5.abs() <= 10).mean()),
        "districts": int(P.landkreis.nunique()), "crops": int(P.crop.nunique()),
        "years": [int(P.year.min()), int(P.year.max())], "rows": int(len(P)),
        "bad_years": int(len(bad)), "bad_detected": int(bad.detected.sum()),
        "bad_share_captured_median": float(bad.share_captured.median()),
        "good_years": int((N.kind == "good").sum()), "good_detected": int(N[N.kind == "good"].detected.sum()),
        "false_alarms": int(((N.kind != "bad") & (N.predicted_vs_normal <= BAD)).sum()),
        "mape_model": float(P.err_predicted.abs().mean()), "mape_avg5": float(P.err_avg5.abs().mean()),
    }
    summary["error_reduction_vs_avg5"] = 100 * (1 - summary["mape_model"] / summary["mape_avg5"])
    # one rule for hits and false alarms: "the model warns" = predicted national yield <= threshold below normal
    ct = []
    for thr in (0.0, -5.0, -10.0):
        warn = N.predicted_vs_normal <= thr if thr < 0 else N.predicted_vs_normal < 0
        isbad = N.kind == "bad"
        ct.append({"warning_rule": "predicted below normal" if thr == 0 else f"predicted ≥{-thr:.0f}% below normal",
                   "bad_years_warned": int((warn & isbad).sum()), "bad_years_missed": int((~warn & isbad).sum()),
                   "warnings_in_other_years": int((warn & ~isbad).sum()), "other_years_quiet": int((~warn & ~isbad).sum())})
    pd.DataFrame(ct).to_csv(OUT / "bad_year_warnings.csv", index=False)
    summary["bad_warned_10"] = ct[2]["bad_years_warned"]
    summary["false_alarms_any"] = ct[0]["warnings_in_other_years"]
    summary["crop_years"] = int(len(N))

    V = pd.DataFrame(VERSIONS, columns=["version", "planned", "result", "title", "change", "hindcast_new",
                                        "hindcast_main", "papers_won", "decision", "lesson"])
    papers = []
    for v, f in PAPER_FILES.items():
        if (MET / f).exists():
            c = pd.read_csv(MET / f).assign(version=v)
            c.loc[c.ours_equal_protocol.isna(), "verdict"] = "not applicable"
            papers.append(c)
            V.loc[V.version == v, "papers_won"] = int((c.verdict == "ours better").sum())
            V.loc[V.version == v, "papers_rows"] = int(c.ours_equal_protocol.notna().sum())
    V.to_csv(OUT / "versions.csv", index=False)
    pd.concat(papers, ignore_index=True).to_csv(OUT / "papers.csv", index=False)
    vers = [p.version.iloc[0] for p in papers]
    # the final model's own runs on the papers' splits once they exist, else v12 (v15's network, all 18 rows)
    summary["papers_model"] = "v18" if "v18" in vers else "v12"
    main_papers = papers[vers.index(summary["papers_model"])]
    summary["papers_won"] = int((main_papers.verdict == "ours better").sum())
    summary["papers_total"] = int(len(main_papers))
    v15p = papers[[p.version.iloc[0] for p in papers].index("v15")]          # rows where error memory is allowed
    summary["papers_won_v15"] = int((v15p.verdict == "ours better").sum())
    summary["papers_total_v15"] = int(v15p.ours_equal_protocol.notna().sum())

    f = ROOT / "results/forecasts/forecast_2026_issued_2026-09-27.csv"
    if f.exists():
        F = pd.read_csv(f)
        F["district"] = F["landkreis"].map(names)
        F["vs_normal"] = 100 * (F.y_pred / F.y_hist - 1)
        F.to_csv(OUT / "forecast_2026.csv", index=False)
    # v16 §4: frozen YieldFormer-DE forecasts (one file per year, never overwritten)
    yf = sorted((ROOT / "results/forecasts").glob("forecast_*_yieldformer*_issued_*.csv"))   # incl. the v18 ensemble
    if yf:
        sha = dict(reversed(l.split()) for l in (ROOT / "results/forecasts/FROZEN.sha256").read_text().splitlines() if l.strip())
        Y = pd.concat([pd.read_csv(f).assign(file=f.name, sha256=sha.get(f.name, ""),
                                             model="YieldFormer-DE (v18)" if "ensemble" in f.name
                                             else ("District normal" if "normal" in pd.read_csv(f, nrows=1).method.iloc[0]
                                                   else "YieldFormer-DE-i (v15)"))
                       for f in yf], ignore_index=True)
        Y["district"] = Y["landkreis"].map(names)
        Y["vs_normal"] = 100 * (Y.y_pred / Y.y_hist - 1)
        Y.to_csv(OUT / "forecast_yieldformer.csv", index=False)
    # 2024/2025 backtests of every transformer (rolling folds; v18 reporting)
    if (MET / "v18_predictions.parquet").exists():
        B = pd.read_parquet(MET / "v18_predictions.parquet")
        B = B[B.year.isin([2024, 2025])]
        lab = {"XE18": "YieldFormer-DE (v18)", "E15": "YieldFormer-DE-i (v15)", "TE17": "YieldFormer-DE-T (v17)",
               "PE17": "YieldFormer-DE-P (v17)", "GE16": "XGBoost + memory (reference)", "b5": "5-year average",
               "y_hist": "Long-term average"}
        rows = []
        for (yr, cr), g in B.groupby(["year", "crop"]):
            for c, n in lab.items():
                rows.append({"year": yr, "crop": cr, "model": n, "real": g.y_true.mean(), "forecast": g[c].mean(),
                             "district_accuracy": 100 - 100 * ((g[c] - g.y_true).abs() / g.y_true).mean(),
                             "rows": len(g)})
        pd.DataFrame(rows).to_csv(OUT / "backtest_2024_2025.csv", index=False)
    # Operational checks (v18 change log): JRC MARS 2025 and the sealed 2026 forecasts vs PRELIMINARY national
    # yields. Both use district forecasts weighted by 2020 census crop area (data/processed/static).
    if (MET / "v18_mars_2025.csv").exists():
        pd.read_csv(MET / "v18_mars_2025.csv")[["crop", "mars_crop", "bulletin", "official_dt_ha", "mars_err_pct",
                                                "ens_err_official_pct", "b5_err_official_pct"]].to_csv(
            OUT / "mars_2025.csv", index=False)
    pre_f, area_f = MET / "v18_2026_vs_preliminary.csv", ROOT / "data/processed/static/crop_area_census.parquet"
    fc = {"YieldFormer-DE (v18)": ("forecast_2026_yieldformer-ensemble_issued_2026-10-11.csv", "y_pred"),
          "YieldFormer-DE-i (v15)": ("forecast_2026_yieldformer_issued_2026-10-09.csv", "y_pred"),
          "Normal year": ("forecast_2026_yieldformer-ensemble_issued_2026-10-11.csv", "y_hist")}
    if pre_f.exists() and area_f.exists() and all((ROOT / "results/forecasts" / v[0]).exists() for v in fc.values()):
        pre = pd.read_csv(pre_f).set_index("crop")
        A = pd.read_parquet(area_f)
        A = A[A.census_year == 2020][["landkreis", "crop", "area_ha"]]
        out = pd.DataFrame({"official_preliminary": pre.prelim_official})
        for lab, (file, col) in fc.items():
            d = pd.read_csv(ROOT / "results/forecasts" / file).merge(A, on=["landkreis", "crop"])
            w = d.groupby("crop").apply(lambda g: (g[col] * g.area_ha).sum() / g.area_ha.sum(), include_groups=False)
            out[lab] = 100 * (w.reindex(out.index) / out.official_preliminary - 1)
        out["AgroForecast-DE July (wheat, ratio basis)"] = pre.get("AgroForecast-DE (Jul) err%")
        out["YieldFormer-DE (v18, ratio basis)"] = pre.get("YF-DE (ensemble) err%")
        out.reset_index().to_csv(OUT / "prelim_2026.csv", index=False)
    for name in ("v16_ablation.csv", "v16_intervals.csv", "v16_cleaned.csv", "v16_flags.csv", "v16_wheat2025.csv",
                 "v16_wheat_inputs.csv", "v16_multiple_testing.csv", "v18_intervals.csv", "v18_cleaned.csv",
                 "v18_comparison.csv", "v18_per_crop.csv", "v17_tft_variable_weights.csv"):
        if (MET / name).exists():
            # model family rule (user, 2026-10-10): transformers + simple baselines + one reference (GE16, XGBoost +
            # memory); the plain G16 variant stays in results/metrics and the pre-registration only
            d = pd.read_csv(MET / name)
            for col in ("arm", "forecast"):
                if col in d:
                    d = d[~d[col].isin(["G16"])]
            d.to_csv(OUT / name, index=False)
    summary["insights"] = analytics(P, N, names)
    model_rows(names)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=1))
    print(N[N.kind == "bad"][["crop", "year", "event", "actual_vs_normal", "predicted_vs_normal",
                               "share_captured", "detected"]].round(1).sort_values("actual_vs_normal").to_string(index=False))


if __name__ == "__main__":
    main()
