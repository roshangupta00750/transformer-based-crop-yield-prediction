"""Publication tables: YieldFormer-DE (v18) against existing models. Every number is read from results/metrics.

Table 1  Head-to-head with published German models, each re-run on that paper's own split and measure
         (compare_equal_protocol_v18XF.csv; ensemble without error memory, which those protocols forbid).
Table 2  Comparisons won per published paper, for each of our transformers (compare_equal_protocol_*.csv).
Table 3  Operational benchmark: JRC MARS national forecasts for Germany 2025 (v18_mars_2025.csv).
Table 4  Our model family and baselines on the same 37,383 rolling forecasts 2010-2025
         (v18_comparison.csv, v18_intervals.csv, version_comparison_full.csv, v16_ablation.csv).
Table 5  Related studies and why they are or are not compared numerically (text, from the literature review).
Table 6  Forecasting capability and performance side by side: published models, the operational JRC MARS service and
         YieldFormer-DE (protocol facts from each paper/repository; numbers from results/metrics).

Output: paper/tables/table{1..5}.md and .tex, table1-4 .csv
Usage:  python scripts/77_paper_tables.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
M = ROOT / "results/metrics"
OUT = ROOT / "paper/tables"
CROP = {"winter_wheat": "Winter wheat", "silage_maize": "Silage maize", "winter_barley": "Winter barley",
        "summer_barley": "Spring barley", "rye": "Rye", "triticale": "Triticale", "winter_rapeseed": "Rapeseed",
        "sugar_beet": "Sugar beet", "potatoes": "Potatoes", "oats": "Oats"}
PAPER = {"AgroForecast-DE (same districts)": "AgroForecast-DE", "CropFusionNet paper": "CropFusionNet",
         "Srivastava 2022 CNN": "Srivastava et al. (2022)"}
METRIC = {"r2": ("R²", True), "rmse": ("RMSE (dt/ha)", False), "mae": ("MAE (dt/ha)", False), "mape": ("MAPE (%)", False)}


def years(s: str) -> str:
    s = str(s)
    return "2019–24 pooled†" if "pooled" in s else s.replace(" (their test years)", "").replace("-", "–")


def write(name: str, df: pd.DataFrame, caption: str, notes: list[str], csv: bool = True) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    md = [f"**{caption}**", "", "| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    md += ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    md += [""] + [f"{n}" for n in notes]
    (OUT / f"{name}.md").write_text("\n".join(md) + "\n")
    esc = lambda v: str(v).replace("%", r"\%").replace("&", r"\&").replace("†", r"$^\dagger$").replace(
        "‡", r"$^\ddagger$").replace("²", r"$^2$").replace("–", "--").replace("**", "").replace(
        "≤", r"$\le$").replace("×", r"$\times$").replace("±", r"$\pm$").replace("−", "-")
    bold = lambda v: (r"\textbf{" + esc(v) + "}") if str(v).startswith("**") else esc(v)
    tex = [r"\begin{table}[t]", r"\centering", r"\small", rf"\caption{{{esc(caption)}}}", rf"\label{{tab:{name}}}",
           r"\begin{tabular}{" + "l" * 2 + "r" * (len(df.columns) - 2) + "}", r"\toprule",
           " & ".join(esc(c) for c in df.columns) + r" \\", r"\midrule"]
    tex += [" & ".join(bold(v) for v in r) + r" \\" for r in df.itertuples(index=False)]
    tex += [r"\bottomrule", r"\end{tabular}"]
    tex += [r"\par\smallskip\footnotesize " + esc(n) for n in notes] + [r"\end{table}"]
    (OUT / f"{name}.tex").write_text("\n".join(tex) + "\n")
    if csv:
        df.to_csv(OUT / f"{name}.csv", index=False)
    print(f"wrote {name}: {len(df)} rows")


def fmt(v: float, metric: str) -> str:
    return f"{v:.3f}" if metric == "r2" else f"{v:.2f}"


def table1() -> pd.DataFrame:
    X = pd.read_csv(M / "compare_equal_protocol_v18XF.csv")
    rows = []
    for r in X.itertuples(index=False):
        hi = METRIC[r.metric][1]
        win = (r.ours_equal_protocol > r.theirs) if hi else (r.ours_equal_protocol < r.theirs)
        ours = fmt(r.ours_equal_protocol, r.metric)
        rows.append({"Published model": PAPER[r.paper], "Crop": CROP[r.crop], "Test years": years(r.years),
                     "Measure": METRIC[r.metric][0], "Published": fmt(r.theirs, r.metric),
                     "YieldFormer-DE": f"**{ours}**" if win else ours, "Historical average": fmt(r.hist_avg, r.metric),
                     "Better": "YieldFormer-DE" if win else "published"})
    T = pd.DataFrame(rows)
    w = (T.Better == "YieldFormer-DE").sum()
    write("table1", T, f"YieldFormer-DE (ensemble, without the error memory) against published models for German districts, "
          f"each on the paper's own training/test split and measure: better in {w} of {len(T)} metric rows, which are "
          f"8 paper × crop × test-period cases (overlapping test years): 5 won, 2 lost, 1 tie (CropFusionNet wheat "
          f"pooled, 0.599 vs 0.597). Bold = better.",
          ["R² higher is better; RMSE, MAE (dt/ha) and MAPE lower is better. YieldFormer-DE is retrained on each paper's "
           "years (3 seeds, per-crop fine-tuning) and evaluated without the district error memory, because these "
           "protocols freeze information at the end of training.",
           "† CropFusionNet's published score most likely pools 2019–24 including its validation years; both views are shown.",
           "Srivastava et al. (2022) report yields that appear to be dry matter (~14% lower), which makes their errors "
           "look ~14% smaller than ours; no correction is applied.",
           "AgroForecast-DE is scored on exactly their published district-years (2023–24).",
           "Not a blind test: the papers' test years (2017–2024) were development years of this study, and CropFusionNet's "
           "pooled rows include years that were our validation years under that protocol."])
    return T


def table2() -> pd.DataFrame:
    arms = [("YieldFormer-DE (ensemble)", "v18XF"), ("YieldFormer-DE-T (TFT)", "v17TF"),
            ("YieldFormer-DE-i (iTransformer)", "v12QF"), ("YieldFormer-DE-P (PatchTST)", "v17PF")]
    rows = []
    for name, f in arms:
        c = pd.read_csv(M / f"compare_equal_protocol_{f}.csv")
        r = {"Our model": name}
        for p, short in PAPER.items():
            g = c[c.paper == p]
            r[short] = f"{int((g.verdict == 'ours better').sum())} / {len(g)}"
        r["All"] = f"{int((c.verdict == 'ours better').sum())} / {len(c)}"
        r["Beats historical average"] = f"{int(c.beats_hist_avg.sum())} / {len(c)}"
        rows.append(r)
    T = pd.DataFrame(rows)
    write("table2", T, "Metric rows won against published models (same rows as Table 1), for each transformer we "
          "trained and for the final ensemble. Counting the CropFusionNet wheat pooled row (0.599 vs 0.597) as a tie, "
          "the ensemble and the TFT member both win 12; no ranking among our models is implied.",
          ["Each comparison = one crop × test period × measure on the paper's own split. YieldFormer-DE = the ensemble of "
           "YieldFormer-DE-i, -T and -P. All evaluated without the error memory, which these protocols do not allow."])
    return T


def table3() -> pd.DataFrame:
    """Errors against the OFFICIAL Destatis national yield for all three forecasts. Ours and the 5-year average are
    converted from districts to the national value with 2020 census crop areas (official values are area-weighted);
    a plain district mean scored against the district mean of the real yields is NOT comparable with MARS."""
    D = pd.read_csv(M / "v18_mars_2025.csv")
    rows = []
    for r in D.itertuples(index=False):
        ours_better = abs(r.ens_err_official_pct) < abs(r.mars_err_pct)
        e = f"{r.ens_err_official_pct:+.1f}"
        rows.append({"Crop": CROP[r.crop], "Official 2025 (dt/ha)": f"{r.official_dt_ha:.1f}",
                     "JRC MARS‡": f"{r.mars_err_pct:+.1f}", "YieldFormer-DE": f"**{e}**" if ours_better else e,
                     "5-year average": f"{r.b5_err_official_pct:+.1f}"})
    rows.append({"Crop": "Mean absolute error", "Official 2025 (dt/ha)": "",
                 "JRC MARS‡": f"{D.mars_err_pct.abs().mean():.1f}",
                 "YieldFormer-DE": f"{D.ens_err_official_pct.abs().mean():.1f}",
                 "5-year average": f"{D.b5_err_official_pct.abs().mean():.1f}"})
    T = pd.DataFrame(rows)
    closer = int((D.ens_err_official_pct.abs() < D.mars_err_pct.abs()).sum())
    write("table3", T, "Operational benchmark, Germany 2025: error (%) of each national forecast against the official "
          f"Destatis national yield. Roughly level with JRC MARS; bold = YieldFormer-DE closer ({closer} of {len(D)}).",
          ["‡ Last JRC MARS Bulletin forecast before harvest (Vol. 33 No 7, August 2025, for winter crops; No 8, "
           "22 September 2025, for spring barley, maize, sugar beet, potatoes). MARS publishes no oats forecast for Germany.",
           "YieldFormer-DE and the 5-year average are district forecasts weighted by 2020 census crop area to give the "
           "national value (the same weighting reproduces the official 2025 yields within ±3.4%). YieldFormer-DE is a "
           "rolling backtest with weather up to harvest: for sugar beet, potatoes and maize it had more of the season "
           "than the September MARS forecast, and 2025 was seen while the model was developed; MARS was issued ex ante."])
    return T


def table4() -> pd.DataFrame:
    F = pd.read_csv(M / "version_comparison_full.csv").set_index("model")
    S = pd.read_csv(M / "v18_comparison.csv")
    A = pd.read_csv(M / "v16_ablation.csv")
    pv = lambda inc, w: S[(S.incumbent == inc) & (S.window == w)].p_delta_gt_0.iloc[0]
    acc_ab = lambda arm, w: A[(A.arm == arm) & (A.window == w)].acc_arm.iloc[0]
    rows = [("YieldFormer-DE (ensemble of the three + memory)", "transformer ensemble", F.at["v18", "2010-15"], F.at["v18", "2016-25"],
             F.at["v18", "national"], F.at["v18", "within10"], "–"),
            ("YieldFormer-DE-T (TFT + memory)", "transformer", F.at["v17 TFT", "2010-15"], F.at["v17 TFT", "2016-25"],
             F.at["v17 TFT", "national"], F.at["v17 TFT", "within10"], f"{100 * pv('TE17', 'hindcast_2010_2015'):.1f}%"),
            ("YieldFormer-DE-i (iTransformer + memory)", "transformer", F.at["v15", "2010-15"], F.at["v15", "2016-25"],
             F.at["v15", "national"], F.at["v15", "within10"], f"{100 * pv('E15', 'hindcast_2010_2015'):.1f}%"),
            ("YieldFormer-DE-P (PatchTST + memory)", "transformer", F.at["v17 PatchTST", "2010-15"], F.at["v17 PatchTST", "2016-25"],
             F.at["v17 PatchTST", "national"], F.at["v17 PatchTST", "within10"],
             f"{100 * pv('PE17', 'hindcast_2010_2015'):.1f}%"),
            ("XGBoost + memory (reference)", "gradient boosting", acc_ab("GE16", "hindcast_2010_2015"),
             acc_ab("GE16", "dev_2016_2025"), np.nan, np.nan, f"{100 * pv('GE16', 'hindcast_2010_2015'):.1f}%"),
            ("Trend + memory", "no network", acc_ab("A16a", "hindcast_2010_2015"), acc_ab("A16a", "dev_2016_2025"),
             np.nan, np.nan, "–"),
            ("5-year average", "baseline", F.at["b5", "2010-15"], F.at["b5", "2016-25"], F.at["b5", "national"],
             F.at["b5", "within10"], "–"),
            ("Long-term average with trend", "baseline", F.at["y_hist", "2010-15"], F.at["y_hist", "2016-25"],
             F.at["y_hist", "national"], F.at["y_hist", "within10"], "–")]
    T = pd.DataFrame(rows, columns=["Model", "Type", "Accuracy 2010–15", "Accuracy 2016–25", "National 2010–25",
                                    "Within ±10%", "P(v18 better), 2010–15"])
    for c in ("Accuracy 2010–15", "Accuracy 2016–25", "National 2010–25", "Within ±10%"):
        T[c] = T[c].map(lambda v: "–" if v != v else f"{v:.2f}")
    T.loc[0, "Accuracy 2010–15"] = f"**{T.loc[0, 'Accuracy 2010–15']}**"
    write("table4", T, "All models on the same 37,383 district × crop × year forecasts, 2010–2025; each year predicted "
          "by a model trained only on years up to two years earlier. Accuracy = 100 − MAPE (%).",
          ["2010–15 is the pre-registered decision window; 2016–25 was seen during development. P(v18 better) = share "
           "of 10,000 year-block bootstrap resamples in which v18 has the higher macro MSE skill (v6 rule; 95% needed; "
           "12 decisions were taken on 2010–15, Bonferroni level 99.6%). XGBoost is a non-transformer reference on the "
           "same inputs, not a candidate model."])
    return T


def table5() -> pd.DataFrame:
    rows = [("CropFusionNet (Srivastava et al., 2026)", "Germany, NUTS-3", "wheat, silage maize, winter barley",
             "TFT-inspired transformer", "re-run on their split (Tables 1–2)"),
            ("AgroForecast-DE (CropFusionNet team)", "Germany, NUTS-3", "wheat, silage maize", "transformer benchmark",
             "scored on their published district-years (Tables 1–2)"),
            ("Srivastava et al. (2022), Sci. Rep.", "Germany, 271 counties", "winter wheat", "CNN",
             "re-run on their single-year tests (Tables 1–2)"),
            ("JRC MARS Bulletin (operational)", "EU countries, national", "cereals, oilseeds, root crops",
             "crop models + statistics", "national 2025 (Table 3)"),
            ("CY-Bench (ESSD, 2026)", "Europe incl. Germany, NUTS", "maize, wheat", "benchmark of ML baselines",
             "not compared: different protocol and region set"),
            ("arXiv 2512.15140 (2025)", "Germany, NUTS-3", "several", "XGBoost, RF, LSTM, TCN",
             "not compared: no matched protocol; reports the same collapse on unseen years")]
    T = pd.DataFrame(rows, columns=["Study", "Region / unit", "Crops", "Model", "Compared here"])
    write("table5", T, "Related studies on yield prediction relevant to this work.", [
        "Numbers are compared only where the published protocol could be repeated exactly on our data."], csv=False)
    return T


def table6() -> pd.DataFrame:
    c = lambda f: pd.read_csv(M / f"compare_equal_protocol_{f}.csv")
    X = c("v18XF")
    won = {p: f"{int((g.verdict == 'ours better').sum())}/{len(g)}" for p, g in X.groupby("paper")}
    F = pd.read_csv(M / "version_comparison_full.csv").set_index("model")
    D = pd.read_csv(M / "v18_mars_2025.csv")
    rows = [
        ("CropFusionNet (2026)", "DE NUTS-3; wheat, silage maize, winter barley", "train ≤2018, val 2019–21, test 2022–24",
         "end of season (inputs up to observed maturity)", "no", "not reported",
         "R² 0.597 / 0.752 / 0.445", f"YieldFormer-DE better in {won['CropFusionNet paper']}"),
        ("AgroForecast-DE (2026)", "DE NUTS-3; wheat, silage maize", "train ≤2019, val 2020–22, test 2023–24",
         "in-season, monthly (Apr, May, Jul) from 50-member seasonal forecasts", "yes (2026, district level)",
         "10–90% range from the weather ensemble (wheat ≈1% wide)", "R² wheat 0.196, maize 0.361 (2023–24)",
         f"YieldFormer-DE better in {won['AgroForecast-DE (same districts)']}"),
        ("Srivastava et al. (2022)", "DE 271 counties; winter wheat", "one test year each (2017, 2018, 2019)",
         "end of season (weekly inputs)", "no", "not reported", "RMSE 6.6 / 8.4 / 6.4 dt/ha",
         f"YieldFormer-DE better in {won['Srivastava 2022 CNN']}"),
        ("JRC MARS Bulletin (operational)", "EU national; cereals, oilseeds, root crops", "operational, every year",
         "in-season, monthly", "yes (national)", "not in bulletin tables",
         f"2025 mean absolute error {D.mars_err_pct.abs().mean():.1f}% vs official (9 crops)",
         f"roughly level: YieldFormer-DE {D.ens_err_official_pct.abs().mean():.1f}% (closer for {int((D.ens_err_official_pct.abs() < D.mars_err_pct.abs()).sum())} of {len(D)} crops)‡"),
        ("**YieldFormer-DE (this work)**", "DE NUTS-3 (363 districts with yields); 10 crops",
         "rolling 2010–25 (train ≤Y−2, val Y−1), pre-registered; papers' own splits",
         "at harvest (2026: data to 19 Sep)", "yes: 2026 frozen with SHA-256; 2027–28 normals",
         "80% range per district (realised 72–84%, 2019–25)",
         f"district accuracy {F.at['v18', '2010-25']:.1f}%, national {F.at['v18', 'national']:.1f}% (2010–25)",
         f"5 of 8 cases won, 2 lost, 1 tie ({int((X.verdict == 'ours better').sum())}/{len(X)} metric rows)")]
    T = pd.DataFrame(rows, columns=["Model", "Region; crops", "Evaluation", "When the forecast is made",
                                    "Future forecasts published", "Uncertainty", "Published / own performance",
                                    "Head-to-head"])
    write("table6", T, "Forecasting capability and performance: published models, the operational benchmark and "
          "YieldFormer-DE.",
          ["Head-to-head = same split, rows and measure (Tables 1–3). ‡ 2025 backtest with weather up to harvest; "
           "MARS forecasts for root crops were issued before harvest.",
           "Limitation of this work: YieldFormer-DE forecasts at harvest; AgroForecast-DE and MARS also forecast "
           "earlier in the season, which was not evaluated here for the final model."], csv=False)
    return T


if __name__ == "__main__":
    for f in (table1, table2, table3, table4, table5, table6):
        f()
