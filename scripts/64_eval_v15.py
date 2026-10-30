"""Evaluate pre-registration v15 arm E15 (district error memory) against v12 Q12, 2010-2025.

No retraining: E15 corrects Q12's rolling predictions with the district's mean log error of
the previous 1-3 years, scaled by lambda_Y chosen on years before Y only (v15 §2). Also writes
the area-weighted accuracy of v15 §3.

Inputs:  results/metrics/v12_vs_actual_2010_2025.csv (Q12 rolling predictions, y_hist, b5)
         data/processed/static/cs_share.csv, NUTS-3 polygons (area weights, reporting only)
Outputs: results/metrics/v15_predictions.parquet, v15_scores.json, v15_per_crop.csv,
         v15_lambda.csv, v15_area_weighted.csv
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
_s = importlib.util.spec_from_file_location("e7", ROOT / "scripts/51_eval_v7.py")
e7 = importlib.util.module_from_spec(_s); _s.loader.exec_module(e7)

WINDOWS = {"hindcast_2010_2015": range(2010, 2016), "dev_2016_2025": range(2016, 2026),
           "dev_2019_2025": range(2019, 2026)}
GRID = np.round(np.arange(0, 1.0001, 0.05), 2)
MEMORY, MIN_FIT_YEARS = 3, 2
CS = {"winter_wheat": "wheat", "winter_barley": "barley", "summer_barley": "barley", "rye": "rye",
      "oats": "oats", "silage_maize": "maize", "triticale": "triticale", "potatoes": "potato",
      "sugar_beet": "sugarbeet", "winter_rapeseed": "rapeseed"}


def mape(p, t) -> float:
    return float(100 * np.mean(np.abs(p - t) / t))


def error_memory(P: pd.DataFrame) -> pd.DataFrame:
    """m_Y: mean log error of the district-crop over years Y-3..Y-1 (rows that exist), else 0."""
    r = np.log(P.Q12 / P.y_true).set_axis(pd.MultiIndex.from_frame(P[["landkreis", "crop", "year"]]))
    m, n = [], []
    for lk, cr, y in P[["landkreis", "crop", "year"]].itertuples(index=False):
        past = [r.get((lk, cr, t)) for t in range(y - MEMORY, y)]
        past = [v for v in past if v is not None and np.isfinite(v)]
        m.append(float(np.mean(past)) if past else 0.0); n.append(len(past))
    return P.assign(m=m, n_memory=n)


def main() -> None:
    P = pd.read_csv(ROOT / "results/metrics/v12_vs_actual_2010_2025.csv").rename(columns={"v12": "Q12"})
    P = error_memory(P)
    lam = {}
    for Y in range(2010, 2026):
        fit = P[(P.year >= 2011) & (P.year < Y)]
        if fit.year.nunique() < MIN_FIT_YEARS:
            lam[Y] = 0.0
            continue
        lam[Y] = float(min(GRID, key=lambda l: mape(fit.Q12 * np.exp(-l * fit.m), fit.y_true)))
    P["lam"] = P.year.map(lam)
    P["E15"] = P.Q12 * np.exp(-P.lam * P.m)
    P.to_parquet(ROOT / "results/metrics/v15_predictions.parquet", index=False)
    pd.Series(lam, name="lambda").rename_axis("year").to_csv(ROOT / "results/metrics/v15_lambda.csv")

    S = P.rename(columns={"E15": "R7", "Q12": "F"})          # e7.score: R7 = challenger, F = incumbent
    res = {w: e7.score(S[S.year.isin(list(y))]) for w, y in WINDOWS.items()}
    h = res["hindcast_2010_2015"]
    h["decision_E15_replaces_Q12"] = bool(h["p_delta_gt_0"] >= 0.95 and h["mape_R7"] < h["mape_F"]
                                          and not h["crops_worse_every_year"])
    res["lambda_by_year"] = lam
    (ROOT / "results/metrics/v15_scores.json").write_text(json.dumps(res, indent=2))

    rows = [{"window": w, "crop": c, **{f"acc_{x}": 100 - mape(g[x], g.y_true) for x in ("E15", "Q12", "b5", "y_hist")}}
            for w, yrs in WINDOWS.items() for c, g in P[P.year.isin(list(yrs))].groupby("crop")]
    pd.DataFrame(rows).to_csv(ROOT / "results/metrics/v15_per_crop.csv", index=False)

    # v15 §3: area-weighted accuracy (reporting only)
    import geopandas as gpd
    g = gpd.read_file(ROOT / "data/raw/geo/NUTS_RG_01M_2021_4326_LEVL_3.shp")
    g = g[g.CNTR_CODE == "DE"].to_crs("EPSG:3035")
    area_ha = pd.Series(g.geometry.area.values / 1e4, index=g.NUTS_ID)
    sh = pd.read_csv(ROOT / "data/processed/static/cs_share.csv").set_index("landkreis")
    P["w"] = [sh.at[lk, CS[c]] * area_ha.get(lk, np.nan) if lk in sh.index else np.nan
              for lk, c in zip(P.landkreis, P.crop)]
    out = []
    for w, yrs in WINDOWS.items():
        d = P[P.year.isin(list(yrs)) & P.w.notna()]
        for x in ("E15", "Q12", "b5", "y_hist"):
            e = (d[x] - d.y_true).abs() / d.y_true
            out.append({"window": w, "forecast": x, "accuracy": 100 - mape(d[x], d.y_true),
                        "area_weighted_accuracy": float(100 - 100 * np.sum(d.w * e) / np.sum(d.w)), "rows": len(d)})
    pd.DataFrame(out).to_csv(ROOT / "results/metrics/v15_area_weighted.csv", index=False)

    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk in ("n_rows", "mape_R7", "mape_F", "mape_b5",
                     "delta_skill_R7_minus_F", "delta_ci90", "p_delta_gt_0", "crops_worse_every_year",
                     "decision_E15_replaces_Q12")} if isinstance(v, dict) and "n_rows" in v else v
                      for k, v in res.items()}, indent=1))
    print(pd.DataFrame(out).round(2).to_string(index=False))


if __name__ == "__main__":
    main()
