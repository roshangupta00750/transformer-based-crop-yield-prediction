"""Published papers vs our models' ROLLING forecasts on the papers' own rows and metrics (v18 reporting).

Rows and published values exactly as scripts/40 (AgroForecast-DE matched districts 2023-24; CropFusionNet
R2 2022-24 and 2019-24 pooled; Srivastava 2022 RMSE/MAE 2017/18/19, winter wheat). Our forecasts are the
rolling ones (train <= Y-2, validate Y-1, test Y): E15, T17/TE17, P17/PE17, X18/XE18 (v18 ensemble).

Fairness, stated per row ("protocol" column):
  Srivastava   single-year tests trained to Y-1: our rolling folds train to Y-2 -> equal or stricter -> FAIR
  AF-DE / CFN  multi-year tests trained to 2019/2018: our rolling folds have seen yields up to Y-2 (and the
               memory up to Y-1) -> favourable to us -> INDICATIVE ONLY, not counted as wins.
Equal-protocol numbers of earlier versions (compare_equal_protocol_*.csv) are shown alongside.

Usage:  CMP_YIELD=data/interim/v5/yield.parquet python scripts/75_paper_compare_v18.py external/AgroForecast-DE/docs/data
Output: results/metrics/v18_paper_comparison.csv
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
M = ROOT / "results/metrics"
s = importlib.util.spec_from_file_location("x40", ROOT / "scripts/40_compare_table.py")
x40 = importlib.util.module_from_spec(s); s.loader.exec_module(x40)
MODELS = ["E15", "TE17", "PE17", "XE18"]


def main() -> None:
    A = Path(sys.argv[1])
    P = pd.read_parquet(M / "v18_predictions.parquet")
    k = ["landkreis", "crop", "year"]
    rows = []

    def add(paper, crop, years, protocol, g, theirs, metric):
        r = {"paper": paper, "crop": crop, "years": years, "protocol": protocol, "n": len(g), "metric": metric,
             "published": theirs, "hist_avg": x40.met(g.y_true, g.y_hist)[metric]}
        for m in MODELS:
            r[m] = x40.met(g.y_true, g[m])[metric]
        rows.append(r)

    for crop in ("winter_wheat", "silage_maize"):
        Mt = x40.afde_matched(A, crop).merge(P, on=k, how="inner")
        assert np.allclose(Mt.obs, Mt.y_true, atol=0.051)
        them = x40.met(Mt.obs, Mt.pred_them)
        for met in ("r2", "rmse", "mape"):
            add("AgroForecast-DE (same districts)", crop, "2023-24", "indicative (we saw newer yields)", Mt, them[met], met)
    for crop, r2 in x40.CFN.items():
        for years, sel in (("2022-24", [2022, 2023, 2024]), ("2019-24 pooled", list(range(2019, 2025)))):
            add("CropFusionNet paper", crop, years, "indicative (we saw newer yields)",
                P[(P.crop == crop) & P.year.isin(sel)], r2, "r2")
    for y, v in x40.SRV.items():
        for met in v:
            add("Srivastava 2022 CNN", "winter_wheat", str(y), "fair (ours trained on less)",
                P[(P.crop == "winter_wheat") & (P.year == y)], v[met], met)

    T = pd.DataFrame(rows)
    hi = T.metric == "r2"
    for m in MODELS:
        d = T[m] - T.published
        tol = np.where(T.paper.str.startswith("Srivastava"), 0.05, np.where(hi, 0.0005, 0.0))
        T[f"{m}_verdict"] = np.where(np.abs(d) <= tol + 1e-9, "tie", np.where(hi == (d > 0), "ours better", "ours worse"))
    eq = pd.read_csv(M / "compare_equal_protocol_v12QF.csv")
    T = T.merge(eq[["paper", "crop", "years", "metric", "ours_equal_protocol"]].rename(
        columns={"ours_equal_protocol": "v12_equal_protocol"}).assign(
        years=lambda d: d.years.str.replace(" (their test years)", "", regex=False)
                             .str.replace(" pooled incl. our validation years 2019-21, as their code", " pooled",
                                          regex=False)),
        on=["paper", "crop", "years", "metric"], how="left")
    T.to_csv(M / "v18_paper_comparison.csv", index=False)
    pd.set_option("display.width", 250)
    print(T[["paper", "crop", "years", "metric", "published", "v12_equal_protocol", "E15", "TE17", "PE17", "XE18",
             "hist_avg", "XE18_verdict"]].round(3).to_string(index=False))
    for prot in T.protocol.unique():
        g = T[T.protocol == prot]
        print(f"\n{prot}: " + ", ".join(f"{m} {int((g[f'{m}_verdict'] == 'ours better').sum())}/{len(g)}" for m in MODELS))


if __name__ == "__main__":
    main()
