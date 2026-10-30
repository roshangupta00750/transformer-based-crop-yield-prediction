"""F, F+R, PG against the 5-year-mean baseline; per-crop national/within decomposition of F;
per-year sign of F+R vs F. Read-only on project."""
import numpy as np, pandas as pd
from pathlib import Path
OUT = Path(__file__).resolve().parent / "out"       # results of this analysis
OUT.mkdir(exist_ok=True)
K = ["landkreis", "crop", "year"]
A = pd.read_parquet(OUT / "all_methods_ens.parquet")
Bs = pd.read_parquet(OUT / "baselines_rows.parquet")
Bs = Bs[Bs.year >= 2016][K + ["B1", "B5", "b2"]]
D = A.merge(Bs, on=K, how="left", validate="one_to_one")
assert D.B5.notna().all()
crops = sorted(D.crop.unique())

def pc_skill(df, p, ref):
    e = df.assign(se=(df[p] - df.y_true) ** 2, sr=(df[ref] - df.y_true) ** 2).groupby("crop")[["se", "sr"]].sum()
    return 1 - e.se / e.sr
def mape(df, p):
    return 100 * np.mean(np.abs(df[p] - df.y_true) / df.y_true)
pd.set_option("display.width", 250)
for lab, lo in [("2016-25", 2016), ("2018-25", 2018), ("2019-25", 2019)]:
    s = D[D.year >= lo]
    t = pd.DataFrame({f"{m}_vs_B5": pc_skill(s, m, "B5") for m in ["F", "F+R", "PG", "F+RS", "b"]})
    t.loc["MACRO"] = t.mean()
    print(f"\n=== {lab}: MSE skill vs 5-yr mean B5 ===\n", t.round(3).to_string())
    print("MAPE:", {m: round(mape(s, m), 2) for m in ["F", "F+R", "F+RS", "PG", "B5", "B1", "b"]})
    w = s[s.crop == "winter_wheat"]
    print("wheat MAPE:", {m: round(mape(w, m), 2) for m in ["F", "F+R", "F+RS", "PG", "B5", "B1", "b"]})

# per-year macro skill F+R vs F, and vs b
rows = []
for y, g in D.groupby("year"):
    rows.append({"year": y, "F_vs_b": pc_skill(g, "F", "b").mean(), "FR_vs_b": pc_skill(g, "F+R", "b").mean(),
                 "FR_vs_F": pc_skill(g, "F+R", "F").mean(), "F_vs_B5": pc_skill(g, "F", "B5").mean(),
                 "FR_vs_B5": pc_skill(g, "F+R", "B5").mean(), "PG_vs_F": pc_skill(g, "PG", "F").mean()})
PY = pd.DataFrame(rows)
print("\nper-year macro skill:\n", PY.round(3).to_string(index=False))
print("F+R better than F in", int((PY.FR_vs_F > 0).sum()), "of 10 years")

# national vs within decomposition of F per crop (2016-25)
def decomp(df, p, ref):
    e = df[p] - df.y_true; h = df[ref] - df.y_true
    g = [df.crop, df.year]
    en, hn = e.groupby(g).transform("mean"), h.groupby(g).transform("mean")
    x = pd.DataFrame({"crop": df.crop, "nat_e": en ** 2, "nat_h": hn ** 2, "w_e": (e - en) ** 2, "w_h": (h - hn) ** 2}).groupby("crop").sum()
    return pd.DataFrame({"nat_share_of_ref_SSE": x.nat_h / (x.nat_h + x.w_h), "NAT_skill": 1 - x.nat_e / x.nat_h,
                         "WITHIN_skill": 1 - x.w_e / x.w_h, "TOTAL_skill": 1 - (x.nat_e + x.w_e) / (x.nat_h + x.w_h)})
dd = decomp(D, "F", "b"); dd.loc["MACRO"] = dd.mean()
print("\nF decomposition vs registered y_hist, 2016-25:\n", dd.round(3).to_string())
dd5 = decomp(D, "F", "B5"); dd5.loc["MACRO"] = dd5.mean()
print("\nF decomposition vs B5, 2016-25:\n", dd5.round(3).to_string())
ddr = decomp(D, "F+R", "b"); ddr.loc["MACRO"] = ddr.mean()
print("\nF+R decomposition vs y_hist, 2016-25:\n", ddr.round(3).to_string())
# national direction right (sign of predicted national anomaly vs actual) for F, per crop, 2016-25
nat = D.groupby(["crop", "year"]).apply(lambda g: pd.Series({"pred": (g.F - g.b).mean(), "act": (g.y_true - g.b).mean()}))
nat["right"] = np.sign(nat.pred) == np.sign(nat.act)
print("\nF national direction right (of 10):", nat.groupby("crop").right.sum().to_dict())
print("corr(pred national anomaly, actual) per crop:", nat.groupby("crop").apply(lambda g: np.corrcoef(g.pred, g.act)[0, 1]).round(2).to_dict())
PY.to_csv(OUT / "per_year_macro.csv", index=False)
dd.to_csv(OUT / "F_decomposition_2016_2025.csv")
