"""Standard baselines at issue time vs the registered y_hist (read-only on project).

b2  = registered rule (district mean + v4 yearly-mean trend rule) fitted on years <= Y-2
      (reconstruction of F's y_hist from Destatis yields; checked against y_hist)
B1  = the same rule fitted on years <= Y-1 (issue-time historical average)
B5  = district mean of the last five years Y-5..Y-1 (JRC-MARS-style recent average),
      needs >= 3 of 5 years, else falls back to b2
Scores F (v4 F ens) against each, 2016-2025, and the baselines against each other
for 2010-2015 as well (baseline-only, no model).
"""
import numpy as np, pandas as pd
from pathlib import Path
from scipy import stats

P = Path(__file__).resolve().parents[3]          # the project folder (crop_yield_benchmark)
OUT = Path(__file__).resolve().parent / "out"       # results of this analysis
OUT.mkdir(exist_ok=True)
K = ["landkreis", "crop", "year"]
AUT = {"winter_wheat", "winter_barley", "rye", "triticale", "winter_rapeseed"}
y03 = pd.read_parquet(P / "data/interim/yield.parquet")
y99 = pd.read_parquet(P / "data/raw/destatis_1999/yield_1999_2002.parquet")
print("1999-2002 file cols:", y99.columns.tolist(), y99.year.unique() if "year" in y99 else "")
y99 = y99.rename(columns={c: "yield_dt_per_ha" for c in y99.columns if "yield" in c})[K + ["yield_dt_per_ha"]]
Y = pd.concat([y99, y03]).dropna().drop_duplicates(K, keep="last").rename(columns={"yield_dt_per_ha": "y"})
Y = Y[~((Y.crop.isin(AUT)) & (Y.year < 2001)) & (Y.year >= 2000)]

def fit(tr):
    dm = tr.groupby(["landkreis", "crop"]).y.mean()
    cm = tr.groupby("crop").y.mean()
    ref = tr.year.mean()
    slopes = {}
    for c, g in tr.groupby("crop"):
        resid = g.y.to_numpy() - dm.reindex(list(zip(g.landkreis, g.crop))).to_numpy()
        yrs = g.year.to_numpy(float) - ref
        ym = pd.Series(resid).groupby(yrs).mean()
        lr = stats.linregress(ym.index.to_numpy(float), ym.to_numpy(float))
        hz = float(np.abs(yrs).max()) or 1.0
        slopes[c] = float(lr.slope) if (lr.pvalue < 0.01 and abs(lr.slope) * hz > 0.25 * float(resid.std())) else 0.0
    return dm, cm, ref, slopes

def pred(rows, f):
    dm, cm, ref, sl = f
    v = dm.reindex(list(zip(rows.landkreis, rows.crop))).to_numpy()
    v = np.where(np.isnan(v), rows.crop.map(cm).to_numpy(), v)
    return v + rows.crop.map(sl).to_numpy() * (rows.year.to_numpy() - ref)

def b5(rows, Yr, fallback):
    w = Y[(Y.year >= Yr - 5) & (Y.year <= Yr - 1)].groupby(["landkreis", "crop"]).y.agg(["mean", "count"])
    m = w.reindex(list(zip(rows.landkreis, rows.crop)))
    v = np.where(m["count"].to_numpy() >= 3, m["mean"].to_numpy(), np.nan)
    return np.where(np.isnan(v), fallback, v)

# rows to score: 2016-2025 = F's rows; 2008-2015 = all district-crop-years with yields
v4 = pd.read_parquet(P / "results/metrics/v4_predictions.parquet")
F = v4[(v4.arm == "F") & (v4.seed == "ens")][K + ["y_true", "y_pred", "y_hist"]].sort_values(K).reset_index(drop=True)
early = Y[(Y.year >= 2008) & (Y.year <= 2015)].rename(columns={"y": "y_true"}).sort_values(K).reset_index(drop=True)
out = []
for Yr in range(2008, 2026):
    rows = (F[F.year == Yr] if Yr >= 2016 else early[early.year == Yr]).copy()
    f2, f1 = fit(Y[Y.year <= Yr - 2]), fit(Y[Y.year <= Yr - 1])
    rows["b2"], rows["B1"] = pred(rows, f2), pred(rows, f1)
    rows["B5"] = b5(rows, Yr, rows["b2"].to_numpy())
    out.append(rows)
D = pd.concat(out, ignore_index=True)
L = D[D.year >= 2016]
d = (L.b2 - L.y_hist).abs()
print(f"reconstruction b2 vs y_hist (2016-25): median |diff| {d.median():.3f}, 90% {d.quantile(.9):.3f}, max {d.max():.2f}; "
      f"rel median {(d / L.y_hist).median() * 100:.3f}%")
print("by crop median |b2 - y_hist|:", L.assign(d=d).groupby("crop").d.median().round(2).to_dict())

crops = sorted(D.crop.unique())
def skill(df, p, ref):
    e = df.assign(se=(df[p] - df.y_true) ** 2, sr=(df[ref] - df.y_true) ** 2).groupby("crop")[["se", "sr"]].sum()
    return 1 - e.se / e.sr
def mape(df, p):
    return 100 * np.mean(np.abs(df[p] - df.y_true) / df.y_true)

pd.set_option("display.width", 250)
for lab, sub in [("2016-25", D[D.year >= 2016]), ("2018-25", D[D.year >= 2018]), ("2019-25", D[D.year >= 2019])]:
    t = pd.DataFrame({"F_vs_yhist": skill(sub, "y_pred", "y_hist"), "F_vs_B1": skill(sub, "y_pred", "B1"),
                      "F_vs_B5": skill(sub, "y_pred", "B5"),
                      "B1_vs_yhist": skill(sub, "B1", "y_hist"), "B5_vs_yhist": skill(sub, "B5", "y_hist")})
    t.loc["MACRO"] = t.mean()
    print(f"\n=== {lab}: MSE skill (row = crop) ===")
    print(t.round(3).to_string())
    print("MAPE: F %.2f | y_hist %.2f | B1 %.2f | B5 %.2f" % (mape(sub, "y_pred"), mape(sub, "y_hist"), mape(sub, "B1"), mape(sub, "B5")))
    ww = sub[sub.crop == "winter_wheat"]
    print("  wheat MAPE: F %.2f | y_hist %.2f | B1 %.2f | B5 %.2f" % (mape(ww, "y_pred"), mape(ww, "y_hist"), mape(ww, "B1"), mape(ww, "B5")))
    t.to_csv(OUT / f"baselines_{lab}.csv")
for lab, sub in [("2010-15 baseline-only", D[(D.year >= 2010) & (D.year <= 2015)]), ("2008-15 baseline-only", D[(D.year <= 2015)])]:
    t = pd.DataFrame({"B1_vs_b2": skill(sub, "B1", "b2"), "B5_vs_b2": skill(sub, "B5", "b2")})
    t.loc["MACRO"] = t.mean()
    print(f"\n=== {lab}: MSE skill ===")
    print(t.round(3).to_string())
    print("MAPE: b2 %.2f | B1 %.2f | B5 %.2f" % (mape(sub, "b2"), mape(sub, "B1"), mape(sub, "B5")))
# per-year macro skill of F vs each baseline (2016-25)
py = []
for Yr, g in D[D.year >= 2016].groupby("year"):
    py.append({"year": Yr, "F_vs_yhist": skill(g, "y_pred", "y_hist").mean(), "F_vs_B1": skill(g, "y_pred", "B1").mean(),
               "F_vs_B5": skill(g, "y_pred", "B5").mean(), "B5_vs_yhist": skill(g, "B5", "y_hist").mean()})
print("\nper-year macro skill:\n", pd.DataFrame(py).round(3).to_string(index=False))
D.to_parquet(OUT / "baselines_rows.parquet", index=False)
