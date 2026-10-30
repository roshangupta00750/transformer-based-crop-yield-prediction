"""Unified metric panel for every design's method on identical rows (read-only on project).

Rows: v4 arm F ens (16,159 district-crop-years 2016-2025). Reference b = F's y_hist for all.
Metrics per window: macro MSE skill (mean over crops of pooled 1-SSE/SSE_b), area-weighted
MSE skill, total dt/ha SSE skill, pooled squared-relative-error skill (MSPE), pooled MAPE,
national-component and within-year skill (macro), wheat MAPE/skill, and the paired
year-block bootstrap of macro skill vs F.
"""
import numpy as np, pandas as pd
from pathlib import Path

R = Path(__file__).resolve().parents[3] / "results" / "metrics"
# inputs: the evidence-* folders written by the v6 workflow run (paper/v6_workflow/), not in the repository
W = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "out"       # results of this analysis
OUT.mkdir(exist_ok=True)
K = ["landkreis", "crop", "year"]

v4 = pd.read_parquet(R / "v4_predictions.parquet")
v5 = pd.read_parquet(R / "v5_predictions.parquet")
base = v4[(v4.arm == "F") & (v4.seed == "ens")][K + ["y_true", "y_hist", "y_pred"]].rename(
    columns={"y_hist": "b", "y_pred": "F"}).sort_values(K).reset_index(drop=True)

def add(df, col, name, key=K):
    global base
    m = base[K].merge(df[key + [col]].rename(columns={col: name}), on=K, how="left", validate="one_to_one")
    assert m[name].notna().all(), name
    base[name] = m[name].to_numpy()

for a in ["T", "A"]:
    add(v4[(v4.arm == a) & (v4.seed == "ens")], "y_pred", a)
for a in ["G", "P", "PG"]:
    add(v5[(v5.arm == a) & (v5.seed == "ens")], "y_pred", a)
cal = pd.read_parquet(W / "evidence-calibration/evidence_predictions.parquet")
for a, n in [("C_N", "C_N"), ("C", "C_prestated")]:
    add(cal[(cal.arm == a) & (cal.seed == "ens")], "y_pred", n)
tns = pd.read_parquet(W / "evidence-selection/tns_fp_predictions.parquet")
for c in ["FP", "TNS_FP"]:
    add(tns, c, c)
psd = pd.read_parquet(W / "evidence-modelling/psd_predictions.parquet")
add(psd[psd.seed == "ens"], "PSD", "PSD")
dom = pd.read_parquet(W / "evidence-domain/layer_F_ens.parquet")
for c, n in [("p_R", "F+R"), ("p_S", "F+S"), ("p_RS", "F+RS"), ("p_HR", "H+R")]:
    add(dom, c, n)
assert np.allclose(dom.set_index(K).loc[list(map(tuple, base[K].values)), "p_F"].to_numpy(), base.F.to_numpy())
dec = pd.read_csv(W / "evidence-evaluation/prequential_decisions.csv")
gcol = [c for c in dec.columns if "FQ" in c and "gate" in c.lower()][0]
g = base[["crop", "year"]].merge(dec[["crop", "year", gcol]], on=["crop", "year"], how="left")[gcol].fillna(0).to_numpy()
nat = (base.F - base.b).groupby([base.crop, base.year]).transform("mean").to_numpy()
base["FQ"] = base.F - g * nat

crops = sorted(base.crop.unique())
AREA = {"winter_wheat": 2900, "silage_maize": 2100, "winter_barley": 1250, "winter_rapeseed": 1050,
        "rye": 600, "sugar_beet": 400, "triticale": 350, "summer_barley": 350, "potatoes": 265, "oats": 160}
wA = np.array([AREA[c] for c in crops], float); wA /= wA.sum()
CI = base.crop.map({c: i for i, c in enumerate(crops)}).to_numpy()
YI = (base.year - 2016).to_numpy()
yt, b = base.y_true.to_numpy(), base.b.to_numpy()

def cells(v):
    A = np.zeros((10, 10)); np.add.at(A, (CI, YI), v); return A

HB = cells((b - yt) ** 2)
# national / within decomposition per crop-year
grp = [base.crop, base.year]
def natwithin(p):
    e = pd.Series(p - yt); h = pd.Series(b - yt)
    en = e.groupby(grp).transform("mean"); hn = h.groupby(grp).transform("mean")
    return cells(en.to_numpy() ** 2), cells(hn.to_numpy() ** 2), cells((e - en).to_numpy() ** 2), cells((h - hn).to_numpy() ** 2)

methods = ["F", "T", "A", "G", "P", "PG", "FP", "C_prestated", "C_N", "TNS_FP", "PSD", "FQ", "F+R", "F+S", "F+RS", "H+R"]
rng = np.random.default_rng(0)
DR = rng.integers(0, 10, size=(10000, 10))

def panel(yrs):
    cols = [y - 2016 for y in yrs]
    m = base.year.isin(yrs).to_numpy()
    CNT = np.stack([np.bincount(r, minlength=len(cols)) for r in rng.integers(0, len(cols), size=(10000, len(cols)))])
    idx = np.array(cols)
    AF = cells((base.F.to_numpy() - yt) ** 2)[:, idx]
    Hc = HB[:, idx]
    bootF = 1 - (CNT @ AF.T) / (CNT @ Hc.T)
    rows = []
    for mth in methods:
        p = base[mth].to_numpy()
        A = cells((p - yt) ** 2)[:, idx]
        pc = 1 - A.sum(1) / Hc.sum(1)
        bootM = 1 - (CNT @ A.T) / (CNT @ Hc.T)
        d = (bootM - bootF).mean(1)
        nA, nH, wAc, wH = natwithin(p)
        nat_sk = 1 - nA[:, idx].sum(1) / nH[:, idx].sum(1)
        win_sk = 1 - wAc[:, idx].sum(1) / wH[:, idx].sum(1)
        rel = ((p[m] - yt[m]) / yt[m]) ** 2; relh = ((b[m] - yt[m]) / yt[m]) ** 2
        wi = crops.index("winter_wheat"); wm = m & (base.crop == "winter_wheat").to_numpy()
        rows.append({"method": mth,
                     "MSESS_macro": pc.mean(), "MSESS_area": (pc * wA).sum(),
                     "SSE_total_skill": 1 - A.sum() / Hc.sum(),
                     "MSPE_skill": 1 - rel.sum() / relh.sum(),
                     "MAPE": 100 * np.mean(np.abs(p[m] - yt[m]) / yt[m]),
                     "NAT_skill_macro": nat_sk.mean(), "WITHIN_skill_macro": win_sk.mean(),
                     "wheat_MAPE": 100 * np.mean(np.abs(p[wm] - yt[wm]) / yt[wm]),
                     "wheat_MSESS": pc[wi],
                     "wheat_years_SSE_lt_b": int((A[wi] < Hc[wi]).sum()),
                     "dMacro_vs_F": pc.mean() - (1 - AF.sum(1) / Hc.sum(1)).mean(),
                     "d_lo90": np.quantile(d, 0.05), "d_hi90": np.quantile(d, 0.95), "P(d>0)": np.mean(d > 0),
                     "cy_SSE_lt_F": int((A < AF - 1e-9 * AF).sum()), "cy_SSE_lt_b": int((A < Hc).sum()),
                     "n_cy": A.size})
    out = pd.DataFrame(rows).set_index("method")
    out.loc["HIST"] = {"MSESS_macro": 0, "MSESS_area": 0, "SSE_total_skill": 0, "MSPE_skill": 0,
                       "MAPE": 100 * np.mean(np.abs(b[m] - yt[m]) / yt[m]),
                       "wheat_MAPE": 100 * np.mean(np.abs(b[m & (base.crop == 'winter_wheat').to_numpy()] - yt[m & (base.crop == 'winter_wheat').to_numpy()]) / yt[m & (base.crop == 'winter_wheat').to_numpy()])}
    return out

pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
res = {}
for lab, yrs in [("2016-25", range(2016, 2026)), ("2018-25", range(2018, 2026)), ("2019-25", range(2019, 2026))]:
    t = panel(list(yrs)); res[lab] = t
    print(f"\n==== window {lab} (ens; ref b = F y_hist) ====")
    print(t.round(4).to_string())
    t.to_csv(OUT / f"panel_{lab}.csv")
base.to_parquet(OUT / "all_methods_ens.parquet", index=False)
