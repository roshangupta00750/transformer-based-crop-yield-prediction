"""Evaluate pre-registration v16 §2-§3 (paper/preregistration_2026-10-09_v16.md). Mac, no training.

§2 ablations: A16a = long-term normal + error memory, A16b = 5-year average + error memory,
   G16 = gradient boosting (scripts/68_gbm_ablation.py, GPU cluster), GE16 = G16 + error memory; YieldFormer-DE (E15)
   must beat each under v6 §6 on the hindcast 2010-2015. The memory rule is v15 §2, applied to
   each arm's own past errors.
§3.1 implausible official records (flag rule), §3.2 year-block bootstrap intervals,
§3.3 multiple-testing table, §3.4 2025 winter wheat (prediction side; inputs: scripts/70).

Inputs:  results/metrics/v15_predictions.parquet, results/metrics/v16_gbm/G16_<Y>.parquet (optional),
         data/interim/v5/yield.parquet
Outputs: results/metrics/v16_predictions.parquet, v16_scores.json, v16_ablation.csv, v16_per_crop.csv,
         v16_flags.csv, v16_cleaned.csv, v16_intervals.csv, v16_multiple_testing.csv, v16_wheat2025.csv
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
_s = importlib.util.spec_from_file_location("v15", ROOT / "scripts/64_eval_v15.py")
v15 = importlib.util.module_from_spec(_s); _s.loader.exec_module(v15)

M = ROOT / "results/metrics"
WINDOWS = {"hindcast_2010_2015": range(2010, 2016), "dev_2016_2025": range(2016, 2026),
           "all_2010_2025": range(2010, 2026)}
KEY = ["year", "crop", "landkreis"]
B = 2000
STATE = {"DE1": "Baden-Württemberg", "DE2": "Bayern", "DE3": "Berlin", "DE4": "Brandenburg", "DE5": "Bremen",
         "DE6": "Hamburg", "DE7": "Hessen", "DE8": "Mecklenburg-Vorpommern", "DE9": "Niedersachsen",
         "DEA": "Nordrhein-Westfalen", "DEB": "Rheinland-Pfalz", "DEC": "Saarland", "DED": "Sachsen",
         "DEE": "Sachsen-Anhalt", "DEF": "Schleswig-Holstein", "DEG": "Thüringen"}
# v16 §3.3: every model-change decision taken on the 2010-2015 hindcast (registered P(Δ>0))
DECISIONS = [("v7", "R7 recent-level baseline", 0.874, False), ("v8", "Y rapeseed model", 0.83, False),
             ("v9", "baseline blend", 0.84, False), ("v10", "D10 corrected data", 0.003, False),
             ("v12", "Q12 vs v4 F (same rows)", 0.50, False), ("v13", "C13 crop-specific satellite", 0.08, False),
             ("v14", "H14 HYRAS weather", 0.24, False), ("v15", "E15 error memory", 0.984, True),
             ("v15", "W15 area-weighted training", 0.39, False)]


def acc(p, t) -> float:
    return float(100 - 100 * np.mean(np.abs(p - t) / t))


def memory(P: pd.DataFrame, col: str, out: str) -> tuple[pd.DataFrame, dict]:
    """v15 §2 applied to column `col`'s own past errors -> column `out`."""
    r = np.log(P[col] / P.y_true).set_axis(pd.MultiIndex.from_frame(P[["landkreis", "crop", "year"]]))
    r = r[np.isfinite(r)].to_dict()
    m = []
    for lk, cr, y in P[["landkreis", "crop", "year"]].itertuples(index=False):
        past = [r[(lk, cr, t)] for t in range(y - v15.MEMORY, y) if (lk, cr, t) in r]
        m.append(float(np.mean(past)) if past else 0.0)
    P = P.assign(**{f"m_{out}": m})
    lam = {}
    for Y in range(2010, 2027):
        fit = P[(P.year >= 2011) & (P.year < Y)]
        lam[Y] = 0.0 if fit.year.nunique() < v15.MIN_FIT_YEARS else float(
            min(v15.GRID, key=lambda l: v15.mape(fit[col] * np.exp(-l * fit[f"m_{out}"]), fit.y_true)))
    P[out] = P[col] * np.exp(-P.year.map(lam) * P[f"m_{out}"])
    return P, lam


def fast_score(W: pd.DataFrame) -> dict:
    """e7.score with identical draws and results, from per crop-year sums instead of re-concatenation."""
    out = {"n_rows": int(len(W))}
    for col in ("R7", "F", "b5", "y_hist"):
        out[f"mape_{col}"] = float(100 * ((W[col] - W.y_true).abs() / W.y_true).mean())
        out[f"skill_{col}_vs_yhist"] = e7.macro_skill(W, col, "y_hist")
        out[f"skill_{col}_vs_b5"] = e7.macro_skill(W, col, "b5")
    years = np.array(sorted(W.year.unique())); rng = np.random.default_rng(0)
    sse = W.assign(R=(W.R7 - W.y_true) ** 2, F_=(W.F - W.y_true) ** 2, H=(W.y_hist - W.y_true) ** 2
                   ).groupby(["crop", "year"])[["R", "F_", "H"]].sum()
    crops = sorted(W.crop.unique())
    A = {k: sse[k].unstack("year").reindex(index=crops, columns=years).fillna(0).to_numpy() for k in ("R", "F_", "H")}
    pos = {y: i for i, y in enumerate(years)}
    d = np.empty(10000)
    for b in range(10000):
        idx = [pos[y] for y in rng.choice(years, len(years))]
        r, f, h = (A[k][:, idx].sum(1) for k in ("R", "F_", "H"))
        ok = h > 0
        d[b] = np.mean(1 - r[ok] / h[ok]) - np.mean(1 - f[ok] / h[ok])
    out["delta_skill_R7_minus_F"] = out["skill_R7_vs_yhist"] - out["skill_F_vs_yhist"]
    out["delta_ci90"] = [float(np.quantile(d, 0.05)), float(np.quantile(d, 0.95))]
    out["p_delta_gt_0"] = float((d > 0).mean())
    s2 = W.assign(eR=(W.R7 - W.y_true) ** 2, eF=(W.F - W.y_true) ** 2).groupby(["crop", "year"])[["eR", "eF"]].sum()
    worse_all = (s2.eR > s2.eF).groupby(level="crop").all()
    out["crops_worse_every_year"] = [c for c, v in worse_all.items() if v]
    return out


def beats(P: pd.DataFrame, arm: str) -> dict:
    """E15 (challenger slot R7) vs `arm` (incumbent slot F) under v6 §6, every window."""
    S = P.rename(columns={"E15": "R7"}).assign(F=P[arm])
    res = {w: fast_score(S[S.year.isin(list(y))]) for w, y in WINDOWS.items()}
    h = res["hindcast_2010_2015"]
    res["E15_beats_" + arm] = bool(h["p_delta_gt_0"] >= 0.95 and h["mape_R7"] < h["mape_F"]
                                   and not h["crops_worse_every_year"])
    return res


def flags(Y: pd.DataFrame) -> pd.DataFrame:
    """v16 §3.1: own-neighbour-years ratio beyond 2x AND not shared by the state's other districts."""
    h = Y.set_index(["landkreis", "crop", "year"])["yield_dt_per_ha"].to_dict()
    rows = []
    for (lk, cr, t), v in h.items():
        nb = [h[(lk, cr, s)] for s in range(t - 3, t + 4) if s != t and (lk, cr, s) in h]
        if len(nb) >= 4 and v > 0:
            rows.append((lk, cr, t, v, float(np.median(nb)), float(np.log(v / np.median(nb)))))
    F = pd.DataFrame(rows, columns=["landkreis", "crop", "year", "y_reported", "neighbour_median", "lr"])
    F["state"] = F.landkreis.str[:3]
    F["state_lr"] = np.nan
    for _, idx in F.groupby(["state", "crop", "year"]).groups.items():
        a = F.loc[idx, "lr"].to_numpy()
        if len(a) >= 4:                           # at least 3 OTHER districts
            F.loc[idx, "state_lr"] = [np.median(np.delete(a, i)) for i in range(len(a))]
    F["flag"] = (F.lr.abs() > np.log(2)) & ((F.lr - F.state_lr).abs() > np.log(2)) & F.state_lr.notna()
    return F


def boot(P: pd.DataFrame, cols: list[str], diffs: list[tuple[str, str]], level="district") -> list[dict]:
    rng = np.random.default_rng(0)
    out = []
    for w, yrs in WINDOWS.items():
        W = P[P.year.isin(list(yrs))]
        if level == "national":
            W = W.groupby(["year", "crop"], as_index=False)[["y_true"] + cols].mean()
        years = np.array(sorted(W.year.unique()))
        err = {c: (W[c] - W.y_true).abs() / W.y_true for c in cols}
        by_year = {y: (W.year == y).to_numpy() for y in years}
        sums = {c: np.array([err[c][by_year[y]].sum() for y in years]) for c in cols}
        n = np.array([by_year[y].sum() for y in years])
        draws = rng.integers(0, len(years), size=(B, len(years)))
        a = {c: 100 - 100 * sums[c][draws].sum(1) / n[draws].sum(1) for c in cols}
        for c in cols:
            out.append({"level": level, "window": w, "quantity": f"accuracy {c}", "estimate": acc(W[c], W.y_true),
                        "lo90": float(np.quantile(a[c], 0.05)), "hi90": float(np.quantile(a[c], 0.95))})
        for x, y in diffs:
            d = a[x] - a[y]
            out.append({"level": level, "window": w, "quantity": f"{x} − {y} (pp)",
                        "estimate": acc(W[x], W.y_true) - acc(W[y], W.y_true),
                        "lo90": float(np.quantile(d, 0.05)), "hi90": float(np.quantile(d, 0.95))})
    return out


def wheat2025(P: pd.DataFrame) -> pd.DataFrame:
    W = P[(P.crop == "winter_wheat") & (P.year == 2025)].assign(state=lambda d: d.landkreis.str[:3].map(STATE))
    cols = ["E15", "Q12", "b5", "y_hist"]
    nat = {"state": "Germany", "districts": len(W), "real": W.y_true.mean(), **{c: W[c].mean() for c in cols}}
    st = [{"state": s, "districts": len(g), "real": g.y_true.mean(), **{c: g[c].mean() for c in cols}}
          for s, g in W.groupby("state")]
    T = pd.DataFrame([nat] + st)
    for c in cols:
        T[f"{c}_err_pct"] = 100 * (T[c] / T.real - 1)
    return T


def decompose(P: pd.DataFrame, col: str, crop: str, year: int) -> dict:
    W = P[(P.crop == crop) & (P.year == year)].assign(state=lambda d: d.landkreis.str[:3])
    e = np.log(W[col] / W.y_true)
    nat = e.mean()
    st = e.groupby(W.state).transform("mean") - nat
    dist = e - nat - st
    tot = (e ** 2).sum()
    return {"forecast": col, "mean_log_error": float(nat), "share_national": float(len(e) * nat ** 2 / tot),
            "share_state": float((st ** 2).sum() / tot), "share_district": float((dist ** 2).sum() / tot)}


def main() -> None:
    P = pd.read_parquet(M / "v15_predictions.parquet")
    P, lam_a = memory(P, "y_hist", "A16a")
    P, lam_b = memory(P, "b5", "A16b")
    arms = ["A16a", "A16b", "y_hist", "b5", "Q12"]
    gfiles = sorted((M / "v16_gbm").glob("G16_*.parquet"))
    lam_g = None
    if len(gfiles) == 16:
        G = pd.concat([pd.read_parquet(f) for f in gfiles], ignore_index=True)
        G = P[KEY + ["y_hist"]].merge(G, on=KEY, how="left", suffixes=("", "_g"))
        miss = G.G16.isna().sum()
        gap = (G.y_hist_g - G.y_hist).abs().max()
        print(f"G16: {len(gfiles)} folds, {miss} of {len(P):,} rows without a G16 forecast, max |y_hist diff| {gap:.2e}")
        if miss:
            sys.exit("G16 does not cover every E15 row; v16 §2 compares on the same rows")
        P["G16"] = G.G16.to_numpy()
        P, lam_g = memory(P, "G16", "GE16")
        arms = ["GE16", "G16"] + arms
    else:
        print(f"G16 not complete ({len(gfiles)}/16 folds): ablation without gradient boosting for now")
    P.to_parquet(M / "v16_predictions.parquet", index=False)

    res = {"lambda": {"A16a": lam_a, "A16b": lam_b, "GE16": lam_g}}
    rows = []
    for a in arms:
        r = beats(P, a)
        res[f"E15_vs_{a}"] = r
        for w in WINDOWS:
            s = r[w]
            rows.append({"window": w, "arm": a, "rows": s["n_rows"], "acc_E15": 100 - s["mape_R7"],
                         "acc_arm": 100 - s["mape_F"], "skill_E15": s["skill_R7_vs_yhist"],
                         "skill_arm": s["skill_F_vs_yhist"], "delta_skill": s["delta_skill_R7_minus_F"],
                         "ci90_lo": s["delta_ci90"][0], "ci90_hi": s["delta_ci90"][1], "p_delta_gt_0": s["p_delta_gt_0"],
                         "crops_E15_worse_every_year": ", ".join(s["crops_worse_every_year"]),
                         "E15_beats_arm_v6": r["E15_beats_" + a] if w == "hindcast_2010_2015" else None})
    if "G16" in P:
        S = P.rename(columns={"G16": "R7"}).assign(F=P.Q12)
        res["G16_vs_Q12"] = {w: fast_score(S[S.year.isin(list(y))]) for w, y in WINDOWS.items()}
    pd.DataFrame(rows).to_csv(M / "v16_ablation.csv", index=False)
    cols = ["E15"] + arms
    pd.DataFrame([{"window": w, "crop": c, **{f"acc_{x}": acc(g[x], g.y_true) for x in cols}}
                  for w, yrs in WINDOWS.items() for c, g in P[P.year.isin(list(yrs))].groupby("crop")]
                 ).to_csv(M / "v16_per_crop.csv", index=False)

    # §3.1 implausible official records
    Y = pd.read_parquet(ROOT / "data/interim/v5/yield.parquet").dropna().drop_duplicates(["landkreis", "crop", "year"])
    F = flags(Y)
    fl = F[F.flag].assign(state=lambda d: d.state.map(STATE)).sort_values(["year", "crop", "landkreis"])
    fl.drop(columns="flag").to_csv(M / "v16_flags.csv", index=False)
    bad = set(map(tuple, fl[["landkreis", "crop", "year"]].to_numpy()))
    P["flagged"] = [(lk, c, y) in bad for lk, c, y in zip(P.landkreis, P.crop, P.year)]
    cl = []
    for w, yrs in WINDOWS.items():
        W = P[P.year.isin(list(yrs))]
        for x in ["E15", "Q12", "b5", "y_hist"] + (["GE16"] if "GE16" in P else []):
            cl.append({"window": w, "forecast": x, "rows": len(W), "flagged": int(W.flagged.sum()),
                       "accuracy_all": acc(W[x], W.y_true), "accuracy_clean": acc(W[x][~W.flagged], W.y_true[~W.flagged])})
    pd.DataFrame(cl).to_csv(M / "v16_cleaned.csv", index=False)
    print(f"§3.1: {len(fl)} flagged official records 2000-2025 ({int(P.flagged.sum())} among scored rows)")

    # §3.2 intervals
    cols_b = ["E15", "Q12", "b5", "y_hist"]
    diffs = [("E15", "b5"), ("E15", "y_hist"), ("E15", "Q12")]
    iv = boot(P, cols_b, diffs) + boot(P, cols_b, diffs, level="national")
    pd.DataFrame(iv).to_csv(M / "v16_intervals.csv", index=False)

    # §3.3 multiple testing
    k = len(DECISIONS)
    mt = pd.DataFrame(DECISIONS, columns=["version", "change", "p_delta_gt_0", "adopted"])
    mt["bonferroni_level"] = 1 - 0.05 / k
    mt["survives_bonferroni"] = mt.p_delta_gt_0 >= 1 - 0.05 / k
    dev = res["E15_vs_Q12"]["dev_2016_2025"]
    mt.loc[mt.change.str.startswith("E15"), "dev_2016_2025_p"] = dev["p_delta_gt_0"]
    mt.to_csv(M / "v16_multiple_testing.csv", index=False)

    # §3.4 2025 winter wheat (prediction side)
    T = wheat2025(P)
    T.to_csv(M / "v16_wheat2025.csv", index=False)
    res["wheat2025_decomposition"] = [decompose(P, c, "winter_wheat", 2025) for c in ("E15", "Q12", "b5", "y_hist")]
    nat = P[P.crop == "winter_wheat"].groupby("year")[["y_true", "E15", "Q12", "b5", "y_hist"]].mean()
    res["wheat_national_error_pct_by_year"] = {int(y): {c: float(100 * (r[c] / r.y_true - 1)) for c in ("E15", "Q12", "b5", "y_hist")}
                                               for y, r in nat.iterrows()}
    (M / "v16_scores.json").write_text(json.dumps(res, indent=2, default=float))

    pd.set_option("display.width", 220)
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    print(pd.DataFrame(cl).round(2).to_string(index=False))
    print(pd.DataFrame(iv).round(2).to_string(index=False))
    print(mt.to_string(index=False))
    print(T.round(1).to_string(index=False))
    print(json.dumps(res["wheat2025_decomposition"], indent=1))


if __name__ == "__main__":
    main()
