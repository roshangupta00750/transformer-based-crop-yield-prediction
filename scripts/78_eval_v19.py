"""Pre-registration v19 (paper/preregistration_2026-10-12_v19.md): swing calibration S19 and interval
calibration C19 on YieldFormer-DE's (v18 XE18) rolling forecasts. No retraining.

S19: p' = n * exp(kappa_Y * abar_{c,Y} + r), a = log(p/n), abar = crop-year mean of a over districts, r = a - abar;
     kappa_Y from {0.5..3.0} minimising pooled MAPE on 2010..Y-1 (>= 3 years, else 1).
C19: per-crop q10/q90 of past log errors log(y/p) (2016..Y-1), half-widths x phi_Y, phi_Y = smallest of {1.0..2.0}
     reaching >= 80% pooled coverage on the earlier years 2017..Y-1 (each rebuilt from its own earlier years).

Outputs: results/metrics/v19_predictions.parquet, v19_scores.json, v19_per_crop.csv, v19_coverage.csv
Usage:   python scripts/78_eval_v19.py
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
M = ROOT / "results/metrics"
KGRID = np.round(np.arange(0.5, 3.0001, 0.1), 2)
PGRID = np.round(np.arange(1.0, 2.0001, 0.1), 2)
WINDOWS = {"hindcast_2010_2015": range(2010, 2016), "dev_2016_2025": range(2016, 2026), "all_2010_2025": range(2010, 2026)}


def _load(name, file):
    s = importlib.util.spec_from_file_location(name, ROOT / "scripts" / file)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
    return m


mape = lambda p, y: float(100 * np.mean(np.abs(p - y) / y))
acc = lambda p, y: 100 - mape(p, y)


def swing(P: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    a = np.log(P.XE18 / P.y_hist)
    abar = a.groupby([P.crop, P.year]).transform("mean")
    P = P.assign(a=a, abar=abar, r=a - abar)
    cal = lambda d, k: d.y_hist * np.exp(k * d.abar + d.r)
    kap = {}
    for Y in range(2010, 2027):
        fit = P[P.year < Y]
        kap[Y] = 1.0 if fit.year.nunique() < 3 else float(min(KGRID, key=lambda k: mape(cal(fit, k), fit.y_true)))
    P["S19"] = cal(P, P.year.map(kap))
    return P, kap


def slope(P, col):
    N = P.groupby(["crop", "year"])[["y_true", col, "y_hist"]].mean()
    return float(np.polyfit(100 * (N.y_true / N.y_hist - 1), 100 * (N[col] / N.y_hist - 1), 1)[0])


def interval(P, col, Y, kind):
    """Per-crop (lo, hi) offsets for year Y from errors 2016..Y-1: 'abs' = current method, 'log' = C19 base."""
    past = P[P.year.between(2016, Y - 1)]
    e = (past.y_true - past[col]) if kind == "abs" else np.log(past.y_true / past[col])
    return e.groupby(past.crop).quantile([0.1, 0.9]).unstack()


def covered(P, col, Y, q, kind, phi=1.0):
    g = P[P.year == Y].join(q, on="crop")
    if kind == "abs":
        lo, hi = g[col] + phi * g[0.1], g[col] + phi * g[0.9]
    else:
        lo, hi = g[col] * np.exp(phi * g[0.1]), g[col] * np.exp(phi * g[0.9])
    return g.y_true.between(lo, hi), g.y_true > hi, (hi - lo) / g[col]


def coverage(P, col):
    rows = []
    for Y in range(2019, 2026):
        cur_in, cur_up, cur_w = covered(P, col, Y, interval(P, col, Y, "abs"), "abs")
        hist = [(Yp, interval(P, col, Yp, "log")) for Yp in range(2017, Y)]
        phi = 1.0
        for f in PGRID:                                   # smallest phi reaching >= 80% on earlier years
            ins = pd.concat([covered(P, col, Yp, q, "log", f)[0] for Yp, q in hist]) if hist else None
            phi = f
            if ins is None or ins.mean() >= 0.80:
                break
        c_in, c_up, c_w = covered(P, col, Y, interval(P, col, Y, "log"), "log", phi)
        rows.append({"year": Y, "current_coverage": 100 * cur_in.mean(), "current_above": 100 * cur_up.mean(),
                     "current_width_pct": 100 * cur_w.mean(), "phi": phi, "c19_coverage": 100 * c_in.mean(),
                     "c19_above": 100 * c_up.mean(), "c19_width_pct": 100 * c_w.mean()})
    return pd.DataFrame(rows)


def main() -> None:
    v16 = _load("v16", "69_eval_v16.py")
    P = pd.read_parquet(M / "v18_predictions.parquet")
    P, kap = swing(P)
    res = {"kappa": kap}
    S = P.rename(columns={"S19": "R7"}).assign(F=P.XE18)
    res["S19_vs_XE18"] = {w: v16.fast_score(S[S.year.isin(list(y))]) for w, y in WINDOWS.items()}
    h = res["S19_vs_XE18"]["hindcast_2010_2015"]
    res["S19_adopted"] = bool(h["p_delta_gt_0"] >= 0.95 and h["mape_R7"] < h["mape_F"] and not h["crops_worse_every_year"])
    N = P.groupby(["crop", "year"]).apply(lambda g: 100 * (g.y_true.mean() / g.y_hist.mean() - 1)).rename("dev")
    P = P.join(N, on=["crop", "year"])
    ext = P.dev.abs() >= 10
    res["extreme"] = {"rows": int(ext.sum()), "acc_S19": acc(P.S19[ext], P.y_true[ext]), "acc_XE18": acc(P.XE18[ext], P.y_true[ext]),
                      "acc_S19_normal": acc(P.S19[~ext], P.y_true[~ext]), "acc_XE18_normal": acc(P.XE18[~ext], P.y_true[~ext])}
    res["swing_slope"] = {"XE18": slope(P, "XE18"), "S19": slope(P, "S19")}
    main_col = "S19" if res["S19_adopted"] else "XE18"
    C = coverage(P, main_col)
    gap = lambda s: float((s - 80).abs().mean())
    res["C19"] = {"forecast": main_col, "current_mean_cov": float(C.current_coverage.mean()), "c19_mean_cov": float(C.c19_coverage.mean()),
                  "current_gap": gap(C.current_coverage), "c19_gap": gap(C.c19_coverage)}
    res["C19_adopted"] = bool(res["C19"]["c19_gap"] < res["C19"]["current_gap"] and 77 <= res["C19"]["c19_mean_cov"] <= 85)
    P.drop(columns=["a", "abar", "r"]).to_parquet(M / "v19_predictions.parquet", index=False)
    (M / "v19_scores.json").write_text(json.dumps(res, indent=2, default=float))
    C.to_csv(M / "v19_coverage.csv", index=False)
    pd.DataFrame([{"window": w, "crop": c, "acc_XE18": acc(g.XE18, g.y_true), "acc_S19": acc(g.S19, g.y_true)}
                  for w, yrs in WINDOWS.items() for c, g in P[P.year.isin(list(yrs))].groupby("crop")]
                 ).to_csv(M / "v19_per_crop.csv", index=False)
    for w, s in res["S19_vs_XE18"].items():
        print(f"{w}: S19 {100 - s['mape_R7']:.2f} vs XE18 {100 - s['mape_F']:.2f}  Δskill {s['delta_skill_R7_minus_F']:+.3f} "
              f"CI {s['delta_ci90'][0]:+.3f}..{s['delta_ci90'][1]:+.3f}  P {s['p_delta_gt_0']:.3f}  worse-every-year {s['crops_worse_every_year']}")
    print("S19 adopted:", res["S19_adopted"], "| kappa:", {k: v for k, v in kap.items() if k >= 2013})
    print("extreme crop-years:", {k: round(v, 2) if isinstance(v, float) else v for k, v in res["extreme"].items()})
    print("swing slope:", {k: round(v, 2) for k, v in res["swing_slope"].items()})
    print(C.round(1).to_string(index=False))
    print("C19:", {k: (round(v, 2) if isinstance(v, float) else v) for k, v in res["C19"].items()}, "adopted:", res["C19_adopted"])


if __name__ == "__main__":
    main()
