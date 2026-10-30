"""Pre-registration v20 (paper/preregistration_2026-10-12_v20.md).

--part A: XE20 = mean(Q20, T20, P20) + v15 §2 error memory on its own errors; decision vs XE18 (v6 §6, 2010-2015).
          Also each member + memory vs its v18 counterpart, national-swing correlation per crop, 2025 national errors.
--part B: in-season forecasts of the v18 members (scripts/73 --cut), ensemble mean x exp(-lambda_Y * m) with XE18's
          memory m and lambda_Y (both known before the season), accuracy per cut date vs the averages.

Outputs: results/metrics/v20_scores.json, v20_predictions.parquet, v20_swing.csv (A); v20_inseason.csv,
         v20_inseason_by_crop.csv (B)
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
M = ROOT / "results/metrics"
K = ["year", "crop", "landkreis"]
WINDOWS = {"hindcast_2010_2015": range(2010, 2016), "dev_2016_2025": range(2016, 2026), "all_2010_2025": range(2010, 2026)}
acc = lambda p, y: float(100 - 100 * np.mean(np.abs(p - y) / y))


def _load(name, file):
    s = importlib.util.spec_from_file_location(name, ROOT / "scripts" / file)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
    return m


def swing_r(P, col):
    N = P.groupby(["crop", "year"])[["y_true", col, "y_hist"]].mean()
    t, p = 100 * (N.y_true / N.y_hist - 1), 100 * (N[col] / N.y_hist - 1)
    return {c: float(np.corrcoef(t.xs(c, level="crop"), p.xs(c, level="crop"))[0, 1]) for c in sorted(N.index.get_level_values(0).unique())}


def part_a() -> None:
    v16 = _load("v16", "69_eval_v16.py")
    P = pd.read_parquet(M / "v18_predictions.parquet")
    for col in ("Q20", "T20", "P20"):
        A = pd.read_parquet(M / f"v20_predictions_{col}.parquet")
        if not (A["n"] == 3).all():
            sys.exit(f"{col}: not every row has all 3 seeds")
        P = P.merge(A[K + [col, "y_hist"]].rename(columns={"y_hist": "yh"}), on=K, how="left")
        if P[col].isna().any() or (P.yh - P.y_hist).abs().max() > 1e-6:
            sys.exit(f"{col} does not cover every row with the same normal")
        P = P.drop(columns="yh")
    lam = {}
    for col, mem in (("Q20", "QE20"), ("T20", "TE20"), ("P20", "PE20")):
        P, lam[mem] = v16.memory(P, col, mem)
    P["X20"] = P[["Q20", "T20", "P20"]].mean(axis=1)
    P, lam["XE20"] = v16.memory(P, "X20", "XE20")
    res = {"lambda": lam}
    for chal, inc in (("XE20", "XE18"), ("QE20", "E15"), ("TE20", "TE17"), ("PE20", "PE17")):
        S = P.rename(columns={chal: "R7"}).assign(F=P[inc])
        r = {w: v16.fast_score(S[S.year.isin(list(y))]) for w, y in WINDOWS.items()}
        h = r["hindcast_2010_2015"]
        r["passes_v6"] = bool(h["p_delta_gt_0"] >= 0.95 and h["mape_R7"] < h["mape_F"] and not h["crops_worse_every_year"])
        res[f"{chal}_vs_{inc}"] = r
        print(f"{chal} vs {inc}: " + " | ".join(f"{w[:4]} {100 - s['mape_R7']:.2f} vs {100 - s['mape_F']:.2f} P {s['p_delta_gt_0']:.3f}"
                                                for w, s in r.items() if isinstance(s, dict)) + f" | passes {r['passes_v6']}")
    res["XE20_adopted"] = res["XE20_vs_XE18"]["passes_v6"]
    sw = pd.DataFrame({"XE18": swing_r(P, "XE18"), "XE20": swing_r(P, "XE20")})
    sw.to_csv(M / "v20_swing.csv")
    g = P[P.year == 2025]; n = g.groupby("crop")[["y_true", "XE18", "XE20"]].mean()
    res["nat2025_err_pct"] = {c: {m: float(100 * (n.at[c, m] / n.at[c, "y_true"] - 1)) for m in ("XE18", "XE20")} for c in n.index}
    P.to_parquet(M / "v20_predictions.parquet", index=False)
    (M / "v20_scores.json").write_text(json.dumps(res, indent=2, default=float))
    print(sw.round(2).to_string()); print("XE20 adopted:", res["XE20_adopted"])


def part_b() -> None:
    P = pd.read_parquet(M / "v18_predictions.parquet")
    lam = {int(k): v for k, v in json.loads((M / "v18_scores.json").read_text())["lambda"].items()}
    rows, crops = [], []
    for cut in ("06-30", "07-31"):
        H = None
        for col in ("Q12", "T17", "P17"):
            d = pd.read_parquet(M / f"horizon_{cut}_{col}.parquet")[K + [col]]
            H = d if H is None else H.merge(d, on=K)
        D = P[K + ["y_true", "y_hist", "b5", "XE18", "m_XE18"]].merge(H, on=K)
        D["XH"] = D[["Q12", "T17", "P17"]].mean(axis=1) * np.exp(-D.year.map(lam) * D.m_XE18)
        for w, yrs in WINDOWS.items():
            g = D[D.year.isin(list(yrs))]
            nat = g.groupby(["crop", "year"])[["y_true", "XH", "XE18", "b5", "y_hist"]].mean()
            rows.append({"cut": cut, "window": w, "rows": len(g), **{f"acc_{c}": acc(g[c], g.y_true) for c in ("XH", "XE18", "b5", "y_hist")},
                         **{f"nat_{c}": acc(nat[c], nat.y_true) for c in ("XH", "XE18", "b5", "y_hist")}})
        crops += [{"cut": cut, "crop": c, "acc_XH": acc(g.XH, g.y_true), "acc_XE18": acc(g.XE18, g.y_true), "acc_b5": acc(g.b5, g.y_true)}
                  for c, g in D.groupby("crop")]
        n25 = D[D.year == 2025].groupby("crop")[["y_true", "XH"]].mean()
        print(f"cut {cut}, 2025 national error %:", (100 * (n25.XH / n25.y_true - 1)).round(1).to_dict())
    T = pd.DataFrame(rows); T.to_csv(M / "v20_inseason.csv", index=False)
    pd.DataFrame(crops).to_csv(M / "v20_inseason_by_crop.csv", index=False)
    print(T.round(2).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument("--part", choices=["A", "B"], required=True)
    (part_a if ap.parse_args().part == "A" else part_b)()
