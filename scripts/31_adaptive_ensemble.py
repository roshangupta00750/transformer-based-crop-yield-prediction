"""Adaptive per-crop, per-harvest-year model weighting from past performance.

Uses results/metrics/rolling_predictions.parquet (scripts/30). For harvest
year Y and crop c, every candidate's weight comes ONLY from its errors in
years < Y for that crop -- the information available before the season:

    candidates   itransformer drv/nodrv (seed-ensemble), xgboost drv/nodrv,
                 historical average (the district baseline itself)
    weighted     w_m ∝ 1 / past_MSE_m, past years discounted with a half-life
                 of HALF_LIFE years (recent seasons count more)
    select       the single candidate with the lowest discounted past MSE

The first MIN_HISTORY years only build history; later years are scored.
Every row is one crop-year. Writes results/metrics/adaptive_eval.csv and
results/metrics/adaptive_weights.csv.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
HALF_LIFE = 3.0
MIN_HISTORY = 2


def candidates(pred: pd.DataFrame) -> pd.DataFrame:
    """Wide table: one row per (year, crop, landkreis), one column per candidate."""
    keep = pred[((pred["model"] == "itransformer") & (pred["seed"] == "ens")) |
                (pred["model"] == "xgboost")].copy()
    keep["cand"] = keep["model"] + "_" + keep["arm"]
    wide = keep.pivot_table(index=["year", "crop", "landkreis"], columns="cand",
                            values="y_pred").reset_index()
    base = (pred.groupby(["year", "crop", "landkreis"], as_index=False)
                .agg(y_true=("y_true", "first"), y_hist=("y_hist", "first")))
    wide = base.merge(wide, on=["year", "crop", "landkreis"], how="left")
    wide["historical_avg"] = wide["y_hist"]
    return wide


def score(yt, yp, yh) -> dict:
    sse, ss_hist = float(((yt - yp) ** 2).sum()), float(((yt - yh) ** 2).sum())
    a, p = (yt.mean() / yh.mean() - 1) * 100, (yp.mean() / yh.mean() - 1) * 100
    return {"rmse": float(np.sqrt(np.mean((yt - yp) ** 2))),
            "skill": 1 - sse / ss_hist if ss_hist > 0 else np.nan,
            "direction_ok": bool(np.sign(a) == np.sign(p))}


def main() -> None:
    path = ROOT / "results/metrics/rolling_predictions.parquet"
    if not path.exists():
        sys.exit(f"{path} missing; run scripts/30_eval_rolling.py first")
    wide = candidates(pd.read_parquet(path))
    cands = [c for c in wide.columns if c.startswith(("itransformer_", "xgboost_"))] + ["historical_avg"]
    years = sorted(wide["year"].unique())
    print(f"candidates: {cands}\nyears: {years[0]}-{years[-1]}")

    # per candidate, crop, year: MSE on that year (the history the weights learn from)
    mse = {(c, cr, y): float(((g["y_true"] - g[c]) ** 2).mean())
           for (y, cr), g in wide.groupby(["year", "crop"]) for c in cands if g[c].notna().all()}

    rows, wrows = [], []
    for (y, cr), g in wide.groupby(["year", "crop"]):
        past = [yy for yy in years if yy < y]
        if len(past) < MIN_HISTORY:
            continue
        disc = {yy: 0.5 ** ((y - yy - 1) / HALF_LIFE) for yy in past}
        past_mse = {}
        for c in cands:
            vals = [(mse[(c, cr, yy)], disc[yy]) for yy in past if (c, cr, yy) in mse]
            if len(vals) >= MIN_HISTORY:
                past_mse[c] = sum(m * w for m, w in vals) / sum(w for _, w in vals)
        if not past_mse:
            continue
        inv = {c: 1 / v for c, v in past_mse.items()}
        tot = sum(inv.values())
        w = {c: v / tot for c, v in inv.items()}
        best = min(past_mse, key=past_mse.get)
        yt, yh = g["y_true"].to_numpy(), g["y_hist"].to_numpy()
        weighted = sum(w[c] * g[c].to_numpy() for c in w)
        out = {"year": y, "crop": cr, "n": len(g), "selected": best}
        for name, yp in (("weighted", weighted), ("select", g[best].to_numpy())):
            s = score(yt, yp, yh)
            out.update({f"{name}_{k}": v for k, v in s.items()})
        for c in cands:
            if g[c].notna().all():
                out[f"{c}_skill"] = score(yt, g[c].to_numpy(), yh)["skill"]
        rows.append(out)
        wrows.append({"year": y, "crop": cr, **{f"w_{c}": round(v, 3) for c, v in w.items()}})

    res = pd.DataFrame(rows)
    res.to_csv(ROOT / "results/metrics/adaptive_eval.csv", index=False)
    pd.DataFrame(wrows).to_csv(ROOT / "results/metrics/adaptive_weights.csv", index=False)

    single = [c for c in cands if f"{c}_skill" in res]
    print(f"\ncrop-years scored: {len(res)}  (years {res.year.min()}-{res.year.max()})")
    print("beats the historical average (count of crop-years):")
    for c in single:
        print(f"  {c:22s} {int((res[f'{c}_skill'] > 0).sum()):3d}")
    for name in ("weighted", "select"):
        print(f"  ADAPTIVE {name:13s} {int((res[f'{name}_skill'] > 0).sum()):3d}   "
              f"direction right {int(res[f'{name}_direction_ok'].sum())}")
    best_single = max(single, key=lambda c: int((res[f"{c}_skill"] > 0).sum()))
    for name in ("weighted", "select"):
        n = int((res[f"{name}_skill"] > res[f"{best_single}_skill"]).sum())
        print(f"  {name} beats best single model ({best_single}) in {n} of {len(res)} crop-years")
    print("\nmodel selected most often per crop:")
    print(res.groupby("crop")["selected"].agg(lambda s: s.value_counts().index[0]).to_string())


if __name__ == "__main__":
    main()
