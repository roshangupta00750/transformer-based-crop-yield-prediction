"""Forecast scoreboard 2023 -> 2027: every prediction next to the actual yield.

Two kinds of rows, never mixed up:

    backtest     2023-2025. Arm D, 3-seed ensemble, trained <= Y-2 and early
                 stopped on Y-1, so the model never saw year Y. Made in
                 September 2026, after those yields were public, and the arm
                 was chosen on 2016-2025 results -- so these are NOT
                 prospective forecasts. 80% interval from earlier test years only.
    prospective  frozen files from scripts/36 (results/forecasts/forecast_*.csv),
                 checked against FROZEN.sha256 before scoring. 2026 is the
                 first year forecast before its yields exist.

Actual yields: data/interim/yield.parquet (Destatis 41241-01-03-4). A year is
scored as soon as it appears there; until then it is listed as awaiting.

Per (year, crop): districts, R², RMSE, MAE (dt/ha), skill vs the historical
baseline (1 - SSE_model/SSE_baseline; > 0 = better than the baseline), whether
the national direction (above/below baseline) was right, 80% interval coverage,
and accuracy % = 100 - mean absolute % error per district.

Usage:
    python scripts/37_forecast_scoreboard.py
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results/forecasts"
BACKTEST_YEARS = [2023, 2024, 2025]


def backtests() -> pd.DataFrame:
    P = pd.read_parquet(ROOT / "results/metrics/prereg_predictions.parquet")
    P = P[(P["arm"] == "D") & (P["seed"] == "ens") & (P["horizon"] == 0)].copy()
    P["err"] = P["y_true"] - P["y_pred"]
    rows = []
    for (crop, year), g in P[P["year"].isin(BACKTEST_YEARS)].groupby(["crop", "year"]):
        past = P[(P["crop"] == crop) & (P["year"] < year)]["err"]
        q10, q90 = np.percentile(past, [10, 90])
        rows.append(g.assign(lo80=g["y_pred"] + q10, hi80=g["y_pred"] + q90))
    B = pd.concat(rows)
    B["kind"], B["issued"], B["lead_days"] = "backtest", "2026-09 (retrospective)", 0
    return B[["kind", "issued", "year", "crop", "landkreis", "y_pred", "y_hist",
              "lo80", "hi80", "lead_days", "y_true"]]


def prospective(actual: pd.DataFrame) -> pd.DataFrame:
    sums = {}
    ledger = OUT / "FROZEN.sha256"
    if ledger.exists():
        for line in ledger.read_text().split("\n"):
            if line.strip():
                sha, name = line.split()
                sums[name] = sha
    parts = []
    for f in sorted(OUT.glob("forecast_*_issued_*.csv")):
        if hashlib.sha256(f.read_bytes()).hexdigest() != sums.get(f.name):
            sys.exit(f"{f.name}: SHA-256 does not match FROZEN.sha256 -- file changed after freezing")
        d = pd.read_csv(f)
        d["kind"] = "prospective" if "baseline" not in f.name else "prospective (baseline)"
        parts.append(d)
    if not parts:
        return pd.DataFrame()
    F = pd.concat(parts, ignore_index=True)
    F = F.merge(actual, on=["landkreis", "crop", "year"], how="left")
    return F[["kind", "issued", "year", "crop", "landkreis", "y_pred", "y_hist",
              "lo80", "hi80", "lead_days", "y_true"]]


def score(g: pd.DataFrame) -> dict:
    g = g.dropna(subset=["y_true"])
    if g.empty:
        return {"status": "awaiting Destatis"}
    yt, yp, yh = g["y_true"].to_numpy(), g["y_pred"].to_numpy(), g["y_hist"].to_numpy()
    sse, sst, ss_hist = ((yt - yp) ** 2).sum(), ((yt - yt.mean()) ** 2).sum(), ((yt - yh) ** 2).sum()
    return {"status": "scored", "districts": len(g),
            "r2": 1 - sse / sst if sst > 0 else np.nan,
            "rmse": np.sqrt(np.mean((yt - yp) ** 2)), "mae": np.mean(np.abs(yt - yp)),
            "skill": 1 - sse / ss_hist if ss_hist > 0 else np.nan,
            "direction_ok": bool(np.sign(yt.mean() - yh.mean()) == np.sign(yp.mean() - yh.mean())),
            "coverage80": ((yt >= g["lo80"]) & (yt <= g["hi80"])).mean(),
            "accuracy_pct": 100 - 100 * np.mean(np.abs(yt - yp) / yt),
            "actual_mean": yt.mean(), "forecast_mean": yp.mean(), "baseline_mean": yh.mean()}


def main() -> None:
    actual = (pd.read_parquet(ROOT / "data/interim/yield.parquet")
                .rename(columns={"yield_dt_per_ha": "y_true"})[["landkreis", "crop", "year", "y_true"]])
    L = pd.concat([backtests(), prospective(actual)], ignore_index=True)
    OUT.mkdir(parents=True, exist_ok=True)
    L.to_parquet(OUT / "ledger.parquet", index=False)
    S = pd.DataFrame([{"year": y, "crop": c, "kind": k, "issued": i, "lead_days": g["lead_days"].iloc[0],
                       "n_forecast": len(g), **score(g)}
                      for (y, c, k, i), g in L.groupby(["year", "crop", "kind", "issued"])])
    S.to_csv(OUT / "scoreboard.csv", index=False)
    Y = (S[S["status"] == "scored"].groupby(["year", "kind"])
           .agg(crops=("crop", "size"), beat_baseline=("skill", lambda s: int((s > 0).sum())),
                median_skill=("skill", "median"), mean_r2=("r2", "mean"), mean_mae=("mae", "mean"),
                direction_ok=("direction_ok", "sum"), coverage80=("coverage80", "mean"),
                accuracy_pct=("accuracy_pct", "mean")))
    Y.to_csv(OUT / "scoreboard_by_year.csv")
    pd.set_option("display.width", 200)
    print(Y.round(3).to_string())
    wait = S[S["status"] != "scored"].groupby(["year", "kind"])["crop"].size()
    if len(wait):
        print("\nawaiting actual yields:\n" + wait.to_string())


if __name__ == "__main__":
    main()
