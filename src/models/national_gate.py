"""National gate (paper/preregistration_2026-09-30_v5.md §2.1).

For a crop whose national yield level is not predictable from seasonal weather
in the TRAINING years, keep the model's district pattern but take the
historical-average national level. Predictability = leave-one-year-out R² of a
ridge regression of the detrended national anomaly on nine national seasonal
indicators. Everything here sees training years only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ALPHA = 5.0
THRESHOLD = 0.20
INDICATORS = ["rain_ond_prev", "rain_mam", "rain_jj", "rain_harvest", "cwb_mam", "cwb_jj",
              "tmax_jj", "heat_jj", "soilm_jj"]


def national_indicators(daily: pd.DataFrame) -> pd.DataFrame:
    """Nine seasonal indicators per calendar year from daily district weather.

    daily: landkreis, date, precipitation_height, cwb, temperature_air_max_2m, swroot7
    (one row per district-day). National value = mean over districts of the
    daily values, summed (rain, cwb, heat days) or averaged (tmax, soil) over
    the period. The Oct-Dec value belongs to the NEXT year.
    """
    d = daily.drop_duplicates(["landkreis", "date"]).copy()
    d["date"] = pd.to_datetime(d["date"])
    d["heat"] = (d["temperature_air_max_2m"] >= 30).astype(float)
    d.loc[d["temperature_air_max_2m"].isna(), "heat"] = np.nan
    cols = {"precipitation_height": "p", "cwb": "c", "temperature_air_max_2m": "t",
            "heat": "h", "swroot7": "s"}
    nat = d.groupby("date")[list(cols)].mean().rename(columns=cols)
    nat["y"], nat["m"], nat["dd"] = nat.index.year, nat.index.month, nat.index.day

    def agg(mask, col, how):
        g = nat[mask].groupby("y")[col]
        return g.sum(min_count=1) if how == "sum" else g.mean()

    m = nat["m"]
    out = pd.DataFrame({
        "rain_mam": agg(m.isin([3, 4, 5]), "p", "sum"),
        "rain_jj": agg(m.isin([6, 7]), "p", "sum"),
        "rain_harvest": agg(((m == 7) & (nat["dd"] >= 15)) | ((m == 8) & (nat["dd"] <= 15)), "p", "sum"),
        "cwb_mam": agg(m.isin([3, 4, 5]), "c", "sum"),
        "cwb_jj": agg(m.isin([6, 7]), "c", "sum"),
        "tmax_jj": agg(m.isin([6, 7]), "t", "mean"),
        "heat_jj": agg(m.isin([6, 7]), "h", "sum"),
        "soilm_jj": agg(m.isin([6, 7]), "s", "mean"),
    })
    ond = nat[m.isin([10, 11, 12])]
    # only a complete Oct-Dec counts (the weather record may start mid-autumn)
    full = ond.groupby("y")["p"].count() >= 90
    prev = ond.groupby("y")["p"].sum()[full]
    prev.index = prev.index + 1
    out["rain_ond_prev"] = prev
    out.index.name = "year"
    return out[INDICATORS]


def national_anomaly(yields: pd.DataFrame) -> pd.Series:
    """Detrended national anomaly (%) per year for ONE crop's training yields.

    yields: landkreis, year, yield (training years only).
    """
    y = yields.dropna(subset=["yield"])
    rel = y["yield"] / y.groupby("landkreis")["yield"].transform("mean") - 1.0
    a = rel.groupby(y["year"]).mean() * 100.0
    t = a.index.to_numpy(float)
    if len(a) >= 3:
        a = a - np.polyval(np.polyfit(t, a.to_numpy(float), 1), t)
    return a


def _ridge_fit_predict(Xtr, ytr, Xte, alpha):
    mu = np.nanmean(Xtr, axis=0)
    Xtr = np.where(np.isnan(Xtr), mu, Xtr); Xte = np.where(np.isnan(Xte), mu, Xte)
    sd = Xtr.std(axis=0); sd[sd == 0] = 1.0
    Ztr, Zte = (Xtr - mu) / sd, (Xte - mu) / sd
    ym = ytr.mean()
    b = np.linalg.solve(Ztr.T @ Ztr + alpha * np.eye(Ztr.shape[1]), Ztr.T @ (ytr - ym))
    return Zte @ b + ym


def loyo_r2(anom: pd.Series, ind: pd.DataFrame, alpha: float = ALPHA) -> float:
    """Leave-one-year-out R² of ridge(anomaly ~ indicators) over the given years."""
    yrs = [y for y in anom.index if y in ind.index]
    if len(yrs) < 6:
        return float("nan")
    y = anom.loc[yrs].to_numpy(float)
    X = ind.loc[yrs, INDICATORS].to_numpy(float)
    pred = np.array([_ridge_fit_predict(np.delete(X, i, 0), np.delete(y, i), X[i:i + 1], alpha)[0]
                     for i in range(len(y))])
    return float(1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def gate_decisions(yields: pd.DataFrame, ind: pd.DataFrame, train_end: int,
                   threshold: float = THRESHOLD, alpha: float = ALPHA) -> pd.DataFrame:
    """Per crop: LOYO R² on training years <= train_end, and whether the gate is on.

    yields: landkreis, crop, year, yield. A crop with too few years (nan R²) is gated.
    """
    rows = []
    for crop, g in yields[yields["year"] <= train_end].groupby("crop"):
        r2 = loyo_r2(national_anomaly(g[["landkreis", "year", "yield"]]), ind, alpha)
        rows.append({"crop": crop, "loyo_r2": r2, "gate": bool(not (r2 >= threshold))})
    return pd.DataFrame(rows)


def apply_gate(pred: pd.DataFrame, gated_crops: set[str]) -> pd.Series:
    """Remove the predicted national anomaly for gated crops.

    pred: crop, year, y_pred, y_hist (one row per district). Returns new y_pred.
    """
    nat = (pred["y_pred"] - pred["y_hist"]).groupby([pred["crop"], pred["year"]]).transform("mean")
    return pred["y_pred"] - np.where(pred["crop"].isin(gated_crops), nat, 0.0)
