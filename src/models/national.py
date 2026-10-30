"""National component (paper/preregistration_2026-09-28_v2.md §2.3).

Per crop, a ridge regression predicts the national anomaly of a harvest year
(the mean over districts of the z-scored yield anomaly) from national season
features: for every continuous daily input, the mean over districts and days
of each calendar month of the crop window (month index relative to the harvest
year, so the previous autumn is -4..-1). With phenology, the national median
day of each observed stage (days from 1 Jan of the harvest year) is added.

The penalty is chosen by leave-one-year-out inside the training years
(sklearn RidgeCV with its efficient LOO). Everything is fitted on the
training years only and stored as plain JSON with the feature pipeline.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ALPHAS = np.logspace(-2, 4, 25)
FLAG_SUFFIXES = ("_missing", "_observed", "_stale")


def _continuous(cols: list[str]) -> list[str]:
    return [c for c in cols if not c.endswith(FLAG_SUFFIXES) and not c.startswith("ph_")
            and not c.endswith("_age_days")]


def season_features(frame: pd.DataFrame, past_cols: list[str], phenology: bool) -> pd.DataFrame:
    """One row per (crop, year): national monthly means (+ stage days)."""
    f = frame[["crop", "year", "date", "landkreis"] + past_cols].copy()
    d = pd.to_datetime(f["date"])
    f["m"] = (d.dt.year - f["year"]) * 12 + d.dt.month
    cont = _continuous(past_cols)
    g = f.groupby(["crop", "year", "m"])[cont].mean()
    wide = g.unstack("m")
    wide.columns = [f"{v}@{m}" for v, m in wide.columns]
    if phenology:
        stage = {}
        for s in [c for c in past_cols if c.startswith("ph_") and not c.endswith("_missing")]:
            hit = f[f[s] == 1].groupby(["crop", "year", "landkreis"])["date"].min()
            day = (pd.to_datetime(hit) - pd.to_datetime(hit.index.get_level_values("year").astype(str) + "-01-01")).dt.days
            stage[f"{s}_day"] = day.groupby(level=["crop", "year"]).median()
        wide = wide.join(pd.DataFrame(stage), how="left")
    return wide


class NationalModel:
    def __init__(self) -> None:
        self.models: dict[str, dict] = {}

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "NationalModel":
        from sklearn.linear_model import RidgeCV
        for crop in sorted(set(X.index.get_level_values("crop"))):
            Xc = X.xs(crop, level="crop").dropna(axis=1, how="all")
            yc = y.xs(crop, level="crop").reindex(Xc.index)
            mu, sd = Xc.mean(), Xc.std(ddof=0).replace(0, 1.0)
            Z = ((Xc - mu) / sd).fillna(0.0)
            r = RidgeCV(alphas=ALPHAS).fit(Z.to_numpy(), yc.to_numpy())
            self.models[crop] = {"cols": list(Xc.columns), "mu": mu.tolist(), "sd": sd.tolist(),
                                 "coef": r.coef_.tolist(), "intercept": float(r.intercept_),
                                 "alpha": float(r.alpha_), "n_years": int(len(yc))}
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        out = {}
        for (crop, year), row in X.iterrows():
            m = self.models.get(crop)
            if m is None:
                out[(crop, year)] = 0.0
                continue
            x = row.reindex(m["cols"]).to_numpy(dtype=float)
            z = np.nan_to_num((x - np.asarray(m["mu"])) / np.asarray(m["sd"]))
            out[(crop, year)] = float(z @ np.asarray(m["coef"]) + m["intercept"])
        s = pd.Series(out)
        s.index.names = ["crop", "year"]
        return s

    def to_dict(self) -> dict:
        return {"models": self.models}

    @classmethod
    def from_dict(cls, d: dict) -> "NationalModel":
        m = cls()
        m.models = d["models"]
        return m
