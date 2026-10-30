"""Score the replication-protocol predictions against each paper (equal protocols).

Input: results/metrics/compare_predictions.parquet (scripts/39, seed="ens").
Competitor values: AgroForecast-DE published per-district predictions (matched
exactly as scripts/38); Srivastava 2022 Table 1 (Proposed CNN, t/ha x 10);
CropFusionNet abstract R² (years not stated; their code pools 2019+).

Output: results/metrics/compare_equal_protocol.csv and a printed table.
Usage: python scripts/40_compare_table.py <AgroForecast-DE docs/data dir>
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SRV = {2017: {"rmse": 6.6, "mae": 5.2}, 2018: {"rmse": 8.4, "mae": 7.1}, 2019: {"rmse": 6.4, "mae": 5.0}}
CFN = {"winter_wheat": 0.597, "silage_maize": 0.752, "winter_barley": 0.445}


def met(yt, yp) -> dict:
    yt, yp = np.asarray(yt, float), np.asarray(yp, float)
    sse = ((yt - yp) ** 2).sum()
    return {"r2": 1 - sse / ((yt - yt.mean()) ** 2).sum(), "rmse": np.sqrt(np.mean((yt - yp) ** 2)),
            "mae": np.mean(np.abs(yt - yp)), "mape": 100 * np.mean(np.abs(yt - yp) / yt)}


def align(a, U):
    """Order-preserving alignment of their rows `a` to district slots `U`.

    U = [(code, value | None)] in NUTS-code order: every district they still
    report in that year; value None = a district we lack, which may absorb one
    of their rows (small score) but never a wrong value (-inf). This stops a
    tie between neighbouring equal yields from shifting a row onto the wrong
    district (e.g. 2023 DE228/DE22A both 7.90 t/ha).
    """
    n, m = len(a), len(U)
    score = lambda v, x: 1.0 if v == x else (0.001 if v is None else -1e9)
    D = np.zeros((n + 1, m + 1))
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i, j] = max(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1] + score(U[j - 1][1], a[i - 1]))
    i, j, out = n, m, {}
    while i > 0 and j > 0:
        s_ = score(U[j - 1][1], a[i - 1])
        if s_ > 0 and np.isclose(D[i, j], D[i - 1, j - 1] + s_):
            if U[j - 1][1] is not None:
                out[U[j - 1][0]] = i - 1
            i -= 1; j -= 1
        elif D[i - 1, j] >= D[i, j - 1]:
            i -= 1
        else:
            j -= 1
    return out


def afde_matched(A: Path, crop: str) -> pd.DataFrame:
    """AgroForecast-DE test predictions with district codes.

    Their per-year arrays are in NUTS-code order over the districts they still
    report (observed_<crop>.json: last reported year >= test year). Slot values
    come from their own file where it holds that year's yield, else from
    Destatis (ours), else a placeholder. Only rows whose yield equals ours
    exactly are kept.
    """
    # CMP_YIELD: the yield table the arm was trained on (v10: corrected AGS matching)
    Y = pd.read_parquet(ROOT / os.environ.get("CMP_YIELD", "data/interim/yield.parquet"))
    last = json.load(open(A / f"observed_{crop}.json"))["districts"]
    rows = []
    for e in json.load(open(A / f"performance_{crop}.json"))["years"]:
        if e["split"] != "test":
            continue
        y = e["year"]
        a = [int(round(v * 100)) for v in e["observed"]]
        ours = dict(zip(*Y[(Y.crop == crop) & (Y.year == y)][["landkreis", "yield_dt_per_ha"]].values.T))
        U = []
        for k in sorted(k for k, v in last.items() if v["year"] >= y):
            if last[k]["year"] == y:
                U.append((k, int(round(last[k]["value"] * 100))))
            else:
                U.append((k, int(round(ours[k] * 10)) if k in ours else None))
        for k, i in align(a, U).items():
            if k in ours and abs(ours[k] - e["observed"][i] * 10) < 0.051:
                rows.append(dict(landkreis=k, year=y, crop=crop, obs=e["observed"][i] * 10,
                                 pred_them=e["predicted"][i] * 10))
    return pd.DataFrame(rows)


SUFFIX = "" if os.environ.get("CMP_ARM", "preD") == "preD" else f"_{os.environ['CMP_ARM']}"


def main() -> None:
    A = Path(sys.argv[1])
    P = pd.read_parquet(ROOT / f"results/metrics/compare_predictions{SUFFIX}.parquet")
    # ensemble from the seed rows; y_hist averaged too (the random-validation
    # protocol fits each seed's baseline on a different 85% of crop-years)
    P = (P[P.seed != "ens"].groupby(["protocol", "role", "year", "crop", "landkreis"], as_index=False)
           [["y_true", "y_pred", "y_hist"]].mean())
    R = pd.read_parquet(ROOT / "results/metrics/prereg_predictions.parquet")
    R = R[(R.arm == "D") & (R.seed == "ens") & (R.horizon == 0)]   # rolling protocol, for reference
    rows = []

    def add(paper, crop, years, n, theirs, ours, hist, rolling):
        for k in theirs:
            rows.append(dict(paper=paper, crop=crop, years=years, n=n, metric=k,
                             theirs=theirs[k], ours_equal_protocol=ours[k], hist_avg=hist[k],
                             ours_rolling=rolling.get(k, np.nan) if rolling else np.nan))

    for crop in ("winter_wheat", "silage_maize"):
        M = afde_matched(A, crop)
        for src, name in ((P[P.protocol == "AFDE"], "eq"), (R, "roll")):
            M = M.merge(src[["landkreis", "crop", "year", "y_pred", "y_true", "y_hist"]]
                        .rename(columns=lambda c: f"{c}_{name}" if c.startswith("y_") else c),
                        on=["landkreis", "crop", "year"], how="inner")
        assert np.allclose(M.obs, M.y_true_eq, atol=0.051)
        pick = lambda d: {k: d[k] for k in ("r2", "rmse", "mape")}
        add("AgroForecast-DE (same districts)", crop, "2023-24", len(M),
            pick(met(M.obs, M.pred_them)), pick(met(M.obs, M.y_pred_eq)),
            pick(met(M.obs, M.y_hist_eq)), pick(met(M.obs, M.y_pred_roll)))

    for crop, r2 in CFN.items():
        for years, sel in (("2022-24 (their test years)", [2022, 2023, 2024]),
                           ("2019-24 pooled incl. our validation years 2019-21, as their code", list(range(2019, 2025)))):
            g = P[(P.protocol == "CFN") & (P.crop == crop) & P.year.isin(sel)]
            gr = R[(R.crop == crop) & R.year.isin(sel)]
            add("CropFusionNet paper", crop, years, len(g), {"r2": r2},
                {"r2": met(g.y_true, g.y_pred)["r2"]}, {"r2": met(g.y_true, g.y_hist)["r2"]},
                {"r2": met(gr.y_true, gr.y_pred)["r2"]})

    for y, v in SRV.items():
        g = P[(P.protocol == f"SRV{y}") & (P.crop == "winter_wheat")]
        gr = R[(R.crop == "winter_wheat") & (R.year == y)]
        o, h, r = met(g.y_true, g.y_pred), met(g.y_true, g.y_hist), met(gr.y_true, gr.y_pred)
        add("Srivastava 2022 CNN", "winter_wheat", str(y), len(g), v,
            {k: o[k] for k in v}, {k: h[k] for k in v}, {k: r[k] for k in v})

    T = pd.DataFrame(rows)
    higher = T.metric == "r2"
    d = T.ours_equal_protocol - T.theirs
    tol = np.where(T.paper.str.startswith("Srivastava"), 0.05, np.where(higher, 0.0005, 0.0))
    T["verdict"] = np.where(np.abs(d) <= tol + 1e-9, "tie", np.where(higher == (d > 0), "ours better", "ours worse"))
    T["beats_hist_avg"] = np.where(higher, T.ours_equal_protocol > T.hist_avg, T.ours_equal_protocol < T.hist_avg)
    T.to_csv(ROOT / f"results/metrics/compare_equal_protocol{SUFFIX}.csv", index=False)
    pd.set_option("display.width", 220)
    print(T.round(3).to_string(index=False))
    print("\n" + T.verdict.value_counts().to_string())


if __name__ == "__main__":
    main()
