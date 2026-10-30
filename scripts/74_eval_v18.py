"""Pre-registration v18 (paper/preregistration_2026-10-11_v18.md): equal-weight transformer ensemble.

X18 = (Q12 + T17 + P17) / 3 on every row (weights fixed, nothing fitted); XE18 = X18 + v15 §2 error memory.
Decision: XE18 vs E15 under v6 §6 on 2010-2015; tie-break with the v17 arms by macro skill.
Reported: 2016-2025, 2010-2025, per crop, XE18 vs TE17 / PE17 / GE16 (reference).

Input:   results/metrics/v17_predictions.parquet (scripts/73 --stage score, needs both v17 arms)
Outputs: results/metrics/v18_predictions.parquet, v18_scores.json, v18_comparison.csv, v18_per_crop.csv
Usage:   python scripts/74_eval_v18.py
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
M = ROOT / "results/metrics"
WINDOWS = {"hindcast_2010_2015": range(2010, 2016), "dev_2016_2025": range(2016, 2026),
           "all_2010_2025": range(2010, 2026)}
N_DECISIONS = 12        # v7-v15 (9) + TE17, PE17 + XE18


def _load(name, file):
    s = importlib.util.spec_from_file_location(name, ROOT / "scripts" / file)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
    return m


def main() -> None:
    v16 = _load("v16", "69_eval_v16.py")
    P = pd.read_parquet(M / "v17_predictions.parquet")
    need = ["Q12", "T17", "P17", "E15", "TE17", "PE17", "GE16"]
    if missing := [c for c in need if c not in P]:
        sys.exit(f"v17_predictions.parquet lacks {missing}: score both v17 arms first")
    P["X18"] = P[["Q12", "T17", "P17"]].mean(axis=1)
    P, lam = v16.memory(P, "X18", "XE18")
    res, rows = {"lambda": lam}, []
    for inc in ("E15", "TE17", "PE17", "GE16"):
        S = P.rename(columns={"XE18": "R7"}).assign(F=P[inc])
        r = {w: v16.fast_score(S[S.year.isin(list(y))]) for w, y in WINDOWS.items()}
        h = r["hindcast_2010_2015"]
        r["passes_v6"] = bool(h["p_delta_gt_0"] >= 0.95 and h["mape_R7"] < h["mape_F"] and not h["crops_worse_every_year"])
        res[f"XE18_vs_{inc}"] = r
        for w in WINDOWS:
            s = r[w]
            rows.append({"window": w, "challenger": "XE18", "incumbent": inc, "rows": s["n_rows"],
                         "acc_challenger": 100 - s["mape_R7"], "acc_incumbent": 100 - s["mape_F"],
                         "delta_skill": s["delta_skill_R7_minus_F"], "ci90_lo": s["delta_ci90"][0],
                         "ci90_hi": s["delta_ci90"][1], "p_delta_gt_0": s["p_delta_gt_0"],
                         "crops_worse_every_year": ", ".join(s["crops_worse_every_year"]),
                         "passes_v6": r["passes_v6"] if w == "hindcast_2010_2015" else None})
    v17 = json.loads((M / "v17_scores.json").read_text())
    cand = {"XE18": res["XE18_vs_E15"]} | {m: v17[f"{m}_vs_E15"] for m in ("TE17", "PE17") if f"{m}_vs_E15" in v17}
    passing = {m: r["hindcast_2010_2015"]["skill_R7_vs_yhist"] for m, r in cand.items() if r["passes_v6"]}
    res["new_main_model"] = max(passing, key=passing.get) if passing else None
    res["bonferroni_level"] = 1 - 0.05 / N_DECISIONS
    P.to_parquet(M / "v18_predictions.parquet", index=False)
    (M / "v18_scores.json").write_text(json.dumps(res, indent=2, default=float))
    pd.DataFrame(rows).to_csv(M / "v18_comparison.csv", index=False)
    cols = ["E15", "TE17", "PE17", "XE18", "GE16"]
    pd.DataFrame([{"window": w, "crop": c, **{f"acc_{x}": 100 - 100 * np.mean(np.abs(g[x] - g.y_true) / g.y_true)
                                               for x in cols}}
                  for w, yrs in WINDOWS.items() for c, g in P[P.year.isin(list(yrs))].groupby("crop")]
                 ).to_csv(M / "v18_per_crop.csv", index=False)
    # reporting (v16 §3.2 / §3.1 applied to the new main model): year-block 90% ranges, doubtful records
    iv = v16.boot(P, ["XE18", "E15", "b5", "y_hist"], [("XE18", "E15"), ("XE18", "b5"), ("XE18", "y_hist")])
    iv += v16.boot(P, ["XE18", "E15", "b5", "y_hist"], [("XE18", "E15"), ("XE18", "b5"), ("XE18", "y_hist")],
                   level="national")
    pd.DataFrame(iv).to_csv(M / "v18_intervals.csv", index=False)
    fl = pd.read_csv(M / "v16_flags.csv")
    bad = set(map(tuple, fl[["landkreis", "crop", "year"]].to_numpy()))
    P["flagged"] = [(lk, c, y) in bad for lk, c, y in zip(P.landkreis, P.crop, P.year)]
    acc = lambda p, y: float(100 - 100 * np.mean(np.abs(p - y) / y))
    pd.DataFrame([{"window": w, "forecast": x, "rows": len(W), "flagged": int(W.flagged.sum()),
                   "accuracy_all": acc(W[x], W.y_true), "accuracy_clean": acc(W[x][~W.flagged], W.y_true[~W.flagged])}
                  for w, yrs in WINDOWS.items() for W in [P[P.year.isin(list(yrs))]] for x in ("XE18", "E15", "b5", "y_hist")]
                 ).to_csv(M / "v18_cleaned.csv", index=False)
    pd.set_option("display.width", 220)
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    print("new main model:", res["new_main_model"], "| Bonferroni level", round(res["bonferroni_level"], 4))


if __name__ == "__main__":
    main()
