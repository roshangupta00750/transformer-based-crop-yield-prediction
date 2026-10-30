"""Pre-registration v18 (papers' protocols): the ensemble's predictions on the other papers' splits.

X18 per protocol = mean of the three 3-seed means (v12QF iTransformer, v17TF TFT, v17PF PatchTST), weights 1/3,
no error memory. Written as compare_predictions_v18XF.parquet with one "seed" row per network, so
scripts/40 (which averages the non-"ens" seed rows) scores exactly that mean on the papers' rows.

Usage:  python scripts/76_paper_ensemble.py
        CMP_ARM=v18XF CMP_YIELD=data/interim/v5/yield.parquet python scripts/40_compare_table.py <AgroForecast-DE data>
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
M = ROOT / "results/metrics"
K = ["protocol", "role", "year", "crop", "landkreis"]


def main() -> None:
    parts = []
    for arm, tag in (("v12QF", "Q"), ("v17TF", "T"), ("v17PF", "P")):
        d = pd.read_parquet(M / f"compare_predictions_{arm}.parquet")
        d = d[d.seed == "ens"][K + ["y_true", "y_hist", "y_pred"]]
        parts.append(d.assign(seed=tag))
    keys = set(map(tuple, parts[0][K].to_numpy()))
    for p in parts[1:]:
        if set(map(tuple, p[K].to_numpy())) != keys:
            raise SystemExit("the three networks do not cover the same protocol rows")
    X = pd.concat(parts, ignore_index=True)
    X.to_parquet(M / "compare_predictions_v18XF.parquet", index=False)
    print(f"wrote {len(X):,} rows ({len(keys):,} protocol rows x 3 networks)")


if __name__ == "__main__":
    main()
