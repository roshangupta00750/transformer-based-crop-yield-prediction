"""Computing cost of the final model (YieldFormer-DE, v18) for the dashboard (Compare > Computing cost).

Reads the SLURM job logs of the training runs and the saved checkpoints, and writes
  results/dashboard/compute_jobs.csv    one row per job: network, purpose, stage, season, seed, node, GPU, seconds
  results/dashboard/compute_models.csv  one row per network: parameters, checkpoint size
Copy the job logs (the *.out and *.err files that scripts/sbatch_*.sh write to slurm_logs/) into one local folder,
then run, for example:
    python scripts/79_compute_cost.py --logs slurm_logs --checkpoints results/checkpoints
Which jobs belong to the final model (see the pre-registrations v12, v15, v16, v17, v18):
  iTransformer member: v12 arm Q12 networks (v15 = those predictions + error memory, no retraining);
                       2026 forecast networks from the v16 forecast jobs
  TFT and PatchTST:    v17 arms T and P, rolling 2010-2025 and the 2026 forecast
  XGBoost reference:   v16 arm G16 (one job per season)
The error memory (v15) and the equal-weight average (v18) run on a laptop CPU in seconds.
"""
from __future__ import annotations

import argparse
import datetime as dt
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "dashboard"
FAMILIES = ("v12-cyp-v12", "v17-cyp-v17", "v16-cyp-v16-fc", "v16-cyp-v16-gbm")
NETWORK = {"itransformer_v12Q": "iTransformer", "tftnet_v17T": "TFT", "patchtst17_v17P": "PatchTST"}
CHECKPOINTS = (("iTransformer", "itransformer_v12Q_roll2025_s42.pt"), ("TFT", "tftnet_v17T_roll2025_s42.pt"),
               ("PatchTST", "patchtst17_v17P_roll2025_s42.pt"))


def read_jobs(logs: Path) -> pd.DataFrame:
    """One row per job log: task line, start, wall-clock and training seconds, node, GPU."""
    rows = []
    for fam in FAMILIES:
        for f in sorted(logs.glob(f"{fam}-*.out")):
            t = f.read_text(errors="ignore").splitlines()
            head = next((line for line in t if line.startswith("=== ") and "DONE" not in line), "")
            m = re.match(r"=== (\S+ \S+) (.*) ===", head)
            md = re.match(r"=== (\S+ \S+) DONE", next((line for line in t if "DONE" in line), ""))
            gpu = next((line.split(":")[1].split("(")[0].strip() for line in t if line.startswith("GPU 0")), "")
            err = f.with_suffix(".err")
            tr = re.search(r"Done in ([0-9.]+)s", err.read_text(errors="ignore")) if err.exists() else None
            sec = ((dt.datetime.fromisoformat(md.group(1)) - dt.datetime.fromisoformat(m.group(1))).total_seconds()
                   if m and md else None)
            rows.append({"file": f.name, "task": m.group(2) if m else "", "start": m.group(1) if m else "",
                         "wall_s": sec, "train_s": float(tr.group(1)) if tr else None,
                         "host": t[1].strip() if len(t) > 1 else "", "gpu": gpu})
    return pd.DataFrame(rows)


def read_sizes(checkpoints: Path) -> pd.DataFrame:
    import torch
    rows = []
    for net, name in CHECKPOINTS:
        f = checkpoints / name
        if f.exists():
            sd = torch.load(f, map_location="cpu", weights_only=False)["state_dict"]
            rows.append({"network": net, "parameters": sum(v.numel() for v in sd.values()),
                         "checkpoint_bytes": f.stat().st_size})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--logs", type=Path, required=True, help="folder with the copied SLURM job logs")
    ap.add_argument("--checkpoints", type=Path, default=ROOT / "results" / "checkpoints",
                    help="folder with the saved network checkpoints")
    args = ap.parse_args()

    d = read_jobs(args.logs)
    if d.empty:
        raise SystemExit(f"no job logs of {', '.join(FAMILIES)} found in {args.logs}")
    d["job"] = d.file.str.extract(r"-(\d+)_\d+\.out", expand=False)
    parts = d.task.fillna("").str.split(" ", expand=True)
    gbm = parts[0].eq("G16")
    d["stage"] = parts[0].where(~gbm, "train")
    d["config"] = parts[1].where(~gbm, "G16")
    d["season"] = pd.to_numeric(parts[2].where(~gbm, parts[1]), errors="coerce")
    d["seed"] = pd.to_numeric(parts[3].str.lstrip("s"), errors="coerce").where(~gbm)
    d["network"] = d.config.map(NETWORK).fillna("XGBoost + memory (reference)")
    size = d.groupby("job").file.transform("size")
    d["purpose"] = "rolling 2010-2025"
    d.loc[d.season.eq(2026), "purpose"] = "2026 forecast"
    d.loc[size.eq(1) & ~gbm & d.season.ne(2026), "purpose"] = "smoke test"
    d = d[d.wall_s.notna() & d.config.isin(list(NETWORK) + ["G16"])]
    cols = ["network", "purpose", "stage", "season", "seed", "job", "host", "gpu", "wall_s", "train_s", "start"]
    OUT.mkdir(parents=True, exist_ok=True)
    d[cols].sort_values(["network", "purpose", "stage", "season", "seed"]).to_csv(OUT / "compute_jobs.csv",
                                                                                 index=False)
    sizes = read_sizes(args.checkpoints)
    if sizes.empty:
        print(f"no checkpoints found in {args.checkpoints}: compute_models.csv left unchanged")
    else:
        sizes.to_csv(OUT / "compute_models.csv", index=False)
    print(d.groupby(["network", "purpose", "stage"]).agg(jobs=("job", "size"), hours=("wall_s", lambda s: s.sum() / 3600),
                                                         gpus=("gpu", lambda s: dict(s.value_counts()))))


if __name__ == "__main__":
    main()
