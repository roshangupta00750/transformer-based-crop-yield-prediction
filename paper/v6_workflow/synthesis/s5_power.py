"""Power of the prospective decision rules if future years behave like 2016-2025.
Draw k future years from the development years (with replacement); apply the registered rule:
pooled per-crop skill over those k years, macro over crops; year-block bootstrap P(d>0) >= 0.95
(exact enumeration of the k^k resamples) and pooled MAPE lower. Read-only on project."""
import itertools, sys, numpy as np, pandas as pd
from pathlib import Path
OUT = Path(__file__).resolve().parent / "out"       # results of this analysis
OUT.mkdir(exist_ok=True)
K = ["landkreis", "crop", "year"]
A = pd.read_parquet(OUT / "all_methods_ens.parquet")
Bs = pd.read_parquet(OUT / "baselines_rows.parquet")
A = A.merge(Bs[Bs.year >= 2016][K + ["B5"]], on=K, how="left")
crops = sorted(A.crop.unique()); yrs = sorted(A.year.unique())
def sse(p):
    return ((A[p] - A.y_true) ** 2).groupby([A.crop, A.year]).sum().unstack().loc[crops, yrs].to_numpy()
def ape(p):
    return (np.abs(A[p] - A.y_true) / A.y_true).groupby(A.year).sum().loc[yrs].to_numpy()
n = A.groupby("year").size().loc[yrs].to_numpy()

def power(sNew, sOld, sRef, aNew, aOld, k, R=3000, seed=0):
    rng = np.random.default_rng(seed)
    combos = np.array(list(itertools.product(range(k), repeat=k)))       # (k^k, k)
    passes = pos = 0
    for _ in range(R):
        fut = rng.integers(0, len(yrs), k)
        cols = fut[combos]                                              # (k^k, k)
        def macro(sM):
            num = sM[:, cols].sum(-1); den = sRef[:, cols].sum(-1)     # (crops, k^k)
            return (1 - num / den).mean(0)
        d = macro(sNew) - macro(sOld)                                   # (k^k,)
        d0 = d[np.all(combos == np.arange(k), axis=1)][0]
        pos += d0 > 0
        passes += (np.mean(d > 0) >= 0.95) and (aNew[fut].sum() < aOld[fut].sum())
    return passes / R, pos / R

sF, sFR, sB, sH, sB5 = sse("F"), sse("F+R"), sse("b"), sse("b"), sse("B5")
aF, aFR, aB5 = ape("F"), ape("F+R"), ape("B5")
for k in (1, 2, 3, 4):
    p, s = power(sFR, sF, sH, aFR, aF, k)
    print(f"k={k}: H16 F+R vs F (ref y_hist)  P(rule passes)={p:.2f}  P(point estimate >0)={s:.2f}", flush=True)
for k in (1, 2, 3, 4):
    # F vs B5: 'old' forecast is B5 itself (skill of B5 vs B5 = 0)
    p, s = power(sF, sB5, sB5, aF, aB5, k, seed=1)
    print(f"k={k}: H15 F vs B5               P(rule passes)={p:.2f}  P(point estimate >0)={s:.2f}", flush=True)
