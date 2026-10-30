"""All rolling-test versions on the same rows (district-crop-years present in every version with the
same reported yield), plus the full corrected panel for v10-v18. Descriptive; decisions stay in the
pre-registrations. Pmain>_* = share of 2,000 year-block bootstrap resamples where the main model (v18,
the transformer ensemble; MAIN below) has the higher macro MSE skill vs y_hist.

Outputs: results/metrics/version_comparison_common.csv, version_comparison_full.csv
Usage:   python scripts/67_version_comparison.py      (~10 min, CPU)
"""
import importlib.util, numpy as np, pandas as pd, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT)); import os; os.chdir(ROOT)
s=importlib.util.spec_from_file_location("e7","scripts/51_eval_v7.py"); e7=importlib.util.module_from_spec(s); s.loader.exec_module(e7)
M="results/metrics/"; K=["year","crop","landkreis"]
base=pd.read_parquet(M+"v18_predictions.parquet")[K+["y_true","y_hist","b5","Q12","E15","TE17","PE17","XE18"]]
add=[("v7_predictions.parquet",{"F":"v4","R7":"v7"}),("v9_predictions.parquet",{"R7":"v9"}),("v10_predictions.parquet",{"D10":"v10"}),
     ("v13_predictions.parquet",{"C13":"v13"}),("v14_predictions.parquet",{"H14":"v14"}),("v15W_predictions.parquet",{"WE15":"v15b"})]
full=base.copy()
for f,ren in add:
    d=pd.read_parquet(M+f)[K+["y_true"]+list(ren)].rename(columns={**ren,"y_true":"yt"})
    full=full.merge(d,on=K,how="left"); full.loc[(full.yt-full.y_true).abs()>=0.05, list(ren.values())]=np.nan; full=full.drop(columns="yt")
full=full.rename(columns={"Q12":"v12","E15":"v15","TE17":"v17 TFT","PE17":"v17 PatchTST","XE18":"v18"})
MODELS=["v4","v7","v9","v10","v12","v13","v14","v15b","v15","v17 TFT","v17 PatchTST","v18"]
MAIN="v18"
acc=lambda p,t: 100-100*np.mean(np.abs(p-t)/t)
def table(D,label):
    rows=[]
    for m in MODELS+["b5","y_hist"]:
        r={"set":label,"model":m}
        for w,(a,b) in {"2010-15":(2010,2015),"2016-25":(2016,2025),"2010-25":(2010,2025)}.items():
            g=D[D.year.between(a,b)]; r[w]=acc(g[m],g.y_true)
        N=D.groupby(["crop","year"])[[m,"y_true"]].mean(); r["national"]=acc(N[m],N.y_true)
        r["within10"]=100*np.mean(np.abs(D[m]-D.y_true)/D.y_true<=0.10)
        if m not in (MAIN,"b5","y_hist"):
            for w,(a,b) in {"Pmain>_10-15":(2010,2015),"Pmain>_16-25":(2016,2025)}.items():
                g=D[D.year.between(a,b)]; yrs=np.array(sorted(g.year.unique())); by={y:g[g.year==y] for y in yrs}; rng=np.random.default_rng(0); dd=[]
                for _ in range(2000):
                    S=pd.concat([by[y] for y in rng.choice(yrs,len(yrs))]); dd.append(e7.macro_skill(S,MAIN,"y_hist")-e7.macro_skill(S,m,"y_hist"))
                r[w]=100*np.mean(np.array(dd)>0)
        rows.append(r)
    return pd.DataFrame(rows)
common=full.dropna(subset=MODELS)
out=pd.concat([table(common,f"common {len(common)} rows, {common.landkreis.nunique()} districts")])
out.to_csv(M+"version_comparison_common.csv",index=False)
print(out.round(2).to_string(index=False))
full_rows=[]
for m in ["v10","v12","v13","v14","v15b","v15","v17 TFT","v17 PatchTST","v18","b5","y_hist"]:
    g=full.dropna(subset=[m]); r={"model":m,"rows":len(g),"districts":g.landkreis.nunique()}
    for w,(a,b) in {"2010-15":(2010,2015),"2016-25":(2016,2025),"2010-25":(2010,2025)}.items():
        h=g[g.year.between(a,b)]; r[w]=acc(h[m],h.y_true)
    N=g.groupby(["crop","year"])[[m,"y_true"]].mean(); r["national"]=acc(N[m],N.y_true)
    r["within10"]=100*np.mean(np.abs(g[m]-g.y_true)/g.y_true<=0.10); full_rows.append(r)
pd.DataFrame(full_rows).to_csv(M+"version_comparison_full.csv",index=False)
print(pd.DataFrame(full_rows).round(2).to_string(index=False))
