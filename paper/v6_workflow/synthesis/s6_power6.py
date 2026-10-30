import numpy as np, pandas as pd
from pathlib import Path
OUT=Path('.')
K=["landkreis","crop","year"]
A=pd.read_parquet(OUT/'all_methods_ens.parquet')
Bs=pd.read_parquet(OUT/'baselines_rows.parquet')
A=A.merge(Bs[Bs.year>=2016][K+['B5']],on=K,how='left')
crops=sorted(A.crop.unique()); yrs=sorted(A.year.unique())
def sse(p): return ((A[p]-A.y_true)**2).groupby([A.crop,A.year]).sum().unstack().loc[crops,yrs].to_numpy()
sF,sH,sB5,sFR=sse('F'),sse('b'),sse('B5'),sse('F+R')
rng=np.random.default_rng(0)
def ci_power(sM,sRef,k,R=1500,B=2000):
    lo_pos=0; pt_pos=0
    for _ in range(R):
        fut=rng.integers(0,len(yrs),k)
        idx=rng.integers(0,k,(B,k)); cols=fut[idx]
        sk=(1-sM[:,cols].sum(-1)/sRef[:,cols].sum(-1)).mean(0)
        pt=(1-sM[:,fut].sum(-1)/sRef[:,fut].sum(-1)).mean()
        pt_pos+=pt>0; lo_pos+=np.quantile(sk,0.05)>0
    return lo_pos/R, pt_pos/R
for k in (3,6):
    print(k,'F vs y_hist: P(90% CI lower>0), P(point>0) =',ci_power(sF,sH,k))
    print(k,'F vs B5   : P(90% CI lower>0), P(point>0) =',ci_power(sF,sB5,k))
    print(k,'F+R vs y_hist:', ci_power(sFR,sH,k))
