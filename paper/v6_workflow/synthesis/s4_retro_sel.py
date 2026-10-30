import pandas as pd, numpy as np
from pathlib import Path
B = str(Path(__file__).resolve().parents[3] / 'results' / 'metrics') + '/'
pd.set_option('display.width',250)
def load(f, arms=None):
    d=pd.read_parquet(B+f)
    if 'horizon' in d:
        print(f, 'horizons', d.horizon.astype(str).unique()[:10])
        h=d.horizon.astype(str)
        d=d[h.isin(['0','eos','None','nan','end'])] if h.isin(['0','eos','None','nan','end']).any() else d[h==h.iloc[0]]
    d=d[d.seed.astype(str)=='ens']
    return d
def table(d, yrs=range(2016,2026)):
    d=d[d.year.isin(yrs)].copy()
    d['se']=(d.y_pred-d.y_true)**2; d['seh']=(d.y_hist-d.y_true)**2
    d['ape']=(d.y_pred-d.y_true).abs()/d.y_true*100
    cy=d.groupby(['arm','crop','year']).agg(sse=('se','sum'),ss_hist=('seh','sum')).reset_index(); cy['skill']=1-cy.sse/cy.ss_hist
    pc=d.groupby(['arm','crop']).agg(sse=('se','sum'),ss_hist=('seh','sum')).reset_index()
    pc['lr']=np.log(pc.sse); pc['psk']=1-pc.sse/pc.ss_hist
    ref=pc.groupby('crop').sse.transform('min')
    pc['rel_to_min']=pc.sse/ref
    out=pd.DataFrame({
      'beats':cy.groupby('arm').apply(lambda g:(g.skill>0).sum()),
      'med_skill':cy.groupby('arm').skill.median(),
      'geo_relSSE':np.exp(pc.groupby('arm').lr.mean()-pc.groupby('arm').lr.mean().min()),
      'mean_pooled_skill':pc.groupby('arm').psk.mean(),
      'total_SSE':d.groupby('arm').se.sum()/d.groupby('arm').se.sum().min(),
      'MAPE':d.groupby('arm').ape.mean()})
    return out.round(4)
for f in ['prereg_predictions.parquet','v2_predictions.parquet','v3_predictions.parquet','v4_predictions.parquet','v5_predictions.parquet']:
    d=load(f)
    print('\n==',f); print(table(d))
    print(' 2019-25:'); print(table(d,range(2019,2026))[['geo_relSSE','total_SSE','MAPE']])
