"""SUPERSEDED by scripts/40_compare_table.py (district-code alignment, fixes a
tie mis-match in 2023). Kept for the record; do not use its numbers."""
"""Head-to-head vs AgroForecast-DE on identical district-years (2023-24 test).

Their per-year arrays carry no district ids but are in NUTS-code order; districts
are recovered by an order-preserving exact-value alignment (LCS) of their observed
yields against Destatis. Districts they no longer report (observed_<crop>.json last year < test year) are
excluded first; with that, the alignment is unique (no ties) and agrees 100% with
the district-code mapping of their 2024 list. Ours = arm D 3-seed ensemble backtests.
"""
import sys, json, numpy as np, pandas as pd
A=sys.argv[1] if len(sys.argv) > 1 else 'external/AgroForecast-DE/docs/data'  # clone of github.com/geonextgis/AgroForecast-DE @ e3d6fdb
P=pd.read_parquet('results/metrics/prereg_predictions.parquet')
P=P[(P.arm=='D')&(P.seed=='ens')&(P.horizon==0)]
Y=pd.read_parquet('data/interim/yield.parquet')
rows=[]
for crop in ['winter_wheat','silage_maize']:
    d=json.load(open(f'{A}/performance_{crop}.json'))
    for e in d['years']:
        if e['split']!='test': continue
        y=e['year']; them=pd.DataFrame({'obs':e['observed'],'pred_them':e['predicted']})
        them['key']=(them.obs*100).round().astype(int)          # t/ha, 2 dp
        yy=Y[(Y.crop==crop)&(Y.year==y)].copy()
        yy['key']=(yy.yield_dt_per_ha*10).round().astype(int)   # dt/ha,1dp -> t/ha*100
        # their data drops districts after their last reported year (e.g. the cities
        # DE241-DE244 end in 2016, while our panel repeats the surrounding district's
        # yield for them); only districts they still report can be matched
        last=json.load(open(f'{A}/observed_{crop}.json'))['districts']
        yy=yy[[last.get(lk,{}).get('year',0)>=y for lk in yy.landkreis]]
        yy=yy.sort_values('landkreis').reset_index(drop=True)
        a,b=them.key.to_numpy(),yy.key.to_numpy()   # both in NUTS-code order
        L=np.zeros((len(a)+1,len(b)+1),int)
        for i in range(len(a)-1,-1,-1):
            for j in range(len(b)-1,-1,-1):
                L[i,j]=L[i+1,j+1]+1 if a[i]==b[j] else max(L[i+1,j],L[i,j+1])
        i=j=0; pairs=[]
        while i<len(a) and j<len(b):
            if a[i]==b[j]: pairs.append((i,j)); i+=1; j+=1
            elif L[i+1,j]>=L[i,j+1]: i+=1
            else: j+=1
        ii,jj=map(list,zip(*pairs))
        m=pd.concat([them.iloc[ii].reset_index(drop=True),yy.iloc[jj][['landkreis','yield_dt_per_ha']].reset_index(drop=True)],axis=1)
        rows.append(m.assign(crop=crop,year=y))
        print(crop,y,'theirs',len(them),'destatis',len(yy),'order-aligned',len(m))
M=pd.concat(rows)
M['pred_them']*=10; M['obs']*=10
M=M.merge(P[['landkreis','crop','year','y_pred','y_hist','y_true']],on=['landkreis','crop','year'],how='inner')
print('matched to our backtest sample:',len(M), M.groupby(['crop','year']).size().to_dict())
assert np.allclose(M.obs,M.y_true,atol=0.051)
def met(yt,yp): 
    sse=((yt-yp)**2).sum(); return 1-sse/((yt-yt.mean())**2).sum(), np.sqrt(np.mean((yt-yp)**2)), 100*np.mean(np.abs(yt-yp)/yt)
out=[]
for (c,y),g in list(M.groupby(['crop','year']))+[((c,'2023-24'),g) for c,g in M.groupby('crop')]:
    for name,col in [('AgroForecast-DE (CropFusionNet release)','pred_them'),('Ours: iTransformer arm D','y_pred'),('Historical average (baseline)','y_hist')]:
        r2,rmse,mape=met(g.y_true.values,g[col].values)
        out.append(dict(crop=c,year=y,model=name,n=len(g),r2=r2,rmse=rmse,mape=mape,skill=1-((g.y_true-g[col])**2).sum()/((g.y_true-g.y_hist)**2).sum()))
O=pd.DataFrame(out); print(O.round(3).to_string(index=False))
O.to_csv('results/metrics/compare_agroforecast_matched.csv',index=False)
