import numpy as np
import pandas as pd
from pathlib import Path
from extinction import fitzpatrick99
from sklearn.model_selection import StratifiedKFold,cross_val_predict
from sklearn.metrics import precision_recall_curve,f1_score,precision_score,recall_score
from xgboost import XGBClassifier
root=Path('mallorn-astronomical-classification-challenge')
out=Path('color_results'); out.mkdir(exist_ok=True)
x=pd.read_csv('timing_results/train_features.csv',index_col='object_id')
y=x.pop('target')
log=pd.read_csv(root/'train_log.csv',index_col='object_id').loc[x.index]
s=log.SpecType
labels=np.select([y.eq(1),s.eq('AGN'),s.str.startswith('SN Ia')],[3,0,1],default=2)
obs=pd.concat(pd.read_csv(p) for p in sorted(root.glob('split_*/train_full_lightcurves.csv')))
obs=obs[obs.Filter.isin(['g','r'])].copy()
obs['snr']=obs.Flux/obs.Flux_err
a=fitzpatrick99(np.array([4704.,6155.]),3.1)
obs['flux']=obs.Flux*10**(obs.object_id.map(log.EBV)*obs.Filter.map(dict(zip('gr',a)))/2.5)
curves={}; colors={}
for key,g in obs.groupby('object_id'):
 r=g[(g.Filter=='r') & (g.snr>=5)]
 if r.empty: continue
 peak=r.loc[r.flux.idxmax(),'Time (MJD)'] if r.index.is_unique else r.sort_values('flux').iloc[-1]['Time (MJD)']
 g=g.assign(day=g['Time (MJD)']-peak)
 curves[key]=g
 w=g[g.day.between(10,40)&g.snr.ge(5)]
 green=w[w.Filter.eq('g')][['day','flux']].sort_values('day')
 red=w[w.Filter.eq('r')][['day','flux']].sort_values('day')
 if green.empty or red.empty: continue
 pairs=pd.merge_asof(green,red,on='day',direction='nearest',tolerance=3,suffixes=('_g','_r')).dropna()
 if len(pairs): colors[key]=np.median(-2.5*np.log10(pairs.flux_g/pairs.flux_r))
c=pd.Series(colors).reindex(x.index)
print('Coverage',c.notna().sum(),c[y.eq(1)].notna().sum(),flush=True)
preds=pd.DataFrame({'target':y,'SpecType':s,'color':c}); results=[]
for name,z in [('baseline',x),('color',x.assign(post_peak_gr=c))]:
 for f,(tr,va) in enumerate(StratifiedKFold(5,shuffle=True,random_state=42).split(z,y),1):
  model=XGBClassifier(n_estimators=250,max_depth=3,learning_rate=.05,subsample=.85,colsample_bytree=.85,random_state=42,n_jobs=2)
  inner=list(StratifiedKFold(4,shuffle=True,random_state=42+f).split(z.iloc[tr],y.iloc[tr]))
  p=cross_val_predict(model,z.iloc[tr],labels[tr],cv=inner,method='predict_proba')[:,3]
  pr,re,t=precision_recall_curve(y.iloc[tr],p)
  cutoff=t[np.argmax(2*pr[:-1]*re[:-1]/np.maximum(pr[:-1]+re[:-1],1e-12))]
  model.fit(z.iloc[tr],labels[tr])
  prob=model.predict_proba(z.iloc[va])[:,3]; pred=prob>=cutoff
  preds.loc[z.index[va],name+'_prediction']=pred.astype(int)
  preds.loc[z.index[va],name+'_probability']=prob
  results.append(dict(model=name,fold=f,f1=f1_score(y.iloc[va],pred),precision=precision_score(y.iloc[va],pred),recall=recall_score(y.iloc[va],pred)))
metrics=pd.DataFrame(results)
metrics.to_csv(out/'fold_metrics.csv',index=False); preds.to_csv(out/'validation_predictions.csv')
print(metrics.to_string(index=False)); print(metrics.groupby('model')[['f1','precision','recall']].mean())
for actual,pred in [(1,0),(1,1),(0,1)]:
 group=preds[(preds.target==actual)&(preds.baseline_prediction==pred)]
 print(actual,pred,len(group),group.color.count(),group.SpecType.value_counts().head(3).to_dict())
 panels=[]
 for key in group[group.color.notna()].sort_values('baseline_probability').iloc[::max(1,group.color.count()//2)].index[:2]:
  g=curves[key]; g=g[g.day.between(-40,80)&g.flux.notna()]
  scale=max(abs(g.flux).max(),1e-12)
  dots=''.join(f'<circle cx="{50+(r.day+40)*4:.1f}" cy="{125-r.flux/scale*85:.1f}" r="3" fill="{dict(g="green",r="red")[r.Filter]}"/>' for r in g.itertuples())
  panels.append(f'<h3>{key} ({s[key]})</h3><svg width="560" height="225" viewBox="0 0 560 225" xmlns="http://www.w3.org/2000/svg"><rect x="250" y="25" width="120" height="185" fill="#eee"/><path d="M 210 25 V 210 M 50 125 H 530" stroke="#999"/>{dots}</svg>')
 if panels:
  (out/f'curves_{actual}_{pred}.html').write_text('<meta charset="utf-8"><p>Green = g, red = r. Observed r peak at vertical line. Shading = days 10–40. Flux scaled per object.</p>'+''.join(panels),encoding='utf-8')

