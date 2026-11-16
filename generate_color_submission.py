"""Train four-class XGBoost and export binary Kaggle predictions."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from xgboost import XGBClassifier


def main():
    results = Path("timing_results")
    data = Path("mallorn-astronomical-classification-challenge")
    train = pd.read_csv(results / "train_features.csv", index_col="object_id")
    test = pd.read_csv(results / "test_features.csv", index_col="object_id")
    log = pd.read_csv(data / "train_log.csv", index_col="object_id")
    sample = pd.read_csv(data / "sample_submission.csv", index_col="object_id")
    target = train.pop("target")
    if (train.shape != (3043, 78) or test.shape != (7135, 78)
            or not train.columns.equals(test.columns)
            or any(not frame.index.is_unique for frame in (train, test, log, sample))
            or set(test.index) != set(sample.index) or list(sample.columns) != ["prediction"]
            or set(log.index) != set(train.index)):
        raise ValueError("Expected matching timing features, metadata, and submission IDs")
    log = log.loc[train.index]
    types = log["SpecType"]
    if types.isna().any() or not log["target"].eq(target).all() or not types.eq("TDE").eq(target.eq(1)).all():
        raise ValueError("Unexpected training labels")
    labels = np.select([target.eq(1), types.eq("AGN"), types.str.startswith("SN Ia")],
                       [3, 0, 1], default=2)
    model = XGBClassifier(
        n_estimators=250, max_depth=3, learning_rate=0.05,
        subsample=0.85, colsample_bytree=0.85, random_state=42, n_jobs=2,
    )
    from extinction import fitzpatrick99
    root = data
    test_log = pd.read_csv(data / 'test_log.csv', index_col='object_id')
    obs=pd.concat(pd.read_csv(p) for p in sorted(root.glob('split_*/test_full_lightcurves.csv')))
    obs=obs[obs.Filter.isin(['g','r'])].copy()
    obs['snr']=obs.Flux/obs.Flux_err
    a=fitzpatrick99(np.array([4704.,6155.]),3.1)
    obs['flux']=obs.Flux*10**(obs.object_id.map(test_log.EBV)*obs.Filter.map(dict(zip('gr',a)))/2.5)
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
    train['post_peak_gr'] = pd.read_csv('color_results/validation_predictions.csv', index_col='object_id')['color'].reindex(train.index)
    test['post_peak_gr'] = pd.Series(colors).reindex(test.index)
    if train.post_peak_gr.count() != 516 or not test.post_peak_gr.notna().any():
        raise ValueError('Unexpected color coverage')
    print(f'Test color coverage: {test.post_peak_gr.count()}', flush=True)
    folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(train, target))
    probabilities = cross_val_predict(model, train, labels, cv=folds, method="predict_proba")[:, 3]
    precision, recall, thresholds = precision_recall_curve(target, probabilities)
    total = precision[:-1] + recall[:-1]
    scores = np.divide(2 * precision[:-1] * recall[:-1], total,
                       out=np.zeros_like(thresholds), where=total > 0)
    threshold = thresholds[np.argmax(scores)]
    model.fit(train, labels)
    tde_column = np.flatnonzero(model.classes_ == 3).item()
    predicted = model.predict_proba(test.loc[sample.index])[:, tde_column] >= threshold
    submission = pd.DataFrame({"prediction": predicted.astype(int)}, index=sample.index)
    output = results / "submission_multiclass_color.csv"
    with output.open("x", encoding="utf-8", newline="") as file:
        submission.to_csv(file)
    if not pd.read_csv(output, index_col="object_id").equals(submission):
        raise ValueError("Written submission differs from predictions")
    print(f"{output}: {len(submission)} rows, {predicted.sum()} TDE predictions, threshold {threshold:.6f}")


if __name__ == "__main__":
    main()

