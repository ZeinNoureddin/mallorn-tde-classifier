import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from extinction import fitzpatrick99

BANDS = list("ugrizy")
WAVELENGTHS = np.array([3641, 4704, 6155, 7504, 8695, 10056], dtype=float)


def timing_features(group):
    detections = group.loc[group["snr"] >= 5].sort_values("Time (MJD)")
    times = detections["Time (MJD)"].to_numpy()
    flux = detections["flux"].to_numpy()
    if len(times) < 2:
        return np.nan, np.nan, np.nan
    peak = int(np.argmax(flux))
    before = times < times[peak]
    after = times > times[peak]
    duration = times[-1] - times[0]
    rise = times[peak] - times[before][0] if before.any() else np.nan
    fade = times[after][-1] - times[peak] if after.any() else np.nan
    return duration, rise, fade


def build_features(root, partition, with_timing=False):
    log = pd.read_csv(root / f"{partition}_log.csv").set_index("object_id", verify_integrity=True)
    observations = pd.concat(
        pd.read_csv(root / split / f"{partition}_full_lightcurves.csv")
        for split in log["split"].unique()
    )
    extinction = fitzpatrick99(WAVELENGTHS, 3.1)
    factor = 10 ** (
        observations["object_id"].map(log["EBV"])
        * observations["Filter"].map(dict(zip(BANDS, extinction)))
        / 2.5
    )
    observations["flux"] = observations["Flux"] * factor
    observations["snr"] = observations["flux"] / (observations["Flux_err"] * factor)
    grouped = observations.groupby(["object_id", "Filter"])
    summary = grouped.agg(
        count=("flux", "size"),
        valid_flux_count=("flux", "count"),
        first=("Time (MJD)", "min"),
        last=("Time (MJD)", "max"),
        flux_std=("flux", "std"),
        flux_skew=("flux", "skew"),
        snr_median=("snr", "median"),
        snr_max=("snr", "max"),
    )
    summary["span"] = summary.pop("last") - summary.pop("first")
    for percentile in (10, 50, 90):
        summary[f"flux_p{percentile}"] = grouped["flux"].quantile(percentile / 100)
    if with_timing:
        timing = pd.DataFrame(
            [timing_features(group) for _, group in grouped],
            index=summary.index,
            columns=["detection_span", "first_detection_to_peak", "peak_to_last_detection"],
        )
        summary = summary.join(timing)
    columns = pd.MultiIndex.from_product([summary.columns, BANDS])
    features = summary.unstack().reindex(index=log.index, columns=columns)
    for count in ("count", "valid_flux_count"):
        features[count] = features[count].fillna(0)
    features.columns = [f"{band}_{stat}" for stat, band in features.columns]
    return features, log


def evaluate(features, target):
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score, precision_recall_curve, precision_score, recall_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from xgboost import XGBClassifier

    models = {
        "logistic_regression": make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True),
            StandardScaler(),
            LogisticRegression(max_iter=2000),
        ),
        "xgboost": XGBClassifier(
            n_estimators=250, max_depth=3, learning_rate=0.05,
            subsample=0.85, colsample_bytree=0.85, random_state=42, n_jobs=2,
        ),
    }
    outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    predictions = pd.DataFrame({"target": target})
    results = []
    for fold, (train, held_out) in enumerate(outer.split(features, target), start=1):
        x_train, y_train = features.iloc[train], target.iloc[train]
        actual = target.iloc[held_out]
        predictions.loc[actual.index, "fold"] = fold
        inner = StratifiedKFold(n_splits=4, shuffle=True, random_state=42 + fold)
        for name, model in models.items():
            oof = cross_val_predict(model, x_train, y_train, cv=inner, method="predict_proba")[:, 1]
            precision, recall, thresholds = precision_recall_curve(y_train, oof)
            denominator = precision[:-1] + recall[:-1]
            scores = np.divide(
                2 * precision[:-1] * recall[:-1], denominator,
                out=np.zeros_like(thresholds), where=denominator > 0,
            )
            threshold = thresholds[np.argmax(scores)]
            model.fit(x_train, y_train)
            probability = model.predict_proba(features.iloc[held_out])[:, 1]
            predicted = probability >= threshold
            predictions.loc[actual.index, f"{name}_probability"] = probability
            predictions.loc[actual.index, f"{name}_prediction"] = predicted.astype(int)
            results.append({
                "model": name, "fold": fold, "threshold": threshold,
                "f1": f1_score(actual, predicted, zero_division=0),
                "precision": precision_score(actual, predicted, zero_division=0),
                "recall": recall_score(actual, predicted, zero_division=0),
            })
    return pd.DataFrame(results), predictions.astype(
        {"fold": int, **{f"{name}_prediction": int for name in models}}
    )


def main():
    parser = argparse.ArgumentParser(description="Build and evaluate MALLORN light-curve baselines.")
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument('--features-only', action='store_true')
    parser.add_argument("--with-timing", action="store_true")
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = Path("timing_results" if args.with_timing else "baseline_results")
    train, train_log = build_features(args.data_dir, "train", args.with_timing)
    test, _ = build_features(args.data_dir, "test", args.with_timing)
    if not train.columns.equals(test.columns) or not train.index.intersection(test.index).empty:
        raise ValueError("Expected identical feature columns and disjoint object IDs")
    if not (np.isfinite(train) | train.isna()).all().all() or not (
        np.isfinite(test) | test.isna()
    ).all().all():
        raise ValueError("Feature tables contain infinite values")
    target = train_log["target"].astype(int)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train.join(target).to_csv(args.output_dir / "train_features.csv")
    test.to_csv(args.output_dir / "test_features.csv")
    print(f'Train: {train.shape}, test: {test.shape} (objects, features)', flush=True)
    if args.features_only:
        return
    metrics, predictions = evaluate(train, target)
    metrics.to_csv(args.output_dir / "fold_metrics.csv", index=False)
    predictions.to_csv(args.output_dir / "validation_predictions.csv")
    if args.with_timing:
        baseline, baseline_predictions = evaluate(train.iloc[:, :60], target)
        baseline.to_csv(args.output_dir / "baseline_fold_metrics.csv", index=False)
        if not baseline_predictions["fold"].equals(predictions["fold"]):
            raise ValueError("Baseline and timing experiments used different folds")
        comparison = baseline.merge(metrics, on=["model", "fold"], suffixes=("_baseline", "_timing"))
        comparison["f1_change"] = comparison["f1_timing"] - comparison["f1_baseline"]
        comparison.to_csv(args.output_dir / "timing_comparison.csv", index=False)
        print(comparison.groupby("model")[["f1_baseline", "f1_timing", "f1_change"]].mean())
    print(metrics.groupby("model")[["f1", "precision", "recall"]].agg(["mean", "std"]).round(4))


if __name__ == "__main__":
    main()

