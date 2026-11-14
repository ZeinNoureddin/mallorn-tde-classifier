"""Reproduce the binary XGBoost Kaggle submission from --with-timing outputs."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve
from xgboost import XGBClassifier


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("timing_results"))
    parser.add_argument("--data", type=Path, default=Path("mallorn-astronomical-classification-challenge"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.results / "submission.csv"

    train = pd.read_csv(args.results / "train_features.csv", index_col="object_id")
    test = pd.read_csv(args.results / "test_features.csv", index_col="object_id")
    validation = pd.read_csv(args.results / "validation_predictions.csv", index_col="object_id")
    sample = pd.read_csv(args.data / "sample_submission.csv", index_col="object_id")
    target = train.pop("target")
    if train.shape != (3043, 78) or test.shape != (7135, 78):
        raise ValueError("Expected the original 78-feature timing experiment")
    if any(not table.index.is_unique for table in (train, test, validation, sample)):
        raise ValueError("Duplicate object IDs")
    if not train.columns.equals(test.columns) or set(test.index) != set(sample.index):
        raise ValueError("Feature columns or Kaggle test IDs do not match")
    if list(sample.columns) != ["prediction"] or set(validation.index) != set(train.index):
        raise ValueError("Unexpected sample submission or validation IDs")
    validation = validation.reindex(train.index)
    probabilities = validation["xgboost_probability"]
    if not validation["target"].eq(target).all() or not probabilities.between(0, 1).all():
        raise ValueError("Invalid validation labels or probabilities")

    precision, recall, thresholds = precision_recall_curve(target, probabilities)
    total = precision[:-1] + recall[:-1]
    f1 = np.divide(2 * precision[:-1] * recall[:-1], total,
                   out=np.zeros_like(thresholds), where=total > 0)
    threshold = thresholds[np.argmax(f1)]
    model = XGBClassifier(
        n_estimators=250, max_depth=3, learning_rate=0.05,
        subsample=0.85, colsample_bytree=0.85, random_state=42, n_jobs=2,
    )
    model.fit(train, target)
    predicted = model.predict_proba(test.reindex(sample.index))[:, 1] >= threshold
    submission = pd.DataFrame({"prediction": predicted.astype(int)}, index=sample.index)
    with output.open("x", encoding="utf-8", newline="") as file:
        submission.to_csv(file)
    print(f"{output}: {len(submission)} rows, {predicted.sum()} TDE predictions, threshold {threshold:.6f}")


if __name__ == "__main__":
    main()
