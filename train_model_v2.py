import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error


DATA_FILE = Path("data/processed/bitcoin_merged_hourly.csv")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

HORIZONS = [1, 3, 6]
N_SPLITS = 5
TEST_SIZE = 24 * 14  # 14 days per test fold

PRICE_FEATURES = [
    "return_1h",
    "return_3h",
    "return_6h",
    "log_volume",
]

SENTIMENT_FEATURES = [
    "tweet_count",
    "average_sentiment",
    "median_sentiment",
    "positive_share",
    "negative_share",
    "has_tweets",
]


def make_model():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("regressor", Ridge(alpha=10.0)),
    ])


def calculate_metrics(actual, predicted):
    actual = np.asarray(actual)
    predicted = np.asarray(predicted)

    # Ignore exact zero predictions when evaluating direction.
    # This makes directional accuracy undefined if a model
    # predicts zero for every observation.
    nonzero_predictions = predicted != 0

    if nonzero_predictions.any():
        directional_accuracy = np.mean(
            np.sign(actual[nonzero_predictions])
            == np.sign(predicted[nonzero_predictions])
        )
    else:
        directional_accuracy = np.nan

    return {
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": np.sqrt(mean_squared_error(actual, predicted)),
        "directional_accuracy": directional_accuracy,
        "observations": len(actual),
    }


df = pd.read_csv(DATA_FILE, parse_dates=["hour"])
df = df.sort_values("hour").reset_index(drop=True)

if df["hour"].duplicated().any():
    raise ValueError("Duplicate timestamps found.")

df["has_tweets"] = (df["tweet_count"] > 0).astype(int)
df["log_volume"] = np.log1p(df["Volume"].clip(lower=0))

feature_sets = {
    "price_only": PRICE_FEATURES,
    "price_plus_sentiment": PRICE_FEATURES + SENTIMENT_FEATURES,
}

fold_results = []
prediction_results = []

for horizon in HORIZONS:
    target = f"future_return_{horizon}h"

    data = df.loc[df[target].notna()].copy()
    data = data.replace([np.inf, -np.inf], np.nan)

    splitter = TimeSeriesSplit(
        n_splits=N_SPLITS,
        test_size=TEST_SIZE,
        gap=horizon,
    )

    print(f"\n===== {horizon}-HOUR HORIZON =====")

    for fold, (train_idx, test_idx) in enumerate(
        splitter.split(data), start=1
    ):
        train = data.iloc[train_idx]
        test = data.iloc[test_idx]

        y_train = train[target].to_numpy(dtype=float)
        y_test = test[target].to_numpy(dtype=float)

        # Zero-return baseline
        zero_predictions = np.zeros(len(test))
        zero_metrics = calculate_metrics(y_test, zero_predictions)

        fold_results.append({
            "horizon_hours": horizon,
            "fold": fold,
            "model": "zero_return_baseline",
            **zero_metrics,
        })

        prediction_results.append(pd.DataFrame({
            "hour": test["hour"].to_numpy(),
            "horizon_hours": horizon,
            "fold": fold,
            "model": "zero_return_baseline",
            "actual_return": y_test,
            "predicted_return": zero_predictions,
        }))

        for model_name, features in feature_sets.items():
            model = make_model()
            model.fit(train[features], y_train)

            predictions = model.predict(test[features])
            metrics = calculate_metrics(y_test, predictions)

            fold_results.append({
                "horizon_hours": horizon,
                "fold": fold,
                "model": model_name,
                **metrics,
            })

            prediction_results.append(pd.DataFrame({
                "hour": test["hour"].to_numpy(),
                "horizon_hours": horizon,
                "fold": fold,
                "model": model_name,
                "actual_return": y_test,
                "predicted_return": predictions,
            }))

            print(
                f"Fold {fold} | {model_name:24s} | "
                f"MAE={metrics['MAE']:.6f} | "
                f"RMSE={metrics['RMSE']:.6f} | "
                f"Direction={metrics['directional_accuracy']:.3f}"
            )

metrics_df = pd.DataFrame(fold_results)
predictions_df = pd.concat(prediction_results, ignore_index=True)

metrics_path = OUTPUT_DIR / "walk_forward_fold_metrics.csv"
predictions_path = OUTPUT_DIR / "walk_forward_fold_predictions.csv"

metrics_df.to_csv(metrics_path, index=False)
predictions_df.to_csv(predictions_path, index=False)

print("\n===== AVERAGE METRICS ACROSS FOLDS =====")

summary = (
    metrics_df
    .groupby(["horizon_hours", "model"], as_index=False)
    .agg({
        "MAE": "mean",
        "RMSE": "mean",
        "directional_accuracy": "mean",
        "observations": "sum",
    })
)

print(summary.to_string(index=False))

print("\nSaved fold metrics:", metrics_path.resolve())
print("Saved fold predictions:", predictions_path.resolve())