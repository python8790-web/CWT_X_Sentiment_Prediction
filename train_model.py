import numpy as np
import pandas as pd
from pathlib import Path
import sys

from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

DATA_FILE = ROOT / "data" / "processed" / "bitcoin_merged_hourly.csv"
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

HORIZONS = [0, 1, 2, 3, 4, 5]
N_SPLITS = 5
TEST_SIZE = 24 * 14

PRICE_FEATURES = ["return_1h", "return_3h", "return_6h", "log_volume"]
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
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    nonzero = predicted != 0
    directional_accuracy = (
        np.mean(np.sign(actual[nonzero]) == np.sign(predicted[nonzero]))
        if np.any(nonzero)
        else np.nan
    )
    return {
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": np.sqrt(mean_squared_error(actual, predicted)),
        "directional_accuracy": directional_accuracy,
        "observations": len(actual),
    }


def main():
    print("Loading merged hourly dataset...")
    df = pd.read_csv(DATA_FILE, parse_dates=["hour"]).sort_values("hour").reset_index(drop=True)

    if df["hour"].duplicated().any():
        raise ValueError("Duplicate timestamps found.")

    required_columns = [
        "Close", "Volume", "tweet_count", "average_sentiment",
        "return_1h", "return_3h", "return_6h",
    ]
    for required in required_columns:
        if required not in df.columns:
            raise ValueError(f"Missing required column: {required}")

    df["has_tweets"] = (df["tweet_count"] > 0).astype(int)
    df["log_volume"] = np.log1p(df["Volume"].clip(lower=0))

    for horizon in HORIZONS:
        target_column = f"future_return_{horizon}h"
        if target_column not in df.columns:
            raise ValueError(f"Missing horizon target: {target_column}")

    print("Rows loaded:", len(df))
    print("Hours with tweets:", int(df["has_tweets"].sum()))

    all_metrics = []
    all_predictions = []

    for horizon in HORIZONS:
        target_column = f"future_return_{horizon}h"
        data = df.loc[df[target_column].notna()].copy().replace([np.inf, -np.inf], np.nan)

        if len(data) <= TEST_SIZE * N_SPLITS:
            raise ValueError(
                f"Not enough rows for {horizon}h horizon. "
                "Reduce TEST_SIZE or N_SPLITS."
            )

        # n+0 is the completed return ending at this timestamp, not a forecast.
        if horizon == 0:
            diagnostic = data["future_return_0h"].to_numpy(dtype=float)
            all_metrics.append({
                "horizon_hours": 0,
                "model": "known_current_return_reference",
                "status": "diagnostic_not_forecast",
                "MAE": np.nan,
                "RMSE": np.nan,
                "directional_accuracy": np.nan,
                "observations": len(diagnostic),
            })
            all_predictions.append(pd.DataFrame({
                "hour": data["hour"],
                "target_hour": data["hour"],
                "signal_time": data["hour"] + pd.Timedelta(hours=1),
                "target_time": data["hour"] + pd.Timedelta(hours=1),
                "horizon_hours": 0,
                "model": "known_current_return_reference",
                "fold": 0,
                "actual_return": diagnostic,
                "predicted_return": np.nan,
            }))
            continue

        splitter = TimeSeriesSplit(
            n_splits=N_SPLITS,
            test_size=TEST_SIZE,
            gap=horizon,
        )
        feature_sets = {
            "zero_return_baseline": None,
            "price_only": PRICE_FEATURES,
            "price_plus_sentiment": PRICE_FEATURES + SENTIMENT_FEATURES,
        }

        print(f"\n===== {horizon}-HOUR HORIZON =====")

        for model_name, features in feature_sets.items():
            fold_actual = []
            fold_predicted = []
            fold_timestamps = []
            fold_numbers = []

            for fold, (train_idx, test_idx) in enumerate(splitter.split(data), start=1):
                train = data.iloc[train_idx]
                test = data.iloc[test_idx]
                y_train = train[target_column].to_numpy(dtype=float)
                y_test = test[target_column].to_numpy(dtype=float)

                if model_name == "zero_return_baseline":
                    predictions = np.zeros(len(test), dtype=float)
                else:
                    model = make_model()
                    model.fit(train[features], y_train)
                    predictions = model.predict(test[features])

                fold_actual.extend(y_test)
                fold_predicted.extend(predictions)
                fold_timestamps.extend(test["hour"].tolist())
                fold_numbers.extend([fold] * len(test))

            metrics = calculate_metrics(fold_actual, fold_predicted)
            result = {
                "horizon_hours": horizon,
                "model": model_name,
                "status": "out_of_sample",
                **metrics,
            }
            all_metrics.append(result)

            print(
                f"{model_name:24s} | "
                f"MAE={metrics['MAE']:.6f} | "
                f"RMSE={metrics['RMSE']:.6f} | "
                f"Direction={metrics['directional_accuracy']:.3f} | "
                f"N={metrics['observations']}"
            )

            predictions_df = pd.DataFrame({
                "hour": fold_timestamps,
                "target_hour": pd.to_datetime(fold_timestamps) + pd.Timedelta(hours=horizon),
                "signal_time": pd.to_datetime(fold_timestamps) + pd.Timedelta(hours=1),
                "target_time": pd.to_datetime(fold_timestamps) + pd.Timedelta(hours=horizon + 1),
                "horizon_hours": horizon,
                "model": model_name,
                "fold": fold_numbers,
                "actual_return": fold_actual,
                "predicted_return": fold_predicted,
            })
            all_predictions.append(predictions_df)

    metrics_df = pd.DataFrame(all_metrics)
    predictions_df = pd.concat(all_predictions, ignore_index=True)

    metrics_path = OUTPUT_DIR / "walk_forward_metrics.csv"
    predictions_path = OUTPUT_DIR / "walk_forward_predictions.csv"
    metrics_df.to_csv(metrics_path, index=False)
    predictions_df.to_csv(predictions_path, index=False)

    print("\n===== FINAL RESULTS =====")
    print(metrics_df.to_string(index=False))
    print("\nSaved metrics to:", metrics_path.resolve())
    print("Saved predictions to:", predictions_path.resolve())


if __name__ == "__main__":
    main()
