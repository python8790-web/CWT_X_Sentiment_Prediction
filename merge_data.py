import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.pipeline_utils import add_return_targets, normalize_utc_timestamps

DATA_DIR = ROOT / "data" / "processed"
SENTIMENT_FILE = DATA_DIR / "hourly_sentiment.csv"
PRICE_FILE = DATA_DIR / "bitcoin_historical_hourly.csv"
OUTPUT_FILE = DATA_DIR / "bitcoin_merged_hourly.csv"


def main() -> None:
    if not SENTIMENT_FILE.exists():
        raise FileNotFoundError(f"Missing hourly sentiment file: {SENTIMENT_FILE}")
    if not PRICE_FILE.exists():
        raise FileNotFoundError(f"Missing hourly price file: {PRICE_FILE}")

    sentiment = pd.read_csv(SENTIMENT_FILE)
    prices = pd.read_csv(PRICE_FILE)
    sentiment["hour"] = normalize_utc_timestamps(sentiment["hour"]).dt.tz_localize(None)
    prices["hour"] = normalize_utc_timestamps(prices["hour"]).dt.tz_localize(None)
    sentiment = sentiment.dropna(subset=["hour"])
    prices = prices.dropna(subset=["hour", "Close"])
    if prices["hour"].duplicated().any():
        raise ValueError("Duplicate price timestamps found.")
    if sentiment["hour"].duplicated().any():
        raise ValueError("Duplicate sentiment timestamps found.")

    merged = prices.merge(
        sentiment,
        on="hour",
        how="left",
        validate="one_to_one",
    )
    merged["tweet_count"] = merged["tweet_count"].fillna(0)
    for column in [
        "average_sentiment",
        "median_sentiment",
        "positive_share",
        "negative_share",
    ]:
        merged[column] = merged[column].fillna(0)

    merged = add_return_targets(merged)
    merged = merged.dropna(subset=["return_1h", "return_3h", "return_6h"]).reset_index(
        drop=True
    )
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUTPUT_FILE, index=False)

    gaps = int(merged["hour"].diff().gt(pd.Timedelta(hours=1)).sum())
    print("\n===== MERGED DATA SUMMARY =====")
    print("Hourly rows:", len(merged))
    print("First hour (UTC):", merged["hour"].min())
    print("Last hour (UTC):", merged["hour"].max())
    print("Hours with tweets:", int((merged["tweet_count"] > 0).sum()))
    print("Hours without tweets:", int((merged["tweet_count"] == 0).sum()))
    print("Missing closing prices:", int(merged["Close"].isna().sum()))
    print("Duplicate timestamps:", int(merged["hour"].duplicated().sum()))
    print("Missing hourly intervals:", gaps)
    print("Saved to:", OUTPUT_FILE.resolve())


if __name__ == "__main__":
    main()
