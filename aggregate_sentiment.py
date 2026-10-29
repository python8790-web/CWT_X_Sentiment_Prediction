
import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.pipeline_utils import aggregate_sentiment_posts

def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate scored tweets by UTC hour.")
    parser.add_argument(
        "--input",
        nargs="+",
        default=["data/processed/bitcoin_tweets_with_sentiment.csv"],
        help="Scored tweet CSV (large inputs are processed in chunks)",
    )
    parser.add_argument(
        "--output", default="data/processed/hourly_sentiment.csv"
    )
    args = parser.parse_args()

    output_file = Path(args.output)
    hourly_chunks = []
    for input_name in args.input:
        input_file = Path(input_name)
        if not input_file.exists():
            raise FileNotFoundError(f"Tweet input not found: {input_file.resolve()}")
        for chunk in pd.read_csv(input_file, chunksize=200_000, low_memory=False):
            if "date" not in chunk.columns and "created_at" in chunk.columns:
                chunk = chunk.rename(columns={"created_at": "date"})
            hourly = aggregate_sentiment_posts(chunk)
            if not hourly.empty:
                hourly_chunks.append(hourly)
    if not hourly_chunks:
        raise ValueError("No valid scored posts found in the selected input files.")

    combined = pd.concat(hourly_chunks, ignore_index=True)
    combined["sentiment_sum"] = (
        combined["average_sentiment"] * combined["tweet_count"]
    )
    combined["positive_count"] = (
        combined["positive_share"] * combined["tweet_count"]
    )
    combined["negative_count"] = (
        combined["negative_share"] * combined["tweet_count"]
    )
    hourly = (
        combined.groupby("hour", as_index=False)
        .agg(
            tweet_count=("tweet_count", "sum"),
            sentiment_sum=("sentiment_sum", "sum"),
            positive_count=("positive_count", "sum"),
            negative_count=("negative_count", "sum"),
            median_sentiment=("median_sentiment", "median"),
            average_followers=("average_followers", "mean"),
        )
    )
    hourly["average_sentiment"] = hourly["sentiment_sum"] / hourly["tweet_count"]
    hourly["positive_share"] = hourly["positive_count"] / hourly["tweet_count"]
    hourly["negative_share"] = hourly["negative_count"] / hourly["tweet_count"]
    hourly = hourly[
        [
            "hour",
            "tweet_count",
            "average_sentiment",
            "median_sentiment",
            "positive_share",
            "negative_share",
            "average_followers",
        ]
    ].sort_values("hour")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    hourly.to_csv(output_file, index=False)
    print(f"Hourly records: {len(hourly)}")
    print(f"UTC coverage: {hourly['hour'].min()} to {hourly['hour'].max()}")
    print(f"Saved to: {output_file.resolve()}")


if __name__ == "__main__":
    main()
