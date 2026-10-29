from pathlib import Path
import os

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[1]
load_dotenv(ROOT_DIR / ".env")

DATA_DIR = ROOT_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUT_DIR = ROOT_DIR / "outputs"
CHARTS_DIR = OUTPUT_DIR / "charts"
RAW_DIR = DATA_DIR / "raw"
NEWS_DIR = RAW_DIR / "news"

RAW_TWEET_FILE = DATA_DIR / "bitcoin_tweets_latest.csv"
PROCESSED_TWEETS = PROCESSED_DIR / "bitcoin_tweets_with_sentiment.csv"
PROCESSED_HOURLY = PROCESSED_DIR / "hourly_sentiment.csv"
PRICE_FILE = PROCESSED_DIR / "bitcoin_historical_hourly.csv"
MERGED_FILE = PROCESSED_DIR / "bitcoin_merged_hourly.csv"
PREDICTIONS_FILE = OUTPUT_DIR / "walk_forward_predictions.csv"
STRATEGY_FILE = OUTPUT_DIR / "strategy_metrics.csv"

DEFAULT_HORIZONS = [0, 1, 2, 3, 4, 5]


def required_api_key(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(
            f"{name} is not configured. Set it in the environment or project .env file."
        )
    return value
