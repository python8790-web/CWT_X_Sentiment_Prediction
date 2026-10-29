from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from src.pipeline_utils import normalize_utc_timestamps
from src.project_config import PROCESSED_DIR, RAW_DIR


def clean_text(text: str) -> str:
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"#(\w+)", r" \1 ", text)
    return re.sub(r"\s+", " ", text.replace("\n", " ")).strip()


def score_posts(posts: pd.DataFrame) -> pd.DataFrame:
    required = {"created_at", "text"}
    missing = required.difference(posts.columns)
    if missing:
        raise ValueError(f"Missing post columns: {', '.join(sorted(missing))}")
    frame = posts.copy()
    frame["date"] = normalize_utc_timestamps(frame["created_at"])
    frame["text"] = frame["text"].astype("string").str.strip()
    frame = frame.dropna(subset=["date", "text"])
    frame = frame[frame["text"].ne("")].copy()
    if "tweet_id" in frame.columns:
        has_id = (
            frame["tweet_id"].notna()
            & frame["tweet_id"].astype(str).str.strip().ne("")
        )
        frame = pd.concat(
            [
                frame.loc[has_id].drop_duplicates("tweet_id", keep="last"),
                frame.loc[~has_id].drop_duplicates(["date", "text"], keep="last"),
            ],
            ignore_index=True,
        )
    else:
        frame = frame.drop_duplicates(["date", "text"], keep="last")
    frame["text_clean"] = frame["text"].map(clean_text)
    scores = frame["text_clean"].map(SentimentIntensityAnalyzer().polarity_scores)
    frame["sentiment_positive"] = scores.map(lambda score: score["pos"])
    frame["sentiment_negative"] = scores.map(lambda score: score["neg"])
    frame["sentiment_neutral"] = scores.map(lambda score: score["neu"])
    frame["sentiment_compound"] = scores.map(lambda score: score["compound"])
    frame["sentiment_label"] = "neutral"
    frame.loc[frame["sentiment_compound"] > 0.05, "sentiment_label"] = "positive"
    frame.loc[frame["sentiment_compound"] < -0.05, "sentiment_label"] = "negative"
    frame["user_followers"] = pd.to_numeric(
        frame.get("author_followers"), errors="coerce"
    )
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clean and score normalized Apify posts with VADER."
    )
    parser.add_argument(
        "--input", default=str(RAW_DIR / "x_posts_apify.csv")
    )
    parser.add_argument(
        "--output", default=str(PROCESSED_DIR / "apify_tweets_with_sentiment.csv")
    )
    args = parser.parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Apify post file not found: {input_path.resolve()}")
    scored = score_posts(pd.read_csv(input_path))
    if scored.empty:
        raise ValueError("No valid Apify posts remained after validation and deduplication.")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(output, index=False)
    print(
        f"Scored {len(scored)} unique posts; UTC coverage "
        f"{scored['date'].min()} to {scored['date'].max()}; saved to {output.resolve()}"
    )


if __name__ == "__main__":
    main()
