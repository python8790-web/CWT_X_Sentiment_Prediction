
import csv
import re
from pathlib import Path

import numpy as np
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

input_file = Path("data/bitcoin_tweets_latest.csv")
output_dir = Path("data/processed")
output_dir.mkdir(parents=True, exist_ok=True)

full_output = output_dir / "bitcoin_tweets_with_sentiment.csv"
sample_output = output_dir / "bitcoin_tweets_clean_sample.csv"

if full_output.exists():
    full_output.unlink()
if sample_output.exists():
    sample_output.unlink()

analyzer = SentimentIntensityAnalyzer()
bit_pattern = r"(?i)(?<!\w)(?:bitcoin|btc)(?!\w)"


def clean_text(text: str) -> str:
    if text is None:
        return ""
    text = str(text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"#(\w+)", r" \1 ", text)
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


print("Loading the latest BTC tweet scrape and preparing the sentiment dataset...")

processed_rows = 0
sample_rows = []
first_chunk = True

for chunk in pd.read_csv(
    input_file,
    chunksize=200_000,
    encoding="utf-8-sig",
    engine="c",
    on_bad_lines="skip",
    quoting=csv.QUOTE_MINIMAL,
    low_memory=False,
):
    chunk = chunk.copy()
    chunk["date"] = pd.to_datetime(chunk["date"], errors="coerce")
    chunk = chunk.dropna(subset=["date", "text"]).copy()
    chunk["text"] = chunk["text"].astype(str).str.strip()
    chunk = chunk[chunk["text"].ne("")].copy()
    chunk["text_clean"] = chunk["text"].apply(clean_text)
    chunk = chunk[chunk["text_clean"].str.contains(bit_pattern, na=False, regex=True)].copy()
    chunk["user_followers"] = pd.to_numeric(chunk["user_followers"], errors="coerce")
    chunk = chunk.drop_duplicates(subset=["date", "text"]).copy()
    chunk = chunk.sort_values("date").reset_index(drop=True)

    if chunk.empty:
        continue

    scores = chunk["text_clean"].apply(analyzer.polarity_scores)
    chunk["sentiment_positive"] = [x["pos"] for x in scores]
    chunk["sentiment_negative"] = [x["neg"] for x in scores]
    chunk["sentiment_neutral"] = [x["neu"] for x in scores]
    chunk["sentiment_compound"] = [x["compound"] for x in scores]
    chunk["sentiment_label"] = np.where(
        chunk["sentiment_compound"] > 0.05,
        "positive",
        np.where(chunk["sentiment_compound"] < -0.05, "negative", "neutral"),
    )

    chunk = chunk[
        [
            "user_name",
            "user_location",
            "user_description",
            "user_created",
            "user_followers",
            "user_friends",
            "user_favourites",
            "user_verified",
            "date",
            "text",
            "hashtags",
            "source",
            "is_retweet",
            "sentiment_positive",
            "sentiment_negative",
            "sentiment_neutral",
            "sentiment_compound",
            "sentiment_label",
        ]
    ]

    chunk.to_csv(full_output, mode="a", header=first_chunk, index=False, encoding="utf-8")
    first_chunk = False

    processed_rows += len(chunk)
    if len(sample_rows) < 100_000:
        remaining = 100_000 - len(sample_rows)
        sample_rows.append(chunk.head(remaining))

if not full_output.exists():
    raise FileNotFoundError(f"No processed tweets were written to {full_output}")

sample_df = pd.concat(sample_rows, ignore_index=True) if sample_rows else pd.DataFrame()
if not sample_df.empty:
    sample_df.to_csv(sample_output, index=False, encoding="utf-8")

print("\n===== CLEANING SUMMARY =====")
print("Processed rows:", processed_rows)
print("Output file:", full_output.resolve())
print("Sample output:", sample_output.resolve())

if sample_df.empty:
    print("No sample rows retained after cleaning.")
else:
    print("Sample cleaned tweets:")
    print(sample_df[["date", "user_followers"]].head(5).to_string(index=False))