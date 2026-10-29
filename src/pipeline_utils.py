from __future__ import annotations

import pandas as pd


def normalize_utc_timestamps(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, utc=True, errors="coerce")


def aggregate_sentiment_posts(posts: pd.DataFrame) -> pd.DataFrame:
    required = {"date", "text", "sentiment_compound"}
    missing = required.difference(posts.columns)
    if missing:
        raise ValueError(f"Missing sentiment columns: {', '.join(sorted(missing))}")

    frame = posts.copy()
    frame["date"] = normalize_utc_timestamps(frame["date"])
    frame["sentiment_compound"] = pd.to_numeric(
        frame["sentiment_compound"], errors="coerce"
    )
    frame = frame.dropna(subset=["date", "text", "sentiment_compound"])
    frame["text"] = frame["text"].astype(str).str.strip()
    frame = frame[frame["text"].ne("")].copy()

    dedup_key = "tweet_id" if "tweet_id" in frame.columns else None
    if dedup_key:
        has_id = frame[dedup_key].notna() & frame[dedup_key].astype(str).str.strip().ne("")
        with_ids = frame.loc[has_id].drop_duplicates(dedup_key, keep="last")
        without_ids = frame.loc[~has_id].drop_duplicates(["date", "text"], keep="last")
        frame = pd.concat([with_ids, without_ids], ignore_index=True)
    else:
        frame = frame.drop_duplicates(["date", "text"], keep="last")

    frame["hour"] = frame["date"].dt.floor("h")
    frame["is_positive"] = (frame["sentiment_compound"] >= 0.05).astype(int)
    frame["is_negative"] = (frame["sentiment_compound"] <= -0.05).astype(int)
    if "user_followers" not in frame.columns:
        frame["user_followers"] = pd.NA
    frame["user_followers"] = pd.to_numeric(frame["user_followers"], errors="coerce")

    return (
        frame.groupby("hour", as_index=False)
        .agg(
            tweet_count=("text", "count"),
            average_sentiment=("sentiment_compound", "mean"),
            median_sentiment=("sentiment_compound", "median"),
            positive_share=("is_positive", "mean"),
            negative_share=("is_negative", "mean"),
            average_followers=("user_followers", "mean"),
        )
        .sort_values("hour")
        .reset_index(drop=True)
    )


def add_return_targets(prices: pd.DataFrame, max_horizon: int = 5) -> pd.DataFrame:
    required = {"hour", "Close"}
    missing = required.difference(prices.columns)
    if missing:
        raise ValueError(f"Missing price columns: {', '.join(sorted(missing))}")

    frame = prices.copy()
    frame["hour"] = normalize_utc_timestamps(frame["hour"])
    frame["Close"] = pd.to_numeric(frame["Close"], errors="coerce")
    frame = frame.dropna(subset=["hour", "Close"]).sort_values("hour")
    if frame["hour"].duplicated().any():
        raise ValueError("Duplicate price timestamps found.")
    if (frame["Close"] <= 0).any():
        raise ValueError("Closing prices must be positive.")

    frame["return_1h"] = frame["Close"].pct_change()
    frame["return_3h"] = frame["Close"].pct_change(3)
    frame["return_6h"] = frame["Close"].pct_change(6)
    for horizon in range(max_horizon + 1):
        if horizon == 0:
            frame["future_return_0h"] = frame["return_1h"]
        else:
            frame[f"future_return_{horizon}h"] = (
                frame["Close"].shift(-horizon) / frame["Close"] - 1
            )
    return frame.reset_index(drop=True)
