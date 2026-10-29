import pandas as pd
from pathlib import Path

file_path = Path("data/processed/bitcoin_merged_hourly.csv")

df = pd.read_csv(file_path, parse_dates=["hour"])

print("===== DATA VALIDATION =====")
print("Rows:", len(df))
print("Columns:", len(df.columns))
print("Duplicate timestamps:", df["hour"].duplicated().sum())
print("Missing timestamps:", df["hour"].isna().sum())

print("\n===== COLUMN TYPES =====")
print(df.dtypes.to_string())

print("\n===== SENTIMENT FEATURE CHECK =====")
sentiment_columns = [
    "tweet_count",
    "average_sentiment",
    "median_sentiment",
    "positive_share",
    "negative_share",
    "average_followers",
]

for column in sentiment_columns:
    print(f"\n{column}:")
    print("  Missing:", df[column].isna().sum())
    print("  Unique values:", df[column].nunique())
    print("  Minimum:", df[column].min())
    print("  Maximum:", df[column].max())

print("\n===== TWEET COUNT CHECK =====")
print("Rows with tweets:", (df["tweet_count"] > 0).sum())
print("Rows with zero tweets:", (df["tweet_count"] == 0).sum())
print("Rows with negative tweet counts:", (df["tweet_count"] < 0).sum())

print("\n===== TARGET CHECK =====")
for horizon in range(1, 7):
    column = f"future_return_{horizon}h"
    print(f"{column}: missing={df[column].isna().sum()}")

print("\n===== SAMPLE ROWS =====")
print(
    df[
        [
            "hour",
            "Close",
            "tweet_count",
            "average_sentiment",
            "return_1h",
            "future_return_1h",
        ]
    ].head(10).to_string(index=False)
)

print("\n===== HOURS WITH TWEETS =====")
print(
    df.loc[
        df["tweet_count"] > 0,
        [
            "hour",
            "tweet_count",
            "average_sentiment",
            "positive_share",
            "negative_share",
        ],
    ].head(10).to_string(index=False)
)