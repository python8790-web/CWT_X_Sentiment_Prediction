
import pandas as pd
from pathlib import Path
import csv

file_path = Path("data/bitcoin_tweets_latest.csv")

print("Inspecting dataset columns using a small sample...\n")

try:
    df = pd.read_csv(
        file_path,
        nrows=10_000,
        encoding="utf-8",
        engine="c",
        on_bad_lines="skip",
        quoting=csv.QUOTE_MINIMAL,
        low_memory=True,
    )

    print("COLUMN NAMES")
    print("-" * 50)

    for column in df.columns:
        print(f"{column}: {df[column].dtype}")

    print("\nMISSING VALUES")
    print("-" * 50)
    print(df.isna().sum().to_string())

    print("\nSAMPLE DATA")
    print("-" * 50)

    available_columns = [
        column
        for column in [
            "user_name",
            "user_followers",
            "user_verified",
            "date",
            "text",
            "hashtags",
            "is_retweet",
        ]
        if column in df.columns
    ]

    print(df[available_columns].head(5).to_string(index=False))

except Exception as error:
    print("Inspection failed:", repr(error))