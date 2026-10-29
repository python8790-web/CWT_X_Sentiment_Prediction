
import pandas as pd
from pathlib import Path
import csv

file_path = Path("data/bitcoin_tweets_latest.csv")

print("Testing CSV parsing methods...\n")

for method in ["c", "python"]:
    print("=" * 60)
    print(f"Parser: {method}")

    try:
        df = pd.read_csv(
            file_path,
            engine=method,
            nrows=5,
            encoding="utf-8",
            on_bad_lines="warn",
            quoting=csv.QUOTE_MINIMAL,
        )

        print("Number of columns:", len(df.columns))
        print("Column names:", df.columns.tolist())

        print("\nSelected fields:")
        print(
            df[["user_name", "user_followers", "date", "text"]]
            .head(5)
            .to_string(index=False)
        )

        print("\nDate values:")
        print(df["date"].tolist())

    except Exception as error:
        print("Parser failed:", repr(error))

print("\nDiagnostic complete.")
