
import pandas as pd
from pathlib import Path
import csv

file_path = Path("data/bitcoin_tweets_latest.csv")

print("Checking dataset quality in chunks...")
print("This may take several minutes for a large CSV.\n")

total_rows = 0
valid_dates = 0
invalid_dates = 0
missing_text = 0
earliest_date = None
latest_date = None
sample_texts = []

try:
    for chunk_number, chunk in enumerate(
        pd.read_csv(
            file_path,
            chunksize=100_000,
            encoding="utf-8",
            engine="c",
            on_bad_lines="warn",
            quoting=csv.QUOTE_MINIMAL,
            low_memory=True,
        ),
        start=1,
    ):
        total_rows += len(chunk)

        dates = pd.to_datetime(chunk["date"], errors="coerce")
        valid = dates.dropna()

        valid_dates += len(valid)
        invalid_dates += int(dates.isna().sum())
        missing_text += int(chunk["text"].isna().sum())

        if not valid.empty:
            chunk_min = valid.min()
            chunk_max = valid.max()

            if earliest_date is None or chunk_min < earliest_date:
                earliest_date = chunk_min

            if latest_date is None or chunk_max > latest_date:
                latest_date = chunk_max

        if len(sample_texts) < 5:
            texts = chunk["text"].dropna().astype(str).tolist()
            sample_texts.extend(texts[: 5 - len(sample_texts)])

        print(
            f"Processed chunk {chunk_number}: "
            f"{total_rows:,} rows so far"
        )

    print("\n===== DATASET QUALITY SUMMARY =====")
    print(f"Rows successfully loaded: {total_rows:,}")
    print(f"Rows with valid dates: {valid_dates:,}")
    print(f"Rows with invalid dates: {invalid_dates:,}")
    print(f"Rows with missing tweet text: {missing_text:,}")
    print(f"Earliest date: {earliest_date}")
    print(f"Latest date: {latest_date}")

    print("\nSample tweet texts:")
    for i, text in enumerate(sample_texts, start=1):
        print(f"{i}. {text[:250]}")

except Exception as error:
    print("\nDataset check failed:", repr(error))
    print("Do not use this dataset for training yet.")
