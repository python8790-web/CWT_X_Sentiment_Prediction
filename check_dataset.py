
import csv
from pathlib import Path
from collections import Counter

file_path = Path("data/bitcoin_tweets_latest.csv")

if not file_path.exists():
    print("File not found:", file_path.resolve())
    raise SystemExit(1)

print("File size:", round(file_path.stat().st_size / (1024 * 1024), 2), "MB")

row_counts = Counter()
date_samples = []
text_samples = []
max_rows_to_check = 10000

with open(
    file_path,
    "r",
    encoding="utf-8-sig",
    errors="replace",
    newline=""
) as file:
    reader = csv.reader(file, strict=True)

    try:
        headers = next(reader)
        print("\nHeader column count:", len(headers))
        print("Headers:", headers)

        for row_number, row in enumerate(reader, start=2):
            if row_number > max_rows_to_check + 1:
                break

            row_counts[len(row)] += 1

            # The date column is index 8 in the header.
            if len(row) == len(headers):
                date_samples.append(row[8])
                text_samples.append(row[9])

    except csv.Error as error:
        print("\nCSV parsing stopped at physical line:", reader.line_num)
        print("Parser error:", error)

print("\n--- COLUMN COUNT CHECK ---")
print("Rows grouped by number of fields:", dict(row_counts))

print("\n--- SAMPLE TIMESTAMPS ---")
for value in date_samples[:10]:
    print(value)

print("\n--- SAMPLE TWEET TEXT ---")
for value in text_samples[:3]:
    print(repr(value[:250]))

print("\nInspection complete.")
