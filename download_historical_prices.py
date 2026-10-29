
import time
import requests
import pandas as pd
from pathlib import Path

BASE_URL = "https://api.binance.com/api/v3/klines"

# Retrieve prices for the period covered by the sentiment dataset.
START = pd.Timestamp("2025-03-11 00:00:00", tz="UTC")
END = pd.Timestamp("2026-03-01 00:00:00", tz="UTC")

start_ms = int(START.timestamp() * 1000)
end_ms = int(END.timestamp() * 1000) - 1

output_dir = Path("data/processed")
output_dir.mkdir(parents=True, exist_ok=True)

output_file = output_dir / "bitcoin_historical_hourly.csv"

all_rows = []
session = requests.Session()

print("Downloading historical BTC/USDT hourly prices...")
print("Start:", START)
print("End:", END)

while start_ms <= end_ms:
    params = {
        "symbol": "BTCUSDT",
        "interval": "1h",
        "startTime": start_ms,
        "endTime": end_ms,
        "limit": 1000,
    }

    response = session.get(
        BASE_URL,
        params=params,
        timeout=30,
    )
    response.raise_for_status()

    rows = response.json()

    if not rows:
        print("No more candles returned.")
        break

    all_rows.extend(rows)

    last_open_ms = int(rows[-1][0])
    next_start_ms = last_open_ms + 60 * 60 * 1000

    print(
        f"Downloaded {len(all_rows):,} candles; "
        f"latest: {pd.to_datetime(last_open_ms, unit='ms', utc=True)}"
    )

    if next_start_ms <= start_ms:
        raise RuntimeError("Pagination did not advance.")

    start_ms = next_start_ms
    time.sleep(0.2)

if not all_rows:
    raise RuntimeError(
        "No price data downloaded. Check your internet connection "
        "or whether the API is accessible from your network."
    )

columns = [
    "open_time",
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "close_time",
    "quote_volume",
    "trade_count",
    "taker_buy_volume",
    "taker_buy_quote_volume",
    "ignore",
]

prices = pd.DataFrame(all_rows, columns=columns)

prices["hour"] = pd.to_datetime(
    pd.to_numeric(prices["open_time"]),
    unit="ms",
    utc=True,
).dt.tz_localize(None)

for column in ["Open", "High", "Low", "Close", "Volume"]:
    prices[column] = pd.to_numeric(prices[column], errors="coerce")

prices = (
    prices[["hour", "Open", "High", "Low", "Close", "Volume"]]
    .dropna(subset=["hour", "Close"])
    .drop_duplicates(subset=["hour"])
    .sort_values("hour")
)

prices.to_csv(output_file, index=False)

print("\n===== HISTORICAL PRICE SUMMARY =====")
print("Candles:", len(prices))
print("First hour:", prices["hour"].min())
print("Last hour:", prices["hour"].max())
print("Missing closing prices:", prices["Close"].isna().sum())
print("Duplicate hours:", prices["hour"].duplicated().sum())
print("Quote currency: USDT")
print("Source: Binance spot BTCUSDT")
print("Saved to:", output_file.resolve())
