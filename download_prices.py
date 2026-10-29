
import yfinance as yf
import pandas as pd
from pathlib import Path

output_dir = Path("data/processed")
output_dir.mkdir(parents=True, exist_ok=True)

output_file = output_dir / "bitcoin_hourly_prices.csv"

print("Downloading recent hourly Bitcoin prices...")

try:
    btc = yf.download(
        "BTC-USD",
        period="60d",
        interval="1h",
        auto_adjust=True,
        progress=False,
    )

    if btc.empty:
        raise RuntimeError("Yahoo Finance returned no price data.")

    # Handle the MultiIndex columns returned by some yfinance versions.
    if isinstance(btc.columns, pd.MultiIndex):
        btc.columns = btc.columns.get_level_values(0)

    btc = btc.reset_index()

    time_column = "Datetime" if "Datetime" in btc.columns else "Date"

    btc = btc.rename(columns={time_column: "hour"})
    btc["hour"] = pd.to_datetime(btc["hour"], errors="coerce")

    # Remove timezone information so timestamps can be merged consistently.
    if btc["hour"].dt.tz is not None:
        btc["hour"] = btc["hour"].dt.tz_convert("UTC").dt.tz_localize(None)

    btc = btc.dropna(subset=["hour", "Close"])
    btc = btc.sort_values("hour")

    btc.to_csv(output_file, index=False)

    print("\n===== BITCOIN PRICE SUMMARY =====")
    print("Price records:", len(btc))
    print("First timestamp:", btc["hour"].min())
    print("Last timestamp:", btc["hour"].max())
    print("\nSample prices:")
    print(btc[["hour", "Close"]].head().to_string(index=False))
    print("\nSaved to:", output_file.resolve())

except Exception as error:
    print("\nPrice download failed:", repr(error))
    print("We can troubleshoot the download if necessary.")