import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

DATA_FILE = Path("data/processed/bitcoin_merged_hourly.csv")
PREDICTIONS_FILE = Path("outputs/walk_forward_predictions.csv")
METRICS_FILE = Path("outputs/walk_forward_metrics.csv")
OUTPUT_DIR = Path("outputs/charts")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


df = pd.read_csv(DATA_FILE, parse_dates=["hour"]).sort_values("hour").reset_index(drop=True)
preds = pd.read_csv(PREDICTIONS_FILE)

# Chart 1: Bitcoin price and hourly sentiment
fig, ax1 = plt.subplots(figsize=(14, 6))
ax1.plot(df["hour"], df["Close"], color="tab:blue", linewidth=1.6, label="BTC close price (USDT)")
ax1.set_xlabel("Date")
ax1.set_ylabel("BTC price (USDT)")
ax1.tick_params(axis="x", rotation=30)

ax2 = ax1.twinx()
mask = df["tweet_count"] > 0
ax2.plot(df.loc[mask, "hour"], df.loc[mask, "average_sentiment"], color="tab:orange", linewidth=1.2, alpha=0.9, label="Avg. hourly VADER sentiment")
ax2.set_ylabel("VADER compound sentiment")

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
ax1.set_title("Bitcoin price and observed X sentiment")
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "bitcoin_price_and_sentiment.png", dpi=160)
plt.close(fig)

# Chart 2: Tweet volume over time
active = df[df["tweet_count"] > 0].copy()
fig, ax = plt.subplots(figsize=(14, 5))
ax.bar(active["hour"], active["tweet_count"], width=0.12, color="tab:green", alpha=0.75)
ax.set_title("Hourly Bitcoin-related tweet volume")
ax.set_xlabel("Date")
ax.set_ylabel("Tweets per hour")
ax.tick_params(axis="x", rotation=30)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "tweet_volume_over_time.png", dpi=160)
plt.close(fig)

# Chart 3: Positive vs negative sentiment
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(df["hour"], df["positive_share"], label="Positive share", color="tab:green")
ax.plot(df["hour"], df["negative_share"], label="Negative share", color="tab:red")
ax.set_title("Positive and negative sentiment shares")
ax.set_xlabel("Date")
ax.set_ylabel("Share of hourly tweets")
ax.legend()
ax.tick_params(axis="x", rotation=30)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "positive_negative_sentiment.png", dpi=160)
plt.close(fig)

# Chart 4: Sentiment vs subsequent returns
scatter = df[df["tweet_count"] > 0].copy()
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(scatter["average_sentiment"], scatter["future_return_1h"] * 100, alpha=0.6, s=18)
ax.axhline(0, color="black", linewidth=1, linestyle="--")
ax.axvline(0, color="black", linewidth=1, linestyle="--")
ax.set_title("Observed sentiment vs next-hour return")
ax.set_xlabel("Average VADER compound sentiment")
ax.set_ylabel("Next-hour return (%)")
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "sentiment_vs_subsequent_returns.png", dpi=160)
plt.close(fig)

# Chart 5: Prediction vs actual
predictions_one_hour = preds[preds["horizon_hours"] == 1].copy()
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(predictions_one_hour["actual_return"] * 100, predictions_one_hour["predicted_return"] * 100, alpha=0.5)
ax.axline((0, 0), slope=1, color="black", linestyle="--", linewidth=1)
ax.set_title("Prediction vs actual return (1-hour horizon)")
ax.set_xlabel("Actual return (%)")
ax.set_ylabel("Predicted return (%)")
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "prediction_vs_actual_return.png", dpi=160)
plt.close(fig)

# Chart 6: Walk-forward model comparison
summary = pd.read_csv(METRICS_FILE)
summary = summary[
    summary["model"].isin(["price_only", "price_plus_sentiment"])
    & (summary["horizon_hours"] > 0)
]
fig, ax = plt.subplots(figsize=(12, 6))
for model_name, group in summary.groupby("model"):
    group = group.sort_values("horizon_hours")
    ax.plot(group["horizon_hours"], group["MAE"] * 100, marker="o", label=model_name)
ax.set_title("Out-of-sample MAE by forecast horizon")
ax.set_xlabel("Horizon (hours)")
ax.set_ylabel("Mean absolute error (return percentage points)")
ax.legend()
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "walk_forward_model_comparison.png", dpi=160)
plt.close(fig)

print("Charts saved to:", OUTPUT_DIR.resolve())
print("1. bitcoin_price_and_sentiment.png")
print("2. tweet_volume_over_time.png")
print("3. positive_negative_sentiment.png")
print("4. sentiment_vs_subsequent_returns.png")
print("5. prediction_vs_actual_return.png")
print("6. walk_forward_model_comparison.png")