from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
import shutil
from urllib.parse import urlparse

import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "processed"
OUTPUT_DIR = ROOT / "outputs"
CHART_NAMES = (
    "bitcoin_price_and_sentiment.png",
    "tweet_volume_over_time.png",
    "positive_negative_sentiment.png",
    "sentiment_vs_subsequent_returns.png",
    "prediction_vs_actual_return.png",
    "walk_forward_model_comparison.png",
)


def read_csv(path: Path) -> pd.DataFrame | None:
    return pd.read_csv(path) if path.exists() else None


def timestamp_coverage(
    path: Path, candidates: tuple[str, ...], chunksize: int = 250_000
) -> dict[str, object] | None:
    if not path.exists():
        return None
    columns = pd.read_csv(path, nrows=0, encoding="utf-8-sig").columns
    date_column = next((name for name in candidates if name in columns), None)
    if date_column is None:
        raise ValueError(f"No supported timestamp column found in {path.name}.")

    rows = 0
    invalid = 0
    earliest = None
    latest = None
    for chunk in pd.read_csv(
        path,
        usecols=[date_column],
        chunksize=chunksize,
        encoding="utf-8-sig",
        low_memory=False,
        on_bad_lines="skip",
    ):
        timestamps = pd.to_datetime(chunk[date_column], utc=True, errors="coerce")
        rows += len(timestamps)
        invalid += int(timestamps.isna().sum())
        valid = timestamps.dropna()
        if not valid.empty:
            chunk_min, chunk_max = valid.min(), valid.max()
            earliest = chunk_min if earliest is None or chunk_min < earliest else earliest
            latest = chunk_max if latest is None or chunk_max > latest else latest
    return {
        "rows": rows,
        "invalid": invalid,
        "earliest": earliest,
        "latest": latest,
        "column": date_column,
    }


def fmt(value: object, digits: int = 6) -> str:
    if pd.isna(value):
        return "—"
    return f"{float(value):.{digits}f}"


def copy_chart_assets() -> str:
    source_dir = OUTPUT_DIR / "charts"
    destination_dir = ROOT / "reports" / "charts"
    destination_dir.mkdir(parents=True, exist_ok=True)
    copied = []
    for name in CHART_NAMES:
        source = source_dir / name
        if source.exists():
            shutil.copy2(source, destination_dir / name)
            copied.append(name)
    if not copied:
        return "Charts have not been generated. Run `python create_charts.py` first."
    return "\n\n".join(
        f"![{Path(name).stem.replace('_', ' ').title()}](reports/charts/{name})"
        for name in copied
    )


def api_status(path: Path, label: str) -> str:
    data = read_csv(path)
    if data is not None and not data.empty and "retrieved_at" in data.columns:
        return f"Retrieved and normalized output present ({len(data)} records; {label})."
    return f"Not verified in this run; no validated output file found ({label})."


def make_report() -> str:
    merged = pd.read_csv(DATA_DIR / "bitcoin_merged_hourly.csv", parse_dates=["hour"])
    metrics = pd.read_csv(OUTPUT_DIR / "walk_forward_metrics.csv")
    strategy = pd.read_csv(OUTPUT_DIR / "strategy_metrics.csv")
    sentiment = pd.read_csv(DATA_DIR / "hourly_sentiment.csv", parse_dates=["hour"])
    now = datetime.now(timezone.utc)
    earliest = merged["hour"].min()
    latest = merged["hour"].max()
    tweet_file = ROOT / "data" / "bitcoin_tweets_latest.csv"
    scored_tweet_file = DATA_DIR / "bitcoin_tweets_with_sentiment.csv"
    raw_coverage = timestamp_coverage(tweet_file, ("date", "created_at"))
    scored_coverage = timestamp_coverage(
        scored_tweet_file, ("date", "created_at")
    )
    tweet_size = (
        f"{tweet_file.stat().st_size / (1024**3):.2f} GiB"
        if tweet_file.exists()
        else "not present"
    )
    required_start = (now - timedelta(days=180)).date()
    scored_coverage_text = (
        f"{scored_coverage['rows']:,} rows; "
        f"{scored_coverage['earliest']} to {scored_coverage['latest']} UTC; "
        f"{scored_coverage['invalid']:,} invalid timestamps."
        if scored_coverage
        else "scored tweet file not present; exact tweet coverage unavailable."
    )
    raw_coverage_text = (
        f"{raw_coverage['rows']:,} rows; "
        f"{raw_coverage['earliest']} to {raw_coverage['latest']} UTC; "
        f"{raw_coverage['invalid']:,} invalid timestamps."
        if raw_coverage
        else "raw tweet file not present."
    )
    latest_scored = scored_coverage["latest"] if scored_coverage else None
    recent_coverage = (
        latest_scored is not None
        and latest_scored >= pd.Timestamp(required_start, tz="UTC")
    )
    live_posts = ROOT / "data" / "raw" / "x_posts_apify.csv"
    live_post_data = read_csv(live_posts)
    apify_status = (
        f"Validated Apify output file present ({len(live_post_data)} records)."
        if live_post_data is not None and not live_post_data.empty
        else "Implemented; not verified because no validated post output is present."
    )
    tv_path = ROOT / "data" / "raw" / "news" / "tavily_news.csv"
    exa_path = ROOT / "data" / "raw" / "news" / "exa_news.csv"
    news_status = (
        f"- Tavily: {api_status(tv_path, 'news retrieval is supplementary only')}\n"
        f"- Exa: {api_status(exa_path, 'news retrieval is supplementary only')}"
    )
    chart_assets = copy_chart_assets()
    metric_lines = [
        "| Horizon | Model | MAE | RMSE | Directional accuracy | N | Status |",
        "|---:|---|---:|---:|---:|---:|---|",
    ]
    for row in metrics.sort_values(["horizon_hours", "model"]).itertuples():
        metric_lines.append(
            f"| {row.horizon_hours} | {row.model} | {fmt(row.MAE)} | "
            f"{fmt(row.RMSE)} | {fmt(row.directional_accuracy, 4)} | "
            f"{row.observations} | {row.status} |"
        )
    strategy_lines = [
        "| Model | Horizon (hours) | Trades | Net return | Annualized Sharpe | "
        "Annualized Sortino | Max drawdown | Cost (bps) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in strategy.sort_values(["horizon_hours", "model"]).itertuples():
        strategy_lines.append(
            f"| {row.model} | {row.horizon_hours} | {row.observations} | "
            f"{row.total_net_return:.2%} | {fmt(row.annualized_sharpe, 3)} | "
            f"{fmt(row.annualized_sortino, 3)} | {row.max_drawdown:.2%} | "
            f"{row.transaction_cost_bps} |"
        )
    top_level = (
        f"- Raw X-post export: {raw_coverage_text} File size {tweet_size}; "
        "excluded from Git.\n"
        f"- Scored X-post data used for sentiment: {scored_coverage_text}\n"
        f"- Latest-180-day requirement: window begins {required_start} UTC; "
        f"{'scored data reaches this window' if recent_coverage else 'scored data does not reach this window'}.\n"
        "- Prices: local hourly BTC/USDT OHLCV file (documented source: Binance).\n"
        f"- Merged modeling sample: {len(merged):,} hourly rows, "
        f"{earliest} through {latest} UTC.\n"
        f"- Aggregated sentiment timestamps: {sentiment['hour'].min()} through "
        f"{sentiment['hour'].max()} UTC.\n"
        "- SQLite default path: `data/bitcoin_project.db`; the existing local database "
        "was not modified or refreshed.\n"
        f"- Apify: {apify_status}\n"
        f"{news_status}"
    )
    return f"""# Evaluation report

Generated: {now.strftime('%Y-%m-%d %H:%M UTC')}

## Objective and architecture

This project tests whether hourly X-post sentiment improves BTC/USDT return forecasts. The offline pipeline reads the supplied historical data, normalizes/aggregates sentiment, joins it to hourly prices, creates forward return targets, performs chronological walk-forward validation, evaluates a simplified non-overlapping strategy, generates charts, and loads tabular outputs into SQLite. Tavily and Exa news are supplementary context and are not model features.

## Data sources, coverage, and integration status

{top_level}

The exact scored-post coverage is reported above from a chunked scan of the timestamp column. The raw export may contain parseable outlier dates; the scored range is the relevant sentiment-processing coverage. No live collection is claimed unless its validated output exists as noted above. Credentials must be configured locally; API keys are never written to reports.

## Cleaning, features, and targets

Tweet timestamps are converted to UTC; malformed timestamps/text/sentiment are dropped; duplicate IDs (or timestamp/text when IDs are absent) are removed within each processing chunk. Sentiment is computed with VADER, then aggregated hourly into count, mean/median sentiment, positive/negative shares, and follower average when available. Hours without tweets receive zero counts/shares/mean sentiment. Hourly prices are sorted and checked for duplicate timestamps. News data is kept outside the predictive feature set.

Price features are 1h/3h/6h historical returns and log volume. Sentiment features are tweet count, mean/median sentiment, positive/negative shares, and a has-tweets indicator. The target for n+h (h=1..5) is `Close[t+h] / Close[t] - 1`. Binance candle timestamps label the candle open; the signal is available at the current candle close (`hour + 1h`), and the target close occurs at `hour + h + 1h`. Prediction output records both candle labels and these physical signal/target times. The n+0 target is the already completed 1-hour return ending at the current candle close. It is emitted only as a known-current-return diagnostic, not trained, forecast, or included in trading metrics.

## Model and walk-forward design

Price-only and price-plus-sentiment Ridge regression pipelines use median imputation and standardization fitted only on each training fold. Validation is chronological, not shuffled: five expanding folds, 336-hour test windows, and a gap equal to the forward horizon. The zero-return baseline is included for each genuine forward horizon. MAE/RMSE are in decimal return units; directional accuracy is the proportion of non-zero model predictions with the correct sign (undefined for the zero-return baseline).

## Out-of-sample prediction metrics

{chr(10).join(metric_lines)}

Compare price-only with price-plus-sentiment on the same horizons. Small differences are not evidence of a robust edge; no improvement is claimed unless the measured results support it.

## Trading simulation

Signals are read at the current candle close and held until the target candle close, exactly h hours later. Returns are selected without overlapping holding periods; n+0 is excluded. Positions are long for positive predictions, short for negative predictions, and flat for zero predictions. Costs are 10 bps per unit of position turnover, including initial entry, changes, exits across gaps, and final liquidation. Net returns are compounded; annualization assumes 365×24/horizon holding periods per year. Short returns use a linear notional approximation. Slippage, funding, taxes, market impact, and execution delay are not modeled.

{chr(10).join(strategy_lines)}

These are historical simulations, not investment advice or evidence of executable profitability. Sharpe/Sortino ratios are highly sample-dependent.

## Charts

The following assessment chart snapshots are copied from the latest generated charts:

{chart_assets}

## Limitations and known issues

- Historical data ends at {latest}; no current latest-180-day post dataset has been verified.
- The raw tweet export and generated outputs are intentionally excluded from Git due to size. Reproducers need the licensed/authorized source data or validated API output.
- Duplicate removal is performed within each processing chunk; repeated records crossing chunk boundaries remain a data-quality risk.
- The selected Apify actor is actor-specific. Inspect its documented input schema and limit/costs; generic search/date input is not assumed.
- API endpoints, credentials, quotas, actor schemas, and service availability were not tested unless a validated output is reported above.
- VADER is a lexicon method and can misread sarcasm, financial context, and non-English text. Sample selection and platform coverage may bias sentiment.
- Model and strategy results do not account for market microstructure, slippage, funding, or operational constraints. Hourly candles may not represent achievable execution prices.
- Hourly aggregation across chunk boundaries can make medians approximate when the same hour spans multiple input chunks.

## Reproduction (Windows PowerShell)

From the project directory:

```powershell
python -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
python -m pip install -r requirements.txt
if (-not (Test-Path .env)) {{ Copy-Item .env.example .env }}
python aggregate_sentiment.py --input data/processed/bitcoin_tweets_with_sentiment.csv
python merge_data.py
python train_model.py
python evaluate_strategy.py
python create_charts.py
python generate_reports.py
python scripts\\load_to_sqlite.py --db data\\bitcoin_project_rebuilt.db
python -m pytest -q
```

For downloaded posts, first run `python scripts\\apify_ingest.py` to inspect actor input and costs. Start a bounded job only after review by passing `--run`; then run `python scripts\\process_apify_posts.py` and aggregate both scored files by passing both paths after `--input`. Retrieve optional news with `python scripts\\tavily_news.py` and `python scripts\\exa_news.py`. Re-running the SQLite loader against an existing database requires explicit `--replace-existing`; `--reset` rebuilds tables.
"""


def make_outlook() -> str:
    merged = pd.read_csv(DATA_DIR / "bitcoin_merged_hourly.csv", parse_dates=["hour"])
    sentiment = pd.read_csv(DATA_DIR / "hourly_sentiment.csv", parse_dates=["hour"])
    now = datetime.now(timezone.utc)
    mean_sentiment = sentiment["average_sentiment"].mean()
    recent = sentiment[sentiment["tweet_count"] > 0].tail(24)
    recent_mean = recent["average_sentiment"].mean() if not recent.empty else float("nan")
    total_posts = int(sentiment["tweet_count"].sum())
    active_hours = int((sentiment["tweet_count"] > 0).sum())
    news_rows = []
    for name in ("tavily", "exa"):
        path = ROOT / "data" / "raw" / "news" / f"{name}_news.csv"
        data = read_csv(path)
        if data is not None and not data.empty:
            for row in data.head(3).itertuples():
                host = urlparse(str(row.url)).netloc
                news_rows.append(
                    f"- [{row.title or host}]({row.url}) ({name}; retrieved "
                    f"{row.retrieved_at}; publication date: {row.published_date or 'not supplied'})"
                )
    news_section = (
        "\n".join(news_rows)
        if news_rows
        else "No validated live Tavily or Exa news output is present; no articles are cited."
    )
    return f"""# Historical Bitcoin sentiment outlook

**Generated:** {now.strftime('%Y-%m-%d %H:%M UTC')}
**Classification:** historical sample only; not a current market outlook.

## Sample snapshot

- Price/sentiment alignment: {merged['hour'].min()} through {merged['hour'].max()} UTC.
- Hourly sentiment source coverage: {sentiment['hour'].min()} through {sentiment['hour'].max()} UTC.
- Posts represented by hourly aggregates: {total_posts:,}; active tweet-hours: {active_hours:,}.
- Mean sentiment across hourly aggregate records: {mean_sentiment:.4f}.
- Mean sentiment for the latest 24 active sentiment hours in the local sample: {recent_mean:.4f}.
- Latest local BTC close: {merged.iloc[-1]['Close']:.2f} USDT at {merged.iloc[-1]['hour']} UTC.

This summarizes only local historical data. It is not a real-time signal, and the sample ends before the latest 180-day requirement.

## Supplementary retrieved news

{news_section}

News is contextual only and is not included in the prediction model. An absent section of validated results is not interpreted as evidence that no news exists.

## Interpretation and caveats

Hourly sample sentiment is descriptive and potentially affected by source sampling, missing posts, duplicate handling, language, and VADER limitations. It does not establish a causal or predictive relationship with BTC returns. Walk-forward metrics and strategy assumptions are in [evaluation_report.md](./evaluation_report.md). This outlook is not investment advice.
"""


def main() -> None:
    (ROOT / "evaluation_report.md").write_text(make_report(), encoding="utf-8")
    (ROOT / "sentiment_outlook.md").write_text(make_outlook(), encoding="utf-8")
    print("Generated evaluation_report.md and sentiment_outlook.md from current outputs.")


if __name__ == "__main__":
    main()
