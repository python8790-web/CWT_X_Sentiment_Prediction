# CrowdWisdomTrading X Sentiment Price Prediction Agent

This reproducible research project evaluates whether hourly Bitcoin-related X-post sentiment improves BTC/USDT return forecasts. The supplied historical dataset remains the offline fallback; API integrations are optional and report success only when a response has been validated.

## What is in this repository

- `data/processed/` — hourly price, sentiment, and merged datasets (small core files only).
- `scripts/` — bounded Apify ingestion, post scoring, Tavily/Exa retrieval, and SQLite loader.
- `outputs/` — generated walk-forward predictions, metrics, strategy trades, and charts (ignored by Git).
- `evaluation_report.md`, `sentiment_outlook.md` — current output summaries, regenerate from CSV results.
- `tests/` — API-response normalization and pipeline/strategy tests; no paid API calls.

Large raw tweet exports, scored tweet exports, SQLite databases, and generated outputs are excluded from Git. The supplied local historical data can be used as-is. Rebuild tweet-derived data only from an authorized raw export or fresh validated API data; price history may be regenerated from Binance using `download_historical_prices.py` if needed. Do not overwrite or redistribute source datasets without authorization.

## Setup (Windows PowerShell; Python 3.10+)

```powershell
cd "C:\Users\HP\Downloads\CWT_X_Sentiment_Prediction"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
```

`.env.example` contains placeholders only. Put rotated service credentials in the local `.env`, which is ignored by Git. Never paste credentials into source, documentation, reports, or issue trackers. If a key has been shared, revoke/rotate it before use.

Expected variables:

- `APIFY_API_TOKEN`, `APIFY_ACTOR_ID`
- `TAVILY_API_KEY`
- `EXA_API_KEY`
- Optional `APIFY_START_DATE`, `APIFY_END_DATE`, `APIFY_MAX_POSTS`, and `APIFY_ACTOR_INPUT_JSON`

External integrations are optional; offline model, strategy, charts, report generation, and tests work without API keys.

## Offline processing and evaluation

The existing scored historical tweets and price data are available locally. Run from the project directory:

```powershell
.\.venv\Scripts\Activate.ps1
python aggregate_sentiment.py --input data\processed\bitcoin_tweets_with_sentiment.csv
python merge_data.py
python train_model.py
python evaluate_strategy.py
python create_charts.py
python generate_reports.py
python scripts\load_to_sqlite.py --db data\bitcoin_project_rebuilt.db
python -m pytest -q
```

The aggregator reads large tweet files in chunks. Its hourly median combines chunk medians when the same UTC hour crosses a chunk boundary, so that edge case can be approximate. Do not run `prepare_data.py` unless the authorized raw export is present and you intend to regenerate the large scored tweet files.

## Optional live data

### Apify X posts

The actor interface is actor-specific. Inspect and validate it first; this performs a schema request but does **not** start a paid actor run:

```powershell
python scripts\apify_ingest.py --show-input-schema
```

The default window is the most recent 180 days; the default maximum fetched posts is 500. Check the actor documentation, input schema, date/search field support, quotas, and expected cost. If the generic input builder cannot match the actor's schema, provide actor-specific JSON through `APIFY_ACTOR_INPUT_JSON` or `--input-json path\to\actor-input.json`. A run is never started unless `--run` is explicitly supplied:

```powershell
python scripts\apify_ingest.py --max-posts 500 --run
python scripts\process_apify_posts.py
python aggregate_sentiment.py --input data\processed\bitcoin_tweets_with_sentiment.csv data\processed\apify_tweets_with_sentiment.csv
python merge_data.py
python train_model.py
python evaluate_strategy.py
python generate_reports.py
```

Apify outputs are schema-validated UTC post records under `data\raw\`. Unsupported actor fields are not fabricated. If the actor lacks a server-side result limit, the script refuses to start it.

### Tavily and Exa news

Each command defaults to the previous 30 days and a bounded result count. News remains supplemental commentary, not a model feature:

```powershell
python scripts\tavily_news.py --max-results 10
python scripts\exa_news.py --max-results 10
python generate_reports.py
```

Use `--start-date YYYY-MM-DD`, `--end-date YYYY-MM-DD`, and (for Tavily) `--search-depth basic|advanced` to configure the date range/depth. API plans impose usage limits and may charge per request/result; check provider pricing and quotas before running. Missing credentials and provider failures are reported without claiming successful retrieval.

## Outputs and interpretation

- `data\processed\bitcoin_merged_hourly.csv` — hourly BTC/USDT prices, sentiment features, returns, and n+0..n+5 labels.
- `outputs\walk_forward_metrics.csv`, `outputs\walk_forward_predictions.csv` — zero-return baseline and chronological out-of-sample results.
- `outputs\strategy_metrics.csv`, `outputs\strategy_hourly_returns.csv` — non-overlapping, positive-horizon strategy evaluation with turnover costs.
- `outputs\charts\` — price/sentiment, volume, sentiment/return, and out-of-sample MAE charts.
- `evaluation_report.md`, `sentiment_outlook.md` — generated from current local outputs, with live-data status explicitly distinguished.
- `data\bitcoin_project_rebuilt.db` — SQLite copy of processed tables; use a different file name to preserve an existing database.

n+0 is the completed hourly return ending at its timestamp. It is included only as a known-return diagnostic, not as a forecast or tradable result. For n+1 through n+5, the target is `Close[t+h] / Close[t] - 1`. The hourly timestamp labels the candle open; signals are available at that candle's close (`hour + 1h`), and each target is realized at the target candle's close. Walk-forward tests are chronological, use a horizon-sized gap, and fit preprocessing inside each training fold. Strategy returns are compounded at non-overlapping holding periods and include 10 bps per unit of position turnover; no slippage, funding, taxes, or execution latency are modeled. Past results do not establish profitability or constitute investment advice.

## Data and coverage limitations

The included local sample covers 2025-03-11 through 2026-02-28 UTC. It does not cover the latest 180-day period ending now. Historical tweet sampling may be incomplete, and no real-time status is inferred from old sample data. The assessment report records current data coverage, actual integration outputs, model metrics, strategy assumptions, and limitations.
