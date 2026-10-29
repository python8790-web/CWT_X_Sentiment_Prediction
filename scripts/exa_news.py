from __future__ import annotations

import argparse
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from scripts.news_utils import normalize_news_records, post_json
from src.project_config import RAW_DIR, required_api_key

ENDPOINT = "https://api.exa.ai/search"


def retrieve_news(
    api_key: str,
    query: str = "Bitcoin BTC market news",
    max_results: int = 10,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    if not 1 <= max_results <= 50:
        raise ValueError("max_results must be between 1 and 50.")
    payload: dict[str, Any] = {
        "query": query,
        "category": "news",
        "numResults": max_results,
        "contents": {"text": {"maxCharacters": 1500}},
    }
    if start_date:
        date.fromisoformat(start_date)
        payload["startPublishedDate"] = f"{start_date}T00:00:00.000Z"
    if end_date:
        date.fromisoformat(end_date)
        payload["endPublishedDate"] = f"{end_date}T23:59:59.999Z"
    if start_date and end_date and start_date > end_date:
        raise ValueError("start_date must be on or before end_date.")

    response = post_json(
        ENDPOINT,
        payload,
        {"x-api-key": api_key, "Content-Type": "application/json"},
        "Exa",
    )
    records = []
    for item in response.get("results", []):
        if not isinstance(item, dict):
            continue
        records.append(
            {
                "title": item.get("title"),
                "url": item.get("url"),
                "published_date": item.get("publishedDate"),
                "source": item.get("author"),
                "content": item.get("text") or item.get("summary"),
            }
        )
    return normalize_news_records(records)


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieve supplementary Bitcoin news via Exa.")
    parser.add_argument("--query", default="Bitcoin BTC market news")
    parser.add_argument("--max-results", type=int, default=10)
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    parser.add_argument("--output", default=str(RAW_DIR / "news" / "exa_news.csv"))
    args = parser.parse_args()

    today = datetime.now(timezone.utc).date()
    start_date = args.start_date or (today - timedelta(days=30)).isoformat()
    end_date = args.end_date or today.isoformat()
    records = retrieve_news(
        required_api_key("EXA_API_KEY"),
        args.query,
        args.max_results,
        start_date,
        end_date,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    records.to_csv(output, index=False)
    print(f"Saved {len(records)} validated Exa records to {output.resolve()}")


if __name__ == "__main__":
    main()
