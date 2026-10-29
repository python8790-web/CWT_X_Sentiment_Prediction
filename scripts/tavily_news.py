from __future__ import annotations

import argparse
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from scripts.news_utils import normalize_news_records, post_json
from src.project_config import RAW_DIR, required_api_key

ENDPOINT = "https://api.tavily.com/search"


def retrieve_news(
    api_key: str,
    query: str = "Bitcoin BTC market",
    search_depth: str = "basic",
    max_results: int = 10,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    if search_depth not in {"basic", "advanced"}:
        raise ValueError("search_depth must be 'basic' or 'advanced'.")
    if not 1 <= max_results <= 20:
        raise ValueError("max_results must be between 1 and 20.")
    payload = {
        "query": query,
        "topic": "news",
        "search_depth": search_depth,
        "max_results": max_results,
        "include_answer": False,
        "include_raw_content": False,
    }
    if start_date:
        date.fromisoformat(start_date)
        payload["start_date"] = start_date
    if end_date:
        date.fromisoformat(end_date)
        payload["end_date"] = end_date
    if start_date and end_date and start_date > end_date:
        raise ValueError("start_date must be on or before end_date.")

    response = post_json(
        ENDPOINT,
        {"api_key": api_key, **payload},
        {"Content-Type": "application/json"},
        "Tavily",
    )
    records = []
    for item in response.get("results", []):
        if not isinstance(item, dict):
            continue
        records.append(
            {
                "title": item.get("title"),
                "url": item.get("url"),
                "published_date": item.get("published_date"),
                "source": item.get("source"),
                "content": item.get("content"),
            }
        )
    return normalize_news_records(records)


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieve recent Bitcoin news via Tavily.")
    parser.add_argument("--query", default="Bitcoin BTC market")
    parser.add_argument("--search-depth", choices=["basic", "advanced"], default="basic")
    parser.add_argument("--max-results", type=int, default=10)
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    parser.add_argument("--output", default=str(RAW_DIR / "news" / "tavily_news.csv"))
    args = parser.parse_args()

    today = datetime.now(timezone.utc).date()
    start_date = args.start_date or (today - timedelta(days=30)).isoformat()
    end_date = args.end_date or today.isoformat()
    records = retrieve_news(
        required_api_key("TAVILY_API_KEY"),
        args.query,
        args.search_depth,
        args.max_results,
        start_date,
        end_date,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    records.to_csv(output, index=False)
    print(f"Saved {len(records)} validated Tavily records to {output.resolve()}")


if __name__ == "__main__":
    main()
