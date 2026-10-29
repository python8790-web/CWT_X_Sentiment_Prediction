from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pandas as pd
import requests

ROOT_DIR = Path(__file__).resolve().parents[1]


def normalize_news_records(records: list[dict[str, Any]]) -> pd.DataFrame:
    normalized = []
    seen_urls: set[str] = set()
    for record in records:
        url = str(record.get("url") or "").strip()
        if not url or url in seen_urls:
            continue
        seen_urls.add(url)
        domain = urlparse(url).netloc.lower()
        normalized.append(
            {
                "title": str(record.get("title") or "").strip(),
                "url": url,
                "published_date": record.get("published_date"),
                "source": str(record.get("source") or domain),
                "content": str(record.get("content") or "").strip(),
                "retrieved_at": datetime.now(timezone.utc).isoformat(),
            }
        )
    return pd.DataFrame(
        normalized,
        columns=[
            "title",
            "url",
            "published_date",
            "source",
            "content",
            "retrieved_at",
        ],
    )


def post_json(
    endpoint: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    service_name: str,
    timeout: int = 30,
    attempts: int = 4,
) -> dict[str, Any]:
    for attempt in range(attempts):
        try:
            response = requests.post(
                endpoint, json=payload, headers=headers, timeout=timeout
            )
        except requests.Timeout as exc:
            if attempt + 1 == attempts:
                raise RuntimeError(f"{service_name} request timed out.") from exc
            time.sleep(min(2**attempt, 8))
            continue
        except requests.RequestException as exc:
            raise RuntimeError(f"{service_name} request failed.") from exc

        if response.status_code == 429 or response.status_code >= 500:
            if attempt + 1 < attempts:
                retry_after = response.headers.get("Retry-After", "")
                delay = int(retry_after) if retry_after.isdigit() else min(2**attempt, 8)
                time.sleep(delay)
                continue
        if not response.ok:
            raise RuntimeError(
                f"{service_name} API returned HTTP {response.status_code}."
            )
        try:
            result = response.json()
        except ValueError as exc:
            raise RuntimeError(f"{service_name} returned invalid JSON.") from exc
        if not isinstance(result, dict):
            raise RuntimeError(f"{service_name} returned an unexpected response.")
        return result
    raise RuntimeError(f"{service_name} request failed after retries.")
