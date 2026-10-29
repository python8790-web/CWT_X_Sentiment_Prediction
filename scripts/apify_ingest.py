from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote

import pandas as pd
import requests

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from src.project_config import RAW_DIR, required_api_key

API_ROOT = "https://api.apify.com/v2"
MAX_RESULT_LIMIT = 5000


def _api_get(path: str, token: str, params: dict[str, Any] | None = None) -> Any:
    for attempt in range(4):
        try:
            response = requests.get(
                f"{API_ROOT}{path}",
                headers={"Authorization": f"Bearer {token}"},
                params=params,
                timeout=30,
            )
        except requests.Timeout as exc:
            if attempt == 3:
                raise RuntimeError("Apify request timed out after retries.") from exc
            time.sleep(min(2**attempt, 8))
            continue
        except requests.RequestException as exc:
            if attempt == 3:
                raise RuntimeError("Apify request failed after retries.") from exc
            time.sleep(min(2**attempt, 8))
            continue

        if response.status_code == 429 or response.status_code >= 500:
            if attempt < 3:
                retry_after = response.headers.get("Retry-After", "")
                delay = int(retry_after) if retry_after.isdigit() else min(2**attempt, 8)
                time.sleep(delay)
                continue
        if not response.ok:
            raise RuntimeError(f"Apify API returned HTTP {response.status_code}.")
        try:
            return response.json()
        except ValueError as exc:
            raise RuntimeError("Apify returned invalid JSON.") from exc
    raise RuntimeError("Apify request failed after retries.")


def get_actor_input_schema(actor_id: str, token: str) -> dict[str, Any]:
    path = f"/acts/{quote(actor_id, safe='~')}/input-schema"
    result = _api_get(path, token)
    if isinstance(result, dict) and isinstance(result.get("data"), dict):
        result = result["data"]
    if isinstance(result, dict) and isinstance(result.get("schema"), dict):
        result = result["schema"]
    if not isinstance(result, dict):
        raise RuntimeError("Apify actor did not return a usable input schema.")
    if not isinstance(result.get("properties"), dict):
        raise RuntimeError(
            "The configured actor has no readable input schema. Configure "
            "APIFY_ACTOR_INPUT_JSON from the actor's documented input instead."
        )
    return result


def _matching_property(
    properties: dict[str, Any], candidates: tuple[str, ...]
) -> str | None:
    lower_names = {name.lower(): name for name in properties}
    for candidate in candidates:
        if candidate.lower() in lower_names:
            return lower_names[candidate.lower()]
    return None


def build_actor_input(
    schema: dict[str, Any],
    search_terms: list[str],
    start_date: str,
    end_date: str,
    max_posts: int,
) -> tuple[dict[str, Any], list[str]]:
    properties = schema.get("properties", {})
    payload: dict[str, Any] = {}
    configured: list[str] = []
    search_key = _matching_property(
        properties, ("searchTerms", "search_terms", "queries", "query", "keywords")
    )
    if search_key is None:
        raise RuntimeError(
            "The actor schema does not expose a recognized search input. "
            "Provide --input-json using the actor's documented schema."
        )
    search_type = properties[search_key].get("type")
    payload[search_key] = search_terms if search_type == "array" else ", ".join(search_terms)
    configured.append(search_key)

    date_fields = (
        (("startDate", "start_date", "since", "fromDate"), start_date),
        (("endDate", "end_date", "until", "toDate"), end_date),
    )
    for aliases, value in date_fields:
        key = _matching_property(properties, aliases)
        if key:
            payload[key] = value
            configured.append(key)

    limit_key = _matching_property(
        properties, ("maxItems", "maxResults", "maxTweets", "maxPosts", "limit")
    )
    if limit_key:
        payload[limit_key] = max_posts
        configured.append(limit_key)

    missing_required = set(schema.get("required", [])) - payload.keys()
    if missing_required:
        raise RuntimeError(
            "The actor requires additional input fields: "
            + ", ".join(sorted(missing_required))
            + ". Provide --input-json from the actor's documentation."
        )
    return payload, configured


def _load_custom_input(input_path: str | None) -> dict[str, Any] | None:
    raw = None
    if input_path:
        try:
            raw = Path(input_path).read_text(encoding="utf-8")
        except OSError as exc:
            raise RuntimeError(f"Could not read actor input file: {input_path}") from exc
    else:
        raw = os.getenv("APIFY_ACTOR_INPUT_JSON")
    if not raw:
        return None
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Actor input must be valid JSON.") from exc
    if not isinstance(value, dict):
        raise RuntimeError("Actor input JSON must be an object.")
    return value


def validate_actor_input(payload: dict[str, Any], schema: dict[str, Any]) -> None:
    properties = schema.get("properties", {})
    missing = set(schema.get("required", [])) - payload.keys()
    unknown = payload.keys() - properties.keys()
    if missing:
        raise ValueError(f"Missing required actor inputs: {', '.join(sorted(missing))}")
    if unknown:
        raise ValueError(f"Unknown actor inputs: {', '.join(sorted(unknown))}")
    for key, value in payload.items():
        specification = properties[key]
        expected = specification.get("type")
        valid = {
            "array": isinstance(value, list),
            "object": isinstance(value, dict),
            "string": isinstance(value, str),
            "integer": isinstance(value, int) and not isinstance(value, bool),
            "number": isinstance(value, (int, float)) and not isinstance(value, bool),
            "boolean": isinstance(value, bool),
        }.get(expected, True)
        if not valid:
            raise ValueError(f"Actor input {key!r} must have type {expected}.")
        if "enum" in specification and value not in specification["enum"]:
            raise ValueError(f"Actor input {key!r} is not an allowed value.")


def _first_value(item: dict[str, Any], *names: str) -> Any:
    for name in names:
        value = item.get(name)
        if value is not None:
            return value
    return None


def normalize_actor_item(item: Any) -> dict[str, Any] | None:
    if not isinstance(item, dict):
        return None
    author = item.get("author") or item.get("user") or {}
    metrics = item.get("public_metrics") or item.get("metrics") or {}
    if not isinstance(author, dict):
        author = {}
    if not isinstance(metrics, dict):
        metrics = {}
    created_at = _first_value(item, "createdAt", "created_at", "date", "timestamp")
    text = _first_value(item, "fullText", "full_text", "text")
    timestamp = pd.to_datetime(created_at, utc=True, errors="coerce")
    if pd.isna(timestamp) or not isinstance(text, str) or not text.strip():
        return None

    def metric(*names: str) -> float | None:
        value = _first_value(item, *names)
        if value is None:
            value = _first_value(metrics, *names)
        try:
            return float(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    followers = _first_value(author, "followers", "followersCount", "followers_count")
    if followers is None:
        followers = _first_value(
            item, "authorFollowers", "author_followers", "user_followers"
        )
    try:
        followers = float(followers) if followers is not None else None
    except (TypeError, ValueError):
        followers = None

    post_id = _first_value(item, "id", "id_str", "tweetId", "rest_id")
    if post_id is not None and not str(post_id).strip():
        post_id = None
    post_url = _first_value(item, "url", "postUrl", "post_url", "permalink")
    return {
        "tweet_id": str(post_id) if post_id is not None else None,
        "created_at": timestamp.isoformat(),
        "text": text.strip(),
        "like_count": metric("likeCount", "like_count", "favorite_count", "likes"),
        "repost_count": metric("retweetCount", "repost_count", "retweet_count", "retweets"),
        "reply_count": metric("replyCount", "reply_count", "replies"),
        "view_count": metric("viewCount", "view_count", "views"),
        "author_followers": followers,
        "author_username": _first_value(author, "userName", "username", "screen_name"),
        "post_url": post_url,
    }


def normalize_actor_items(items: list[Any]) -> tuple[pd.DataFrame, int]:
    rows = [normalized for item in items if (normalized := normalize_actor_item(item))]
    frame = pd.DataFrame(
        rows,
        columns=[
            "tweet_id",
            "created_at",
            "text",
            "like_count",
            "repost_count",
            "reply_count",
            "view_count",
            "author_followers",
            "author_username",
            "post_url",
        ],
    )
    if not frame.empty:
        has_id = frame["tweet_id"].notna() & frame["tweet_id"].astype(str).str.strip().ne("")
        frame = pd.concat(
            [
                frame.loc[has_id].drop_duplicates("tweet_id", keep="last"),
                frame.loc[~has_id].drop_duplicates(["created_at", "text"], keep="last"),
            ],
            ignore_index=True,
        ).sort_values("created_at")
    return frame.reset_index(drop=True), len(items) - len(rows)


def _poll_run(run: dict[str, Any], token: str, timeout_seconds: int = 600) -> dict[str, Any]:
    data = run.get("data", run)
    run_id = data.get("id")
    dataset_id = data.get("defaultDatasetId")
    if not run_id:
        raise RuntimeError("Apify accepted the run request without returning a run ID.")
    deadline = time.monotonic() + timeout_seconds
    while data.get("status") in {"READY", "RUNNING"} and time.monotonic() < deadline:
        time.sleep(5)
        result = _api_get(f"/actor-runs/{quote(str(run_id), safe='')}", token)
        data = result.get("data", result)
        dataset_id = data.get("defaultDatasetId") or dataset_id
    if data.get("status") != "SUCCEEDED":
        raise RuntimeError(
            f"Apify actor run ended with status {data.get('status', 'unknown')}."
        )
    if not dataset_id:
        raise RuntimeError("Successful Apify run did not provide a dataset ID.")
    return {"run_id": run_id, "dataset_id": dataset_id}


def fetch_dataset_items(
    dataset_id: str, token: str, max_posts: int, page_size: int = 1000
) -> list[Any]:
    items: list[Any] = []
    offset = 0
    while offset < max_posts:
        limit = min(page_size, max_posts - offset)
        batch = _api_get(
            f"/datasets/{quote(dataset_id, safe='')}/items",
            token,
            {"offset": offset, "limit": limit, "clean": "true", "format": "json"},
        )
        if not isinstance(batch, list):
            raise RuntimeError("Apify dataset returned an unexpected item format.")
        if not batch:
            break
        items.extend(batch)
        offset += len(batch)
        if len(batch) < limit:
            break
    return items


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect an Apify actor input schema; start a bounded job only with --run."
    )
    parser.add_argument("--start-date", default=os.getenv("APIFY_START_DATE"))
    parser.add_argument("--end-date", default=os.getenv("APIFY_END_DATE"))
    parser.add_argument("--max-posts", type=int, default=int(os.getenv("APIFY_MAX_POSTS", "500")))
    parser.add_argument("--search-term", action="append", dest="search_terms")
    parser.add_argument("--input-json", help="Path to actor-specific JSON input")
    parser.add_argument("--output", default=str(RAW_DIR / "x_posts_apify.csv"))
    parser.add_argument("--run", action="store_true", help="Start a potentially billable actor run")
    parser.add_argument("--show-input-schema", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 1 <= args.max_posts <= MAX_RESULT_LIMIT:
        raise ValueError(f"--max-posts must be between 1 and {MAX_RESULT_LIMIT}.")
    now = datetime.now(timezone.utc).date()
    end_date = args.end_date or now.isoformat()
    start_date = args.start_date or (now - timedelta(days=180)).isoformat()
    start = datetime.fromisoformat(start_date).date()
    end = datetime.fromisoformat(end_date).date()
    if start > end:
        raise ValueError("--start-date must be on or before --end-date.")

    token = required_api_key("APIFY_API_TOKEN")
    actor_id = required_api_key("APIFY_ACTOR_ID")
    schema = get_actor_input_schema(actor_id, token)
    if args.show_input_schema:
        print(json.dumps(schema, indent=2))
        return

    terms = args.search_terms or ["Bitcoin", "BTC", "#Bitcoin"]
    payload = _load_custom_input(args.input_json)
    if payload is None:
        payload, configured_fields = build_actor_input(
            schema, terms, start.isoformat(), end.isoformat(), args.max_posts
        )
        if not any(field in configured_fields for field in ("startDate", "start_date", "since", "fromDate")):
            print("Note: actor schema exposes no recognized start-date field.")
        if not any(field in configured_fields for field in ("endDate", "end_date", "until", "toDate")):
            print("Note: actor schema exposes no recognized end-date field.")
    validate_actor_input(payload, schema)
    if args.run and _matching_property(
        schema["properties"],
        ("maxItems", "maxResults", "maxTweets", "maxPosts", "limit"),
    ) is None:
        raise RuntimeError(
            "The actor schema exposes no recognized maximum-result input. "
            "Refusing an unbounded actor run."
        )
    print(
        f"Actor input validated for {start.isoformat()} through {end.isoformat()}, "
        f"with a maximum of {args.max_posts} fetched posts."
    )
    if not args.run:
        print("Dry run only. Review actor input and cost limits, then pass --run to start the actor.")
        print("Actor input fields:", ", ".join(sorted(payload)))
        return

    response = None
    for attempt in range(4):
        try:
            response = requests.post(
                f"{API_ROOT}/acts/{quote(actor_id, safe='~')}/runs",
                headers={"Authorization": f"Bearer {token}"},
                json=payload,
                params={"waitForFinish": 300},
                timeout=330,
            )
        except requests.Timeout as exc:
            raise RuntimeError(
                "Apify run request timed out; its status is unknown. Check the Apify console "
                "before retrying to avoid starting a duplicate job."
            ) from exc
        except requests.RequestException as exc:
            raise RuntimeError("Apify run request failed; check the Apify console before retrying.") from exc
        if response.status_code == 429 and attempt < 3:
            retry_after = response.headers.get("Retry-After", "")
            delay = int(retry_after) if retry_after.isdigit() else min(2**attempt, 8)
            time.sleep(delay)
            continue
        break
    if response is None:
        raise RuntimeError("Apify run request did not return a response.")
    if not response.ok:
        raise RuntimeError(f"Apify run request returned HTTP {response.status_code}.")
    try:
        run = response.json()
    except ValueError as exc:
        raise RuntimeError("Apify returned invalid run JSON.") from exc
    run_info = _poll_run(run, token)
    items = fetch_dataset_items(run_info["dataset_id"], token, args.max_posts)
    posts, rejected = normalize_actor_items(items)
    if not posts.empty:
        timestamps = pd.to_datetime(posts["created_at"], utc=True, errors="coerce")
        in_window = timestamps.dt.date.between(start, end)
        out_of_window = int((~in_window).sum())
        posts = posts.loc[in_window].reset_index(drop=True)
    else:
        out_of_window = 0
    if posts.empty:
        raise RuntimeError(
            "Apify completed, but no valid records fell inside the requested date window; "
            "no data was saved."
        )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    posts.to_csv(output, index=False)
    print(
        f"Validated {len(posts)} posts ({rejected} malformed and "
        f"{out_of_window} out-of-window records skipped); "
        f"saved to {output.resolve()}. Apify run ID: {run_info['run_id']}."
    )


if __name__ == "__main__":
    main()
