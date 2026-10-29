import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluate_strategy import evaluate_model, max_drawdown, select_non_overlapping_trades
from scripts.apify_ingest import (
    build_actor_input,
    normalize_actor_item,
    normalize_actor_items,
    validate_actor_input,
)
from scripts.news_utils import normalize_news_records, post_json
from scripts.process_apify_posts import score_posts
from scripts.tavily_news import retrieve_news as retrieve_tavily
from scripts.exa_news import retrieve_news as retrieve_exa
from src.pipeline_utils import (
    add_return_targets,
    aggregate_sentiment_posts,
    normalize_utc_timestamps,
)
from src.project_config import required_api_key


def test_timestamp_normalization_handles_offsets_and_bad_values():
    result = normalize_utc_timestamps(
        pd.Series(["2025-03-11T01:00:00+01:00", "not-a-date"])
    )
    assert str(result.iloc[0]) == "2025-03-11 00:00:00+00:00"
    assert pd.isna(result.iloc[1])


def test_sentiment_hourly_aggregation_normalizes_and_deduplicates():
    posts = pd.DataFrame(
        {
            "tweet_id": ["1", "1", "2"],
            "date": [
                "2025-03-11T01:10:00+01:00",
                "2025-03-11T01:10:00+01:00",
                "2025-03-11T00:50:00Z",
            ],
            "text": ["Bitcoin is up", "Bitcoin is up", "BTC is down"],
            "sentiment_compound": [0.8, 0.8, -0.5],
            "user_followers": [10, 10, None],
        }
    )
    hourly = aggregate_sentiment_posts(posts)
    assert len(hourly) == 1
    assert hourly.loc[0, "tweet_count"] == 2
    assert hourly.loc[0, "hour"] == pd.Timestamp("2025-03-11T00:00:00Z")
    assert hourly.loc[0, "positive_share"] == 0.5


def test_apify_post_normalization_rejects_malformed_and_keeps_real_metrics():
    record = normalize_actor_item(
        {
            "id": "p1",
            "createdAt": "2025-03-11T01:00:00+01:00",
            "text": "Bitcoin post",
            "likeCount": "4",
            "author": {"followersCount": 99, "userName": "alice"},
        }
    )
    assert record["created_at"] == "2025-03-11T00:00:00+00:00"
    assert record["like_count"] == 4.0
    assert record["author_followers"] == 99.0
    assert normalize_actor_item({"text": "missing timestamp"}) is None
    normalized, rejected = normalize_actor_items([record, None, {"text": "bad"}])
    assert len(normalized) == 1
    assert rejected == 2


def test_actor_input_uses_discovered_schema_and_validates_bounds():
    schema = {
        "type": "object",
        "required": ["queries", "startDate", "endDate", "maxItems"],
        "properties": {
            "queries": {"type": "array", "items": {"type": "string"}},
            "startDate": {"type": "string"},
            "endDate": {"type": "string"},
            "maxItems": {"type": "integer"},
        },
    }
    payload, configured = build_actor_input(
        schema, ["Bitcoin", "BTC"], "2025-01-01", "2025-06-30", 400
    )
    validate_actor_input(payload, schema)
    assert payload["queries"] == ["Bitcoin", "BTC"]
    assert payload["maxItems"] == 400
    assert len(configured) == 4
    payload["maxItems"] = "unbounded"
    with pytest.raises(ValueError, match="must have type integer"):
        validate_actor_input(payload, schema)


def test_apify_scoring_deduplicates_and_handles_optional_author_fields():
    scored = score_posts(
        pd.DataFrame(
            {
                "tweet_id": ["1", "1"],
                "created_at": [
                    "2025-03-11T00:00:00Z",
                    "2025-03-11T00:00:00Z",
                ],
                "text": ["Bitcoin is great!", "Bitcoin is great!"],
            }
        )
    )
    assert len(scored) == 1
    assert scored.iloc[0]["sentiment_compound"] > 0
    assert pd.isna(scored.iloc[0]["user_followers"])


def test_news_normalization_deduplicates_and_uses_url_domain():
    news = normalize_news_records(
        [
            {"url": "https://news.example/a", "title": "BTC", "content": "text"},
            {"url": "https://news.example/a", "title": "duplicate"},
            {"title": "no URL"},
        ]
    )
    assert len(news) == 1
    assert news.iloc[0]["source"] == "news.example"
    assert news.iloc[0]["retrieved_at"]


def test_tavily_response_normalization_uses_supported_fields(monkeypatch):
    def fake_post(endpoint, payload, headers, service_name):
        assert payload["topic"] == "news"
        assert payload["max_results"] == 1
        return {
            "results": [
                {
                    "title": "BTC update",
                    "url": "https://source.example/story",
                    "published_date": "2025-03-11",
                    "content": "Snippet",
                }
            ]
        }

    monkeypatch.setattr("scripts.tavily_news.post_json", fake_post)
    news = retrieve_tavily("test-key", max_results=1)
    assert news.iloc[0]["published_date"] == "2025-03-11"
    assert news.iloc[0]["content"] == "Snippet"


def test_exa_response_normalization_uses_supported_fields(monkeypatch):
    def fake_post(endpoint, payload, headers, service_name):
        assert payload["category"] == "news"
        assert payload["startPublishedDate"].startswith("2025-03-01")
        return {
            "results": [
                {
                    "title": "BTC report",
                    "url": "https://publisher.example/report",
                    "publishedDate": "2025-03-11T00:00:00Z",
                    "text": "Retrieved content",
                }
            ]
        }

    monkeypatch.setattr("scripts.exa_news.post_json", fake_post)
    news = retrieve_exa(
        "test-key", max_results=1, start_date="2025-03-01", end_date="2025-03-11"
    )
    assert news.iloc[0]["published_date"] == "2025-03-11T00:00:00Z"
    assert news.iloc[0]["content"] == "Retrieved content"


def test_api_helpers_never_require_or_echo_missing_keys(monkeypatch):
    monkeypatch.delenv("EXA_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="EXA_API_KEY is not configured"):
        required_api_key("EXA_API_KEY")


def test_post_json_retries_rate_limit_without_exposing_request_payload(monkeypatch):
    class Response:
        def __init__(self, status, payload):
            self.status_code = status
            self.headers = {}
            self.ok = status < 400
            self._payload = payload

        def json(self):
            return self._payload

    responses = iter([Response(429, {}), Response(200, {"results": []})])
    monkeypatch.setattr("scripts.news_utils.requests.post", lambda *args, **kwargs: next(responses))
    monkeypatch.setattr("scripts.news_utils.time.sleep", lambda _: None)
    assert post_json("https://example.test", {}, {}, "Test API") == {"results": []}


def test_targets_are_hour_aligned_and_n0_is_current_completed_return():
    prices = pd.DataFrame(
        {
            "hour": pd.date_range("2025-01-01", periods=8, freq="h"),
            "Close": [100, 101, 99, 102, 103, 105, 104, 106],
        }
    )
    result = add_return_targets(prices)
    assert result.loc[2, "future_return_0h"] == pytest.approx(99 / 101 - 1)
    assert result.loc[2, "future_return_1h"] == pytest.approx(102 / 99 - 1)
    assert result.loc[2, "hour"] + pd.Timedelta(hours=1) == result.loc[3, "hour"]


def test_strategy_excludes_overlap_checks_target_alignment_and_costs_round_trip():
    hours = pd.date_range("2025-01-01", periods=5, freq="h")
    frame = pd.DataFrame(
        {
            "hour": hours,
            "target_hour": hours + pd.Timedelta(hours=1),
            "signal_time": hours + pd.Timedelta(hours=1),
            "target_time": hours + pd.Timedelta(hours=2),
            "horizon_hours": 1,
            "model": "price_only",
            "predicted_return": [0.1] * 5,
            "actual_return": [0.01] * 5,
        }
    )
    chosen = select_non_overlapping_trades(frame, 1)
    assert len(chosen) == 5
    summary, trades = evaluate_model(frame, 1)
    assert summary["observations"] == 5
    assert trades.iloc[0]["transaction_cost"] == pytest.approx(0.001)
    assert trades.iloc[-1]["transaction_cost"] == pytest.approx(0.001)
    assert summary["total_net_return"] == pytest.approx(
        np.prod([1.009, 1.01, 1.01, 1.01, 1.009]) - 1
    )


def test_strategy_rejects_misaligned_horizon_and_drawdown_includes_starting_cash():
    frame = pd.DataFrame(
        {
            "hour": pd.to_datetime(["2025-01-01T00:00:00"]),
            "target_hour": pd.to_datetime(["2025-01-01T02:00:00"]),
            "signal_time": pd.to_datetime(["2025-01-01T01:00:00"]),
            "target_time": pd.to_datetime(["2025-01-01T03:00:00"]),
            "model": ["price_only"],
            "predicted_return": [0.1],
            "actual_return": [0.02],
        }
    )
    with pytest.raises(ValueError, match="target timestamps"):
        select_non_overlapping_trades(frame, 1)
    assert max_drawdown(pd.Series([-0.2, 0.1])) == pytest.approx(-0.2)
    frame["target_hour"] = frame["hour"] + pd.Timedelta(hours=1)
    frame["target_time"] = frame["target_hour"] + pd.Timedelta(hours=1)
    summary, _ = evaluate_model(frame, 1)
    assert np.isnan(summary["annualized_sharpe"])
