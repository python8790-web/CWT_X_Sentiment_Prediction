from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MERGED_FILE = ROOT / "data" / "processed" / "bitcoin_merged_hourly.csv"
METRICS_FILE = ROOT / "outputs" / "walk_forward_metrics.csv"
STRATEGY_FILE = ROOT / "outputs" / "strategy_metrics.csv"


def test_merged_dataset_has_all_assignment_horizons():
    df = pd.read_csv(MERGED_FILE)
    expected = {f"future_return_{h}h" for h in range(6)}
    assert expected.issubset(df.columns)
    assert not df["hour"].duplicated().any()


def test_walk_forward_metrics_are_numeric_and_complete():
    metrics = pd.read_csv(METRICS_FILE)
    assert {"horizon_hours", "model", "MAE", "RMSE", "directional_accuracy", "observations"}.issubset(metrics.columns)
    assert len(metrics) > 0
    scored = metrics[metrics["horizon_hours"] > 0]
    assert scored["MAE"].notna().all()
    assert scored["RMSE"].notna().all()
    assert (scored["directional_accuracy"].between(0, 1, inclusive="both") | scored["directional_accuracy"].isna()).all()
    diagnostic = metrics[metrics["horizon_hours"] == 0]
    assert diagnostic["status"].eq("diagnostic_not_forecast").all()
    assert diagnostic["MAE"].isna().all()


def test_strategy_metrics_include_models_and_costs():
    strategy = pd.read_csv(STRATEGY_FILE)
    assert {"model", "horizon_hours", "total_net_return", "annualized_sharpe", "annualized_sortino", "max_drawdown"}.issubset(strategy.columns)
    assert len(strategy) > 0
    assert strategy["model"].isin(["price_only", "price_plus_sentiment"]).all()
    assert (strategy["horizon_hours"] > 0).all()
