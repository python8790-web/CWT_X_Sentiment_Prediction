import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "data" / "processed" / "bitcoin_merged_hourly.csv"
PREDICTIONS_FILE = ROOT / "outputs" / "walk_forward_predictions.csv"
OUTPUT_DIR = ROOT / "outputs"

TRANSACTION_COST_BPS = 10
PERIODS_PER_YEAR = 365 * 24
COST_RATE = TRANSACTION_COST_BPS / 10_000


def max_drawdown(returns: pd.Series) -> float:
    wealth = np.concatenate(([1.0], (1 + returns.to_numpy(dtype=float)).cumprod()))
    running_peak = np.maximum.accumulate(wealth)
    return float(np.min(wealth / running_peak - 1))


def select_non_overlapping_trades(group: pd.DataFrame, horizon: int) -> pd.DataFrame:
    if horizon < 1:
        raise ValueError("Strategy evaluation requires a positive forecast horizon.")
    ordered = group.sort_values("signal_time").copy()
    expected_target = ordered["hour"] + pd.to_timedelta(horizon, unit="h")
    if not ordered["target_hour"].equals(expected_target):
        raise ValueError("Prediction target timestamps do not match their horizon.")
    expected_signal = ordered["hour"] + pd.Timedelta(hours=1)
    if not ordered["signal_time"].equals(expected_signal):
        raise ValueError("Prediction signal timestamps do not match candle close times.")
    expected_target_time = ordered["target_hour"] + pd.Timedelta(hours=1)
    if not ordered["target_time"].equals(expected_target_time):
        raise ValueError("Realized target timestamps do not match candle close times.")
    if not (
        ordered["target_time"] - ordered["signal_time"]
    ).eq(pd.Timedelta(hours=horizon)).all():
        raise ValueError("Holding periods do not match the selected horizon.")

    selected = []
    next_entry = None
    for row_index, row in ordered.iterrows():
        if next_entry is None or row["signal_time"] >= next_entry:
            selected.append(row_index)
            next_entry = row["target_time"]
    return ordered.loc[selected].reset_index(drop=True)


def evaluate_model(group: pd.DataFrame, horizon: int) -> tuple[dict | None, pd.DataFrame]:
    trades = select_non_overlapping_trades(group, horizon)
    if trades.empty:
        return None, trades

    trades["position"] = np.sign(trades["predicted_return"])
    trades["turnover"] = 0.0
    trades.loc[0, "turnover"] = abs(trades.loc[0, "position"])
    for index in range(1, len(trades)):
        previous = trades.iloc[index - 1]
        current = trades.iloc[index]
        if previous["target_hour"] == current["hour"]:
            trades.loc[index, "turnover"] = abs(
                current["position"] - previous["position"]
            )
        else:
            trades.loc[index - 1, "turnover"] += abs(previous["position"])
            trades.loc[index, "turnover"] += abs(current["position"])
    trades.loc[trades.index[-1], "turnover"] += abs(trades["position"].iloc[-1])

    trades["gross_strategy_return"] = (
        trades["position"] * trades["actual_return"]
    )
    trades["transaction_cost"] = trades["turnover"] * COST_RATE
    trades["net_strategy_return"] = (
        trades["gross_strategy_return"] - trades["transaction_cost"]
    )
    returns = trades["net_strategy_return"].astype(float)
    if not np.isfinite(returns).all() or (returns <= -1).any():
        raise ValueError("Strategy returns must be finite and greater than -100%.")

    mean_return = float(returns.mean())
    volatility = float(returns.std(ddof=1))
    periods_per_year = PERIODS_PER_YEAR / horizon
    sharpe = (
        np.sqrt(periods_per_year) * mean_return / volatility
        if len(returns) > 1 and volatility > 0
        else np.nan
    )
    downside_deviation = float(
        np.sqrt(np.mean(np.minimum(returns.to_numpy(), 0.0) ** 2))
    )
    sortino = (
        np.sqrt(periods_per_year) * mean_return / downside_deviation
        if downside_deviation > 0
        else np.nan
    )
    equity = (1 + returns).cumprod()

    summary = {
        "model": trades["model"].iloc[0],
        "horizon_hours": horizon,
        "observations": int(len(returns)),
        "transaction_cost_bps": TRANSACTION_COST_BPS,
        "total_net_return": float(equity.iloc[-1] - 1),
        "annualized_sharpe": float(sharpe) if pd.notna(sharpe) else np.nan,
        "annualized_sortino": float(sortino) if pd.notna(sortino) else np.nan,
        "max_drawdown": max_drawdown(returns),
        "win_rate": float((returns > 0).mean()),
        "average_trade_net_return": mean_return,
        "annualization_periods_per_year": periods_per_year,
        "selection": "non_overlapping_holding_periods",
    }
    return summary, trades


def main() -> None:
    prices = pd.read_csv(DATA_FILE, parse_dates=["hour"])
    predictions = pd.read_csv(
        PREDICTIONS_FILE, parse_dates=[        "hour", "target_hour", "signal_time", "target_time"]
    )
    predictions = predictions[
        predictions["model"].isin(["price_only", "price_plus_sentiment"])
        & (predictions["horizon_hours"] > 0)
    ].copy()
    if predictions.empty:
        raise RuntimeError("No forward-horizon predictions available for strategy evaluation.")

    close_by_hour = prices[["hour", "Close"]].rename(
        columns={"hour": "hour", "Close": "entry_close"}
    )
    target_close = prices[["hour", "Close"]].rename(
        columns={"hour": "target_hour", "Close": "target_close"}
    )
    data = predictions.merge(
        close_by_hour, on="hour", how="left", validate="many_to_one"
    ).merge(target_close, on="target_hour", how="left", validate="many_to_one")
    data = data.dropna(
        subset=[
            "entry_close",
            "target_close",
            "actual_return",
            "predicted_return",
        ]
    )
    aligned_returns = data["target_close"] / data["entry_close"] - 1
    if not np.allclose(
        aligned_returns.to_numpy(),
        data["actual_return"].to_numpy(),
        rtol=1e-8,
        atol=1e-10,
    ):
        raise ValueError("Predicted labels are misaligned with realized target prices.")

    summaries = []
    trade_frames = []
    for horizon in sorted(data["horizon_hours"].unique().tolist()):
        horizon_data = data[data["horizon_hours"] == horizon]
        for _, group in horizon_data.groupby("model"):
            summary, trades = evaluate_model(group, int(horizon))
            if summary:
                summaries.append(summary)
                trade_frames.append(trades)
    if not summaries:
        raise RuntimeError("No valid non-overlapping strategy results were produced.")

    summary_file = OUTPUT_DIR / "strategy_metrics.csv"
    trades_file = OUTPUT_DIR / "strategy_hourly_returns.csv"
    pd.DataFrame(summaries).to_csv(summary_file, index=False)
    pd.concat(trade_frames, ignore_index=True).to_csv(trades_file, index=False)
    print("\n===== NON-OVERLAPPING STRATEGY RESULTS =====")
    print(pd.DataFrame(summaries).to_string(index=False))
    print("\nSaved strategy metrics:", summary_file.resolve())
    print("Saved trades:", trades_file.resolve())
    print(
        "\nAssumptions: signal observed at the prediction timestamp; trades hold "
        "until target timestamp; no overlapping positions; 10 bps per unit of "
        "position turnover, including entry and final liquidation; no slippage."
    )
    print("n+0 is excluded because it is a completed current-period return, not a forecast.")


if __name__ == "__main__":
    main()
