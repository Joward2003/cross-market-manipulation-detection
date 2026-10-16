#!/usr/bin/env python3
"""Build start- and end-aligned anomaly trajectories for cases and controls."""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/posttraining_analysis.yaml")
    parser.add_argument("--trading-days", type=int, default=60)
    return parser.parse_args()


def make_anchors(cases: pd.DataFrame, controls: pd.DataFrame) -> pd.DataFrame:
    case = cases[["event_id", "symbol6", "start_date", "end_date"]].copy()
    case["window_id"] = case.event_id + "::case"
    case["sample"] = "case"
    control = controls[["event_id", "symbol6", "start_date", "end_date", "control_rank"]].copy()
    control["window_id"] = control.event_id + "::control_" + control.control_rank.astype(str)
    control["sample"] = "control"
    windows = pd.concat([case, control.drop(columns="control_rank")], ignore_index=True)
    parts = []
    for anchor_type, date_column in (("start", "start_date"), ("end", "end_date")):
        part = windows.copy()
        part["anchor_type"] = anchor_type
        part["anchor_date"] = part[date_column].astype(int)
        anchor_ts = pd.to_datetime(part.anchor_date.astype(str), format="%Y%m%d")
        part["lower_date"] = (anchor_ts - pd.Timedelta(days=120)).dt.strftime("%Y%m%d").astype(int)
        part["upper_date"] = (anchor_ts + pd.Timedelta(days=120)).dt.strftime("%Y%m%d").astype(int)
        parts.append(part[["event_id", "window_id", "sample", "symbol6", "anchor_type", "anchor_date", "lower_date", "upper_date"]])
    return pd.concat(parts, ignore_index=True)


def extract_daily(runs_root: Path, dataset: str, anchors: pd.DataFrame) -> pd.DataFrame:
    pattern = str(runs_root / dataset / "seed_*" / "scores" / "scores_with_percentile.parquet").replace("'", "''")
    con = duckdb.connect()
    con.execute("PRAGMA threads=4")
    con.register("anchors", anchors)
    result = con.execute(f"""
      SELECT a.event_id, a.window_id, a.sample, a.symbol6, a.anchor_type, a.anchor_date,
             s.trade_date, AVG(s.svdd_percentile_same_day) AS ensemble_percentile
      FROM anchors a
      JOIN (
        SELECT *, regexp_extract(symbol, '([0-9]{{6}})', 1) AS symbol6
        FROM read_parquet('{pattern}')
      ) s ON a.symbol6 = s.symbol6 AND s.trade_date BETWEEN a.lower_date AND a.upper_date
      GROUP BY ALL ORDER BY a.window_id, a.anchor_type, s.trade_date
    """).df()
    con.close()
    result["dataset"] = dataset
    return result


def add_event_time(frame: pd.DataFrame, limit: int) -> pd.DataFrame:
    pieces = []
    for _, block in frame.groupby(["dataset", "window_id", "anchor_type"], sort=False):
        block = block.sort_values("trade_date").copy()
        dates = block.trade_date.to_numpy(int)
        anchor = int(block.anchor_date.iloc[0])
        position = int(np.searchsorted(dates, anchor, side="left"))
        if position >= len(dates):
            continue
        block["event_time"] = np.arange(len(block)) - position
        pieces.append(block.loc[block.event_time.between(-limit, limit)])
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()


def event_equal_weight(frame: pd.DataFrame) -> pd.DataFrame:
    case = frame.loc[frame["sample"] == "case"].copy()
    controls = frame.loc[frame["sample"] == "control"].groupby(
        ["dataset", "anchor_type", "event_id", "event_time"], as_index=False
    ).ensemble_percentile.mean()
    controls["sample"] = "control"
    return pd.concat([
        case[["dataset", "anchor_type", "event_id", "event_time", "sample", "ensemble_percentile"]],
        controls[["dataset", "anchor_type", "event_id", "event_time", "sample", "ensemble_percentile"]],
    ], ignore_index=True)


def summarize(frame: pd.DataFrame, repetitions: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for keys, block in frame.groupby(["dataset", "anchor_type", "sample", "event_time"]):
        values = block.ensemble_percentile.dropna().to_numpy(float)
        if not len(values):
            continue
        draws = rng.integers(0, len(values), size=(repetitions, len(values)))
        boot = values[draws].mean(axis=1)
        low, high = np.quantile(boot, [0.025, 0.975])
        rows.append({"dataset": keys[0], "anchor_type": keys[1], "sample": keys[2], "event_time": keys[3],
                     "n_events": len(values), "mean_percentile": float(values.mean()),
                     "ci_low": float(low), "ci_high": float(high)})
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    output = ROOT / cfg["paths"]["output_root"]
    cases = pd.read_parquet(output / "data" / "matched_cases.parquet")
    controls = pd.read_parquet(output / "data" / "difficult_controls.parquet")
    anchors = make_anchors(cases, controls)
    runs_root = ROOT / cfg["paths"]["runs_root"]
    daily = pd.concat([extract_daily(runs_root, dataset, anchors) for dataset in ("a_only", "a_us_coral")], ignore_index=True)
    timed = add_event_time(daily, args.trading_days)
    event_weighted = event_equal_weight(timed)
    event_weighted.to_parquet(output / "data" / "event_time_scores.parquet", index=False)
    summary = summarize(event_weighted, int(cfg["inference"]["bootstrap_repetitions"]),
                        int(cfg["inference"]["random_seed"]) + 40_000)
    summary.to_csv(output / "tables" / "event_time_trajectories.csv", index=False, encoding="utf-8-sig")
    print({"daily_rows": len(daily), "event_time_rows": len(event_weighted), "summary_rows": len(summary)})


if __name__ == "__main__":
    main()

