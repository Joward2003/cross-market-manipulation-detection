#!/usr/bin/env python3
"""Run same-stock, same-duration randomly shifted event-time placebos."""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
METRICS = ["max_svdd_percentile", "mean_svdd_percentile", "share_top_1pct", "share_top_5pct"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/posttraining_analysis.yaml")
    parser.add_argument("--placebos-per-event", type=int, default=10)
    return parser.parse_args()


def make_windows(cases: pd.DataFrame, n_placebos: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    lower = pd.Timestamp("2018-01-01")
    upper = pd.Timestamp("2025-12-31")
    rows = []
    for case in cases.itertuples(index=False):
        start = pd.to_datetime(str(int(case.start_date)), format="%Y%m%d")
        end = pd.to_datetime(str(int(case.end_date)), format="%Y%m%d")
        duration = end - start
        rows.append({"event_id": case.event_id, "window_id": f"{case.event_id}::actual", "window_type": "actual",
                     "symbol6": case.symbol6, "start_date": int(case.start_date), "end_date": int(case.end_date)})
        latest_start = upper - duration
        if latest_start < lower:
            continue
        candidates = pd.date_range(lower, latest_start, freq="D")
        if len(candidates) == 0:
            continue
        order = rng.permutation(len(candidates))
        accepted = []
        embargo_start = start - pd.Timedelta(days=60)
        embargo_end = end + pd.Timedelta(days=60)
        for idx in order:
            pseudo_start = candidates[int(idx)]
            pseudo_end = pseudo_start + duration
            nonoverlap = pseudo_end < embargo_start or pseudo_start > embargo_end
            separated = all(abs((pseudo_start - prior).days) >= 60 for prior in accepted)
            if nonoverlap and separated:
                accepted.append(pseudo_start)
                rank = len(accepted)
                rows.append({"event_id": case.event_id, "window_id": f"{case.event_id}::placebo_{rank:02d}",
                             "window_type": "placebo", "symbol6": case.symbol6,
                             "start_date": int(pseudo_start.strftime("%Y%m%d")),
                             "end_date": int(pseudo_end.strftime("%Y%m%d"))})
            if len(accepted) >= n_placebos:
                break
    return pd.DataFrame(rows)


def aggregate_dataset(runs_root: Path, dataset: str, windows: pd.DataFrame) -> pd.DataFrame:
    pattern = str(runs_root / dataset / "seed_*" / "scores" / "scores_with_percentile.parquet").replace("'", "''")
    con = duckdb.connect()
    con.execute("PRAGMA threads=4")
    con.register("windows", windows)
    result = con.execute(f"""
      WITH daily AS (
        SELECT w.event_id, w.window_id, w.window_type, w.symbol6,
               w.start_date, w.end_date, s.trade_date,
               AVG(s.svdd_percentile_same_day) AS ensemble_percentile
        FROM windows w
        LEFT JOIN (
          SELECT *, regexp_extract(symbol, '([0-9]{{6}})', 1) AS symbol6
          FROM read_parquet('{pattern}')
        ) s ON w.symbol6 = s.symbol6 AND s.trade_date BETWEEN w.start_date AND w.end_date
        GROUP BY ALL
      )
      SELECT event_id, window_id, window_type, symbol6, start_date, end_date,
             COUNT(trade_date)::INTEGER AS n_scored_days,
             MAX(ensemble_percentile) AS max_svdd_percentile,
             AVG(ensemble_percentile) AS mean_svdd_percentile,
             AVG(CASE WHEN ensemble_percentile >= 0.99 THEN 1.0 ELSE 0.0 END) AS share_top_1pct,
             AVG(CASE WHEN ensemble_percentile >= 0.95 THEN 1.0 ELSE 0.0 END) AS share_top_5pct
      FROM daily GROUP BY ALL ORDER BY event_id, window_type, window_id
    """).df()
    con.close()
    result["dataset"] = dataset
    return result


def inference(frame: pd.DataFrame, repetitions: int, permutations: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for dataset, block in frame.groupby("dataset"):
        for metric in METRICS:
            actual = block.loc[(block.window_type == "actual") & (block.n_scored_days > 0), ["event_id", metric]].rename(columns={metric: "actual"})
            placebo = block.loc[(block.window_type == "placebo") & (block.n_scored_days > 0)].groupby("event_id", as_index=False)[metric].mean().rename(columns={metric: "placebo"})
            paired = actual.merge(placebo, on="event_id", how="inner").dropna()
            values = (paired.actual - paired.placebo).to_numpy(float)
            observed = float(values.mean()) if len(values) else np.nan
            if len(values):
                draws = rng.integers(0, len(values), size=(repetitions, len(values)))
                boot = values[draws].mean(axis=1)
                signs = rng.choice(np.array([-1.0, 1.0]), size=(permutations, len(values)))
                null = (signs * values).mean(axis=1)
                ci_low, ci_high = np.quantile(boot, [0.025, 0.975])
                p_one = (1 + np.sum(null >= observed)) / (permutations + 1)
            else:
                ci_low = ci_high = p_one = np.nan
            rows.append({"dataset": dataset, "metric": metric, "n_events": len(values),
                         "actual_minus_placebo": observed, "ci_low": float(ci_low), "ci_high": float(ci_high),
                         "permutation_p_one_sided": float(p_one)})
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    output = ROOT / cfg["paths"]["output_root"]
    cases = pd.read_parquet(output / "data" / "matched_cases.parquet")
    windows = make_windows(cases, args.placebos_per_event, int(cfg["inference"]["random_seed"]))
    windows.to_csv(output / "data" / "time_placebo_windows.csv", index=False, encoding="utf-8-sig")
    runs_root = ROOT / cfg["paths"]["runs_root"]
    scores = pd.concat([aggregate_dataset(runs_root, dataset, windows) for dataset in ("a_only", "a_us_coral")], ignore_index=True)
    scores.to_parquet(output / "data" / "time_placebo_scores.parquet", index=False)
    result = inference(scores, int(cfg["inference"]["bootstrap_repetitions"]),
                       int(cfg["inference"]["permutation_repetitions"]), int(cfg["inference"]["random_seed"]) + 30_000)
    result.to_csv(output / "inference" / "time_placebo_inference.csv", index=False, encoding="utf-8-sig")
    counts = scores.loc[(scores.window_type == "placebo") & (scores.n_scored_days > 0)].groupby("dataset").size().to_dict()
    print({"events": int(cases.event_id.nunique()), "placebo_windows_with_scores": counts})


if __name__ == "__main__":
    main()

