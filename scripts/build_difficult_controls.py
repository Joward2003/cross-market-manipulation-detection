#!/usr/bin/env python3
"""Build same-calendar, abnormal A-share controls for each CSRC episode.

The matching target uses information observed during the candidate event
window but never post-event outcomes.  Controls are not called legal negatives:
they are non-penalized candidates as of the enforcement-panel coverage date.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/posttraining_analysis.yaml")
    parser.add_argument("--force-rebuild-aggregates", action="store_true")
    return parser.parse_args()


def normalise_symbol(value: object) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip().upper()
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    match = re.search(r"(?<!\d)(\d{6})(?!\d)", text)
    return match.group(1) if match else None


def load_events(cfg: dict) -> tuple[pd.DataFrame, set[str]]:
    workbook = ROOT / cfg["paths"]["events_workbook"]
    sheet = cfg["paths"]["events_sheet"]
    strict = pd.read_excel(workbook, sheet_name=sheet, dtype=object)
    all_regulated: set[str] = set()
    for candidate_sheet in ("A股严格可用", "A股宽口径"):
        frame = pd.read_excel(workbook, sheet_name=candidate_sheet, dtype=object)
        all_regulated.update(x for x in frame["证券代码（自动）"].map(normalise_symbol) if x)
    events = pd.DataFrame({
        "event_id": strict["事件ID"].astype(str),
        "case_name": strict["被操纵标的"].astype(str),
        "case_symbol6": strict["证券代码（自动）"].map(normalise_symbol),
        "start_ts": pd.to_datetime(strict["操纵开始日"], errors="coerce"),
        "end_ts": pd.to_datetime(strict["操纵结束日"], errors="coerce"),
        "document_id": strict["决定书编号"].astype(str),
    }).dropna(subset=["case_symbol6", "start_ts", "end_ts"])
    events = events[events.start_ts.dt.year >= int(cfg["sample"]["primary_start_year"])].copy()
    events["start_date"] = events.start_ts.dt.strftime("%Y%m%d").astype("int32")
    events["end_date"] = events.end_ts.dt.strftime("%Y%m%d").astype("int32")
    return events, all_regulated


def aggregate_event_windows(cfg: dict, events: pd.DataFrame, output: Path, force: bool) -> pd.DataFrame:
    aggregate_path = output / "data" / "event_candidate_aggregates.parquet"
    aggregate_path.parent.mkdir(parents=True, exist_ok=True)
    if aggregate_path.exists() and not force:
        return pd.read_parquet(aggregate_path)
    feature_path = str(ROOT / cfg["paths"]["feature_panel"]).replace("'", "''")
    target_path = str(aggregate_path).replace("'", "''")
    registered = events[["event_id", "start_ts", "end_ts"]].copy()
    con = duckdb.connect()
    con.execute("PRAGMA threads=4")
    con.execute("PRAGMA memory_limit='12GB'")
    con.register("events", registered)
    con.execute(f"""
        COPY (
          SELECT
            e.event_id,
            p.symbol,
            regexp_extract(p.symbol, '([0-9]{{6}})', 1) AS symbol6,
            COUNT(*)::INTEGER AS n_days,
            SUM(p.log_return) AS cumulative_log_return,
            SUM(p.market_adjusted_return) AS cumulative_market_adjusted_return,
            STDDEV_SAMP(p.log_return) AS realized_return_volatility,
            AVG(p.turnover_shock_60) AS mean_turnover_shock,
            AVG(p.relative_volume_60) AS mean_relative_volume,
            AVG(p.intraday_range) AS mean_intraday_range,
            MEDIAN(p.circ_mv) AS median_circulating_market_value,
            MEDIAN(p.listing_age_days) AS median_listing_age,
            AVG(p.log_amihud_20) AS mean_log_amihud,
            AVG(CASE WHEN ABS(p.market_adjusted_return) >= 0.08 THEN 1.0 ELSE 0.0 END) AS extreme_return_share,
            AVG(ABS(p.ma_deviation_20)) AS mean_absolute_ma_deviation,
            ARG_MAX(p.exchange, p.trade_date) AS exchange,
            ARG_MAX(p.industry, p.trade_date) AS industry
          FROM read_parquet('{feature_path}', hive_partitioning=1) p
          JOIN events e ON p.trade_date BETWEEN e.start_ts AND e.end_ts
          WHERE p.feature_ready = TRUE AND p.market = 'cn'
          GROUP BY e.event_id, p.symbol
        ) TO '{target_path}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)
    con.close()
    return pd.read_parquet(aggregate_path)


def robust_location_scale(values: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    location = values.median(axis=0, skipna=True)
    scale = (values.sub(location).abs().median(axis=0, skipna=True) * 1.4826).replace(0, np.nan)
    fallback = values.std(axis=0, ddof=0).replace(0, 1.0)
    return location, scale.fillna(fallback).fillna(1.0)


def weighted_stats(values: pd.Series, weights: pd.Series) -> tuple[float, float]:
    mask = values.notna() & weights.notna() & (weights > 0)
    x = values.loc[mask].astype(float).to_numpy()
    w = weights.loc[mask].astype(float).to_numpy()
    if not len(x):
        return np.nan, np.nan
    w = w / w.sum()
    mean = float(np.sum(w * x))
    variance = float(np.sum(w * np.square(x - mean)))
    return mean, variance


def standardized_difference(case: pd.DataFrame, control: pd.DataFrame, feature: str, control_weight: str) -> dict:
    case_w = pd.Series(np.repeat(1.0 / len(case), len(case)), index=case.index)
    control_w = control[control_weight].astype(float)
    cm, cv = weighted_stats(case[feature], case_w)
    tm, tv = weighted_stats(control[feature], control_w)
    pooled = np.sqrt(max((cv + tv) / 2.0, 1e-12))
    return {"feature": feature, "case_mean": cm, "control_mean": tm, "smd": (cm - tm) / pooled}


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    output = ROOT / cfg["paths"]["output_root"]
    (output / "data").mkdir(parents=True, exist_ok=True)
    (output / "diagnostics").mkdir(parents=True, exist_ok=True)
    events, regulated_symbols = load_events(cfg)
    aggregates = aggregate_event_windows(cfg, events, output, args.force_rebuild_aggregates)
    aggregates["log_median_circulating_market_value"] = np.log1p(aggregates.median_circulating_market_value.clip(lower=0))
    aggregates["log_median_listing_age"] = np.log1p(aggregates.median_listing_age.clip(lower=0))
    features = list(cfg["matching"]["features"])
    weights = pd.Series(cfg["matching"]["features"], dtype=float)
    matches: list[pd.DataFrame] = []
    cases: list[pd.Series] = []
    unmatched_pools: list[pd.DataFrame] = []
    audit: list[dict] = []
    n_controls = int(cfg["matching"]["controls_per_event"])
    min_days = int(cfg["sample"]["minimum_scored_days"])
    coverage = float(cfg["sample"]["candidate_coverage_ratio"])
    for event in events.itertuples(index=False):
        block = aggregates.loc[aggregates.event_id == event.event_id].copy()
        case_rows = block.loc[block.symbol6 == event.case_symbol6]
        if case_rows.empty:
            audit.append({"event_id": event.event_id, "status": "missing_case_features", "n_candidates": 0})
            continue
        case = case_rows.sort_values("n_days", ascending=False).iloc[0].copy()
        candidates = block.loc[
            (block.symbol6 != event.case_symbol6)
            & (~block.symbol6.isin(regulated_symbols))
            & (block.n_days >= max(min_days, int(np.ceil(float(case.n_days) * coverage))))
        ].copy()
        candidates = candidates.dropna(subset=features)
        if bool(cfg["matching"]["exact_exchange"]):
            exact = candidates.loc[candidates.exchange == case.exchange]
            if len(exact) >= int(cfg["matching"]["minimum_exact_exchange_candidates"]):
                candidates = exact
        if len(candidates) < n_controls:
            audit.append({"event_id": event.event_id, "status": "insufficient_candidates", "n_candidates": len(candidates)})
            continue
        candidates.loc[:, features] = candidates[features].apply(pd.to_numeric, errors="coerce")
        case_values = pd.to_numeric(case[features], errors="coerce").astype(float)
        scale_frame = pd.concat([candidates[features].astype(float), pd.DataFrame([case_values])], ignore_index=True)
        location, scale = robust_location_scale(scale_frame)
        delta = candidates[features].astype(float).sub(case_values).div(scale.astype(float))
        weighted_square = np.square(delta.to_numpy(dtype=float)) * weights.to_numpy(dtype=float)[None, :]
        candidates["distance_continuous"] = np.sqrt(weighted_square.sum(axis=1) / float(weights.sum()))
        candidates["industry_mismatch"] = (candidates.industry.fillna("") != str(case.industry)).astype(float)
        candidates["exchange_mismatch"] = (candidates.exchange.fillna("") != str(case.exchange)).astype(float)
        candidates["match_distance"] = (
            candidates.distance_continuous
            + candidates.industry_mismatch * float(cfg["matching"]["industry_mismatch_penalty"])
            + candidates.exchange_mismatch * float(cfg["matching"]["exchange_mismatch_penalty"])
        )
        selected = candidates.nsmallest(n_controls, "match_distance").copy()
        selected["control_rank"] = np.arange(1, len(selected) + 1)
        selected["case_symbol6"] = event.case_symbol6
        selected["case_name"] = event.case_name
        selected["start_date"] = event.start_date
        selected["end_date"] = event.end_date
        selected["document_id"] = event.document_id
        selected["control_weight"] = 1.0 / (len(events) * len(selected))
        matches.append(selected)
        case["case_name"] = event.case_name
        case["start_date"] = event.start_date
        case["end_date"] = event.end_date
        case["document_id"] = event.document_id
        cases.append(case)
        pool = candidates.copy()
        pool["control_weight"] = 1.0 / (len(events) * len(pool))
        unmatched_pools.append(pool)
        audit.append({"event_id": event.event_id, "status": "matched", "n_candidates": len(candidates),
                      "n_controls": len(selected), "best_distance": float(selected.match_distance.min()),
                      "worst_selected_distance": float(selected.match_distance.max())})
    matched = pd.concat(matches, ignore_index=True) if matches else pd.DataFrame()
    case_frame = pd.DataFrame(cases)
    unmatched = pd.concat(unmatched_pools, ignore_index=True) if unmatched_pools else pd.DataFrame()
    matched.to_parquet(output / "data" / "difficult_controls.parquet", index=False)
    matched.to_csv(output / "data" / "difficult_controls.csv", index=False, encoding="utf-8-sig")
    case_frame.to_parquet(output / "data" / "matched_cases.parquet", index=False)
    pd.DataFrame(audit).to_csv(output / "diagnostics" / "matching_audit.csv", index=False, encoding="utf-8-sig")
    balance_rows: list[dict] = []
    for sample_name, control in (("unmatched_candidate_pool", unmatched), ("matched_controls", matched)):
        if control.empty:
            continue
        for feature in features:
            row = standardized_difference(case_frame, control, feature, "control_weight")
            row["sample"] = sample_name
            balance_rows.append(row)
    balance = pd.DataFrame(balance_rows)
    balance.to_csv(output / "diagnostics" / "matching_balance.csv", index=False, encoding="utf-8-sig")
    summary = {
        "events_requested": int(len(events)),
        "events_matched": int(case_frame.event_id.nunique()) if not case_frame.empty else 0,
        "events_unmatched": int(len(events) - case_frame.event_id.nunique()) if not case_frame.empty else int(len(events)),
        "controls_selected": int(len(matched)),
        "unique_control_symbols": int(matched.symbol6.nunique()) if not matched.empty else 0,
        "controls_per_event": n_controls,
        "candidate_definition": "same calendar interval; >=80% case trading-day coverage; all known regulated symbols excluded",
        "legal_label_warning": "controls are non-penalized candidates, not proven legal negatives",
    }
    (output / "diagnostics" / "matching_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
