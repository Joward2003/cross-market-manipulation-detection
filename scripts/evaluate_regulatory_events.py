#!/usr/bin/env python3
"""Externally evaluate already-produced A-share scores against CSRC cases.

This is deliberately a post-training evaluation command.  It never writes to
the model, the scaler, or the anomaly boundary.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--events", type=Path, default=ROOT / "outputs/csrc_a_share_manipulation_event_panel_2012_2025.xlsx")
    parser.add_argument("--sheet", default="A股严格可用")
    parser.add_argument("--start-year", type=int, default=2018, help="Primary held-out event window starts in 2018.")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def pick_column(frame: pd.DataFrame, expected: str) -> str:
    if expected in frame.columns:
        return expected
    candidates = [x for x in frame.columns if expected.replace("（自动）", "") in str(x)]
    if len(candidates) == 1:
        return candidates[0]
    raise KeyError(f"Cannot locate required column {expected!r}; columns are {list(frame.columns)}")


def normalise_symbol(value: object) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip().upper()
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    # The CSRC workbook intentionally preserves the six-digit historical
    # code; price files use the same code plus a market suffix (e.g.
    # 600876.SH).  Exchange is unnecessary for this A-share-only join, but a
    # non-six-digit string must be audited rather than silently matched.
    match = re.search(r"(?<!\d)(\d{6})(?!\d)", text)
    return match.group(1) if match else None


def normalise_date(series: pd.Series) -> pd.Series:
    """Accept Excel datetimes, YYYYMMDD numbers, and already-formatted text."""
    numeric = pd.to_numeric(series, errors="coerce")
    out = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
    yyyymmdd = numeric.between(19000101, 21001231)
    if yyyymmdd.any():
        out.loc[yyyymmdd] = pd.to_datetime(numeric.loc[yyyymmdd].astype("int64").astype(str), format="%Y%m%d", errors="coerce")
    remainder = ~yyyymmdd
    # pandas handles normal datetime cells and date strings here.  A rare raw
    # Excel serial is handled explicitly rather than being misread as ns since
    # 1970.
    excel_serial = numeric.between(20_000, 80_000) & remainder
    if excel_serial.any():
        out.loc[excel_serial] = pd.to_datetime(numeric.loc[excel_serial], unit="D", origin="1899-12-30", errors="coerce")
    parse_text = remainder & ~excel_serial
    if parse_text.any():
        out.loc[parse_text] = pd.to_datetime(series.loc[parse_text], errors="coerce")
    return out.dt.strftime("%Y%m%d")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw = pd.read_excel(args.events, sheet_name=args.sheet, dtype=object)
    symbol_col = pick_column(raw, "证券代码（自动）")
    start_col = pick_column(raw, "操纵开始日")
    end_col = pick_column(raw, "操纵结束日")
    id_col = pick_column(raw, "事件ID")
    events = pd.DataFrame({
        "event_id": raw[id_col].astype(str),
        "symbol": raw[symbol_col].map(normalise_symbol),
        "start_date": normalise_date(raw[start_col]),
        "end_date": normalise_date(raw[end_col]),
    }).dropna()
    events["start_date"] = events["start_date"].astype("int32")
    events["end_date"] = events["end_date"].astype("int32")
    events = events[events["start_date"] >= args.start_year * 10000].copy()
    if events.empty:
        raise ValueError("No strict events remain after the requested held-out start year.")
    con = duckdb.connect()
    con.register("events", events)
    score_path = str(args.scores).replace("'", "''")
    result = con.execute(f"""
        SELECT e.event_id, e.symbol, e.start_date, e.end_date,
               COUNT(s.trade_date) AS n_scored_days,
               MAX(s.svdd_percentile_same_day) AS max_svdd_percentile,
               AVG(s.svdd_percentile_same_day) AS mean_svdd_percentile,
               AVG(CASE WHEN s.svdd_percentile_same_day >= 0.99 THEN 1.0 ELSE 0.0 END) AS share_top_1pct,
               AVG(CASE WHEN s.svdd_percentile_same_day >= 0.95 THEN 1.0 ELSE 0.0 END) AS share_top_5pct,
               MAX(s.iforest_percentile_same_day) AS max_iforest_percentile,
               AVG(s.iforest_percentile_same_day) AS mean_iforest_percentile
        FROM events e
        LEFT JOIN (
          SELECT *, regexp_extract(symbol, '([0-9]{{6}})', 1) AS symbol6
          FROM read_parquet('{score_path}')
        ) s
          ON e.symbol = s.symbol6 AND s.trade_date BETWEEN e.start_date AND e.end_date
        GROUP BY ALL
        ORDER BY e.event_id
    """).df()
    con.close()
    result.to_parquet(args.output_dir / "event_level_scores.parquet", index=False)
    result.to_csv(args.output_dir / "event_level_scores.csv", index=False, encoding="utf-8-sig")
    events.to_csv(args.output_dir / "events_normalized_for_join.csv", index=False, encoding="utf-8-sig")
    valid = result.loc[result.n_scored_days > 0].copy()
    summary = {
        "events_requested": int(len(events)), "events_with_valid_six_digit_code": int(events.symbol.notna().sum()),
        "events_with_scored_days": int(len(valid)),
        "events_without_scored_days": int(len(events) - len(valid)),
        "event_ids_without_scored_days": result.loc[result.n_scored_days <= 0, "event_id"].astype(str).tolist(),
        "metric_denominator": int(len(valid)),
        "mean_event_max_svdd_percentile": float(valid.max_svdd_percentile.mean()),
        "median_event_max_svdd_percentile": float(valid.max_svdd_percentile.median()),
        "mean_event_mean_svdd_percentile": float(valid.mean_svdd_percentile.mean()),
        "hit_at_1_percent": float((valid.max_svdd_percentile >= 0.99).mean()),
        "hit_at_5_percent": float((valid.max_svdd_percentile >= 0.95).mean()),
        "mean_share_top_1pct": float(valid.share_top_1pct.mean()),
        "mean_share_top_5pct": float(valid.share_top_5pct.mean()),
        "score_file": str(args.scores), "event_file": str(args.events), "primary_start_year": args.start_year,
    }
    (args.output_dir / "event_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
