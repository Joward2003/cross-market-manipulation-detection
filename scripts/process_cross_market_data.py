#!/usr/bin/env python3
"""Build auditable US/HK cross-market clean layers.

Raw Yahoo files are never modified.  The script creates:

* yearly HK liquidity universes selected using the previous calendar year;
* a cleaned static-US layer and a cleaned point-in-time-US layer where files exist;
* cleaned HK rows for the selected yearly universe;
* quality and attrition reports.

The output is deliberately feature-level, not a model-training dataset.  No
forward filling, interpolation, winsorisation, or label information is used.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed" / "cross_market"


def yahoo_files(market: str, pit: bool = False) -> list[Path]:
    if pit:
        return sorted((RAW / "us" / "yfinance_daily_point_in_time").glob("year=*/*.parquet"))
    return sorted((RAW / market / "yfinance_daily").glob("*.parquet"))


def normalize_symbol(s: pd.Series, market: str) -> pd.Series:
    out = s.astype("string").str.strip().str.upper()
    if market == "hk":
        # Yahoo and Tushare both use five-digit HK codes, but Yahoo can omit
        # leading zeros in some downloads.
        code = out.str.replace(r"\.HK$", "", regex=True).str.zfill(5)
        return code + ".HK"
    return out.str.replace(".", "-", regex=False)


def load_yahoo(path: Path, market: str, pit: bool = False) -> pd.DataFrame:
    cols = ["trade_date", "yahoo_symbol", "adj_close", "close", "high", "low", "open", "vol"]
    extra = ["membership_year", "membership_asof"] if pit else []
    df = pd.read_parquet(path, columns=cols + extra)
    df = df.rename(columns={"yahoo_symbol": "symbol", "vol": "volume"})
    df["symbol"] = normalize_symbol(df["symbol"], market)
    df["trade_date"] = pd.to_datetime(df["trade_date"].astype(str), format="%Y%m%d", errors="coerce")
    for c in ["adj_close", "close", "high", "low", "open", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["market"] = market
    if pit:
        df["membership_year"] = pd.to_numeric(df["membership_year"], errors="coerce").astype("Int64")
    return df


def add_quality_flags(df: pd.DataFrame) -> pd.DataFrame:
    core = ["adj_close", "close", "high", "low", "open", "volume"]
    df["missing_core"] = df[core].isna().any(axis=1)
    df["bad_positive"] = (
        (df[["adj_close", "close", "high", "low", "open"]] <= 0).any(axis=1)
        | (df["volume"] < 0)
    )
    df["bad_ohlc"] = (
        (df["high"] < df[["open", "close", "low"]].max(axis=1))
        | (df["low"] > df[["open", "close", "high"]].min(axis=1))
    )
    df["valid_row"] = ~(df["missing_core"] | df["bad_positive"] | df["bad_ohlc"])
    df["zero_volume"] = df["volume"].fillna(0).eq(0)
    df["dollar_turnover"] = (df["close"] * df["volume"]).where(df["valid_row"] & (df["volume"] > 0))
    return df


def yearly_metrics(paths: list[Path], market: str) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for path in paths:
        df = add_quality_flags(load_yahoo(path, market))
        df["year"] = df["trade_date"].dt.year.astype("Int64")
        g = df.groupby(["symbol", "year"], dropna=False)
        x = g.agg(
            n_rows=("trade_date", "size"),
            n_valid=("valid_row", "sum"),
            n_bad_ohlc=("bad_ohlc", "sum"),
            n_nonzero=("zero_volume", lambda s: (~s).sum()),
            median_turnover=("dollar_turnover", "median"),
            first_date=("trade_date", "min"),
            last_date=("trade_date", "max"),
        ).reset_index()
        rows.append(x)
    metrics = pd.concat(rows, ignore_index=True)
    # Duplicate source rows are not silently aggregated: keep a quality count.
    metrics = metrics.groupby(["symbol", "year"], as_index=False).agg(
        n_rows=("n_rows", "sum"), n_valid=("n_valid", "sum"),
        n_bad_ohlc=("n_bad_ohlc", "sum"), n_nonzero=("n_nonzero", "sum"),
        median_turnover=("median_turnover", "median"), first_date=("first_date", "min"),
        last_date=("last_date", "max"),
    )
    max_rows = metrics.groupby("year")["n_rows"].transform("max")
    metrics["coverage_ratio"] = metrics["n_rows"] / max_rows
    metrics["nonzero_ratio"] = metrics["n_nonzero"] / metrics["n_rows"].replace(0, np.nan)
    metrics["bad_ohlc_ratio"] = metrics["n_bad_ohlc"] / metrics["n_rows"].replace(0, np.nan)
    return metrics


def hk_ordinary_symbols() -> set[str]:
    master_path = RAW / "reference" / "hk_master.csv"
    master = pd.read_csv(master_path, dtype=str)
    master["ts_code"] = normalize_symbol(master["ts_code"], "hk")
    name = master["name"].fillna("").astype(str).str.upper()
    # Ordinary main-board/GEM shares only.  RMB double counters are excluded
    # by their explicit suffix; this avoids guessing from the numeric code.
    excluded = name.str.contains(
        r"ETF|REIT|基金|信托|债|票据|权证|优先|优先股|-R$|-SWR$", regex=True
    )
    return set(master.loc[~excluded, "ts_code"].dropna())


def select_hk_universe(metrics: pd.DataFrame, top_n: int, min_coverage: float, min_nonzero: float) -> pd.DataFrame:
    ordinary = hk_ordinary_symbols()
    m = metrics[metrics["symbol"].isin(ordinary)].copy()
    selected: list[pd.DataFrame] = []
    for year in range(int(m["year"].min()), int(m["year"].max()) + 1):
        prior = m[m["year"].eq(year - 1)].copy()
        if prior.empty:
            # 2011 is a warm-up year: its pool is not used for confirmatory
            # inference because no prior-year liquidity information exists.
            current = m[m["year"].eq(year)].sort_values("median_turnover", ascending=False).head(top_n).copy()
            current["selection_asof"] = str(year)
            current["selection_status"] = "warmup"
        else:
            eligible = prior[
                (prior["n_rows"] >= 120)
                & (prior["coverage_ratio"] >= min_coverage)
                & (prior["nonzero_ratio"] >= min_nonzero)
                & prior["median_turnover"].gt(0)
                & (prior["bad_ohlc_ratio"] <= 0.01)
            ].copy()
            current = eligible.sort_values("median_turnover", ascending=False).head(top_n)
            current["selection_asof"] = str(year - 1)
            current["selection_status"] = "point_in_time"
        current["selection_year"] = year
        selected.append(current)
    if not selected:
        return pd.DataFrame()
    out = pd.concat(selected, ignore_index=True)
    return out[["symbol", "selection_year", "selection_asof", "selection_status", "n_rows", "coverage_ratio", "nonzero_ratio", "bad_ohlc_ratio", "median_turnover", "first_date", "last_date"]]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["symbol", "trade_date"]).copy()
    valid = df["valid_row"]
    # Invalid observations are retained as flags but never used in numerical
    # features.  No forward filling is performed.
    # Mask invalid observations before taking logarithms.  This prevents an
    # invalid vendor row from contaminating the next valid return.
    adj = df["adj_close"].where(valid)
    high = df["high"].where(valid)
    low = df["low"].where(valid)
    close = df["close"].where(valid)
    open_ = df["open"].where(valid)
    df["ret_1d"] = np.log(adj).groupby(df["symbol"]).diff()
    df["range_log"] = np.log(high / low)
    df["close_open_log"] = np.log(close / open_)
    df["log_turnover"] = np.log(df["dollar_turnover"].where(df["dollar_turnover"] > 0))
    grp = df.groupby("symbol", group_keys=False)
    df["volume_shock_60"] = df["log_turnover"] - grp["log_turnover"].transform(lambda s: s.shift(1).rolling(60, min_periods=20).median())
    df["volatility_20"] = grp["ret_1d"].transform(lambda s: s.shift(1).rolling(20, min_periods=10).std())
    df["amihud_20"] = grp.apply(lambda x: (x["ret_1d"].abs() / x["dollar_turnover"]).shift(1).rolling(20, min_periods=10).mean(), include_groups=False).reset_index(level=0, drop=True)
    df["ma_dev_20"] = df["adj_close"] / grp["adj_close"].transform(lambda s: s.shift(1).rolling(20, min_periods=10).median()) - 1
    df["feature_valid"] = valid & df[["ret_1d", "range_log", "close_open_log"]].notna().all(axis=1)
    return df


def write_clean(paths: list[Path], market: str, universe: pd.DataFrame | None, out_name: str, pit: bool = False) -> pd.DataFrame:
    out_root = OUT / out_name
    out_root.mkdir(parents=True, exist_ok=True)
    quality_rows: list[dict] = []
    selected_map = None
    if universe is not None and not universe.empty:
        selected_map = {(str(r.symbol), int(r.selection_year)): r for r in universe.itertuples()}
    for path in paths:
        df = add_quality_flags(load_yahoo(path, market, pit=pit))
        df["year"] = df["trade_date"].dt.year.astype(int)
        if selected_map is not None:
            keep = []
            for r in df[["symbol", "year"]].itertuples(index=False):
                keep.append((str(r.symbol), int(r.year)) in selected_map)
            df = df.loc[keep].copy()
            if df.empty:
                continue
            df["selection_year"] = df["year"]
            df["selection_status"] = df.apply(lambda r: selected_map[(str(r.symbol), int(r.year))].selection_status, axis=1)
            df["selection_asof"] = df.apply(lambda r: selected_map[(str(r.symbol), int(r.year))].selection_asof, axis=1)
        elif pit:
            df["selection_year"] = df["membership_year"].astype("Int64")
            df["selection_status"] = "point_in_time"
            df["selection_asof"] = df["membership_asof"].astype(str)
        else:
            df["selection_year"] = df["year"]
            df["selection_status"] = "current_static_survivor_pool"
            df["selection_asof"] = "2026-07-27"
        df = add_features(df)
        for year, x in df.groupby("year"):
            quality_rows.append({
                "market": market, "dataset": out_name, "year": int(year), "rows": len(x),
                "symbols": int(x["symbol"].nunique()), "bad_ohlc": int(x["bad_ohlc"].sum()),
                "missing_core": int(x["missing_core"].sum()), "feature_valid": int(x["feature_valid"].sum()),
            })
        for year, x in df.groupby("year"):
            target = out_root / f"year={int(year)}" / f"{path.stem}.parquet"
            target.parent.mkdir(parents=True, exist_ok=True)
            x.to_parquet(target, index=False)
    return pd.DataFrame(quality_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hk-top-n", type=int, default=1000)
    parser.add_argument("--hk-min-coverage", type=float, default=0.80)
    parser.add_argument("--hk-min-nonzero", type=float, default=0.80)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    us_paths = yahoo_files("us")
    hk_paths = yahoo_files("hk")
    pit_paths = yahoo_files("us", pit=True)
    if not us_paths or not hk_paths:
        raise SystemExit("Expected raw US/HK Yahoo parquet files were not found")

    hk_metrics = yearly_metrics(hk_paths, "hk")
    hk_universe = select_hk_universe(hk_metrics, args.hk_top_n, args.hk_min_coverage, args.hk_min_nonzero)
    hk_universe.to_csv(OUT / "hk_dynamic_universe.csv", index=False)
    hk_metrics.to_csv(OUT / "hk_liquidity_metrics.csv", index=False)
    us_quality = write_clean(us_paths, "us", None, "us_static_clean")
    hk_quality = write_clean(hk_paths, "hk", hk_universe, "hk_dynamic_clean")
    quality = pd.concat([us_quality, hk_quality], ignore_index=True)
    if pit_paths:
        pit_quality = write_clean(pit_paths, "us", None, "us_pit_clean", pit=True)
        quality = pd.concat([quality, pit_quality], ignore_index=True)
    quality.to_csv(OUT / "quality_summary.csv", index=False)
    summary = {
        "hk_top_n": args.hk_top_n,
        "hk_min_coverage": args.hk_min_coverage,
        "hk_min_nonzero": args.hk_min_nonzero,
        "hk_universe_rows": int(len(hk_universe)),
        "hk_years": sorted(hk_universe["selection_year"].unique().tolist()) if not hk_universe.empty else [],
        "us_static_files": len(us_paths),
        "us_pit_files_found": len(pit_paths),
        "outputs": str(OUT),
        "no_forward_fill": True,
        "raw_files_modified": False,
    }
    (OUT / "processing_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
