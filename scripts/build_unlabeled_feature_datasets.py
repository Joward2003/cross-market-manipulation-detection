#!/usr/bin/env python3
"""Build label-free, daily OHLCV feature panels and experiment manifests.

The script deliberately keeps a single physical, partitioned feature store and
uses manifests for A-only / A+US / A+HK / A+HK+US combinations.  Copying the
same A-share panel four times is wasteful and makes audits harder.  CORAL and
DANN are fit-time transformations, not globally transformed data files; their
specifications are therefore written as experiment manifests.

No enforcement labels, event dates, forward fills, global standardisation, or
global CORAL transforms enter this script.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed" / "unlabeled_feature_datasets"

FEATURE_COLUMNS = [
    "log_return",
    "market_adjusted_return",
    "intraday_range",
    "log_rolling_volatility_20",
    "relative_volume_60",
    "turnover_shock_60",
    "log_amihud_20",
    "price_volume_corr_20",
    "ma_deviation_20",
    "listing_age_log_days",
    "zero_volume_flag",
    "suspension_resume_flag",
    "close_location_value",
    "downside_volatility_20",
    "tail_return_20",
]

CONTINUOUS_FEATURES = [x for x in FEATURE_COLUMNS if not x.endswith("_flag")]
SHARED_FEATURE_COLUMNS = [
    "log_return", "market_adjusted_return", "intraday_range", "log_rolling_volatility_20",
    "relative_volume_60", "turnover_shock_60", "log_amihud_20", "price_volume_corr_20",
    "ma_deviation_20", "close_location_value", "downside_volatility_20", "tail_return_20",
]
LOCAL_FEATURE_COLUMNS = ["listing_age_log_days", "zero_volume_flag", "suspension_resume_flag"]
RAW_BASE_COLUMNS = [
    "symbol", "trade_date", "market", "open", "high", "low", "close", "adj_close", "volume", "turnover_amount",
    "turnover_rate", "circ_mv", "listing_age_days", "listing_age_source", "exchange", "industry", "panel_origin",
]


def file_map(directory: Path, prefix: str) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for p in directory.glob(f"{prefix}_*.parquet"):
        out[p.stem.rsplit("_", 1)[-1]] = p
    return out


def valid_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    for c in ["adj_close", "close", "high", "low", "open", "volume", "turnover_amount"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["missing_core"] = df[["adj_close", "close", "high", "low", "open", "volume"]].isna().any(axis=1)
    df["bad_positive"] = (
        (df[["adj_close", "close", "high", "low", "open"]] <= 0).any(axis=1)
        | (df["volume"] < 0)
    )
    df["bad_ohlc"] = (
        (df["high"] < df[["open", "close", "low"]].max(axis=1))
        | (df["low"] > df[["open", "close", "high"]].min(axis=1))
    )
    df["valid_row"] = ~(df["missing_core"] | df["bad_positive"] | df["bad_ohlc"])
    df["zero_volume_flag"] = df["volume"].fillna(0).eq(0).astype("int8")
    return df


def market_reference_from_batches(batches: list[Path], market: str, hk_universe: set[tuple[str, int]] | None) -> pd.DataFrame:
    """Clipped equal-weight daily return, aggregated without retaining all rows."""
    sums: dict[pd.Timestamp, float] = defaultdict(float)
    counts: dict[pd.Timestamp, int] = defaultdict(int)
    for p in batches:
        df = load_external_raw(p, market, hk_universe)
        if df.empty:
            continue
        df = valid_ohlcv(df.sort_values(["symbol", "trade_date"]))
        ret = np.log(df["adj_close"].where(df["valid_row"])).groupby(df["symbol"]).diff()
        use = ret.where(df["valid_row"]).clip(-0.30, 0.30)
        x = pd.DataFrame({"trade_date": df["trade_date"], "ret": use}).dropna()
        agg = x.groupby("trade_date")["ret"].agg(["sum", "count"])
        for dt, r in agg.iterrows():
            sums[dt] += float(r["sum"])
            counts[dt] += int(r["count"])
    out = pd.DataFrame({"trade_date": list(sums), "market_reference_return": [sums[k] / counts[k] for k in sums], "market_reference_n": [counts[k] for k in sums]})
    return out.sort_values("trade_date")


def add_common_features(df: pd.DataFrame, market_reference: pd.DataFrame, calendar_index: dict[pd.Timestamp, int]) -> pd.DataFrame:
    """Construct current-day and strictly lagged rolling features by security."""
    # Reset after sorting: the subsequent market-reference merge creates a new
    # RangeIndex.  Without this reset, Series derived before the merge (notably
    # MA deviation) can be assigned back by stale labels rather than row order.
    df = df.sort_values(["symbol", "trade_date"]).reset_index(drop=True).copy()
    df = valid_ohlcv(df)
    valid = df["valid_row"]
    adj = df["adj_close"].where(valid)
    high = df["high"].where(valid)
    low = df["low"].where(valid)
    close = df["close"].where(valid)
    open_ = df["open"].where(valid)
    vol = df["volume"].where(valid & df["volume"].gt(0))
    amount = df["turnover_amount"].where(valid & df["turnover_amount"].gt(0))

    df["log_return"] = np.log(adj).groupby(df["symbol"]).diff()
    df["intraday_range"] = np.log(high / low)
    df["close_location_value"] = ((2 * close - high - low) / (high - low)).where(high.gt(low))
    df["log_volume"] = np.log(vol)
    df["log_turnover"] = np.log(amount)
    df = df.merge(market_reference, on="trade_date", how="left", validate="many_to_one")
    df["market_adjusted_return"] = df["log_return"] - df["market_reference_return"]

    grp = df.groupby("symbol", sort=False)
    # All rolling statistics exclude t itself, preventing target-time leakage.
    med_volume = grp["log_volume"].transform(lambda s: s.shift(1).rolling(60, min_periods=20).median())
    med_turnover = grp["log_turnover"].transform(lambda s: s.shift(1).rolling(60, min_periods=20).median())
    df["relative_volume_60"] = df["log_volume"] - med_volume
    df["turnover_shock_60"] = df["log_turnover"] - med_turnover
    df["rolling_volatility_20"] = grp["log_return"].transform(lambda s: s.shift(1).rolling(20, min_periods=10).std())
    df["log_rolling_volatility_20"] = np.log(df["rolling_volatility_20"] + 1e-6)
    downside = df["log_return"].clip(upper=0).pow(2)
    df["downside_volatility_20"] = np.sqrt(downside.groupby(df["symbol"]).transform(lambda s: s.shift(1).rolling(20, min_periods=10).mean()))
    historical_ma = adj.groupby(df["symbol"]).transform(lambda s: s.shift(1).rolling(20, min_periods=10).median())
    # Log deviation is symmetric for proportional increases/decreases and is
    # invariant to the arbitrary level of a split/dividend-adjusted price.
    df["ma_deviation_20"] = np.log(adj) - np.log(historical_ma)
    illiq = (df["log_return"].abs() / amount).replace([np.inf, -np.inf], np.nan)
    df["amihud_20"] = illiq.groupby(df["symbol"]).transform(lambda s: s.shift(1).rolling(20, min_periods=10).mean())
    # Keep raw Amihud for interpretation, but use a declared epsilon-log
    # version in model features because the raw scale is highly right-skewed.
    df["log_amihud_20"] = np.log(df["amihud_20"] + 1e-14)

    # Lagged rolling price-volume correlation, expressed through moments for
    # speed and to avoid per-group Python apply calls.
    r = grp["log_return"].shift(1)
    v = grp["log_turnover"].shift(1)
    key = df["symbol"]
    mean_r = r.groupby(key).transform(lambda s: s.rolling(20, min_periods=10).mean())
    mean_v = v.groupby(key).transform(lambda s: s.rolling(20, min_periods=10).mean())
    mean_rv = (r * v).groupby(key).transform(lambda s: s.rolling(20, min_periods=10).mean())
    mean_r2 = r.pow(2).groupby(key).transform(lambda s: s.rolling(20, min_periods=10).mean())
    mean_v2 = v.pow(2).groupby(key).transform(lambda s: s.rolling(20, min_periods=10).mean())
    cov = mean_rv - mean_r * mean_v
    denom = np.sqrt((mean_r2 - mean_r.pow(2)) * (mean_v2 - mean_v.pow(2)))
    df["price_volume_corr_20"] = (cov / denom).replace([np.inf, -np.inf], np.nan).clip(-1, 1)
    df["tail_return_20"] = (df["log_return"] / df["rolling_volatility_20"]).replace([np.inf, -np.inf], np.nan)

    df["market_day_index"] = df["trade_date"].map(calendar_index).astype("Int64")
    prior_idx = df.groupby("symbol")["market_day_index"].diff()
    df["suspension_gap_trading_days"] = (prior_idx - 1).clip(lower=0).fillna(0).astype("int16")
    df["suspension_resume_flag"] = df["suspension_gap_trading_days"].gt(0).astype("int8")
    df["listing_age_log_days"] = np.log1p(df["listing_age_days"].clip(lower=0))
    df["feature_ready"] = valid & df[CONTINUOUS_FEATURES].notna().all(axis=1)
    return df


def build_cn_base() -> pd.DataFrame:
    daily_root = RAW / "cn" / "daily"
    basic_root = RAW / "cn" / "daily_basic"
    adj_root = RAW / "cn" / "adj_factor"
    master = pd.read_csv(RAW / "reference" / "a_share_master.csv", dtype={"ts_code": str})
    master["list_date"] = pd.to_datetime(master["list_date"], format="%Y%m%d", errors="coerce")
    master = master[["ts_code", "list_date", "market", "exchange", "industry"]]

    all_dates = sorted(pd.to_datetime([p.stem.rsplit("_", 1)[-1] for p in daily_root.glob("year=*/*.parquet")], format="%Y%m%d"))
    calendar_index = {dt: i for i, dt in enumerate(all_dates)}
    previous_tail = pd.DataFrame()
    summary: list[dict] = []
    for year in range(2011, 2026):
        dmap = file_map(daily_root / f"year={year}", "daily")
        bmap = file_map(basic_root / f"year={year}", "daily_basic")
        amap = file_map(adj_root / f"year={year}", "adj_factor")
        parts: list[pd.DataFrame] = []
        for date_key, daily_path in sorted(dmap.items()):
            d = pd.read_parquet(daily_path, columns=["ts_code", "trade_date", "open", "high", "low", "close", "vol", "amount"])
            d = d[~d["ts_code"].astype(str).str.endswith(".BJ")]
            b = pd.read_parquet(bmap[date_key], columns=["ts_code", "trade_date", "turnover_rate", "circ_mv"]) if date_key in bmap else pd.DataFrame(columns=["ts_code", "trade_date", "turnover_rate", "circ_mv"])
            a = pd.read_parquet(amap[date_key], columns=["ts_code", "trade_date", "adj_factor"]) if date_key in amap else pd.DataFrame(columns=["ts_code", "trade_date", "adj_factor"])
            x = d.merge(b, on=["ts_code", "trade_date"], how="left", validate="one_to_one").merge(a, on=["ts_code", "trade_date"], how="left", validate="one_to_one")
            parts.append(x)
        current = pd.concat(parts, ignore_index=True)
        current = current.merge(master, on="ts_code", how="left", validate="many_to_one")
        current = current.rename(columns={"ts_code": "symbol", "vol": "volume"})
        current["trade_date"] = pd.to_datetime(current["trade_date"].astype(str), format="%Y%m%d")
        current["market"] = "cn"
        current["adj_close"] = pd.to_numeric(current["close"], errors="coerce") * pd.to_numeric(current["adj_factor"], errors="coerce")
        # Tushare amount is in thousand CNY; rescaling does not affect shocks,
        # and makes the Amihud unit explicit.
        current["turnover_amount"] = pd.to_numeric(current["amount"], errors="coerce") * 1000.0
        current["listing_age_days"] = (current["trade_date"] - current["list_date"]).dt.days
        current["listing_age_source"] = "exchange_listing_date"
        current["panel_origin"] = "current"
        current = current[RAW_BASE_COLUMNS]
        combined = pd.concat([previous_tail, current], ignore_index=True) if not previous_tail.empty else current
        # Market reference is a clipped equal-weight return among all A-share
        # observations available that day; this is a reproducible proxy pending
        # a separate index-series robustness check.
        tmp = valid_ohlcv(combined.sort_values(["symbol", "trade_date"]))
        tmp_ret = np.log(tmp["adj_close"].where(tmp["valid_row"])).groupby(tmp["symbol"]).diff().where(tmp["valid_row"]).clip(-0.30, 0.30)
        ref = pd.DataFrame({"trade_date": tmp["trade_date"], "r": tmp_ret}).dropna().groupby("trade_date")["r"].agg(["mean", "count"]).reset_index().rename(columns={"mean": "market_reference_return", "count": "market_reference_n"})
        featured = add_common_features(combined, ref, calendar_index)
        out = featured[featured["panel_origin"].eq("current")].copy()
        out["year"] = out["trade_date"].dt.year.astype("int16")
        target = OUT / "base_feature_panel" / "market=cn" / f"year={year}" / "part.parquet"
        target.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(target, index=False)
        summary.append({"market": "cn", "year": year, "rows": len(out), "symbols": out["symbol"].nunique(), "feature_ready": int(out["feature_ready"].sum()), "bad_ohlc": int(out["bad_ohlc"].sum())})
        # Retain only raw columns as history.  Keeping old derived columns
        # would duplicate market-reference fields when the next year is merged.
        history_cols = [c for c in RAW_BASE_COLUMNS if c != "panel_origin"]
        previous_tail = featured.groupby("symbol", sort=False).tail(80)[history_cols].copy()
    return pd.DataFrame(summary)


def normalize_hk_symbol(s: pd.Series) -> pd.Series:
    code = s.astype("string").str.upper().str.replace(r"\.HK$", "", regex=True).str.zfill(5)
    return code + ".HK"


def load_external_raw(path: Path, market: str, hk_universe: set[tuple[str, int]] | None) -> pd.DataFrame:
    x = pd.read_parquet(path, columns=["trade_date", "yahoo_symbol", "adj_close", "close", "high", "low", "open", "vol"])
    x = x.rename(columns={"yahoo_symbol": "symbol", "vol": "volume"})
    x["trade_date"] = pd.to_datetime(x["trade_date"].astype(str), format="%Y%m%d", errors="coerce")
    x["symbol"] = normalize_hk_symbol(x["symbol"]) if market == "hk" else x["symbol"].astype("string").str.upper().str.replace(".", "-", regex=False)
    x["year"] = x["trade_date"].dt.year.astype(int)
    if hk_universe is not None:
        mask = [(str(s), int(y)) in hk_universe for s, y in zip(x["symbol"], x["year"])]
        x = x.loc[mask].copy()
    x["market"] = market
    x["turnover_amount"] = pd.to_numeric(x["close"], errors="coerce") * pd.to_numeric(x["volume"], errors="coerce")
    return x


def external_listing_metadata(market: str) -> pd.DataFrame:
    if market == "hk":
        m = pd.read_csv(RAW / "reference" / "hk_master.csv", dtype=str)
        m["symbol"] = normalize_hk_symbol(m["ts_code"])
        m["list_date"] = pd.to_datetime(m["list_date"], format="%Y%m%d", errors="coerce")
        return m[["symbol", "list_date"]].drop_duplicates("symbol")
    return pd.DataFrame(columns=["symbol", "list_date"])


def build_external_base(market: str) -> pd.DataFrame:
    raw_dir = RAW / market / "yfinance_daily"
    batches = sorted(raw_dir.glob("*.parquet"))
    hk_universe = None
    if market == "hk":
        u = pd.read_csv(ROOT / "data" / "processed" / "cross_market" / "hk_dynamic_universe.csv", dtype={"symbol": str})
        hk_universe = {(str(s), int(y)) for s, y in zip(u["symbol"], u["selection_year"])}
    reference = market_reference_from_batches(batches, market, hk_universe)
    calendar_index = {dt: i for i, dt in enumerate(reference["trade_date"].sort_values())}
    metadata = external_listing_metadata(market)
    summary: list[dict] = []
    for path in batches:
        current = load_external_raw(path, market, hk_universe)
        if current.empty:
            continue
        current = current.merge(metadata, on="symbol", how="left", validate="many_to_one")
        current["list_date"] = pd.to_datetime(current["list_date"], errors="coerce")
        first_seen = current.groupby("symbol")["trade_date"].transform("min")
        actual_age = (current["trade_date"] - current["list_date"]).dt.days
        current["listing_age_days"] = actual_age.where(current["list_date"].notna(), (current["trade_date"] - first_seen).dt.days)
        current["listing_age_source"] = np.where(current["list_date"].notna(), "exchange_listing_date", "first_observed_date_proxy")
        current["panel_origin"] = "current"
        keep = ["symbol", "trade_date", "market", "open", "high", "low", "close", "adj_close", "volume", "turnover_amount", "listing_age_days", "listing_age_source", "panel_origin"]
        featured = add_common_features(current[keep], reference, calendar_index)
        featured["year"] = featured["trade_date"].dt.year.astype("int16")
        for year, out in featured.groupby("year"):
            target = OUT / "base_feature_panel" / f"market={market}" / f"year={int(year)}" / f"{path.stem}.parquet"
            target.parent.mkdir(parents=True, exist_ok=True)
            out.to_parquet(target, index=False)
            summary.append({"market": market, "year": int(year), "rows": len(out), "symbols": out["symbol"].nunique(), "feature_ready": int(out["feature_ready"].sum()), "bad_ohlc": int(out["bad_ohlc"].sum())})
    reference.to_parquet(OUT / "market_reference_returns" / f"market={market}.parquet", index=False)
    return pd.DataFrame(summary)


def write_manifests(summary: pd.DataFrame) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "base_feature_panel"
    datasets = {
        "a_only": ["cn"],
        "a_us_static": ["cn", "us"],
        "a_hk": ["cn", "hk"],
        "a_hk_us_static": ["cn", "hk", "us"],
    }
    catalog = {
        "base_path": str(path.relative_to(ROOT)),
        "feature_columns": FEATURE_COLUMNS,
        "shared_feature_columns": SHARED_FEATURE_COLUMNS,
        "local_feature_columns": LOCAL_FEATURE_COLUMNS,
        "continuous_features": CONTINUOUS_FEATURES,
        "row_filter": "feature_ready == True; do not impute raw feature values before fold-specific scaling",
        "datasets": {},
        "adaptation": {},
        "notes": [
            "All datasets are label-free. CSRC case labels are intentionally absent.",
            "Each combination is a manifest over one physical partitioned store; no duplicate A-share panels are written.",
            "CORAL and DANN are fit within each training fold only. A globally transformed CORAL panel would leak evaluation-period distributional information.",
            "market_adjusted_return uses the same-market clipped equal-weight daily return proxy. A benchmark-index version is reserved for robustness once index data are added.",
        ],
    }
    for name, markets in datasets.items():
        catalog["datasets"][name] = {
            "markets": markets,
            "domain_ids": {m: i for i, m in enumerate(markets)},
            "feature_columns": FEATURE_COLUMNS,
            "alignment_feature_columns": SHARED_FEATURE_COLUMNS,
            "local_feature_columns": LOCAL_FEATURE_COLUMNS,
        }
        if len(markets) > 1:
            source_markets = [m for m in markets if m != "cn"]
            representation_mix = {"cn": 0.50}
            for source in source_markets:
                representation_mix[source] = 0.50 / len(source_markets)
            catalog["adaptation"][f"{name}_coral"] = {
                "base_dataset": name, "method": "CORAL", "target_market": "cn", "source_markets": source_markets,
                "alignment_feature_columns": SHARED_FEATURE_COLUMNS,
                "fit_scope": "training fold only",
                "preprocess": "fit per-market robust scaler on training fold; align each source covariance to cn covariance separately; balance markets in minibatches",
                "window_sampling": {
                    "window_length": 60, "train_stride": 5, "hierarchy": ["market", "calendar_year", "symbol", "window_start"],
                    "representation_market_mix": representation_mix, "one_window_per_symbol_per_epoch": True,
                    "pairwise_source_target_loss_weight": "equal across source markets", "covariance_ridge": 1e-4,
                },
            }
            catalog["adaptation"][f"{name}_dann"] = {
                "base_dataset": name, "method": "DANN", "target_market": "cn", "source_markets": source_markets,
                "alignment_feature_columns": SHARED_FEATURE_COLUMNS,
                "fit_scope": "training fold only",
                "preprocess": "fit training-fold scaler only; use market domain_id for adversarial head; do not feed domain_id to anomaly head",
                "window_sampling": {
                    "window_length": 60, "train_stride": 5, "hierarchy": ["market", "calendar_year", "symbol", "window_start"],
                    "representation_market_mix": representation_mix, "domain_loss_market_mix": "equal per market",
                    "one_window_per_symbol_per_epoch": True, "gradient_reversal_schedule": "linear warm-up over first 30% of steps",
                    "domain_head": "shared representation only", "collapse_checks": ["representation_variance", "self_supervised_loss", "domain_accuracy"],
                },
            }
    catalog["adaptation"]["a_only"] = {"method": "none", "reason": "A single market has no cross-domain alignment problem."}
    (OUT / "dataset_catalog.json").write_text(json.dumps(catalog, ensure_ascii=False, indent=2), encoding="utf-8")
    summary.to_csv(OUT / "feature_panel_quality_summary.csv", index=False)


def write_readme() -> None:
    text = """# 无标签特征数据集

本目录包含一个按 `market/year` 分区的物理特征面板，以及四个逻辑数据集视图：
`a_only`、`a_us_static`、`a_hk`、`a_hk_us_static`。具体市场筛选和特征列见
`dataset_catalog.json`。四个视图不复制A股数据，以避免同一数据有多份难以审计的副本。

## 特征口径

- `log_return`：复权收盘价对数收益率。
- `market_adjusted_return`：减去当日同市场、截尾[-30%,30%]的等权收益率。
- `intraday_range`：`log(high / low)`；`close_location_value` 表示收盘相对日内区间的位置。
- `log_rolling_volatility_20`、`downside_volatility_20`、`tail_return_20`：波动率模型列取对数，且仅使用t-1及更早收益率；原始 `rolling_volatility_20` 同时保留。
- `relative_volume_60`、`turnover_shock_60`：相对过去60个交易日中位数的对数偏离。
- `log_amihud_20`：原始 `amihud_20`（过去20日 `|return| / amount` 均值）的epsilon对数模型列；两者均保留。
- `price_volume_corr_20`：过去20日收益率与对数成交额的相关系数。
- `ma_deviation_20`：`log(AdjClose_t) - log(过去20日复权价格中位数)`。
- `listing_age_log_days`、`zero_volume_flag`、`suspension_resume_flag`：年龄、零成交及停牌后恢复交易标记。

`suspension_resume_flag` 不把缺失日伪造为零收益交易；它在恢复交易日记录前一次可观察交易与市场日历之间的缺口。美股缺少可靠的逐股上市日期，年龄改用首次可观察行情日代理，并由 `listing_age_source` 标记。

## 方法约束

CORAL和DANN并非额外的价格数据集：它们必须在每个时间训练折内拟合缩放、协方差对齐或域对抗网络。详见 `dataset_catalog.json`。不得使用CSRC处罚标签、处罚期间或全样本标准化参数来构造本目录。

## 文献依据

Amihud (2002) 为基于收益冲击与成交额的非流动性指标提供依据；Karpoff (1987) 讨论价格变化与交易量关系；Corwin and Schultz (2012) 说明高低价范围可提供交易成本/流动性信息；Liu et al. (2021) 是使用证监会处罚案例与量价变量的直接A股基准。完整链接见处理报告。
"""
    (OUT / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-cn", action="store_true")
    parser.add_argument("--skip-external", action="store_true")
    args = parser.parse_args()
    (OUT / "market_reference_returns").mkdir(parents=True, exist_ok=True)
    summaries: list[pd.DataFrame] = []
    if not args.skip_cn:
        summaries.append(build_cn_base())
    if not args.skip_external:
        summaries.extend([build_external_base("us"), build_external_base("hk")])
    summary = pd.concat([x for x in summaries if not x.empty], ignore_index=True)
    existing_path = OUT / "feature_panel_quality_summary.csv"
    if existing_path.exists() and not summary.empty:
        existing = pd.read_csv(existing_path)
        summary = pd.concat([existing[~existing["market"].isin(summary["market"].unique())], summary], ignore_index=True)
    write_manifests(summary)
    write_readme()
    print(summary.groupby("market").agg(rows=("rows", "sum"), symbols=("symbols", "sum"), feature_ready=("feature_ready", "sum")).to_string())


if __name__ == "__main__":
    main()
