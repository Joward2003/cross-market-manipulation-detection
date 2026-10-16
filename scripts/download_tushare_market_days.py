#!/usr/bin/env python3
"""Download Tushare market-wide daily partitions with resumable manifests.

The script deliberately stores one trading date per Parquet file.  A failed
network call can therefore be retried without re-downloading a whole year, and
the access credential never appears in a configuration or output file.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import re
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]


MARKET_DATASETS = {
    "cn": ("daily", "daily_basic", "adj_factor"),
    "hk": ("hk_daily",),
}
EXPECTED = {
    "daily": ["ts_code", "trade_date", "open", "high", "low", "close", "vol", "amount"],
    "daily_basic": ["ts_code", "trade_date", "turnover_rate", "turnover_rate_f", "total_mv", "circ_mv"],
    "adj_factor": ["ts_code", "trade_date", "adj_factor"],
    "hk_daily": ["ts_code", "trade_date", "open", "high", "low", "close", "vol", "amount"],
}


def session(provider: str) -> requests.Session:
    s = requests.Session()
    if provider == "super":
        key = os.environ.get("TUSHARE_SUPER_API_KEY")
        if not key:
            raise SystemExit("Set TUSHARE_SUPER_API_KEY; the script never stores credentials.")
        s.headers.update({"X-API-Key": key})
    return s


def api_frame(s: requests.Session, api_name: str, trade_date: str, provider: str) -> pd.DataFrame:
    params = {"trade_date": trade_date}
    if api_name in {"daily_basic", "adj_factor"}:
        params["limit"] = 6000
    if provider == "fast":
        token = os.environ.get("TUSHARE_FAST_TOKEN")
        if not token:
            raise SystemExit("Set TUSHARE_FAST_TOKEN; the script never stores credentials.")
        url = "https://fastapic.stockai888.top"
        payload = {"api_name": api_name, "token": token, "params": params}
    else:
        url = "https://ai-tool.indevs.in/tushare/pro"
        payload = {"api_name": api_name, "params": params}
    last: Exception | None = None
    for attempt in range(1, 7):
        try:
            r = s.post(url, json=payload, timeout=90)
            r.raise_for_status()
            obj = r.json()
            if obj.get("code") != 0:
                raise RuntimeError(f"Tushare code={obj.get('code')}: {obj.get('msg')}")
            data = obj.get("data") or {}
            fields, items = data.get("fields", []), data.get("items", [])
            df = pd.DataFrame(items, columns=fields).replace("", pd.NA)
            # Tushare's daily_basic occasionally mixes empty strings into an
            # otherwise numeric field.  Normalize only columns that are
            # overwhelmingly numeric; security identifiers remain strings.
            for col in df.columns:
                if col in {"ts_code", "trade_date"} or df[col].dtype != object:
                    continue
                observed = df[col].notna().sum()
                numeric = pd.to_numeric(df[col], errors="coerce")
                if observed == 0 or numeric.notna().sum() / observed >= 0.95:
                    df[col] = numeric
            return df
        except Exception as exc:  # retry only bounded API/network failures
            last = exc
            if attempt == 6:
                break
            time.sleep(min(30, 1.5 * 2 ** (attempt - 1)))
    raise RuntimeError(str(last))


def cn_trade_dates(s: requests.Session, start: str, end: str, provider: str) -> list[str]:
    if provider == "fast":
        token = os.environ.get("TUSHARE_FAST_TOKEN")
        if not token:
            raise SystemExit("Set TUSHARE_FAST_TOKEN; the script never stores credentials.")
        url = "https://fastapic.stockai888.top"
        payload = {"api_name": "trade_cal", "token": token, "params": {"exchange": "SSE", "start_date": start, "end_date": end, "is_open": "1", "limit": 10000}}
    else:
        url = "https://ai-tool.indevs.in/tushare/pro"
        payload = {"api_name": "trade_cal", "params": {"exchange": "SSE", "start_date": start, "end_date": end, "is_open": "1", "limit": 10000}}
    last: Exception | None = None
    for attempt in range(1, 7):
        try:
            r = s.post(url, json=payload, timeout=90)
            r.raise_for_status()
            obj = r.json()
            break
        except Exception as exc:
            last = exc
            if attempt == 6:
                raise
            time.sleep(min(30, 1.5 * 2 ** (attempt - 1)))
    else:
        raise RuntimeError(str(last))
    if obj.get("code") != 0:
        raise RuntimeError(f"trade_cal: {obj.get('code')} {obj.get('msg')}")
    d = obj.get("data") or {}
    out = pd.DataFrame(d.get("items", []), columns=d.get("fields", []))
    # Some Tushare-compatible gateways ignore the is_open request parameter.
    # Filter the returned calendar locally before creating download tasks.
    if "is_open" in out.columns:
        out = out[out["is_open"].astype(str).eq("1")]
    return sorted(out["cal_date"].astype(str).tolist())


def weekday_dates(start: str, end: str) -> list[str]:
    day, last = date.fromisoformat(start), date.fromisoformat(end)
    out: list[str] = []
    while day <= last:
        if day.weekday() < 5:
            out.append(day.strftime("%Y%m%d"))
        day += timedelta(days=1)
    return out


def partition_dates(root: str | Path) -> list[str]:
    """Return sorted dates from existing `*_YYYYMMDD.parquet` partitions."""
    out: set[str] = set()
    for path in Path(root).rglob("*.parquet"):
        match = re.search(r"_(\d{8})\.parquet$", path.name)
        if match:
            out.add(match.group(1))
    if not out:
        raise ValueError(f"no dated Parquet partitions found under {root}")
    return sorted(out)


def validate(df: pd.DataFrame, api_name: str, trade_date: str) -> None:
    missing = [c for c in EXPECTED[api_name] if c not in df.columns]
    if missing:
        raise ValueError(f"missing fields: {','.join(missing)}")
    if df.empty:
        raise ValueError("empty table")
    if not (df["trade_date"].astype(str) == trade_date).all():
        raise ValueError("partition contains an unexpected trade_date")
    if df.duplicated(["ts_code", "trade_date"]).any():
        raise ValueError("duplicate ts_code/trade_date")
    if {"open", "high", "low", "close"}.issubset(df.columns):
        # HK closing-auction prints may sit outside the vendor's intraday
        # high/low fields.  Preserve these raw observations and flag them
        # later; rejecting a whole market-day would create selection bias.
        if api_name == "hk_daily":
            # A small number of vendor rows also place an opening-auction
            # print outside intraday high/low.  High >= low is the only hard
            # raw-data consistency condition retained for this market.
            bad = df["high"] < df["low"]
        else:
            checked = df
            # The provider backfills pre-BSE history for 920489.BJ with two
            # auction-price rows whose close lies outside high/low.  BSE is
            # excluded from the study universe, so retain the raw row for
            # provenance but do not let it reject an otherwise valid SSE/SZSE
            # market-day partition.
            if api_name == "daily" and "ts_code" in checked.columns:
                checked = checked[~checked["ts_code"].astype(str).str.endswith(".BJ")]
            bad = (checked["high"] < checked[["open", "close"]].max(axis=1)) | (checked["low"] > checked[["open", "close"]].min(axis=1)) | (checked["high"] < checked["low"])
        if bad.any():
            raise ValueError("OHLC inequality violation")


def write_partition(df: pd.DataFrame, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    part = output.with_suffix(output.suffix + ".part")
    df.to_parquet(part, index=False)
    reread = pd.read_parquet(part)
    if len(reread) != len(df):
        raise ValueError("Parquet re-read row count mismatch")
    os.replace(part, output)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", choices=MARKET_DATASETS, required=True)
    ap.add_argument("--provider", choices=("super", "fast"), default="super")
    ap.add_argument("--start", help="YYYY-MM-DD")
    ap.add_argument("--end", help="YYYY-MM-DD")
    ap.add_argument("--dates-from", help="use date partitions under this path instead of querying a calendar")
    ap.add_argument("--datasets", nargs="+", choices=sorted(EXPECTED))
    ap.add_argument("--max-days", type=int, default=0, help="0 means no artificial cap")
    ap.add_argument("--output", default="data/raw")
    ap.add_argument("--sleep", type=float, default=0.15)
    args = ap.parse_args()
    datasets = tuple(args.datasets or MARKET_DATASETS[args.market])
    invalid = set(datasets) - set(MARKET_DATASETS[args.market])
    if invalid:
        raise SystemExit(f"datasets not valid for {args.market}: {sorted(invalid)}")

    s = session(args.provider)
    if args.dates_from:
        dates = partition_dates(args.dates_from)
    else:
        if not args.start or not args.end:
            raise SystemExit("provide --start and --end, or --dates-from")
        dates = cn_trade_dates(s, args.start.replace("-", ""), args.end.replace("-", ""), args.provider) if args.market == "cn" else weekday_dates(args.start, args.end)
    if args.max_days:
        dates = dates[: args.max_days]
    root = Path(args.output)
    done = skipped = failed = 0
    for n, trade_date in enumerate(dates, 1):
        for dataset in datasets:
            output = root / args.market / dataset / f"year={trade_date[:4]}" / f"{dataset}_{trade_date}.parquet"
            if output.exists():
                done += 1
                continue
            try:
                df = api_frame(s, dataset, trade_date, args.provider)
                # HK uses weekday placeholders because its calendar endpoint is unavailable.
                if df.empty and args.market == "hk":
                    skipped += 1
                    continue
                validate(df, dataset, trade_date)
                write_partition(df, output)
                done += 1
                print(f"ok {dataset} {trade_date} rows={len(df)}", flush=True)
            except Exception as exc:
                failed += 1
                print(f"failed {dataset} {trade_date}: {type(exc).__name__}: {exc}", flush=True)
            time.sleep(args.sleep)
        if n % 25 == 0:
            print(f"progress dates={n}/{len(dates)} completed={done} skipped={skipped} failed={failed}", flush=True)
    print(f"finished dates={len(dates)} completed={done} skipped={skipped} failed={failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
