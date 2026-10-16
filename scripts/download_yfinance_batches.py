#!/usr/bin/env python3
"""Batch-download reference-market daily OHLCV from Yahoo Finance.

This is intended for the optional cross-market representation module only.
It writes long-format, per-batch Parquet files and never writes proxy settings
or Yahoo credentials to disk.  The caller may set HTTPS_PROXY/HTTP_PROXY.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yfinance as yf


def ticker_column(df: pd.DataFrame) -> str:
    for col in ("ts_code", "symbol", "ticker"):
        if col in df.columns:
            return col
    raise ValueError("input must include ts_code, symbol, or ticker")


def yahoo_symbol(raw: str, market: str) -> str:
    s = str(raw).upper()
    if market == "hk":
        # Tushare stores a five-digit code (00005.HK), while Yahoo uses four
        # digits (0005.HK).  Do not apply this rule to a non-HK ticker.
        code = s.removesuffix(".HK")
        return f"{int(code):04d}.HK" if code.isdigit() else s
    return s.replace(".", "-")


def to_long(frame: pd.DataFrame, symbols: list[str]) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    if not isinstance(frame.columns, pd.MultiIndex):
        frame.columns = pd.MultiIndex.from_product([frame.columns, [symbols[0]]])
    # yfinance returns either field/ticker or ticker/field depending on version.
    level0 = set(map(str, frame.columns.get_level_values(0)))
    fields = {"Open", "High", "Low", "Close", "Adj Close", "Volume"}
    if level0 & fields:
        wide = frame.stack(level=1, future_stack=True).rename_axis(["trade_date", "yahoo_symbol"]).reset_index()
    else:
        wide = frame.stack(level=0, future_stack=True).rename_axis(["trade_date", "yahoo_symbol"]).reset_index()
    renamed = {"Open": "open", "High": "high", "Low": "low", "Close": "close", "Adj Close": "adj_close", "Volume": "vol"}
    wide = wide.rename(columns=renamed)
    wide["trade_date"] = pd.to_datetime(wide["trade_date"]).dt.strftime("%Y%m%d")
    return wide.dropna(subset=["close"], how="all")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", choices=("hk", "us"), required=True)
    ap.add_argument("--master", required=True)
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--batch-size", type=int, default=80)
    ap.add_argument("--max-batches", type=int, default=0)
    ap.add_argument("--output", default="data/raw")
    ap.add_argument("--eq-only", action="store_true", help="for US, retain classify == EQ when available")
    args = ap.parse_args()

    master = pd.read_csv(args.master, dtype=str)
    if args.eq_only and "classify" in master:
        master = master[master["classify"].eq("EQ")].copy()
    code_col = ticker_column(master)
    master["yahoo_symbol"] = master[code_col].map(lambda x: yahoo_symbol(x, args.market))
    symbols = master["yahoo_symbol"].dropna().drop_duplicates().tolist()
    outdir = Path(args.output) / args.market / "yfinance_daily"
    outdir.mkdir(parents=True, exist_ok=True)

    written = 0
    for batch_no, start_i in enumerate(range(0, len(symbols), args.batch_size), 1):
        if args.max_batches and batch_no > args.max_batches:
            break
        out = outdir / f"batch={batch_no:04d}_{args.start.replace('-', '')}_{args.end.replace('-', '')}.parquet"
        if out.exists():
            print(f"skip batch={batch_no} exists", flush=True)
            continue
        batch = symbols[start_i : start_i + args.batch_size]
        try:
            wide = yf.download(batch, start=args.start, end=args.end, interval="1d", auto_adjust=False, actions=False, progress=False, threads=False, group_by="column", timeout=60)
            long = to_long(wide, batch)
            if long.empty:
                raise ValueError("Yahoo returned no valid rows")
            long.to_parquet(out, index=False)
            written += 1
            print(f"ok batch={batch_no} symbols={len(batch)} rows={len(long)}", flush=True)
        except Exception as exc:
            print(f"failed batch={batch_no}: {type(exc).__name__}: {exc}", flush=True)
    print(f"finished requested_symbols={len(symbols)} written_batches={written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
