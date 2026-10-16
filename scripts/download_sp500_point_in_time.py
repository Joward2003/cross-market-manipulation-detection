#!/usr/bin/env python3
"""Download annual point-in-time S&P 500 OHLCV with yfinance."""
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import yfinance as yf
from tqdm import tqdm

from download_yfinance_batches import to_long, yahoo_symbol


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--membership", default="data/raw/reference/sp500_point_in_time")
    ap.add_argument("--start-year", type=int, default=2011)
    ap.add_argument("--end-year", type=int, default=2025)
    ap.add_argument("--batch-size", type=int, default=80)
    ap.add_argument("--output", default="data/raw/us/yfinance_daily_point_in_time")
    args = ap.parse_args()
    root = Path(args.membership); outroot = Path(args.output); outroot.mkdir(parents=True, exist_ok=True)
    for year in tqdm(range(args.start_year, args.end_year + 1), desc="years", unit="year"):
        pool = pd.read_csv(root / f"sp500_{year}.csv", dtype=str)
        symbols = [yahoo_symbol(x, "us") for x in pool["ticker"].dropna().drop_duplicates()]
        outdir = outroot / f"year={year}"; outdir.mkdir(parents=True, exist_ok=True)
        batches = list(enumerate(range(0, len(symbols), args.batch_size), 1))
        for batch_no, start in tqdm(batches, desc=f"{year} batches", unit="batch", leave=False):
            out = outdir / f"batch={batch_no:04d}_{year}.parquet"
            if out.exists():
                continue
            batch = symbols[start:start + args.batch_size]
            try:
                wide = yf.download(batch, start=f"{year}-01-01", end=f"{year + 1}-01-01", interval="1d", auto_adjust=False, actions=False, progress=False, threads=True, group_by="column", timeout=60)
                long = to_long(wide, batch)
                if long.empty:
                    tqdm.write(f"skip year={year} batch={batch_no}: no valid rows")
                    continue
                long["membership_year"] = year
                long["membership_asof"] = pool["asof_date"].iloc[0]
                long.to_parquet(out, index=False)
                tqdm.write(f"ok year={year} batch={batch_no} symbols={len(batch)} rows={len(long)}")
            except Exception as exc:
                tqdm.write(f"failed year={year} batch={batch_no}: {type(exc).__name__}: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
