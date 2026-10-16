#!/usr/bin/env python3
"""Materialize annual point-in-time S&P 500 data, reusing static history."""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import yfinance as yf
from download_yfinance_batches import to_long, yahoo_symbol

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--membership', default='data/raw/reference/sp500_point_in_time')
    ap.add_argument('--static', default='data/raw/us/yfinance_daily')
    ap.add_argument('--output', default='data/raw/us/yfinance_daily_point_in_time')
    ap.add_argument('--start-year', type=int, default=2011)
    ap.add_argument('--end-year', type=int, default=2025)
    ap.add_argument('--batch-size', type=int, default=60)
    args = ap.parse_args()
    outroot = Path(args.output); outroot.mkdir(parents=True, exist_ok=True)
    static_files = sorted(Path(args.static).glob('*.parquet'))
    static = pd.concat((pd.read_parquet(p) for p in static_files), ignore_index=True)
    static['trade_date'] = static['trade_date'].astype(str)
    static['yahoo_symbol'] = static['yahoo_symbol'].astype(str)
    have = set(static['yahoo_symbol'].unique())
    for year in range(args.start_year, args.end_year + 1):
        pool = pd.read_csv(Path(args.membership) / f'sp500_{year}.csv', dtype=str)
        symbols = [yahoo_symbol(x, 'us') for x in pool['ticker'].dropna().drop_duplicates()]
        ystart, yend = f'{year}0101', f'{year+1}0101'
        rows = [static[(static.yahoo_symbol == s) & (static.trade_date >= ystart) & (static.trade_date < yend)] for s in symbols if s in have]
        base = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
        missing = [s for s in symbols if s not in have]
        for batch_no, start in enumerate(range(0, len(missing), args.batch_size), 1):
            batch = missing[start:start + args.batch_size]
            try:
                wide = yf.download(batch, start=f'{year}-01-01', end=f'{year+1}-01-01', interval='1d', auto_adjust=False, actions=False, progress=False, threads=False, group_by='column', timeout=60)
                got = to_long(wide, batch)
                if not got.empty:
                    got['trade_date'] = got['trade_date'].astype(str)
                    base = pd.concat([base, got], ignore_index=True)
                print(f'year={year} missing_batch={batch_no} requested={len(batch)} got={len(got)}', flush=True)
            except Exception as exc:
                print(f'failed year={year} missing_batch={batch_no}: {type(exc).__name__}: {exc}', flush=True)
        if base.empty:
            print(f'year={year}: no rows'); continue
        base = base.drop_duplicates(['yahoo_symbol', 'trade_date']).sort_values(['trade_date', 'yahoo_symbol'])
        base['membership_year'] = year
        base['membership_asof'] = pool['asof_date'].iloc[0]
        base.to_parquet(outroot / f'year={year}.parquet', index=False)
        print(f'OK year={year} members={len(symbols)} rows={len(base)} static_members={len(symbols)-len(missing)}', flush=True)
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
