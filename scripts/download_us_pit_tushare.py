#!/usr/bin/env python3
"""Download US daily adjusted prices for all PIT S&P members via Tushare gateway.

Set TUSHARE_SUPER_API_KEY in the environment. Credentials are never written.
Each symbol is requested once for 2011--2025, then filtered into annual PIT pools.
"""
from __future__ import annotations
import argparse, os, time
from pathlib import Path
import pandas as pd
import requests

def fetch(session, api, params):
    key = os.environ.get('TUSHARE_SUPER_API_KEY')
    if not key: raise SystemExit('Set TUSHARE_SUPER_API_KEY')
    last = None
    for attempt in range(1, 7):
        try:
            r = session.post('https://ai-tool.indevs.in/tushare/pro',
                             json={'api_name': api, 'params': params}, timeout=90)
            r.raise_for_status(); obj = r.json()
            if obj.get('code') != 0: raise RuntimeError(obj.get('msg'))
            d = obj.get('data') or {}
            return pd.DataFrame(d.get('items', []), columns=d.get('fields', []))
        except Exception as exc:
            last = exc; time.sleep(min(30, 2 ** (attempt - 1)))
    raise RuntimeError(str(last))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--membership', default='data/raw/reference/sp500_point_in_time/sp500_point_in_time_membership.csv')
    ap.add_argument('--output', default='data/raw/us/tushare_point_in_time')
    ap.add_argument('--start-year', type=int, default=2011); ap.add_argument('--end-year', type=int, default=2025)
    args = ap.parse_args(); out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    mem = pd.read_csv(args.membership, dtype=str)
    years = set(range(args.start_year, args.end_year + 1)); mem = mem[mem.year.astype(int).isin(years)]
    symbols = sorted(mem.ticker.dropna().drop_duplicates())
    key = os.environ.get('TUSHARE_SUPER_API_KEY')
    if not key: raise SystemExit('Set TUSHARE_SUPER_API_KEY')
    s = requests.Session()
    s.headers.update({'X-API-Key': key})
    s.trust_env = False
    s.proxies = {'http': '', 'https': ''}
    chunks = []
    for i, code in enumerate(symbols, 1):
        f = out / f'{code}.parquet'
        if f.exists(): df = pd.read_parquet(f)
        else:
            try:
                df = fetch(s, 'us_daily_adj', {'ts_code': code, 'start_date': f'{args.start_year}0101', 'end_date': f'{args.end_year+1}0101', 'limit': 8000})
                if not df.empty: df.to_parquet(f, index=False)
                print(f'{i}/{len(symbols)} {code} rows={len(df)}', flush=True)
            except Exception as exc:
                print(f'{i}/{len(symbols)} {code} FAILED {exc}', flush=True); continue
        if not df.empty: chunks.append(df)
    if not chunks: return
    all_df = pd.concat(chunks, ignore_index=True); all_df['trade_date'] = all_df.trade_date.astype(str)
    for year in sorted(years):
        pool = mem[mem.year.astype(int).eq(year)]; syms = set(pool.ticker)
        part = all_df[all_df.ts_code.isin(syms) & all_df.trade_date.between(f'{year}0101', f'{year}1231')].copy()
        if part.empty: continue
        part['membership_year'] = year; part['membership_asof'] = pool.asof_date.iloc[0]
        part.to_parquet(out / f'year={year}.parquet', index=False)
        print(f'YEAR {year} members={len(syms)} rows={len(part)}', flush=True)

if __name__ == '__main__': main()
