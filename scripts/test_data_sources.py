#!/usr/bin/env python3
"""Run only the explicitly bounded Stage-0 provider probes."""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.providers.tushare_provider import TushareProvider
from src.data.providers.akshare_provider import AKShareProvider
from src.data.providers.yfinance_provider import YFinanceProvider

def probe(name, provider, symbols, years):
    out=[]
    for s in symbols:
        for y in years:
            try:
                df=provider.daily_prices(s, f"{y}-01-01", f"{y}-12-31")
                out.append({"provider":name,"symbol":s,"year":y,"status":"needs_review" if df.empty else "success","rows":len(df),"columns":list(df.columns),"error":"empty table" if df.empty else None})
            except Exception as e:
                out.append({"provider":name,"symbol":s,"year":y,"status":"skipped","error":f"{type(e).__name__}: {e}"})
    return out

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--report", default="reports/01_stage0_provider_report.json"); a=ap.parse_args()
    years=[2023,2024]; rows=[]
    # No token is guessed or logged. The CN probe is intentionally bounded to
    # two requested dates and is skipped when credentials are absent.
    if os.getenv("TUSHARE_TOKEN"):
        rows += probe("tushare", TushareProvider(), ["000001"], [2024])
    else: rows.append({"provider":"tushare","status":"skipped","error":"TUSHARE_TOKEN is not set"})
    # Hong Kong Stage-0 now uses Yahoo as the primary source.  Symbols must
    # carry Yahoo's `.HK` suffix; AKShare remains an explicit backup adapter.
    rows += probe("yfinance_hk", YFinanceProvider(), ["0005.HK","0700.HK","0941.HK"], years)
    rows += probe("yfinance", YFinanceProvider(), ["AAPL","MSFT","BRK-B"], years)
    Path(a.report).parent.mkdir(parents=True, exist_ok=True); Path(a.report).write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(rows, ensure_ascii=False, indent=2)); return 0
if __name__ == "__main__": raise SystemExit(main())
