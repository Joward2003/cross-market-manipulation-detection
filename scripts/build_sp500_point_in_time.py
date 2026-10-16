#!/usr/bin/env python3
"""Build annual point-in-time S&P 500 membership from dated snapshots."""
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/raw/reference/sp500_historical_components.csv")
    ap.add_argument("--start-year", type=int, default=2011)
    ap.add_argument("--end-year", type=int, default=2025)
    ap.add_argument("--output", default="data/raw/reference/sp500_point_in_time")
    args = ap.parse_args()
    src = pd.read_csv(args.input, dtype=str)
    src["date"] = pd.to_datetime(src["date"])
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    rows = []
    for year in range(args.start_year, args.end_year + 1):
        hist = src[src["date"] <= pd.Timestamp(f"{year}-01-01")]
        if hist.empty:
            raise ValueError(f"no historical membership snapshot before {year}")
        asof = hist.iloc[-1]
        tickers = sorted({x.strip().upper() for x in str(asof["tickers"]).split(",") if x.strip()})
        frame = pd.DataFrame({"year": year, "asof_date": asof["date"].strftime("%Y-%m-%d"), "ticker": tickers})
        frame.to_csv(out / f"sp500_{year}.csv", index=False)
        rows.extend(frame.to_dict("records"))
        print(year, asof["date"].date(), len(frame))
    pd.DataFrame(rows).to_csv(out / "sp500_point_in_time_membership.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
