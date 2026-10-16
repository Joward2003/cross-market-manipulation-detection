#!/usr/bin/env python3
from __future__ import annotations
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.download.state_store import StateStore

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--db",default="data/manifests/download_state.sqlite"); ap.add_argument("--market",choices=["cn","hk","us"]); ap.add_argument("--max-tasks",type=int,default=0); ap.add_argument("--dry-run",action="store_true"); a=ap.parse_args()
    if a.dry_run: print("dry-run: no tasks created"); return 0
    store=StateStore(a.db); count=0
    # Full-universe task generation is intentionally not implemented in Stage 0.
    print("Stage 0 guard: refusing full task generation. Use the bounded test scripts."); return 2
if __name__ == "__main__": raise SystemExit(main())
