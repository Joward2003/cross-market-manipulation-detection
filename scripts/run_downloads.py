#!/usr/bin/env python3
from __future__ import annotations
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.download.state_store import StateStore

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--db",default="data/manifests/download_state.sqlite"); ap.add_argument("--max-tasks",type=int); ap.add_argument("--retry-failed",action="store_true"); ap.add_argument("--dry-run",action="store_true"); a=ap.parse_args()
    if not a.dry_run: print("Guard: real task execution requires an explicit Stage-0 provider test and is disabled by default."); return 2
    s=StateStore(a.db); print(f"dry-run: {len(s.list_tasks())} tasks in SQLite; no downloads run"); return 0
if __name__ == "__main__": raise SystemExit(main())
