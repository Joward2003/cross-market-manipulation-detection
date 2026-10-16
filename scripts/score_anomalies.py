#!/usr/bin/env python3
"""Fit label-free anomaly heads and stream A-share scores from a trained encoder."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from sklearn.ensemble import IsolationForest
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.modeling.network import TS2VecDANN
from src.modeling.store import SequenceStore, WindowDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--allow-cpu", action="store_true", help="Diagnostic only; production scoring requires CUDA.")
    return parser.parse_args()


def loader(dataset: WindowDataset, batch_size: int) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True,
                      persistent_workers=True, prefetch_factor=2)


@torch.no_grad()
def encode(model: TS2VecDANN, data_loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    vectors, endpoints = [], []
    model.eval()
    for x, _, end in tqdm(data_loader, desc="Encoding", unit="batch"):
        _, z = model.encode(x.to(device, non_blocking=True))
        vectors.append(z.float().cpu().numpy())
        endpoints.append(end.numpy())
    return np.concatenate(vectors), np.concatenate(endpoints)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError("CUDA is unavailable. Refusing CPU scoring; run scripts/verify_cuda.py first.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    saved = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg, scope = saved["config"], saved["config"]["datasets"][saved["dataset"]]
    store = SequenceStore(ROOT / cfg["data"]["sequence_store"])
    enc = cfg["encoder"]
    model = TS2VecDANN(enc["input_dim"], enc["hidden_dim"], enc["output_dim"], enc["dilations"], enc["dropout"], len(store.market_to_domain)).to(device)
    model.load_state_dict(saved["model"])
    scalers = saved["scalers"]
    clip = tuple(map(float, cfg["scaling"]["clip"]))
    split = cfg["split"]["primary"]
    rng = np.random.default_rng(int(saved["seed"]))
    train_ends = store.select_window_ends("train", ["cn"], split["train_years"])
    reservoir_size = int(cfg["anomaly_heads"]["fit_sample_size"])
    if len(train_ends) > reservoir_size:
        train_ends = rng.choice(train_ends, size=reservoir_size, replace=False)
    train_data = WindowDataset(store, train_ends, scalers, clip=clip, return_endpoint=True)
    train_z, _ = encode(model, loader(train_data, args.batch_size), device)
    center = np.median(train_z, axis=0)
    scale = np.maximum(np.median(np.abs(train_z - center), axis=0) * float(cfg["scaling"]["mad_consistency_constant"]), 1e-6)
    forest = None
    if cfg["anomaly_heads"]["isolation_forest"]["enabled"]:
        forest_cfg = cfg["anomaly_heads"]["isolation_forest"]
        forest = IsolationForest(
            n_estimators=int(forest_cfg["n_estimators"]), max_samples=int(forest_cfg["max_samples"]),
            contamination="auto", n_jobs=-1, random_state=int(saved["seed"]),
        ).fit(train_z)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_dir / "robust_svdd_boundary.npz", center=center.astype("float32"), scale=scale.astype("float32"))
    # All daily A-share endpoints in the held-out period.  The score stream does
    # not retain the full representation matrix, which would be several GiB.
    score_ends = store.select_window_ends("eval", ["cn"], split["external_test_years"])
    score_data = WindowDataset(store, score_ends, scalers, clip=clip, return_endpoint=True)
    score_loader = loader(score_data, args.batch_size)
    raw_path = args.output_dir / "scores_raw.parquet"
    schema = pa.schema([
        pa.field("symbol", pa.string()), pa.field("trade_date", pa.int32()), pa.field("market", pa.string()),
        pa.field("svdd_score", pa.float32()), pa.field("iforest_score", pa.float32()),
        pa.field("dataset", pa.string()), pa.field("seed", pa.int32()),
    ])
    writer = pq.ParquetWriter(raw_path, schema=schema, compression="zstd")
    model.eval()
    with torch.no_grad():
        for x, _, ends in tqdm(score_loader, desc="Scoring A-share windows", unit="batch"):
            _, z = model.encode(x.to(device, non_blocking=True))
            value = z.float().cpu().numpy()
            svdd_score = np.sum(np.square((value - center) / scale), axis=1).astype("float32")
            iforest_score = (-forest.decision_function(value)).astype("float32") if forest is not None else np.full(len(value), np.nan, dtype="float32")
            end_np = ends.numpy().astype(np.int64)
            table = pa.table({
                "symbol": store.endpoint_symbols(end_np).astype(str),
                "trade_date": np.asarray(store.dates[end_np], dtype=np.int32),
                "market": np.repeat("cn", len(end_np)), "svdd_score": svdd_score,
                "iforest_score": iforest_score, "dataset": np.repeat(saved["dataset"], len(end_np)),
                "seed": np.repeat(int(saved["seed"]), len(end_np)).astype(np.int32),
            }, schema=schema)
            writer.write_table(table)
    writer.close()
    # A same-day percentile avoids falsely calling broad market turbulence a
    # stock-specific anomaly and is the paper's primary ranking outcome.
    final_path = args.output_dir / "scores_with_percentile.parquet"
    con = duckdb.connect()
    raw = str(raw_path).replace("'", "''")
    final = str(final_path).replace("'", "''")
    con.execute(f"""
        COPY (
          SELECT *,
             cume_dist() OVER (PARTITION BY trade_date ORDER BY svdd_score) AS svdd_percentile_same_day,
             cume_dist() OVER (PARTITION BY trade_date ORDER BY iforest_score) AS iforest_percentile_same_day
          FROM read_parquet('{raw}')
        ) TO '{final}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)
    con.close()
    manifest = {"checkpoint": str(args.checkpoint), "output": str(final_path), "n_scored": int(len(score_ends)),
                "anomaly_boundary_fit_rows": int(len(train_ends)), "scope": scope, "dataset": saved["dataset"], "seed": int(saved["seed"])}
    (args.output_dir / "scoring_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
