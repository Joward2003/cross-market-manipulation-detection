#!/usr/bin/env python3
"""Convert the partitioned feature panel into a compact mmap sequence store.

This script is intentionally label-blind.  It reads only feature_ready OHLCV
features and derives legal 60-trading-day endpoints; regulator cases are never
opened here.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/model_training_memory_efficient.yaml")
    parser.add_argument("--output", type=Path, default=None, help="Override data.sequence_store")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing generated sequence store")
    return parser.parse_args()


def sql_quote(path: Path) -> str:
    return str(path).replace("'", "''")


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    data_cfg = config["data"]
    source = ROOT / data_cfg["parquet_root"]
    output = args.output or ROOT / data_cfg["sequence_store"]
    if not source.exists():
        raise FileNotFoundError(f"Feature panel not found: {source}")
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output} exists. Verify it, or explicitly use --overwrite.")
        shutil.rmtree(output)
    output.mkdir(parents=True)
    sorted_file = output / "_sorted_feature_rows.parquet"
    features = list(data_cfg["main_features"])
    select_columns = ", ".join([
        "CAST(symbol AS VARCHAR) AS symbol",
        "CAST(strftime(CAST(trade_date AS DATE), '%Y%m%d') AS INTEGER) AS trade_date",
        "CAST(market AS VARCHAR) AS market",
        "CAST(COALESCE(suspension_resume_flag, 0) AS UTINYINT) AS suspension_resume_flag",
        *[f"CAST({name} AS FLOAT) AS {name}" for name in features],
    ])
    glob = sql_quote(source / "**" / "*.parquet")
    con = duckdb.connect()
    con.execute("SET preserve_insertion_order=false")
    con.execute("SET threads TO 4")
    query = f"""
        COPY (
          SELECT {select_columns}
          FROM read_parquet('{glob}', hive_partitioning=1, union_by_name=1)
          WHERE feature_ready = TRUE
          ORDER BY market, symbol, trade_date
        ) TO '{sql_quote(sorted_file)}' (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 250000)
    """
    print("Sorting eligible feature rows once; this can take several minutes.", flush=True)
    con.execute(query)
    row_count = int(con.execute(f"SELECT COUNT(*) FROM read_parquet('{sql_quote(sorted_file)}')").fetchone()[0])
    print(f"Writing mmap store for {row_count:,} eligible daily rows.", flush=True)

    files = {
        "features": "features.f32.bin",
        "dates": "dates.i32.bin",
        "domains": "domains.u8.bin",
        "resume_flags": "resume_flags.u8.bin",
        "offsets": "symbol_offsets.parquet",
        "windows_train": "windows_train.i64.bin",
        "windows_eval": "windows_eval.i64.bin",
    }
    feature_mm = np.memmap(output / files["features"], dtype=np.float32, mode="w+", shape=(row_count, len(features)))
    date_mm = np.memmap(output / files["dates"], dtype=np.int32, mode="w+", shape=(row_count,))
    domain_mm = np.memmap(output / files["domains"], dtype=np.uint8, mode="w+", shape=(row_count,))
    resume_mm = np.memmap(output / files["resume_flags"], dtype=np.uint8, mode="w+", shape=(row_count,))
    domain_map = {"cn": 0, "hk": 1, "us": 2}
    segments: list[dict[str, object]] = []
    previous_key: tuple[str, str] | None = None
    segment_start = 0
    cursor = 0
    parquet = pq.ParquetFile(sorted_file)
    columns = ["symbol", "trade_date", "market", "suspension_resume_flag", *features]
    for batch in tqdm(parquet.iter_batches(batch_size=250_000, columns=columns), desc="Writing rows", unit="batch"):
        frame = batch.to_pandas()
        n = len(frame)
        symbols = frame["symbol"].astype(str).to_numpy()
        markets = frame["market"].astype(str).to_numpy()
        keys = np.char.add(np.char.add(markets, "\x1f"), symbols)
        # Write core arrays in vectorized form.
        feature_mm[cursor:cursor + n] = frame[features].to_numpy(dtype=np.float32, copy=False)
        date_mm[cursor:cursor + n] = frame["trade_date"].to_numpy(dtype=np.int32, copy=False)
        try:
            domain_mm[cursor:cursor + n] = np.asarray([domain_map[x] for x in markets], dtype=np.uint8)
        except KeyError as exc:
            raise ValueError(f"Unexpected market in feature panel: {exc}") from exc
        resume_mm[cursor:cursor + n] = frame["suspension_resume_flag"].to_numpy(dtype=np.uint8, copy=False)
        boundaries = np.flatnonzero(keys[1:] != keys[:-1]) + 1
        starts = np.r_[0, boundaries]
        ends = np.r_[boundaries, n]
        for local_start, local_end in zip(starts, ends):
            key = (str(markets[local_start]), str(symbols[local_start]))
            absolute_start = cursor + int(local_start)
            absolute_end = cursor + int(local_end)
            if previous_key is not None and key == previous_key:
                # The first segment of this record continues the last batch.
                segments[-1]["end"] = absolute_end
                segments[-1]["last_trade_date"] = int(date_mm[absolute_end - 1])
            else:
                segments.append({
                    "market": key[0], "symbol": key[1], "start": absolute_start, "end": absolute_end,
                    "first_trade_date": int(date_mm[absolute_start]), "last_trade_date": int(date_mm[absolute_end - 1]),
                })
            previous_key = key
        cursor += n
    if cursor != row_count:
        raise RuntimeError(f"Expected {row_count} rows but wrote {cursor}")
    for mm in (feature_mm, date_mm, domain_mm, resume_mm):
        mm.flush()
    pd.DataFrame(segments).to_parquet(output / files["offsets"], index=False)

    # A legal endpoint has 60 contiguous observed trading rows for one symbol
    # and does not cross a suspension/resumption gap.  Train endpoints are a
    # stride-five subset; evaluation endpoints retain every eligible day.
    window_length = int(data_cfg["window_length"])
    train_stride = int(data_cfg["train_stride"])
    eval_endpoints: list[np.ndarray] = []
    train_endpoints: list[np.ndarray] = []
    flags = np.asarray(resume_mm)
    for row in tqdm(segments, desc="Indexing legal windows", unit="symbol"):
        start, end = int(row["start"]), int(row["end"])
        if end - start < window_length:
            continue
        local_bad = np.convolve(flags[start:end].astype(np.int16), np.ones(window_length, dtype=np.int16), mode="valid") > 0
        legal = np.flatnonzero(~local_bad) + start + window_length - 1
        if len(legal):
            eval_endpoints.append(legal.astype(np.int64))
            train_endpoints.append(legal[::train_stride].astype(np.int64))
    eval_arr = np.concatenate(eval_endpoints) if eval_endpoints else np.empty(0, dtype=np.int64)
    train_arr = np.concatenate(train_endpoints) if train_endpoints else np.empty(0, dtype=np.int64)
    for name, values in (("windows_eval", eval_arr), ("windows_train", train_arr)):
        mm = np.memmap(output / files[name], dtype=np.int64, mode="w+", shape=(len(values),))
        mm[:] = values
        mm.flush()
        del mm
    del feature_mm, date_mm, domain_mm, resume_mm
    sorted_file.unlink(missing_ok=True)
    metadata = {
        "n_rows": row_count, "n_features": len(features), "feature_columns": features,
        "domain_map": domain_map, "files": files,
        "window_counts": {"train": int(len(train_arr)), "eval": int(len(eval_arr))},
        "window_length": window_length,
    }
    (output / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), **metadata}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
