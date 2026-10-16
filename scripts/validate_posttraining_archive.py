#!/usr/bin/env python3
"""Fail fast when the final post-training archive is incomplete or inconsistent."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/posttraining_analysis.yaml")
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    output = ROOT / cfg["paths"]["output_root"]
    runs = pd.read_csv(output / "tables" / "all_15_runs.csv")
    require(len(runs) == 15, f"expected 15 completed runs, found {len(runs)}")
    require(runs.dataset.nunique() == 5 and runs.seed.nunique() == 3, "run grid is not 5 specifications × 3 seeds")
    require(runs.complete.fillna(False).all(), "at least one training/scoring run is incomplete")

    cases = pd.read_parquet(output / "data" / "matched_cases.parquet")
    controls = pd.read_parquet(output / "data" / "difficult_controls.parquet")
    require(len(cases) == 68, f"expected 68 strict events, found {len(cases)}")
    require(controls.event_id.nunique() == 68, "not every strict event has difficult controls")
    require((controls.groupby("event_id").size() == 5).all(), "expected exactly five controls per event")
    require(set(cases.symbol6.astype(str).str.zfill(6)).isdisjoint(set(controls.symbol6.astype(str).str.zfill(6))),
            "a regulated symbol leaked into the difficult-control pool")

    diag = pd.read_csv(output / "diagnostics" / "windowshap_local_accuracy.csv")
    require(len(diag) >= 530, "too few WindowSHAP explanations")
    require(diag.local_accuracy_error.abs().max() < 1e-4, "WindowSHAP local accuracy constraint failed")
    require(diag.surrogate_r2.dropna().median() > 0.90, "WindowSHAP median surrogate quality is too weak")
    require((diag.surrogate_r2.dropna() > 0.70).mean() >= 0.75, "too many low-fidelity WindowSHAP surrogates")

    expected_stems = [f"fig{i:02d}_" for i in range(1, 12)]
    png_names = [path.name for path in (output / "figures").glob("*.png")]
    pdf_names = [path.name for path in (output / "figures").glob("*.pdf")]
    for stem in expected_stems:
        require(any(name.startswith(stem) for name in png_names), f"missing PNG figure {stem}")
        require(any(name.startswith(stem) for name in pdf_names), f"missing PDF figure {stem}")
    require((output / "file_manifest_sha256.csv").exists(), "missing SHA-256 manifest")
    print(json.dumps({"status": "PASS", "runs": len(runs), "events": len(cases),
                      "controls": len(controls), "windowshap_explanations": len(diag),
                      "figures_png": len(png_names), "figures_pdf": len(pdf_names)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
