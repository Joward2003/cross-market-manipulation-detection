#!/usr/bin/env python3
"""Approximate fixed WindowSHAP for frozen representative anomaly scorers.

Thirty players combine six ten-day bins with five economic feature groups.
Absent blocks are copied from difficult-control background windows.  A weighted
KernelSHAP regression explains the raw robust-SVDD distance; normalized absolute
shares are used only for cross-model comparisons.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.modeling.network import TS2VecDANN
from src.modeling.store import SequenceStore

TIME_LABELS = ["-60:-51", "-50:-41", "-40:-31", "-30:-21", "-20:-11", "-10:-1"]
FEATURE_GROUPS = {
    "return_trend": [0, 1, 8, 11],
    "volatility_pressure": [2, 3, 9, 10],
    "trading_activity": [4, 5],
    "liquidity": [6],
    "price_volume_relation": [7],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/posttraining_analysis.yaml")
    parser.add_argument("--coalitions", type=int, default=256)
    parser.add_argument("--background-windows", type=int, default=32)
    parser.add_argument("--device", choices=["auto", "mps", "cpu"], default="auto")
    return parser.parse_args()


def choose_device(requested: str) -> torch.device:
    if requested == "mps" or (requested == "auto" and torch.backends.mps.is_available()):
        return torch.device("mps")
    return torch.device("cpu")


def target_windows(cases: pd.DataFrame, controls: pd.DataFrame) -> pd.DataFrame:
    case = cases[["event_id", "symbol6", "start_date", "end_date"]].copy()
    case["sample"] = "case"
    control = controls.loc[controls.control_rank == 1, ["event_id", "symbol6", "start_date", "end_date"]].copy()
    control["sample"] = "control"
    return pd.concat([case, control], ignore_index=True)


def peak_dates(score_path: Path, windows: pd.DataFrame) -> pd.DataFrame:
    con = duckdb.connect()
    con.register("windows", windows)
    path = str(score_path).replace("'", "''")
    result = con.execute(f"""
      SELECT w.event_id, w.sample, w.symbol6, w.start_date, w.end_date,
             ARG_MAX(s.trade_date, s.svdd_percentile_same_day) AS anchor_date,
             MAX(s.svdd_percentile_same_day) AS anchor_percentile
      FROM windows w
      LEFT JOIN (
        SELECT *, regexp_extract(symbol, '([0-9]{{6}})', 1) AS symbol6
        FROM read_parquet('{path}')
      ) s ON w.symbol6 = s.symbol6 AND s.trade_date BETWEEN w.start_date AND w.end_date
      GROUP BY ALL ORDER BY w.event_id, w.sample
    """).df()
    con.close()
    return result.dropna(subset=["anchor_date"])


def symbol_offset_map(store: SequenceStore) -> dict[str, tuple[int, int]]:
    mapping = {}
    for row in store.offsets.loc[store.offsets.market == "cn"].itertuples(index=False):
        mapping[str(row.symbol)[:6]] = (int(row.start), int(row.end))
    return mapping


def endpoint_at_or_before(store: SequenceStore, offsets: dict[str, tuple[int, int]], symbol6: str, date: int) -> int | None:
    bounds = offsets.get(str(symbol6).zfill(6))
    if bounds is None:
        return None
    start, end = bounds
    dates = np.asarray(store.dates[start:end])
    position = int(np.searchsorted(dates, int(date), side="right") - 1)
    endpoint = start + position
    if position < store.meta.window_length - 1 or endpoint >= end:
        return None
    return endpoint


def normalized_window(store: SequenceStore, endpoint: int, scaler: dict, clip: tuple[float, float]) -> np.ndarray:
    start = endpoint - store.meta.window_length + 1
    x = np.array(store.features[start:endpoint + 1], dtype=np.float32, copy=True)
    median = np.asarray(scaler["median"], dtype=np.float32)
    scale = np.asarray(scaler["scale"], dtype=np.float32)
    return np.clip((x - median) / scale, clip[0], clip[1])


def coalition_design(n_players: int, budget: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    # Empty/full plus every singleton and complement guarantee that every
    # player is independently identified.  The remaining budget is filled by
    # antithetic random pairs, which materially improves conditioning over a
    # purely random KernelSHAP design at the same inference cost.
    budget = max(budget, 2 * n_players + 2)
    masks = np.zeros((budget, n_players), dtype=np.float32)
    weights = np.zeros(budget, dtype=np.float64)
    masks[1] = 1.0
    weights[:2] = 1e3
    cursor = 2
    for player in range(n_players):
        masks[cursor, player] = 1.0
        masks[cursor + 1] = 1.0
        masks[cursor + 1, player] = 0.0
        cursor += 2
    while cursor < budget:
        size = int(rng.integers(1, n_players))
        chosen = rng.choice(n_players, size=size, replace=False)
        masks[cursor, chosen] = 1.0
        cursor += 1
        if cursor < budget:
            masks[cursor] = 1.0 - masks[cursor - 1]
            cursor += 1
    for i in range(2, budget):
        size = int(masks[i].sum())
        log_comb = math.lgamma(n_players + 1) - math.lgamma(size + 1) - math.lgamma(n_players - size + 1)
        weights[i] = math.exp(-log_comb) * (n_players - 1) / (size * (n_players - size))
    positive = weights[2:]
    weights[2:] = positive / max(positive.max(), 1e-12)
    return masks, weights


def apply_masks(target: np.ndarray, backgrounds: np.ndarray, masks: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n = len(masks)
    batch = backgrounds[rng.integers(0, len(backgrounds), size=n)].copy()
    groups = list(FEATURE_GROUPS.values())
    for player in range(masks.shape[1]):
        time_index = player // len(groups)
        feature_index = player % len(groups)
        present = masks[:, player] > 0.5
        if present.any():
            time_slice = slice(time_index * 10, (time_index + 1) * 10)
            feature_ids = groups[feature_index]
            batch[np.ix_(present, np.arange(time_slice.start, time_slice.stop), feature_ids)] = target[np.ix_(np.arange(time_slice.start, time_slice.stop), feature_ids)][None, :, :]
    return batch


@torch.no_grad()
def risk_scores(model: TS2VecDANN, values: np.ndarray, center: np.ndarray, scale: np.ndarray,
                device: torch.device, batch_size: int = 128) -> np.ndarray:
    outputs = []
    for start in range(0, len(values), batch_size):
        x = torch.from_numpy(values[start:start + batch_size]).to(device)
        _, z = model.encode(x)
        vector = z.float().cpu().numpy()
        outputs.append(np.square((vector - center) / scale).sum(axis=1))
    return np.concatenate(outputs)


def explain_one(model: TS2VecDANN, target: np.ndarray, backgrounds: np.ndarray, center: np.ndarray,
                scale: np.ndarray, device: torch.device, coalitions: int, seed: int) -> tuple[np.ndarray, dict]:
    rng = np.random.default_rng(seed)
    masks, weights = coalition_design(30, coalitions, rng)
    perturbed = apply_masks(target, backgrounds, masks, rng)
    values = risk_scores(model, perturbed, center, scale, device)
    target_value = float(risk_scores(model, target[None, :, :], center, scale, device)[0])
    base_value = float(values[0])
    # Fit changes relative to the empty coalition.  Scaling the response and
    # using an SVD ridge solver prevents the rare catastrophic coefficients
    # caused by an ill-conditioned weighted design.  The final correction
    # enforces the SHAP efficiency/local-accuracy identity exactly.
    delta = values.astype(np.float64) - base_value
    response_scale = max(float(np.std(delta)), 1e-6)
    reg = Ridge(alpha=1e-3, fit_intercept=False, solver="svd").fit(
        masks.astype(np.float64), delta / response_scale, sample_weight=weights
    )
    coefficients = reg.coef_.astype(np.float64) * response_scale
    residual = (target_value - base_value) - float(coefficients.sum())
    allocation = np.abs(coefficients)
    if float(allocation.sum()) <= 1e-12:
        allocation = np.ones_like(coefficients)
    coefficients += residual * allocation / allocation.sum()
    phi = coefficients.reshape(6, 5)
    fitted = base_value + masks.astype(np.float64) @ coefficients
    weighted_mean = float(np.average(values, weights=weights))
    denominator = float(np.sum(weights * np.square(values - weighted_mean)))
    numerator = float(np.sum(weights * np.square(values - fitted)))
    r2 = 1.0 - numerator / denominator if denominator > 1e-12 else np.nan
    approximation = float(base_value + coefficients.sum())
    return phi, {"target_score": target_value, "base_score": base_value, "surrogate_full_score": approximation,
                 "local_accuracy_error": approximation - target_value, "surrogate_r2": r2}


def similarity_inference(attributions: pd.DataFrame, repetitions: int, permutations: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    event_rows = []
    summary_rows = []
    for dataset, block in attributions.groupby("dataset"):
        for basis, value_column in (("signed_phi", "phi"), ("absolute_share", "absolute_contribution_share")):
            matrices = {}
            for (event_id, sample), values in block.groupby(["event_id", "sample"]):
                matrices[(event_id, sample)] = values.sort_values(["time_index", "feature_index"])[value_column].to_numpy(float)
            case_ids = sorted(event_id for event_id, sample in matrices if sample == "case")
            similarities = []
            for event_id in case_ids:
                other = [matrices[(other_id, "case")] for other_id in case_ids if other_id != event_id]
                if not other:
                    continue
                template = np.mean(other, axis=0)
                for sample in ("case", "control"):
                    key = (event_id, sample)
                    if key not in matrices:
                        continue
                    vector = matrices[key]
                    denom = np.linalg.norm(vector) * np.linalg.norm(template)
                    similarity = float(vector @ template / denom) if denom > 0 else np.nan
                    event_rows.append({"dataset": dataset, "basis": basis, "event_id": event_id,
                                       "sample": sample, "similarity": similarity})
                    similarities.append((event_id, sample, similarity))
            similarity_frame = pd.DataFrame(similarities, columns=["event_id", "sample", "similarity"])
            case = similarity_frame.loc[similarity_frame["sample"] == "case", ["event_id", "similarity"]].rename(columns={"similarity": "case"})
            control = similarity_frame.loc[similarity_frame["sample"] == "control", ["event_id", "similarity"]].rename(columns={"similarity": "control"})
            paired = case.merge(control, on="event_id").dropna()
            differences = (paired.case - paired.control).to_numpy(float)
            observed = float(differences.mean()) if len(differences) else np.nan
            if len(differences):
                draws = rng.integers(0, len(differences), size=(repetitions, len(differences)))
                boot = differences[draws].mean(axis=1)
                signs = rng.choice(np.array([-1.0, 1.0]), size=(permutations, len(differences)))
                null = (signs * differences).mean(axis=1)
                low, high = np.quantile(boot, [0.025, 0.975])
                p_one = (1 + np.sum(null >= observed)) / (permutations + 1)
            else:
                low = high = p_one = np.nan
            summary_rows.append({"dataset": dataset, "basis": basis, "n_pairs": len(differences),
                                 "mean_case_similarity": float(paired.case.mean()),
                                 "mean_control_similarity": float(paired.control.mean()),
                                 "paired_difference": observed, "ci_low": float(low), "ci_high": float(high),
                                 "permutation_p_one_sided": float(p_one)})
    return pd.DataFrame(event_rows), pd.DataFrame(summary_rows)


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    output = ROOT / cfg["paths"]["output_root"]
    run_root = ROOT / cfg["paths"]["runs_root"]
    store = SequenceStore(ROOT / "data/analysis/sequence_store")
    offsets = symbol_offset_map(store)
    cases = pd.read_parquet(output / "data" / "matched_cases.parquet")
    controls = pd.read_parquet(output / "data" / "difficult_controls.parquet")
    targets = target_windows(cases, controls)
    # Fixed difficult-control backgrounds, shared across all four models.
    background_rows = controls.copy()
    background_rows["endpoint"] = [endpoint_at_or_before(store, offsets, s, d) for s, d in zip(background_rows.symbol6, background_rows.start_date)]
    background_rows = background_rows.dropna(subset=["endpoint"]).copy()
    background_rows["endpoint"] = background_rows.endpoint.astype(int)
    device = choose_device(args.device)
    records = []
    diagnostics = []
    for model_index, (dataset, spec) in enumerate(cfg["representative_runs"].items()):
        run_dir = run_root / dataset / f"seed_{int(spec['seed'])}"
        checkpoint = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
        enc = checkpoint["config"]["encoder"]
        model = TS2VecDANN(enc["input_dim"], enc["hidden_dim"], enc["output_dim"], enc["dilations"], enc["dropout"], len(store.market_to_domain))
        model.load_state_dict(checkpoint["model"])
        model.eval().to(device)
        boundary = np.load(run_dir / "scores" / "robust_svdd_boundary.npz")
        center = boundary["center"].astype(np.float32)
        scale = boundary["scale"].astype(np.float32)
        scaler = checkpoint["scalers"]["cn"]
        clip = tuple(map(float, checkpoint["config"]["scaling"]["clip"]))
        peak = peak_dates(run_dir / "scores" / "scores_with_percentile.parquet", targets)
        peak["endpoint"] = [endpoint_at_or_before(store, offsets, s, int(d)) for s, d in zip(peak.symbol6, peak.anchor_date)]
        peak = peak.dropna(subset=["endpoint"]).copy()
        for sample_index, row in enumerate(peak.itertuples(index=False)):
            target = normalized_window(store, int(row.endpoint), scaler, clip)
            year = int(row.anchor_date) // 10000
            candidates = background_rows.loc[background_rows.event_id != row.event_id].copy()
            candidates["year_distance"] = (candidates.start_date // 10000 - year).abs()
            candidates = candidates.sort_values(["year_distance", "match_distance"]).head(args.background_windows)
            backgrounds = np.stack([normalized_window(store, int(endpoint), scaler, clip) for endpoint in candidates.endpoint])
            phi, diagnostic = explain_one(model, target, backgrounds, center, scale, device, args.coalitions,
                                          int(cfg["inference"]["random_seed"]) + model_index * 100_000 + sample_index)
            denom = max(float(np.abs(phi).sum()), 1e-12)
            for time_index, time_label in enumerate(TIME_LABELS):
                for feature_index, feature_group in enumerate(FEATURE_GROUPS):
                    records.append({"dataset": dataset, "seed": int(spec["seed"]), "event_id": row.event_id,
                                    "sample": row.sample, "symbol6": row.symbol6, "anchor_date": int(row.anchor_date),
                                    "time_index": time_index, "time_bin": time_label,
                                    "feature_index": feature_index, "feature_group": feature_group,
                                    "phi": float(phi[time_index, feature_index]),
                                    "absolute_contribution_share": float(abs(phi[time_index, feature_index]) / denom)})
            diagnostic.update({"dataset": dataset, "seed": int(spec["seed"]), "event_id": row.event_id,
                               "sample": row.sample, "anchor_date": int(row.anchor_date),
                               "background_windows": len(backgrounds)})
            diagnostics.append(diagnostic)
            print(f"WindowSHAP {dataset} {row.sample} {row.event_id}", flush=True)
        model.to("cpu")
        if device.type == "mps":
            torch.mps.empty_cache()
    attributions = pd.DataFrame(records)
    attributions.to_parquet(output / "data" / "fixed_windowshap_attributions.parquet", index=False)
    pd.DataFrame(diagnostics).to_csv(output / "diagnostics" / "windowshap_local_accuracy.csv", index=False, encoding="utf-8-sig")
    event_similarity, similarity_summary = similarity_inference(
        attributions, int(cfg["inference"]["bootstrap_repetitions"]),
        int(cfg["inference"]["permutation_repetitions"]), int(cfg["inference"]["random_seed"]) + 50_000
    )
    event_similarity.to_csv(output / "data" / "windowshap_event_similarity.csv", index=False, encoding="utf-8-sig")
    similarity_summary.to_csv(output / "inference" / "windowshap_similarity_inference.csv", index=False, encoding="utf-8-sig")
    print(json.dumps({"device": str(device), "attribution_rows": len(attributions),
                      "explained_samples": len(diagnostics)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
