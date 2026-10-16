#!/usr/bin/env python3
"""Evaluate every completed run on CSRC episodes and difficult controls.

This command never retrains or modifies a checkpoint.  It produces a complete
15-run result table, event-cluster bootstrap intervals, matched-set permutation
tests, and a separately labelled representative-scenario table.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
WINDOW_METRICS = ["max_svdd_percentile", "mean_svdd_percentile", "share_top_1pct", "share_top_5pct"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/posttraining_analysis.yaml")
    return parser.parse_args()


def discover_runs(root: Path) -> list[dict]:
    runs = []
    for score_path in sorted(root.glob("*/seed_*/scores/scores_with_percentile.parquet")):
        dataset = score_path.parents[2].name
        seed = int(score_path.parents[1].name.replace("seed_", ""))
        run_dir = score_path.parents[1]
        required = [run_dir / "best.pt", run_dir / "metrics.jsonl", run_dir / "scores" / "scoring_manifest.json"]
        runs.append({"dataset": dataset, "seed": seed, "score_path": score_path,
                     "run_dir": run_dir, "complete": all(p.exists() for p in required)})
    return runs


def aggregate_scores(score_path: Path, cases: pd.DataFrame, controls: pd.DataFrame) -> pd.DataFrame:
    case_input = cases[["event_id", "symbol6", "start_date", "end_date"]].copy()
    case_input["window_id"] = case_input.event_id + "::case"
    case_input["sample"] = "case"
    case_input["rank"] = 0
    control_input = controls[["event_id", "symbol6", "start_date", "end_date", "control_rank"]].copy()
    control_input["window_id"] = control_input.event_id + "::control_" + control_input.control_rank.astype(str)
    control_input["sample"] = "control"
    control_input["rank"] = control_input.control_rank.astype(int)
    windows = pd.concat([
        case_input[["event_id", "window_id", "sample", "rank", "symbol6", "start_date", "end_date"]],
        control_input[["event_id", "window_id", "sample", "rank", "symbol6", "start_date", "end_date"]],
    ], ignore_index=True)
    con = duckdb.connect()
    con.execute("PRAGMA threads=4")
    con.register("windows", windows)
    path = str(score_path).replace("'", "''")
    result = con.execute(f"""
        SELECT
          w.event_id, w.window_id, w.sample, w.rank, w.symbol6,
          w.start_date, w.end_date,
          COUNT(s.trade_date)::INTEGER AS n_scored_days,
          MAX(s.svdd_percentile_same_day) AS max_svdd_percentile,
          AVG(s.svdd_percentile_same_day) AS mean_svdd_percentile,
          AVG(CASE WHEN s.svdd_percentile_same_day >= 0.99 THEN 1.0 ELSE 0.0 END) AS share_top_1pct,
          AVG(CASE WHEN s.svdd_percentile_same_day >= 0.95 THEN 1.0 ELSE 0.0 END) AS share_top_5pct,
          MAX(s.iforest_percentile_same_day) AS max_iforest_percentile,
          AVG(s.iforest_percentile_same_day) AS mean_iforest_percentile
        FROM windows w
        LEFT JOIN (
          SELECT *, regexp_extract(symbol, '([0-9]{{6}})', 1) AS symbol6
          FROM read_parquet('{path}')
        ) s ON w.symbol6 = s.symbol6 AND s.trade_date BETWEEN w.start_date AND w.end_date
        GROUP BY ALL
        ORDER BY w.event_id, w.sample, w.rank
    """).df()
    con.close()
    return result


def paired_event_frame(windows: pd.DataFrame, metric: str) -> pd.DataFrame:
    valid = windows.loc[(windows.n_scored_days > 0) & windows[metric].notna()].copy()
    case = valid.loc[valid["sample"] == "case", ["event_id", metric]].rename(columns={metric: "case"})
    control = valid.loc[valid["sample"] == "control"].groupby("event_id", as_index=False)[metric].mean().rename(columns={metric: "control"})
    paired = case.merge(control, on="event_id", how="inner")
    paired["difference"] = paired.case - paired.control
    return paired


def classification_metrics(windows: pd.DataFrame, score: str) -> tuple[float, float]:
    valid = windows.loc[(windows.n_scored_days > 0) & windows[score].notna()].copy()
    labels = (valid["sample"] == "case").astype(int)
    if labels.nunique() < 2:
        return np.nan, np.nan
    return float(average_precision_score(labels, valid[score])), float(roc_auc_score(labels, valid[score]))


def infer_run(windows: pd.DataFrame, repetitions: int, permutations: int, seed: int) -> tuple[dict, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    case = windows.loc[(windows["sample"] == "case") & (windows.n_scored_days > 0)].copy()
    control = windows.loc[(windows["sample"] == "control") & (windows.n_scored_days > 0)].copy()
    summary: dict[str, float | int] = {
        "events_with_scores": int(case.event_id.nunique()),
        "control_windows_with_scores": int(len(control)),
        "mean_case_max_percentile": float(case.max_svdd_percentile.mean()),
        "median_case_max_percentile": float(case.max_svdd_percentile.median()),
        "case_hit_at_1pct": float((case.max_svdd_percentile >= 0.99).mean()),
        "case_hit_at_5pct": float((case.max_svdd_percentile >= 0.95).mean()),
        "control_false_alert_at_1pct": float((control.max_svdd_percentile >= 0.99).mean()),
        "control_false_alert_at_5pct": float((control.max_svdd_percentile >= 0.95).mean()),
    }
    auprc, auroc = classification_metrics(windows, "max_svdd_percentile")
    summary["matched_auprc_max_percentile"] = auprc
    summary["matched_auroc_max_percentile"] = auroc
    inference_rows = []
    for metric in WINDOW_METRICS:
        paired = paired_event_frame(windows, metric)
        observed = float(paired.difference.mean()) if len(paired) else np.nan
        if len(paired):
            values = paired.difference.to_numpy(float)
            draws = rng.integers(0, len(values), size=(repetitions, len(values)))
            boot = values[draws].mean(axis=1)
            signs = rng.choice(np.array([-1.0, 1.0]), size=(permutations, len(values)))
            null = (signs * values).mean(axis=1)
            ci_low, ci_high = np.quantile(boot, [0.025, 0.975])
            p_one = (1.0 + np.sum(null >= observed)) / (permutations + 1.0)
            p_two = (1.0 + np.sum(np.abs(null) >= abs(observed))) / (permutations + 1.0)
        else:
            ci_low = ci_high = p_one = p_two = np.nan
        summary[f"paired_difference_{metric}"] = observed
        summary[f"paired_p_one_sided_{metric}"] = float(p_one)
        inference_rows.append({
            "metric": metric, "n_events": int(len(paired)), "estimate": observed,
            "ci_low": float(ci_low), "ci_high": float(ci_high),
            "permutation_p_one_sided": float(p_one), "permutation_p_two_sided": float(p_two),
        })
    return summary, pd.DataFrame(inference_rows)


def training_diagnostic(run_dir: Path) -> dict:
    rows = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines() if line.strip()]
    frame = pd.DataFrame(rows)
    best_idx = frame.validation_ts2vec_loss.idxmin()
    best = frame.loc[best_idx]
    last = frame.iloc[-1]
    return {
        "epochs_completed": int(frame.epoch.max()),
        "best_epoch": int(best.epoch),
        "best_validation_loss": float(best.validation_ts2vec_loss),
        "last_embedding_dimension_std": float(last.embedding_dimension_std),
        "last_effective_rank": float(last.effective_rank),
        "last_domain_head_accuracy": float(last.domain_head_accuracy),
    }


def benjamini_hochberg(p_values: pd.Series) -> pd.Series:
    values = p_values.to_numpy(float)
    order = np.argsort(values)
    ranked = values[order]
    adjusted = np.minimum.accumulate((ranked * len(values) / np.arange(1, len(values) + 1))[::-1])[::-1]
    output = np.empty_like(adjusted)
    output[order] = np.clip(adjusted, 0.0, 1.0)
    return pd.Series(output, index=p_values.index)


def cross_market_incremental_tests(window_frame: pd.DataFrame, repetitions: int, permutations: int, seed: int) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(seed)
    seeds = sorted(window_frame.seed.unique())
    targets = sorted(set(window_frame.dataset.unique()) - {"a_only"})
    for target in targets:
        for run_seed in seeds:
            base_windows = window_frame.loc[(window_frame.dataset == "a_only") & (window_frame.seed == run_seed)]
            target_windows = window_frame.loc[(window_frame.dataset == target) & (window_frame.seed == run_seed)]
            if base_windows.empty or target_windows.empty:
                continue
            for metric in WINDOW_METRICS:
                base = paired_event_frame(base_windows, metric)[["event_id", "difference"]].rename(columns={"difference": "base_effect"})
                cross = paired_event_frame(target_windows, metric)[["event_id", "difference"]].rename(columns={"difference": "cross_effect"})
                paired = base.merge(cross, on="event_id", how="inner")
                delta = (paired.cross_effect - paired.base_effect).to_numpy(float)
                observed = float(delta.mean()) if len(delta) else np.nan
                if len(delta):
                    draws = rng.integers(0, len(delta), size=(repetitions, len(delta)))
                    boot = delta[draws].mean(axis=1)
                    signs = rng.choice(np.array([-1.0, 1.0]), size=(permutations, len(delta)))
                    null = (signs * delta).mean(axis=1)
                    ci_low, ci_high = np.quantile(boot, [0.025, 0.975])
                    p_one = (1.0 + np.sum(null >= observed)) / (permutations + 1.0)
                    p_two = (1.0 + np.sum(np.abs(null) >= abs(observed))) / (permutations + 1.0)
                else:
                    ci_low = ci_high = p_one = p_two = np.nan
                rows.append({
                    "target_dataset": target, "baseline_dataset": "a_only", "seed": run_seed,
                    "metric": metric, "n_events": len(delta), "incremental_estimate": observed,
                    "ci_low": float(ci_low), "ci_high": float(ci_high),
                    "permutation_p_one_sided": float(p_one), "permutation_p_two_sided": float(p_two),
                })
    result = pd.DataFrame(rows)
    if not result.empty:
        result["fdr_q_one_sided_within_metric"] = result.groupby("metric", group_keys=False)["permutation_p_one_sided"].apply(benjamini_hochberg)
    return result


def seed_ensemble_analysis(window_frame: pd.DataFrame, repetitions: int, permutations: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    id_columns = ["dataset", "event_id", "window_id", "sample", "rank", "symbol6", "start_date", "end_date"]
    value_columns = ["n_scored_days", *WINDOW_METRICS, "max_iforest_percentile", "mean_iforest_percentile"]
    ensemble = window_frame.groupby(id_columns, as_index=False)[value_columns].mean()
    summaries = []
    inferences = []
    for index, (dataset, block) in enumerate(ensemble.groupby("dataset", sort=True)):
        summary, inference = infer_run(block, repetitions, permutations, seed + index)
        summary["dataset"] = dataset
        summary["seed"] = "three_seed_ensemble"
        summaries.append(summary)
        inference["dataset"] = dataset
        inference["seed"] = "three_seed_ensemble"
        inferences.append(inference)
    incremental_rows = []
    rng = np.random.default_rng(seed + 1000)
    baseline = ensemble.loc[ensemble.dataset == "a_only"]
    for target in sorted(set(ensemble.dataset) - {"a_only"}):
        target_frame = ensemble.loc[ensemble.dataset == target]
        for metric in WINDOW_METRICS:
            base_effect = paired_event_frame(baseline, metric)[["event_id", "difference"]].rename(columns={"difference": "base_effect"})
            target_effect = paired_event_frame(target_frame, metric)[["event_id", "difference"]].rename(columns={"difference": "target_effect"})
            paired = base_effect.merge(target_effect, on="event_id", how="inner").dropna()
            delta = (paired.target_effect - paired.base_effect).to_numpy(float)
            observed = float(delta.mean()) if len(delta) else np.nan
            if len(delta):
                draws = rng.integers(0, len(delta), size=(repetitions, len(delta)))
                boot = delta[draws].mean(axis=1)
                signs = rng.choice(np.array([-1.0, 1.0]), size=(permutations, len(delta)))
                null = (signs * delta).mean(axis=1)
                ci_low, ci_high = np.quantile(boot, [0.025, 0.975])
                p_one = (1.0 + np.sum(null >= observed)) / (permutations + 1.0)
                p_two = (1.0 + np.sum(np.abs(null) >= abs(observed))) / (permutations + 1.0)
            else:
                ci_low = ci_high = p_one = p_two = np.nan
            incremental_rows.append({
                "target_dataset": target, "baseline_dataset": "a_only", "seed": "three_seed_ensemble",
                "metric": metric, "n_events": len(delta), "incremental_estimate": observed,
                "ci_low": float(ci_low), "ci_high": float(ci_high),
                "permutation_p_one_sided": float(p_one), "permutation_p_two_sided": float(p_two),
            })
    incremental = pd.DataFrame(incremental_rows)
    if not incremental.empty:
        incremental["fdr_q_one_sided_within_metric"] = incremental.groupby("metric", group_keys=False)["permutation_p_one_sided"].apply(benjamini_hochberg)
    return ensemble, pd.DataFrame(summaries), pd.concat(inferences, ignore_index=True), incremental


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    output = ROOT / cfg["paths"]["output_root"]
    for folder in ("data", "tables", "diagnostics", "inference"):
        (output / folder).mkdir(parents=True, exist_ok=True)
    cases = pd.read_parquet(output / "data" / "matched_cases.parquet")
    controls = pd.read_parquet(output / "data" / "difficult_controls.parquet")
    runs = discover_runs(ROOT / cfg["paths"]["runs_root"])
    if not runs:
        raise FileNotFoundError("No completed score files were found.")
    all_windows = []
    summaries = []
    all_inference = []
    rep = int(cfg["inference"]["bootstrap_repetitions"])
    permutations = int(cfg["inference"]["permutation_repetitions"])
    base_seed = int(cfg["inference"]["random_seed"])
    for index, run in enumerate(runs):
        windows = aggregate_scores(run["score_path"], cases, controls)
        windows["dataset"] = run["dataset"]
        windows["seed"] = run["seed"]
        all_windows.append(windows)
        summary, inference = infer_run(windows, rep, permutations, base_seed + index)
        summary.update({"dataset": run["dataset"], "seed": run["seed"], "complete": run["complete"], **training_diagnostic(run["run_dir"])})
        summaries.append(summary)
        inference["dataset"] = run["dataset"]
        inference["seed"] = run["seed"]
        all_inference.append(inference)
        print(f"evaluated {run['dataset']} seed={run['seed']}", flush=True)
    window_frame = pd.concat(all_windows, ignore_index=True)
    run_summary = pd.DataFrame(summaries).sort_values(["dataset", "seed"])
    inference_frame = pd.concat(all_inference, ignore_index=True)
    incremental = cross_market_incremental_tests(window_frame, rep, permutations, base_seed + 10_000)
    ensemble_windows, ensemble_summary, ensemble_inference, ensemble_incremental = seed_ensemble_analysis(
        window_frame, rep, permutations, base_seed + 20_000
    )
    window_frame.to_parquet(output / "data" / "all_event_control_scores.parquet", index=False)
    ensemble_windows.to_parquet(output / "data" / "seed_ensemble_event_control_scores.parquet", index=False)
    run_summary.to_csv(output / "tables" / "all_15_runs.csv", index=False, encoding="utf-8-sig")
    inference_frame.to_csv(output / "inference" / "event_cluster_inference.csv", index=False, encoding="utf-8-sig")
    incremental.to_csv(output / "inference" / "cross_market_incremental.csv", index=False, encoding="utf-8-sig")
    ensemble_summary.to_csv(output / "tables" / "seed_ensemble_summary.csv", index=False, encoding="utf-8-sig")
    ensemble_inference.to_csv(output / "inference" / "seed_ensemble_event_inference.csv", index=False, encoding="utf-8-sig")
    ensemble_incremental.to_csv(output / "inference" / "seed_ensemble_incremental.csv", index=False, encoding="utf-8-sig")
    numeric = [c for c in run_summary.select_dtypes(include="number").columns if c != "seed"]
    seed_summary = run_summary.groupby("dataset")[numeric].agg(["mean", "std", "min", "max"])
    seed_summary.columns = [f"{a}_{b}" for a, b in seed_summary.columns]
    seed_summary.reset_index().to_csv(output / "tables" / "seed_aggregate.csv", index=False, encoding="utf-8-sig")
    representatives = []
    for dataset, spec in cfg["representative_runs"].items():
        row = run_summary.loc[(run_summary.dataset == dataset) & (run_summary.seed == int(spec["seed"]))].copy()
        if row.empty:
            continue
        row["display_label"] = spec["label"]
        row["selection_rule"] = spec["selection_rule"]
        row["reporting_tier"] = "descriptive representative scenario; full 15-run table is mandatory"
        representatives.append(row)
    representative_frame = pd.concat(representatives, ignore_index=True) if representatives else pd.DataFrame()
    representative_frame.to_csv(output / "tables" / "representative_scenarios.csv", index=False, encoding="utf-8-sig")
    manifest = {
        "analysis_id": cfg["analysis_id"], "runs_discovered": len(runs),
        "complete_runs": int(sum(run["complete"] for run in runs)),
        "matched_events": int(cases.event_id.nunique()), "control_windows": int(len(controls)),
        "bootstrap_repetitions": rep, "permutation_repetitions": permutations,
        "primary_reporting": "full seed distribution",
        "representative_reporting": "explicitly labelled descriptive scenario",
    }
    (output / "analysis_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
