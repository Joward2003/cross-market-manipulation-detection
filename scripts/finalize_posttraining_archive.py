#!/usr/bin/env python3
"""Create compact paper tables, a result digest, and an integrity manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]

LABELS = {
    "a_only": "A only",
    "a_us_coral": "A + US (CORAL)",
    "a_hk_coral": "A + HK (CORAL)",
    "a_hk_us_coral": "A + HK + US (CORAL)",
    "a_hk_us_dann": "A + HK + US (DANN)",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/posttraining_analysis.yaml")
    return parser.parse_args()


def markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    header = "| " + " | ".join(columns) + " |"
    separator = "|" + "|".join(["---"] * len(columns)) + "|"
    rows = ["| " + " | ".join(str(value) for value in row) + " |" for row in frame.itertuples(index=False, name=None)]
    return "\n".join([header, separator, *rows])


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    output = ROOT / cfg["paths"]["output_root"]
    tables = output / "tables"
    inference = output / "inference"

    ensemble = pd.read_csv(tables / "seed_ensemble_summary.csv")
    ensemble["model"] = ensemble.dataset.map(LABELS)
    main_table = ensemble[[
        "model", "events_with_scores", "mean_case_max_percentile", "case_hit_at_1pct",
        "case_hit_at_5pct", "matched_auprc_max_percentile", "matched_auroc_max_percentile",
        "paired_difference_max_svdd_percentile", "paired_difference_mean_svdd_percentile",
        "paired_difference_share_top_5pct",
    ]].copy()
    main_table = main_table.rename(columns={
        "events_with_scores": "N events", "mean_case_max_percentile": "Case mean max pct",
        "case_hit_at_1pct": "Hit@1%", "case_hit_at_5pct": "Hit@5%",
        "matched_auprc_max_percentile": "Matched AUPRC", "matched_auroc_max_percentile": "Matched AUROC",
        "paired_difference_max_svdd_percentile": "Case-control max",
        "paired_difference_mean_svdd_percentile": "Case-control mean",
        "paired_difference_share_top_5pct": "Case-control top5 share",
    })
    numeric = [column for column in main_table.columns if column not in ("model", "N events")]
    main_table[numeric] = main_table[numeric].round(4)
    main_table.to_csv(tables / "paper_main_results.csv", index=False, encoding="utf-8-sig")

    incremental = pd.read_csv(inference / "seed_ensemble_incremental.csv")
    incremental["model"] = incremental.target_dataset.map(LABELS)
    incremental = incremental[["model", "metric", "n_events", "incremental_estimate", "ci_low", "ci_high",
                               "permutation_p_one_sided", "fdr_q_one_sided_within_metric"]]
    incremental.to_csv(tables / "paper_cross_market_incremental.csv", index=False, encoding="utf-8-sig")

    placebo = pd.read_csv(inference / "time_placebo_inference.csv")
    placebo["model"] = placebo.dataset.map(LABELS)
    placebo[["model", "metric", "n_events", "actual_minus_placebo", "ci_low", "ci_high",
             "permutation_p_one_sided"]].to_csv(tables / "paper_time_placebo.csv", index=False, encoding="utf-8-sig")

    shap = pd.read_csv(inference / "windowshap_similarity_inference.csv")
    shap["model"] = shap.dataset.map(LABELS)
    shap[["model", "basis", "n_pairs", "mean_case_similarity", "mean_control_similarity",
          "paired_difference", "ci_low", "ci_high", "permutation_p_one_sided"]].to_csv(
        tables / "paper_windowshap_similarity.csv", index=False, encoding="utf-8-sig")

    us_inc = incremental.loc[(incremental.model == "A + US (CORAL)")]
    us_mean = us_inc.loc[us_inc.metric == "mean_svdd_percentile"].iloc[0]
    us_top5 = us_inc.loc[us_inc.metric == "share_top_5pct"].iloc[0]
    placebo_mean = placebo.loc[(placebo.dataset == "a_us_coral") & (placebo.metric == "mean_svdd_percentile")].iloc[0]
    placebo_max = placebo.loc[(placebo.dataset == "a_us_coral") & (placebo.metric == "max_svdd_percentile")].iloc[0]
    shap_signed_us = shap.loc[(shap.dataset == "a_us_coral") & (shap.basis == "signed_phi")].iloc[0]
    digest = f"""# Post-training result digest

Generated from all 15 completed runs (five specifications × three seeds). Representative-seed plots are descriptive only and never replace the full-seed tables.

## Main findings

- The three-seed A+US CORAL ensemble raises the regulator-event versus difficult-control contrast in mean event-window percentile by **{us_mean.incremental_estimate:.4f}** (one-sided permutation p={us_mean.permutation_p_one_sided:.3f}); its 95% event-bootstrap interval is [{us_mean.ci_low:.4f}, {us_mean.ci_high:.4f}].
- Its incremental share of event days in the top 5% is **{us_top5.incremental_estimate:.4f}** (p={us_top5.permutation_p_one_sided:.3f}; 95% CI [{us_top5.ci_low:.4f}, {us_top5.ci_high:.4f}]). The within-metric FDR-adjusted q-value is {us_top5.fdr_q_one_sided_within_metric:.3f}, so this is supportive but not multiplicity-robust evidence.
- In the same-stock, same-duration time placebo, A+US scores are higher in the documented conduct interval by **{placebo_mean.actual_minus_placebo:.4f}** for mean percentile (p={placebo_mean.permutation_p_one_sided:.3f}) and **{placebo_max.actual_minus_placebo:.4f}** for maximum percentile (p={placebo_max.permutation_p_one_sided:.3f}). The A-only placebo contrasts are not significant.
- HK-only and DANN specifications do not show stable incremental gains. Joint A+HK+US CORAL is positive on some metrics but weaker and less stable than A+US.
- In the representative A+US model, signed WindowSHAP patterns are more similar among regulator events than matched controls by **{shap_signed_us.paired_difference:.4f}** (one-sided permutation p={shap_signed_us.permutation_p_one_sided:.3f}), but its bootstrap interval [{shap_signed_us.ci_low:.4f}, {shap_signed_us.ci_high:.4f}] crosses zero and the absolute-importance version does not confirm the result. This is exploratory dynamic-pattern evidence, not a causal state sequence.

## Interpretation boundary

These results support a narrower claim: US high-liquidity unlabeled data can add useful out-of-market information for ranking regulator-confirmed A-share manipulation intervals under a fixed representation/anomaly pipeline. They do not establish that US/HK markets are manipulation-free, that public OHLCV legally identifies manipulation, or that SHAP contributions are causal behavioral states.
"""
    (output / "results_digest.md").write_text(digest, encoding="utf-8")

    table_map = pd.DataFrame([
        ("Main three-seed results", "tables/paper_main_results.csv", "fig01_all_seed_overview; fig04_matched_case_control_effects"),
        ("Cross-market increment", "tables/paper_cross_market_incremental.csv", "fig07_seed_ensemble_incremental"),
        ("Time placebo", "tables/paper_time_placebo.csv", "fig08_time_placebo"),
        ("Event-time dynamics", "tables/event_time_trajectories.csv", "fig09_event_time_trajectories"),
        ("WindowSHAP", "tables/paper_windowshap_similarity.csv", "fig10_windowshap_heatmaps; fig11_windowshap_similarity"),
        ("Matching balance", "diagnostics/matching_balance.csv", "fig03_matching_balance"),
    ], columns=["paper_component", "table", "figure_stem"])
    table_map.to_csv(output / "table_figure_map.csv", index=False, encoding="utf-8-sig")

    readme = f"""# Final post-training archive (`{cfg['analysis_id']}`)

This directory is the frozen analysis package built from the 15 completed OpenBayes runs.

## Read first

- `results_digest.md`: concise, evidence-bounded findings.
- `tables/paper_main_results.csv`: mandatory three-seed main table.
- `tables/all_15_runs.csv`: every run; nothing is dropped because it is unfavorable.
- `tables/representative_scenarios.csv`: explicitly selected descriptive runs.
- `inference/`: event-cluster bootstrap, paired permutation, incremental, placebo, and WindowSHAP tests.
- `figures/`: 300-dpi PNG and vector PDF outputs.
- `file_manifest_sha256.csv`: reproducibility checksums and file sizes.

Figures follow the restrained palette, typography, spine, and dual-format conventions documented by the figures4papers reference repository at commit `{cfg['figures']['reference_commit']}`. No repository code was copied.
"""
    (output / "README.md").write_text(readme, encoding="utf-8")

    manifest_rows = []
    excluded = {output / "file_manifest_sha256.csv"}
    for path in sorted(p for p in output.rglob("*") if p.is_file() and p not in excluded):
        manifest_rows.append({"path": str(path.relative_to(output)), "bytes": path.stat().st_size, "sha256": sha256(path)})
    pd.DataFrame(manifest_rows).to_csv(output / "file_manifest_sha256.csv", index=False, encoding="utf-8-sig")
    print(json.dumps({"archive": str(output), "manifest_files": len(manifest_rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
