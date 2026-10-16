#!/usr/bin/env python3
"""Create publication figures from the frozen post-training result tables."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.analysis.figure_style import PALETTE, apply_publication_style, finalize_figure
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

ORDER = ["a_only", "a_us_coral", "a_hk_coral", "a_hk_us_coral", "a_hk_us_dann"]
LABELS = {
    "a_only": "A only",
    "a_us_coral": "A + US\nCORAL",
    "a_hk_coral": "A + HK\nCORAL",
    "a_hk_us_coral": "A + HK + US\nCORAL",
    "a_hk_us_dann": "A + HK + US\nDANN",
}
COLORS = {
    "a_only": PALETTE["neutral"],
    "a_us_coral": PALETTE["blue_main"],
    "a_hk_coral": PALETTE["green"],
    "a_hk_us_coral": PALETTE["teal"],
    "a_hk_us_dann": PALETTE["violet"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/posttraining_analysis.yaml")
    return parser.parse_args()


def all_seed_overview(frame: pd.DataFrame, output: Path, formats: list[str], dpi: int) -> list[Path]:
    metrics = [
        ("mean_case_max_percentile", "Mean event maximum percentile"),
        ("case_hit_at_1pct", "Event hit rate: top 1%"),
        ("case_hit_at_5pct", "Event hit rate: top 5%"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.2), sharey=True)
    rng = np.random.default_rng(20260819)
    for ax, (metric, title) in zip(axes, metrics):
        for i, dataset in enumerate(ORDER):
            values = frame.loc[frame.dataset == dataset, metric].dropna().to_numpy(float)
            if not len(values):
                continue
            jitter = rng.uniform(-0.08, 0.08, size=len(values))
            ax.scatter(np.repeat(i, len(values)) + jitter, values, s=42, color=COLORS[dataset],
                       edgecolor="white", linewidth=0.7, alpha=0.9, zorder=3)
            mean = values.mean()
            sd = values.std(ddof=1) if len(values) > 1 else 0.0
            ax.errorbar(i, mean, yerr=sd, fmt="_", markersize=18, color=PALETTE["dark"],
                        linewidth=1.5, capsize=4, zorder=4)
        ax.set_title(title, fontweight="bold", fontsize=12)
        ax.set_xticks(range(len(ORDER)), [LABELS[x] for x in ORDER], fontsize=9)
        ax.set_ylim(0, 1)
        ax.set_yticks(np.linspace(0, 1, 6))
        ax.grid(axis="y", color="#E6E6E6", linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Proportion / percentile")
    fig.suptitle("All completed runs: individual seeds and mean ± SD", y=1.02, fontsize=14, fontweight="bold")
    return finalize_figure(fig, output / "fig01_all_seed_overview", formats, dpi)


def representative_bars(frame: pd.DataFrame, output: Path, formats: list[str], dpi: int) -> list[Path]:
    ordered = frame.set_index("dataset").loc[[x for x in ORDER if x in frame.dataset.values]].reset_index()
    names = ordered.display_label.tolist()
    values = np.array([
        ordered.mean_case_max_percentile.to_numpy(float),
        ordered.case_hit_at_1pct.to_numpy(float),
        ordered.case_hit_at_5pct.to_numpy(float),
    ]).T
    metric_labels = ["Mean max percentile", "Hit@1%", "Hit@5%"]
    metric_colors = [PALETTE["blue_main"], PALETTE["red_light"], PALETTE["green"]]
    x = np.arange(len(names))
    width = 0.23
    fig, ax = plt.subplots(figsize=(12.5, 5.2))
    for j, label in enumerate(metric_labels):
        bars = ax.bar(x + (j - 1) * width, values[:, j], width, label=label,
                      color=metric_colors[j], edgecolor=PALETTE["dark"], linewidth=1.0)
        for bar, value in zip(bars, values[:, j]):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.018, f"{value:.3f}",
                    ha="center", va="bottom", fontsize=9, rotation=0)
    ax.set_xticks(x, names)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Proportion / percentile")
    ax.set_title("Preselected representative scenarios", fontweight="bold")
    ax.legend(ncol=3, loc="upper center")
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.text(0.0, -0.22, "Descriptive selected-seed comparison; not a substitute for the full seed distribution.",
            transform=ax.transAxes, fontsize=9, color="#555555")
    return finalize_figure(fig, output / "fig02_representative_scenarios", formats, dpi)


def balance_plot(balance: pd.DataFrame, output: Path, formats: list[str], dpi: int) -> list[Path]:
    pivot = balance.pivot(index="feature", columns="sample", values="smd")
    pivot = pivot.reindex(pivot.matched_controls.abs().sort_values().index)
    y = np.arange(len(pivot))
    fig, ax = plt.subplots(figsize=(8.5, 6.0))
    ax.axvspan(-0.1, 0.1, color=PALETTE["green"], alpha=0.18, zorder=0)
    ax.axvline(0, color=PALETTE["dark"], linewidth=1)
    if "unmatched_candidate_pool" in pivot:
        ax.scatter(pivot.unmatched_candidate_pool, y, color=PALETTE["neutral"], s=45,
                   label="Unmatched candidate pool", zorder=3)
    ax.scatter(pivot.matched_controls, y, color=PALETTE["blue_main"], s=52,
               label="Matched controls", zorder=4)
    ax.set_yticks(y, [x.replace("_", " ") for x in pivot.index])
    ax.set_xlabel("Standardized mean difference (case − control)")
    ax.set_title("Covariate balance before and after difficult-control matching", fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(axis="x", color="#E6E6E6", linewidth=0.8)
    return finalize_figure(fig, output / "fig03_matching_balance", formats, dpi)


def paired_effect_plot(inference: pd.DataFrame, output: Path, formats: list[str], dpi: int) -> list[Path]:
    data = inference.loc[inference.metric == "max_svdd_percentile"].copy()
    data["dataset"] = pd.Categorical(data.dataset, ORDER, ordered=True)
    data = data.sort_values(["dataset", "seed"])
    y_map = {dataset: i for i, dataset in enumerate(reversed(ORDER))}
    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    offsets = {-1: -0.14, 0: 0.0, 1: 0.14}
    seeds = sorted(data.seed.unique())
    for dataset in ORDER:
        sub = data.loc[data.dataset == dataset]
        for j, row in enumerate(sub.itertuples(index=False)):
            y = y_map[dataset] + offsets.get(j - 1, 0)
            ax.errorbar(row.estimate, y,
                        xerr=[[row.estimate - row.ci_low], [row.ci_high - row.estimate]],
                        fmt="o", color=COLORS[dataset], capsize=3, markersize=6, linewidth=1.2,
                        label=f"Seed {row.seed}" if dataset == ORDER[0] else None)
    ax.axvline(0, color=PALETTE["dark"], linewidth=1)
    ax.set_yticks([y_map[d] for d in ORDER], [LABELS[d].replace("\n", " ") for d in ORDER])
    ax.set_xlabel("Case − mean matched-control maximum percentile")
    ax.set_title("Matched event contrast with 95% event-bootstrap intervals", fontweight="bold")
    ax.grid(axis="x", color="#E6E6E6", linewidth=0.8)
    ax.legend(loc="lower right", ncol=3, fontsize=8)
    return finalize_figure(fig, output / "fig04_matched_case_control_effects", formats, dpi)


def case_control_distribution(windows: pd.DataFrame, representatives: pd.DataFrame, output: Path, formats: list[str], dpi: int) -> list[Path]:
    selected = representatives[["dataset", "seed", "display_label"]]
    data = windows.merge(selected, on=["dataset", "seed"], how="inner")
    datasets = selected.dataset.tolist()
    fig, axes = plt.subplots(1, len(datasets), figsize=(4.0 * len(datasets), 4.5), sharey=True)
    axes = np.atleast_1d(axes)
    rng = np.random.default_rng(20260819)
    for ax, dataset in zip(axes, datasets):
        sub = data.loc[(data.dataset == dataset) & (data.n_scored_days > 0)]
        groups = [sub.loc[sub["sample"] == s, "max_svdd_percentile"].dropna().to_numpy(float) for s in ("case", "control")]
        bp = ax.boxplot(groups, positions=[0, 1], widths=0.55, patch_artist=True, showfliers=False,
                        medianprops={"color": PALETTE["dark"], "linewidth": 1.5})
        for patch, color in zip(bp["boxes"], [PALETTE["red_light"], PALETTE["blue_secondary"]]):
            patch.set_facecolor(color); patch.set_alpha(0.75); patch.set_edgecolor(PALETTE["dark"])
        for i, values in enumerate(groups):
            take = values if len(values) <= 180 else rng.choice(values, 180, replace=False)
            ax.scatter(i + rng.uniform(-0.11, 0.11, len(take)), take, s=9, alpha=0.28,
                       color=PALETTE["dark"], linewidth=0)
        ax.set_xticks([0, 1], ["CSRC events", "Difficult controls"])
        ax.set_title(LABELS[dataset].replace("\n", " "), fontsize=11, fontweight="bold")
        ax.grid(axis="y", color="#E6E6E6", linewidth=0.8)
        ax.set_ylim(0, 1.02)
    axes[0].set_ylabel("Event-window maximum percentile")
    fig.suptitle("Regulator-confirmed events versus matched abnormal windows", y=1.02, fontsize=14, fontweight="bold")
    return finalize_figure(fig, output / "fig05_case_control_distributions", formats, dpi)


def training_curves(runs_root: Path, representatives: pd.DataFrame, output: Path, formats: list[str], dpi: int) -> list[Path]:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for row in representatives.itertuples(index=False):
        path = runs_root / row.dataset / f"seed_{row.seed}" / "metrics.jsonl"
        metrics = pd.DataFrame([json.loads(line) for line in path.read_text().splitlines() if line.strip()])
        color = COLORS.get(row.dataset, PALETTE["dark"])
        label = LABELS.get(row.dataset, row.dataset).replace("\n", " ")
        axes[0].plot(metrics.epoch, metrics.validation_ts2vec_loss, label=label, color=color, linewidth=2)
        axes[1].plot(metrics.epoch, metrics.effective_rank, label=label, color=color, linewidth=2)
    axes[0].set_title("Validation contrastive loss", fontweight="bold")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[1].set_title("Embedding effective rank", fontweight="bold")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Effective rank")
    for ax in axes:
        ax.grid(color="#E6E6E6", linewidth=0.8)
    axes[1].legend(loc="lower right", fontsize=9)
    return finalize_figure(fig, output / "fig06_training_diagnostics", formats, dpi)


def ensemble_incremental_plot(frame: pd.DataFrame, output: Path, formats: list[str], dpi: int) -> list[Path]:
    metrics = [
        ("mean_svdd_percentile", "Mean event-window percentile"),
        ("share_top_5pct", "Share of days ranked in top 5%"),
    ]
    targets = [x for x in ORDER if x != "a_only" and x in frame.target_dataset.unique()]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5), sharey=True)
    y = np.arange(len(targets))
    for ax, (metric, title) in zip(axes, metrics):
        block = frame.loc[frame.metric == metric].set_index("target_dataset").reindex(targets)
        estimates = block.incremental_estimate.to_numpy(float)
        ax.errorbar(estimates, y,
                    xerr=np.vstack([estimates - block.ci_low.to_numpy(float), block.ci_high.to_numpy(float) - estimates]),
                    fmt="o", color=PALETTE["blue_main"], markersize=7, capsize=4, linewidth=1.5)
        ax.axvline(0, color=PALETTE["dark"], linewidth=1)
        ax.set_title(title, fontweight="bold", fontsize=11)
        ax.set_xlabel("Increment over A-only case–control contrast")
        ax.grid(axis="x", color="#E6E6E6", linewidth=0.8)
        for i, row in enumerate(block.itertuples()):
            ax.text(row.ci_high + 0.004, i, f"p={row.permutation_p_one_sided:.3f}", va="center", fontsize=8)
    axes[0].set_yticks(y, [LABELS[x].replace("\n", " ") for x in targets])
    fig.suptitle("Cross-market incremental tests using three-seed ensemble scores", y=1.02,
                 fontsize=14, fontweight="bold")
    return finalize_figure(fig, output / "fig07_seed_ensemble_incremental", formats, dpi)


def time_placebo_plot(frame: pd.DataFrame, output: Path, formats: list[str], dpi: int) -> list[Path]:
    metrics = [
        ("max_svdd_percentile", "Maximum percentile"),
        ("mean_svdd_percentile", "Mean percentile"),
    ]
    datasets = ["a_only", "a_us_coral"]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2), sharey=True)
    y = np.arange(len(datasets))
    for ax, (metric, title) in zip(axes, metrics):
        block = frame.loc[frame.metric == metric].set_index("dataset").reindex(datasets)
        estimates = block.actual_minus_placebo.to_numpy(float)
        colors = [COLORS[x] for x in datasets]
        for i, dataset in enumerate(datasets):
            row = block.loc[dataset]
            ax.errorbar(row.actual_minus_placebo, i,
                        xerr=[[row.actual_minus_placebo - row.ci_low], [row.ci_high - row.actual_minus_placebo]],
                        fmt="o", color=colors[i], markersize=8, capsize=4, linewidth=1.5)
            ax.text(row.ci_high + 0.004, i, f"p={row.permutation_p_one_sided:.3f}", va="center", fontsize=9)
        ax.axvline(0, color=PALETTE["dark"], linewidth=1)
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Actual event − same-stock time placebo")
        ax.grid(axis="x", color="#E6E6E6", linewidth=0.8)
    axes[0].set_yticks(y, [LABELS[x].replace("\n", " ") for x in datasets])
    fig.suptitle("Same-duration randomly shifted placebo tests", y=1.02, fontsize=14, fontweight="bold")
    return finalize_figure(fig, output / "fig08_time_placebo", formats, dpi)


def event_time_plot(frame: pd.DataFrame, output: Path, formats: list[str], dpi: int) -> list[Path]:
    datasets = ["a_only", "a_us_coral"]
    anchors = ["start", "end"]
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 7.8), sharex=True, sharey=True)
    sample_style = {
        "case": (PALETTE["red"], "CSRC events"),
        "control": (PALETTE["blue_secondary"], "Difficult controls"),
    }
    for row, anchor in enumerate(anchors):
        for col, dataset in enumerate(datasets):
            ax = axes[row, col]
            for sample in ("control", "case"):
                block = frame.loc[(frame.anchor_type == anchor) & (frame.dataset == dataset) & (frame["sample"] == sample)].sort_values("event_time")
                color, label = sample_style[sample]
                x = block.event_time.to_numpy(float)
                mean = block.mean_percentile.to_numpy(float)
                ax.plot(x, mean, color=color, linewidth=2.0, label=label)
                ax.fill_between(x, block.ci_low.to_numpy(float), block.ci_high.to_numpy(float), color=color, alpha=0.13, linewidth=0)
            ax.axvline(0, color=PALETTE["dark"], linewidth=1, linestyle="--")
            ax.set_xlim(-60, 60)
            ax.set_ylim(0.35, 0.75)
            ax.grid(color="#E6E6E6", linewidth=0.8)
            ax.set_title(f"{LABELS[dataset].replace(chr(10), ' ')} — aligned on conduct {anchor}", fontweight="bold", fontsize=11)
            if row == 1:
                ax.set_xlabel("Trading days from documented conduct anchor")
            if col == 0:
                ax.set_ylabel("Mean same-day anomaly percentile")
    axes[0, 1].legend(loc="upper left", fontsize=9)
    fig.suptitle("Event-time anomaly trajectories: three-seed ensemble", y=1.01, fontsize=14, fontweight="bold")
    return finalize_figure(fig, output / "fig09_event_time_trajectories", formats, dpi)


def windowshap_heatmaps(frame: pd.DataFrame, output: Path, formats: list[str], dpi: int) -> list[Path]:
    """Compare when and through which channels the representative models react."""
    datasets = ["a_only", "a_us_coral"]
    time_order = frame.sort_values("time_index").time_bin.drop_duplicates().tolist()
    feature_order = frame.sort_values("feature_index").feature_group.drop_duplicates().tolist()
    sample_max = 0.0
    difference_max = 0.0
    matrices: dict[tuple[str, str], np.ndarray] = {}
    for dataset in datasets:
        block = frame.loc[frame.dataset == dataset]
        sample_matrices = {}
        for sample in ("case", "control"):
            matrix = (block.loc[block["sample"] == sample]
                      .pivot_table(index="feature_group", columns="time_bin",
                                   values="absolute_contribution_share", aggfunc="mean")
                      .reindex(index=feature_order, columns=time_order).to_numpy(float))
            matrices[(dataset, sample)] = matrix
            sample_matrices[sample] = matrix
            sample_max = max(sample_max, float(np.nanmax(matrix)))
        difference = sample_matrices["case"] - sample_matrices["control"]
        matrices[(dataset, "difference")] = difference
        difference_max = max(difference_max, float(np.nanmax(np.abs(difference))))
    fig, axes = plt.subplots(2, 3, figsize=(14.0, 7.0), constrained_layout=True)
    sequential = None
    divergent = None
    for row, dataset in enumerate(datasets):
        for col, sample in enumerate(("case", "control", "difference")):
            ax = axes[row, col]
            matrix = matrices[(dataset, sample)]
            if sample == "difference":
                scale = max(difference_max, 1e-5)
                image = ax.imshow(matrix, cmap="RdBu_r", norm=TwoSlopeNorm(vmin=-scale, vcenter=0, vmax=scale), aspect="auto")
                divergent = image
            else:
                image = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=sample_max, aspect="auto")
                sequential = image
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    value = matrix[i, j]
                    color = "white" if (sample != "difference" and value > sample_max * 0.6) else PALETTE["dark"]
                    ax.text(j, i, f"{value * 100:.1f}", ha="center", va="center", fontsize=8, color=color)
            ax.set_xticks(range(len(time_order)), time_order, rotation=35, ha="right", fontsize=8)
            if col == 0:
                ax.set_yticks(range(len(feature_order)), [x.replace("_", " ") for x in feature_order], fontsize=9)
                ax.set_ylabel(LABELS[dataset].replace("\n", " "), fontweight="bold")
            else:
                ax.set_yticks(range(len(feature_order)), [])
            if row == 0:
                ax.set_title({"case": "CSRC events", "control": "Difficult controls",
                              "difference": "Event − control"}[sample], fontweight="bold")
    if sequential is not None:
        cb = fig.colorbar(sequential, ax=axes[:, :2], shrink=0.78, pad=0.015)
        cb.set_label("Mean absolute contribution share")
    if divergent is not None:
        cb = fig.colorbar(divergent, ax=axes[:, 2], shrink=0.78, pad=0.015)
        cb.set_label("Difference in contribution share")
    fig.suptitle("Fixed WindowSHAP attribution structure around peak-risk dates", fontsize=14, fontweight="bold")
    return finalize_figure(fig, output / "fig10_windowshap_heatmaps", formats, dpi)


def windowshap_similarity_plot(frame: pd.DataFrame, output: Path, formats: list[str], dpi: int) -> list[Path]:
    basis_order = ["absolute_share", "signed_phi"]
    datasets = [x for x in ORDER if x in frame.dataset.unique() and x != "a_hk_us_dann"]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), sharey=True)
    y = np.arange(len(datasets))
    for ax, basis in zip(axes, basis_order):
        block = frame.loc[frame.basis == basis].set_index("dataset").reindex(datasets)
        for i, dataset in enumerate(datasets):
            row = block.loc[dataset]
            ax.errorbar(row.paired_difference, i,
                        xerr=[[row.paired_difference - row.ci_low], [row.ci_high - row.paired_difference]],
                        fmt="o", color=COLORS[dataset], markersize=7, capsize=4, linewidth=1.4)
            ax.text(row.ci_high + 0.008, i, f"p={row.permutation_p_one_sided:.3f}", va="center", fontsize=8)
        ax.axvline(0, color=PALETTE["dark"], linewidth=1)
        ax.grid(axis="x", color="#E6E6E6", linewidth=0.8)
        ax.set_xlabel("Case − control similarity to leave-one-event-out case template")
        ax.set_title("Absolute importance structure" if basis == "absolute_share" else "Signed contribution structure",
                     fontweight="bold", fontsize=11)
    axes[0].set_yticks(y, [LABELS[x].replace("\n", " ") for x in datasets])
    fig.suptitle("Do regulator-confirmed events share a common attribution pattern?", y=1.02,
                 fontsize=14, fontweight="bold")
    return finalize_figure(fig, output / "fig11_windowshap_similarity", formats, dpi)


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    output = ROOT / cfg["paths"]["output_root"]
    figure_dir = output / "figures"
    formats = list(cfg["figures"]["formats"])
    dpi = int(cfg["figures"]["dpi"])
    apply_publication_style()
    all_runs = pd.read_csv(output / "tables" / "all_15_runs.csv")
    reps = pd.read_csv(output / "tables" / "representative_scenarios.csv")
    balance = pd.read_csv(output / "diagnostics" / "matching_balance.csv")
    inference = pd.read_csv(output / "inference" / "event_cluster_inference.csv")
    ensemble_incremental = pd.read_csv(output / "inference" / "seed_ensemble_incremental.csv")
    time_placebo = pd.read_csv(output / "inference" / "time_placebo_inference.csv")
    event_time = pd.read_csv(output / "tables" / "event_time_trajectories.csv")
    windowshap = pd.read_parquet(output / "data" / "fixed_windowshap_attributions.parquet")
    windowshap_similarity = pd.read_csv(output / "inference" / "windowshap_similarity_inference.csv")
    windows = pd.read_parquet(output / "data" / "all_event_control_scores.parquet")
    generated = []
    generated += all_seed_overview(all_runs, figure_dir, formats, dpi)
    generated += representative_bars(reps, figure_dir, formats, dpi)
    generated += balance_plot(balance, figure_dir, formats, dpi)
    generated += paired_effect_plot(inference, figure_dir, formats, dpi)
    generated += case_control_distribution(windows, reps, figure_dir, formats, dpi)
    generated += training_curves(ROOT / cfg["paths"]["runs_root"], reps, figure_dir, formats, dpi)
    generated += ensemble_incremental_plot(ensemble_incremental, figure_dir, formats, dpi)
    generated += time_placebo_plot(time_placebo, figure_dir, formats, dpi)
    generated += event_time_plot(event_time, figure_dir, formats, dpi)
    generated += windowshap_heatmaps(windowshap, figure_dir, formats, dpi)
    generated += windowshap_similarity_plot(windowshap_similarity, figure_dir, formats, dpi)
    manifest = pd.DataFrame({"file": [str(path.relative_to(ROOT)) for path in generated]})
    manifest["reference_repository"] = cfg["figures"]["reference_repository"]
    manifest["reference_commit"] = cfg["figures"]["reference_commit"]
    manifest.to_csv(output / "figure_manifest.csv", index=False, encoding="utf-8-sig")
    print(f"generated {len(generated)} figure files in {figure_dir}")


if __name__ == "__main__":
    main()
