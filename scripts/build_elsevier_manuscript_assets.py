#!/usr/bin/env python3
"""Build auditable tables and figures for the Elsevier manuscript.

Every numerical manuscript asset is generated from the frozen
``results/final_20260819`` archive.  The script never edits the archived
results and never chooses runs based on their direction.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "results" / "final_20260819"
PAPER = ROOT / "paper" / "elsevier_submission"
TABLES = PAPER / "tables"
FIGURES = PAPER / "figures"
SOURCE_DATA = PAPER / "source_data"

MODEL_LABELS = {
    "a_only": "A only",
    "a_us_coral": "A + US (CORAL)",
    "a_hk_coral": "A + HK (CORAL)",
    "a_hk_us_coral": "A + HK + US (CORAL)",
    "a_hk_us_dann": "A + HK + US (DANN)",
}


def latex_escape(value: object) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def write_table(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def fmt(value: object, digits: int = 3) -> str:
    if pd.isna(value):
        return "--"
    return f"{float(value):.{digits}f}"


def p_fmt(value: object) -> str:
    if pd.isna(value):
        return "--"
    value = float(value)
    return "$<0.001$" if value < 0.001 else f"{value:.3f}"


def build_sample_table() -> None:
    rows = [
        ("A shares", "All eligible A shares excluding BSE", "2011--2025", "12,564,184", "Training target domain"),
        ("US equities", "Current S\\&P 500 constituents with available history", "2011--2025", "1,773,029", "Unlabeled reference domain"),
        ("Hong Kong equities", "Prior-year dynamically selected liquid ordinary shares", "2011--2025", "2,616,376", "Unlabeled reference domain"),
        ("CSRC strict panel", "Stock--conduct-interval episodes", "Documents released 2012--2025", "226", "External labels only"),
        ("Held-out CSRC panel", "Conduct beginning in 2018 or later", "2018--2025 scoring period", "68 (67 scoreable)", "Primary evaluation"),
        ("Difficult controls", "Five same-calendar non-penalized abnormal windows per event", "2018--2023 conduct starts", "340 (302 stocks)", "Matched comparison"),
    ]
    lines = [
        r"\begin{tabularx}{\textwidth}{@{}lXlll@{}}",
        r"\toprule",
        r"Sample & Construction & Period & Observations & Role \\",
        r"\midrule",
    ]
    lines += [" & ".join(row) + r" \\" for row in rows]
    lines += [r"\bottomrule", r"\end{tabularx}"]
    write_table(TABLES / "sample_construction.tex", lines)


def build_main_results() -> None:
    df = pd.read_csv(ARCHIVE / "tables" / "paper_main_results.csv", encoding="utf-8-sig")
    lines = [
        r"\begin{tabular}{@{}lrrrrrr@{}}",
        r"\toprule",
        r"Model & Max pct. & Hit@5\% & AUPRC & AUROC & $\Delta$ mean & $\Delta$ top 5\% \\",
        r"\midrule",
    ]
    for row in df.itertuples(index=False):
        lines.append(
            f"{latex_escape(row[0])} & {fmt(row[2])} & {fmt(row[4])} & {fmt(row[5])} & "
            f"{fmt(row[6])} & {fmt(row[8])} & {fmt(row[9])} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    write_table(TABLES / "main_results.tex", lines)


def build_incremental_results() -> None:
    df = pd.read_csv(ARCHIVE / "tables" / "paper_cross_market_incremental.csv", encoding="utf-8-sig")
    metric_labels = {
        "max_svdd_percentile": "Maximum percentile",
        "mean_svdd_percentile": "Mean percentile",
        "share_top_1pct": "Share in top 1\\%",
        "share_top_5pct": "Share in top 5\\%",
    }
    lines = [
        r"\begin{tabular}{@{}llrrrr@{}}",
        r"\toprule",
        r"Reference domain & Metric & Increment & 95\% CI low & 95\% CI high & $q$ \\",
        r"\midrule",
    ]
    for row in df.itertuples(index=False):
        lines.append(
            f"{latex_escape(row.model)} & {metric_labels[row.metric]} & {fmt(row.incremental_estimate)} & "
            f"{fmt(row.ci_low)} & {fmt(row.ci_high)} & {fmt(row.fdr_q_one_sided_within_metric)} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    write_table(TABLES / "incremental_results.tex", lines)


def build_time_placebo() -> None:
    df = pd.read_csv(ARCHIVE / "tables" / "paper_time_placebo.csv", encoding="utf-8-sig")
    metric_labels = {
        "max_svdd_percentile": "Maximum percentile",
        "mean_svdd_percentile": "Mean percentile",
        "share_top_1pct": "Share in top 1\\%",
        "share_top_5pct": "Share in top 5\\%",
    }
    lines = [
        r"\begin{tabular}{@{}llrrrr@{}}",
        r"\toprule",
        r"Model & Metric & Actual $-$ placebo & 95\% CI low & 95\% CI high & $p$ \\",
        r"\midrule",
    ]
    for row in df.itertuples(index=False):
        lines.append(
            f"{latex_escape(row.model)} & {metric_labels[row.metric]} & {fmt(row.actual_minus_placebo)} & "
            f"{fmt(row.ci_low)} & {fmt(row.ci_high)} & {p_fmt(row.permutation_p_one_sided)} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    write_table(TABLES / "time_placebo.tex", lines)


def build_windowshap() -> None:
    df = pd.read_csv(ARCHIVE / "tables" / "paper_windowshap_similarity.csv", encoding="utf-8-sig")
    basis = {"signed_phi": "Signed attribution", "absolute_share": "Absolute share"}
    lines = [
        r"\begin{tabular}{@{}llrrrr@{}}",
        r"\toprule",
        r"Model & Basis & Case sim. & Control sim. & Difference & $p$ \\",
        r"\midrule",
    ]
    for row in df.itertuples(index=False):
        lines.append(
            f"{latex_escape(row.model)} & {basis[row.basis]} & {fmt(row.mean_case_similarity)} & "
            f"{fmt(row.mean_control_similarity)} & {fmt(row.paired_difference)} & "
            f"{p_fmt(row.permutation_p_one_sided)} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    write_table(TABLES / "windowshap_similarity.tex", lines)


def build_matching_balance() -> None:
    df = pd.read_csv(ARCHIVE / "diagnostics" / "matching_balance.csv", encoding="utf-8-sig")
    labels = {
        "cumulative_market_adjusted_return": "Cumulative market-adjusted return",
        "realized_return_volatility": "Realized return volatility",
        "mean_turnover_shock": "Mean turnover shock",
        "mean_relative_volume": "Mean relative volume",
        "mean_intraday_range": "Mean intraday range",
        "log_median_circulating_market_value": "Log circulating market value",
        "log_median_listing_age": "Log listing age",
        "mean_log_amihud": "Mean log Amihud illiquidity",
        "extreme_return_share": "Extreme-return share",
        "mean_absolute_ma_deviation": "Mean absolute MA deviation",
    }
    wide = df.pivot(index="feature", columns="sample", values="smd")
    lines = [
        r"\begin{tabular}{@{}lrr@{}}",
        r"\toprule",
        r"Covariate & Before matching & After matching \\",
        r"\midrule",
    ]
    for feature in labels:
        lines.append(
            f"{labels[feature]} & {fmt(wide.loc[feature, 'unmatched_candidate_pool'])} & "
            f"{fmt(wide.loc[feature, 'matched_controls'])} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    write_table(TABLES / "matching_balance.tex", lines)


def build_all_runs() -> None:
    df = pd.read_csv(ARCHIVE / "tables" / "all_15_runs.csv", encoding="utf-8-sig")
    columns = [
        "dataset", "seed", "best_epoch", "mean_case_max_percentile", "case_hit_at_1pct",
        "case_hit_at_5pct", "matched_auprc_max_percentile", "matched_auroc_max_percentile",
        "paired_difference_mean_svdd_percentile", "paired_difference_share_top_5pct",
    ]
    df = df[columns]
    lines = [
        r"\begin{longtable}{@{}llrrrrrrrr@{}}",
        r"\caption{Complete run-level results (all specifications and seeds).}\label{tab:all-runs}\\",
        r"\toprule",
        r"Model & Seed & Epoch & Max pct. & Hit@1\% & Hit@5\% & AUPRC & AUROC & $\Delta$ mean & $\Delta$ top5 \\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        r"Model & Seed & Epoch & Max pct. & Hit@1\% & Hit@5\% & AUPRC & AUROC & $\Delta$ mean & $\Delta$ top5 \\",
        r"\midrule",
        r"\endhead",
    ]
    for row in df.itertuples(index=False):
        lines.append(
            f"{MODEL_LABELS[row.dataset]} & {row.seed} & {int(row.best_epoch)} & {fmt(row.mean_case_max_percentile)} & "
            f"{fmt(row.case_hit_at_1pct)} & {fmt(row.case_hit_at_5pct)} & {fmt(row.matched_auprc_max_percentile)} & "
            f"{fmt(row.matched_auroc_max_percentile)} & {fmt(row.paired_difference_mean_svdd_percentile)} & "
            f"{fmt(row.paired_difference_share_top_5pct)} \\\\"
        )
    lines += [r"\bottomrule", r"\end{longtable}"]
    write_table(TABLES / "all_15_runs.tex", lines)


def build_feature_table() -> None:
    rows = [
        ("Log return", r"$\log(P_t^a/P_{t-1}^a)$", "Adjusted close"),
        ("Market-adjusted return", r"$r_{it}-\bar r_{m(i),t}$", "Same-market equal-weight return"),
        ("Intraday range", r"$\log(H_t/L_t)$", "Raw OHLC ratio"),
        ("Rolling volatility", r"$\log\{\mathrm{sd}(r_{t-20:t-1})+\epsilon\}$", "Lagged 20-day window"),
        ("Relative volume", r"$\log V_t-\log\mathrm{median}(V_{t-60:t-1})$", "Lagged 60-day benchmark"),
        ("Turnover shock", r"$\log A_t-\log\mathrm{median}(A_{t-60:t-1})$", "Trading amount $A_t$"),
        ("Amihud illiquidity", r"$\log\{\mathrm{mean}(|r|/A)_{t-20:t-1}+\epsilon\}$", "Lagged 20-day window"),
        ("Price--volume correlation", r"$\mathrm{corr}(r,A)_{t-20:t-1}$", "Lagged 20-day window"),
        ("MA deviation", r"$\log P_t^a-\log\mathrm{median}(P^a_{t-20:t-1})$", "Log deviation"),
        ("Close location", r"$(2C_t-H_t-L_t)/(H_t-L_t)$", "Set missing if range is zero"),
        ("Downside volatility", r"$\mathrm{sd}(r\,1\{r<0\})_{t-20:t-1}$", "Lagged 20-day window"),
        ("Tail-return shock", r"$r_t/\mathrm{sd}(r_{t-20:t-1})$", "Current return, lagged scale"),
    ]
    lines = [
        r"\begin{longtable}{@{}p{0.23\textwidth}p{0.43\textwidth}p{0.26\textwidth}@{}}",
        r"\caption{Encoder input variables.}\label{tab:variables}\\",
        r"\toprule",
        r"Feature & Construction & Information convention \\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule Feature & Construction & Information convention \\ \midrule",
        r"\endhead",
    ]
    lines += [" & ".join(row) + r" \\" for row in rows]
    lines += [r"\bottomrule", r"\end{longtable}"]
    write_table(TABLES / "feature_definitions.tex", lines)


def build_architecture_table() -> None:
    rows = [
        ("Window", "60 trading days $\times$ 12 features; training stride 5, scoring stride 1"),
        ("Encoder", "Six dilated residual TCN blocks; hidden width 64; dilations 1, 2, 4, 8, 16, 32"),
        ("Representation", "Last/mean/max pooling followed by a 128-dimensional normalized projection"),
        ("Self-supervision", "Two mask-only views; instance and multiscale temporal contrastive losses"),
        ("Domain sampling", "A share 50%; remaining batch share split equally across reference markets"),
        ("CORAL", "Covariance alignment to A-share representations; weight 5,000"),
        ("DANN", "Gradient reversal; maximum coefficient 0.10 after warm-up"),
        ("Optimization", "AdamW, learning rate 0.001, batch 128, cosine schedule, FP16, at most 30 epochs"),
        ("Model selection", "Unlabeled 2017 contrastive validation loss; patience 5"),
        ("One-class boundary", "Coordinate median and scaled MAD on 250,000 sampled 2011--2016 A-share embeddings"),
        ("Evaluation", "Same-day A-share percentile; enforcement labels never used for fitting or selection"),
    ]
    lines = [r"\begin{tabularx}{\textwidth}{@{}lX@{}}", r"\toprule", r"Component & Frozen specification \\", r"\midrule"]
    lines += [" & ".join(row) + r" \\" for row in rows]
    lines += [r"\bottomrule", r"\end{tabularx}"]
    write_table(TABLES / "architecture.tex", lines)


def build_flow_figure() -> None:
    fig, ax = plt.subplots(figsize=(11.2, 3.7))
    ax.set_axis_off()
    boxes = [
        (0.02, 0.53, 0.16, 0.30, "Unlabeled panels\nA / US / HK", "#DDEBF7"),
        (0.22, 0.53, 0.17, 0.30, "60-day windows\n12 price-volume features", "#EAF2F8"),
        (0.43, 0.53, 0.18, 0.30, "TS2Vec-style TCN\nCORAL or DANN", "#D5E8D4"),
        (0.65, 0.53, 0.16, 0.30, "A-share robust\none-class boundary", "#FFF2CC"),
        (0.85, 0.53, 0.13, 0.30, "Daily A-share\nrisk rank", "#FCE4D6"),
    ]
    for x, y, w, h, label, color in boxes:
        patch = plt.Rectangle((x, y), w, h, transform=ax.transAxes, facecolor=color,
                              edgecolor="#3B4A5A", linewidth=1.2)
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, label, transform=ax.transAxes, ha="center", va="center", fontsize=10)
    for x1, x2 in [(0.18, 0.22), (0.39, 0.43), (0.61, 0.65), (0.81, 0.85)]:
        ax.annotate("", xy=(x2, 0.68), xytext=(x1, 0.68), xycoords=ax.transAxes,
                    arrowprops=dict(arrowstyle="-|>", lw=1.3, color="#3B4A5A"))
    ax.text(0.50, 0.34, "Labels are withheld throughout fitting and model selection", transform=ax.transAxes,
            ha="center", va="center", fontsize=10, color="#8B1E1E", fontweight="semibold")
    ax.annotate("", xy=(0.91, 0.49), xytext=(0.91, 0.34), xycoords=ax.transAxes,
                arrowprops=dict(arrowstyle="-|>", lw=1.2, color="#3B4A5A"))
    ax.text(0.50, 0.13,
            "Held-out external validation: 67 scoreable CSRC episodes, 340 difficult controls, time placebos, and fixed WindowSHAP",
            transform=ax.transAxes, ha="center", va="center", fontsize=10.5,
            bbox=dict(boxstyle="round,pad=0.45", facecolor="#F4F4F4", edgecolor="#666666"))
    fig.savefig(FIGURES / "research_design.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / "research_design.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def copy_frozen_assets() -> None:
    mapping = {
        "fig03_matching_balance.pdf": "matching_balance.pdf",
        "fig04_matched_case_control_effects.pdf": "matched_case_control_effects.pdf",
        "fig07_seed_ensemble_incremental.pdf": "cross_market_incremental.pdf",
        "fig08_time_placebo.pdf": "time_placebo.pdf",
        "fig09_event_time_trajectories.pdf": "event_time_trajectories.pdf",
        "fig10_windowshap_heatmaps.pdf": "windowshap_heatmaps.pdf",
        "fig11_windowshap_similarity.pdf": "windowshap_similarity.pdf",
        "fig01_all_seed_overview.pdf": "all_seed_overview.pdf",
        "fig06_training_diagnostics.pdf": "training_diagnostics.pdf",
    }
    for source, target in mapping.items():
        shutil.copy2(ARCHIVE / "figures" / source, FIGURES / target)

    source_files = [
        ARCHIVE / "tables" / "paper_main_results.csv",
        ARCHIVE / "tables" / "paper_cross_market_incremental.csv",
        ARCHIVE / "tables" / "paper_time_placebo.csv",
        ARCHIVE / "tables" / "paper_windowshap_similarity.csv",
        ARCHIVE / "tables" / "all_15_runs.csv",
        ARCHIVE / "diagnostics" / "matching_balance.csv",
        ARCHIVE / "diagnostics" / "matching_summary.json",
    ]
    for source in source_files:
        shutil.copy2(source, SOURCE_DATA / source.name)


def write_manifest() -> None:
    rows = []
    for path in sorted(p for p in PAPER.rglob("*") if p.is_file() and "template_sources" not in p.parts and p.name != "asset_manifest.csv"):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        rows.append({"path": str(path.relative_to(PAPER)), "bytes": path.stat().st_size, "sha256": digest})
    pd.DataFrame(rows).to_csv(PAPER / "asset_manifest.csv", index=False)


def main() -> None:
    for directory in (TABLES, FIGURES, SOURCE_DATA):
        directory.mkdir(parents=True, exist_ok=True)
    build_sample_table()
    build_main_results()
    build_incremental_results()
    build_time_placebo()
    build_windowshap()
    build_matching_balance()
    build_all_runs()
    build_feature_table()
    build_architecture_table()
    build_flow_figure()
    copy_frozen_assets()
    write_manifest()
    print(json.dumps({"paper_root": str(PAPER), "tables": len(list(TABLES.glob('*.tex'))),
                      "figures": len(list(FIGURES.glob('*')))}, indent=2))


if __name__ == "__main__":
    main()
