#!/usr/bin/env python3
"""从冻结结果生成中文论文图（PDF与300 dpi PNG）。"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "results/final_20260819"
OUT = ROOT / "paper/chinese_elsevier_submission/figures"

ORDER = ["a_only", "a_us_coral", "a_hk_coral", "a_hk_us_coral", "a_hk_us_dann"]
LABEL = {
    "a_only": "仅A股",
    "a_us_coral": "A股+美股\nCORAL",
    "a_hk_coral": "A股+港股\nCORAL",
    "a_hk_us_coral": "A股+港股+美股\nCORAL",
    "a_hk_us_dann": "A股+港股+美股\nDANN",
}
COLORS = {
    "a_only": "#6B7280", "a_us_coral": "#1F5A94", "a_hk_coral": "#3A7D44",
    "a_hk_us_coral": "#0F766E", "a_hk_us_dann": "#6D5A9E",
}
BLUE, RED, DARK, GRID = "#1F5A94", "#B5483A", "#28323C", "#E5E7EB"


def setup() -> None:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/System/Library/Fonts/Supplemental/Songti.ttc",
    ]
    font_path = next((Path(x) for x in candidates if Path(x).exists()), None)
    if font_path:
        name = font_manager.FontProperties(fname=font_path).get_name()
        plt.rcParams["font.family"] = name
    plt.rcParams.update({
        "axes.unicode_minus": False, "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": DARK, "axes.labelcolor": DARK, "text.color": DARK,
        "xtick.color": DARK, "ytick.color": DARK, "font.size": 10,
    })


def save(fig: plt.Figure, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def research_design() -> None:
    fig, ax = plt.subplots(figsize=(11.5, 4.2))
    ax.set_axis_off()
    boxes = [
        (0.02, 0.56, 0.16, 0.27, "未标记行情\nA股 / 美股 / 港股", "#DDEBF7"),
        (0.22, 0.56, 0.17, 0.27, "60日窗口\n12个共同量价特征", "#EAF2F8"),
        (0.43, 0.56, 0.18, 0.27, "TS2Vec式TCN\nCORAL或DANN", "#D5E8D4"),
        (0.65, 0.56, 0.16, 0.27, "仅用A股拟合\n稳健单类边界", "#FFF2CC"),
        (0.85, 0.56, 0.13, 0.27, "A股每日\n异常分位", "#FCE4D6"),
    ]
    for x, y, w, h, label, color in boxes:
        ax.add_patch(plt.Rectangle((x, y), w, h, transform=ax.transAxes,
                                   facecolor=color, edgecolor=DARK, linewidth=1.2))
        ax.text(x + w / 2, y + h / 2, label, transform=ax.transAxes,
                ha="center", va="center", fontsize=10)
    for x1, x2 in [(0.18, 0.22), (0.39, 0.43), (0.61, 0.65), (0.81, 0.85)]:
        ax.annotate("", xy=(x2, 0.695), xytext=(x1, 0.695), xycoords=ax.transAxes,
                    arrowprops=dict(arrowstyle="-|>", lw=1.3, color=DARK))
    ax.text(0.50, 0.39, "处罚标签不参与特征、训练、对齐、早停和异常边界",
            transform=ax.transAxes, ha="center", va="center", color="#8B1E1E",
            fontsize=10.5, fontweight="semibold")
    ax.annotate("", xy=(0.91, 0.53), xytext=(0.91, 0.39), xycoords=ax.transAxes,
                arrowprops=dict(arrowstyle="-|>", lw=1.2, color=DARK))
    ax.text(0.50, 0.13, "外部验证：67条可评分监管事件 + 340个困难对照 + 同股票时间安慰剂 + 固定WindowSHAP",
            transform=ax.transAxes, ha="center", va="center", fontsize=10.5,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#F4F4F4", edgecolor="#666666"))
    save(fig, "research_design")


def matching_balance() -> None:
    d = pd.read_csv(ARCHIVE / "diagnostics/matching_balance.csv", encoding="utf-8-sig")
    p = d.pivot(index="feature", columns="sample", values="smd")
    labels = {
        "cumulative_market_adjusted_return": "累计市场调整收益",
        "realized_return_volatility": "已实现波动率",
        "mean_turnover_shock": "平均成交额冲击",
        "mean_relative_volume": "平均相对成交量",
        "mean_intraday_range": "平均日内振幅",
        "log_median_circulating_market_value": "流通市值对数",
        "log_median_listing_age": "上市年龄对数",
        "mean_log_amihud": "Amihud非流动性对数",
        "extreme_return_share": "极端收益日占比",
        "mean_absolute_ma_deviation": "绝对均线偏离",
    }
    p = p.reindex(p.matched_controls.abs().sort_values().index)
    y = np.arange(len(p))
    fig, ax = plt.subplots(figsize=(8.8, 6.0))
    ax.axvspan(-0.1, 0.1, color="#B7D7A8", alpha=0.28)
    ax.axvline(0, color=DARK, lw=1)
    ax.scatter(p.unmatched_candidate_pool, y, color="#9CA3AF", s=46, label="匹配前候选池", zorder=3)
    ax.scatter(p.matched_controls, y, color=BLUE, s=54, label="匹配后困难对照", zorder=4)
    ax.set_yticks(y, [labels[x] for x in p.index])
    ax.set_xlabel("标准化均值差（案例−对照）")
    ax.set_title("困难对照匹配前后的协变量平衡", fontweight="bold")
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.legend(loc="lower right", frameon=False)
    save(fig, "matching_balance")


def incremental() -> None:
    d = pd.read_csv(ARCHIVE / "tables/paper_cross_market_incremental.csv", encoding="utf-8-sig")
    key = {"A + HK (CORAL)": "a_hk_coral", "A + HK + US (CORAL)": "a_hk_us_coral",
           "A + HK + US (DANN)": "a_hk_us_dann", "A + US (CORAL)": "a_us_coral"}
    d["target"] = d.model.map(key)
    targets = ["a_hk_us_dann", "a_hk_us_coral", "a_hk_coral", "a_us_coral"]
    metrics = [("mean_svdd_percentile", "事件期间平均异常分位"),
               ("share_top_5pct", "Top 5%异常交易日占比")]
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), sharey=True)
    y = np.arange(len(targets))
    for ax, (metric, title) in zip(axes, metrics):
        b = d[d.metric == metric].set_index("target").reindex(targets)
        est = b.incremental_estimate.to_numpy(float)
        ax.errorbar(est, y, xerr=np.vstack([est-b.ci_low.to_numpy(float), b.ci_high.to_numpy(float)-est]),
                    fmt="o", color=BLUE, ms=8, capsize=4, lw=1.5)
        ax.axvline(0, color=DARK, lw=1)
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("相对于仅A股案例—对照差的增量")
        ax.grid(axis="x", color=GRID, lw=0.8)
        for i, row in enumerate(b.itertuples()):
            ax.text(row.ci_high + 0.004, i, f"p={row.permutation_p_one_sided:.3f}", va="center", fontsize=9)
    axes[0].set_yticks(y, [LABEL[x].replace("\n", " ") for x in targets])
    fig.suptitle("三随机种子集成的跨市场增量检验", fontsize=14, fontweight="bold", y=1.02)
    save(fig, "incremental")


def time_placebo() -> None:
    d = pd.read_csv(ARCHIVE / "tables/paper_time_placebo.csv", encoding="utf-8-sig")
    model = {"A only": "a_only", "A + US (CORAL)": "a_us_coral"}
    d["dataset"] = d.model.map(model)
    datasets = ["a_only", "a_us_coral"]
    metrics = [("max_svdd_percentile", "期间最大异常分位"),
               ("mean_svdd_percentile", "期间平均异常分位")]
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.3), sharey=True)
    y = np.arange(2)
    for ax, (metric, title) in zip(axes, metrics):
        b = d[d.metric == metric].set_index("dataset").reindex(datasets)
        for i, dataset in enumerate(datasets):
            r = b.loc[dataset]
            ax.errorbar(r.actual_minus_placebo, i,
                        xerr=[[r.actual_minus_placebo-r.ci_low], [r.ci_high-r.actual_minus_placebo]],
                        fmt="o", color=COLORS[dataset], ms=8, capsize=4, lw=1.5)
            ax.text(r.ci_high + 0.004, i, f"p={r.permutation_p_one_sided:.3f}", va="center", fontsize=9)
        ax.axvline(0, color=DARK, lw=1)
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("真实期间−同股票随机伪期间")
        ax.grid(axis="x", color=GRID, lw=0.8)
    axes[0].set_yticks(y, ["仅A股", "A股+美股 CORAL"])
    fig.suptitle("同股票、同长度时间安慰剂检验", fontsize=14, fontweight="bold", y=1.02)
    save(fig, "time_placebo")


def event_time() -> None:
    d = pd.read_csv(ARCHIVE / "tables/event_time_trajectories.csv", encoding="utf-8-sig")
    datasets, anchors = ["a_only", "a_us_coral"], ["start", "end"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.8), sharex=True, sharey=True)
    for i, anchor in enumerate(anchors):
        for j, dataset in enumerate(datasets):
            ax = axes[i, j]
            for sample, color, label in [("case", RED, "监管事件"), ("control", BLUE, "困难对照")]:
                b = d[(d.anchor_type == anchor) & (d.dataset == dataset) & (d["sample"] == sample)].sort_values("event_time")
                x, m = b.event_time.to_numpy(), b.mean_percentile.to_numpy()
                ax.plot(x, m, color=color, lw=2, label=label)
                ax.fill_between(x, b.ci_low.to_numpy(), b.ci_high.to_numpy(), color=color, alpha=0.13, lw=0)
            ax.axvline(0, color=DARK, ls="--", lw=1)
            ax.grid(color=GRID, lw=0.8)
            anchor_label = "操纵开始日" if anchor == "start" else "操纵结束日"
            ax.set_title(f"{LABEL[dataset].replace(chr(10), ' ')}：以{anchor_label}对齐", fontweight="bold", fontsize=11)
            if i == 1: ax.set_xlabel("距监管记载日期的交易日")
            if j == 0: ax.set_ylabel("同日异常分位均值")
    axes[0, 1].legend(frameon=False, loc="upper left")
    fig.suptitle("事件时间异常分位轨迹（三随机种子集成）", fontsize=14, fontweight="bold", y=1.01)
    save(fig, "event_time")


def windowshap() -> None:
    d = pd.read_parquet(ARCHIVE / "data/fixed_windowshap_attributions.parquet")
    datasets = ["a_only", "a_us_coral"]
    times = d.sort_values("time_index").time_bin.drop_duplicates().tolist()
    features = d.sort_values("feature_index").feature_group.drop_duplicates().tolist()
    feature_cn = {"return_trend": "收益趋势", "volatility_pressure": "波动压力",
                  "trading_activity": "交易活跃度", "liquidity": "流动性",
                  "price_volume_relation": "价量关系"}
    mats, seq_max, diff_max = {}, 0.0, 0.0
    for dataset in datasets:
        pairs = {}
        for sample in ["case", "control"]:
            m = (d[(d.dataset == dataset) & (d["sample"] == sample)]
                 .pivot_table(index="feature_group", columns="time_bin", values="absolute_contribution_share", aggfunc="mean")
                 .reindex(index=features, columns=times).to_numpy(float))
            mats[(dataset, sample)] = pairs[sample] = m
            seq_max = max(seq_max, np.nanmax(m))
        mats[(dataset, "difference")] = pairs["case"] - pairs["control"]
        diff_max = max(diff_max, np.nanmax(np.abs(mats[(dataset, "difference")])))
    fig, axes = plt.subplots(2, 3, figsize=(14.5, 7.2), constrained_layout=True)
    seq_img = diff_img = None
    for i, dataset in enumerate(datasets):
        for j, sample in enumerate(["case", "control", "difference"]):
            ax, m = axes[i, j], mats[(dataset, sample)]
            if sample == "difference":
                diff_img = ax.imshow(m, cmap="RdBu_r", norm=TwoSlopeNorm(vmin=-diff_max, vcenter=0, vmax=diff_max), aspect="auto")
            else:
                seq_img = ax.imshow(m, cmap="Blues", vmin=0, vmax=seq_max, aspect="auto")
            for y in range(m.shape[0]):
                for x in range(m.shape[1]):
                    ax.text(x, y, f"{100*m[y,x]:.1f}", ha="center", va="center", fontsize=8,
                            color="white" if sample != "difference" and m[y,x] > 0.6*seq_max else DARK)
            ax.set_xticks(range(len(times)), [x.replace(":", "至") for x in times], rotation=30, ha="right")
            if j == 0:
                ax.set_yticks(range(len(features)), [feature_cn[x] for x in features])
                ax.set_ylabel(LABEL[dataset].replace("\n", " "), fontweight="bold")
            else:
                ax.set_yticks(range(len(features)), [])
            if i == 0:
                ax.set_title({"case": "监管事件", "control": "困难对照", "difference": "事件−对照"}[sample], fontweight="bold")
    fig.colorbar(seq_img, ax=axes[:, :2], shrink=0.78, pad=0.015, label="平均绝对贡献占比")
    fig.colorbar(diff_img, ax=axes[:, 2], shrink=0.78, pad=0.015, label="贡献占比差")
    fig.suptitle("峰值风险日前60个交易日的固定WindowSHAP结构", fontsize=14, fontweight="bold")
    save(fig, "windowshap")


def windowshap_similarity() -> None:
    d = pd.read_csv(ARCHIVE / "tables/paper_windowshap_similarity.csv", encoding="utf-8-sig")
    key = {"A only": "a_only", "A + US (CORAL)": "a_us_coral", "A + HK (CORAL)": "a_hk_coral",
           "A + HK + US (CORAL)": "a_hk_us_coral"}
    d["dataset"] = d.model.map(key)
    datasets = ["a_only", "a_us_coral", "a_hk_coral", "a_hk_us_coral"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    y = np.arange(len(datasets))
    for ax, basis, title in zip(axes, ["absolute_share", "signed_phi"], ["绝对贡献结构", "有符号贡献结构"]):
        b = d[d.basis == basis].set_index("dataset").reindex(datasets)
        for i, dataset in enumerate(datasets):
            r = b.loc[dataset]
            ax.errorbar(r.paired_difference, i,
                        xerr=[[r.paired_difference-r.ci_low], [r.ci_high-r.paired_difference]],
                        fmt="o", color=COLORS[dataset], ms=7, capsize=4, lw=1.4)
            ax.text(r.ci_high + 0.007, i, f"p={r.permutation_p_one_sided:.3f}", va="center", fontsize=9)
        ax.axvline(0, color=DARK, lw=1)
        ax.grid(axis="x", color=GRID, lw=0.8)
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("案例−对照的模板相似度差")
    axes[0].set_yticks(y, [LABEL[x].replace("\n", " ") for x in datasets])
    fig.suptitle("WindowSHAP留一案例模板相似度", fontsize=14, fontweight="bold", y=1.02)
    save(fig, "windowshap_similarity")


def all_seeds() -> None:
    d = pd.read_csv(ARCHIVE / "tables/all_15_runs.csv", encoding="utf-8-sig")
    metrics = [("mean_case_max_percentile", "案例期间最大分位均值"),
               ("case_hit_at_1pct", "事件Hit@1%"),
               ("case_hit_at_5pct", "事件Hit@5%")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    rng = np.random.default_rng(20260819)
    for ax, (metric, title) in zip(axes, metrics):
        for i, dataset in enumerate(ORDER):
            values = d[d.dataset == dataset][metric].to_numpy(float)
            ax.scatter(i+rng.uniform(-0.07, 0.07, len(values)), values, s=45, color=COLORS[dataset],
                       edgecolor="white", lw=0.7, zorder=3)
            ax.errorbar(i, values.mean(), yerr=values.std(ddof=1), fmt="_", color=DARK, ms=18, capsize=4)
        ax.set_xticks(range(len(ORDER)), [LABEL[x] for x in ORDER], fontsize=8)
        ax.set_title(title, fontweight="bold")
        ax.set_ylim(0, 1)
        ax.grid(axis="y", color=GRID, lw=0.8)
    axes[0].set_ylabel("比例 / 分位")
    fig.suptitle("全部15次运行：单一种子及均值±标准差", fontsize=14, fontweight="bold", y=1.02)
    save(fig, "all_seeds")


def main() -> None:
    setup()
    research_design()
    matching_balance()
    incremental()
    time_placebo()
    event_time()
    windowshap()
    windowshap_similarity()
    all_seeds()
    print(f"中文图形已生成：{OUT}")


if __name__ == "__main__":
    main()
