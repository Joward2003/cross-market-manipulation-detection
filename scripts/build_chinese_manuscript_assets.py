#!/usr/bin/env python3
"""从冻结结果档案生成中文论文表格，并复制矢量图。"""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "results" / "final_20260819"
PAPER = ROOT / "paper" / "chinese_elsevier_submission"
TABLES = PAPER / "tables"
FIGURES = PAPER / "figures"
SOURCE = PAPER / "source_data"

MODEL = {
    "a_only": "仅A股",
    "a_us_coral": "A股+美股（CORAL）",
    "a_hk_coral": "A股+港股（CORAL）",
    "a_hk_us_coral": "A股+港股+美股（CORAL）",
    "a_hk_us_dann": "A股+港股+美股（DANN）",
}


def fmt(x: object, n: int = 3) -> str:
    return "--" if pd.isna(x) else f"{float(x):.{n}f}"


def pval(x: object) -> str:
    if pd.isna(x):
        return "--"
    p = float(x)
    base = "<0.001" if p < 0.001 else f"{p:.3f}"
    stars = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""
    return f"${base}^{{{stars}}}$" if stars else f"${base}$"


def pplain(x: object) -> str:
    if pd.isna(x):
        return "--"
    p = float(x)
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def coef(x: object, p: object, n: int = 3) -> str:
    """Put significance stars on the tested estimate, leaving p as a plain value."""
    if pd.isna(x):
        return "--"
    if pd.isna(p):
        return fmt(x, n)
    pv = float(p)
    stars = "***" if pv < 0.01 else "**" if pv < 0.05 else "*" if pv < 0.10 else ""
    return f"{float(x):.{n}f}$^{{{stars}}}$" if stars else f"{float(x):.{n}f}"


def write(name: str, lines: list[str]) -> None:
    (TABLES / name).write_text("\n".join(lines) + "\n", encoding="utf-8")


def sample_table() -> None:
    rows = [
        ("A股", "剔除北交所后的全部可用A股", "2011--2025", "12,564,184", "目标域训练"),
        ("美股", "当前标普500成分股的可得历史行情", "2011--2025", "1,773,029", "未标记参考域"),
        ("港股", "按上年流动性动态筛选普通股", "2011--2025", "2,616,376", "未标记参考域"),
        ("证监会严格事件表", "股票--操纵期间", "文书公开于2012--2025", "226", "仅作外部标签"),
        ("样本外事件", "操纵开始日不早于2018年", "2018--2025评分期", "68（67条可评分）", "主评价样本"),
        ("困难对照", "每事件5个同期未处罚异动窗口", "对应事件期间", "340（302只股票）", "匹配比较"),
    ]
    lines = [r"\begin{tabularx}{\textwidth}{@{}p{0.13\textwidth}X p{0.15\textwidth}p{0.14\textwidth}p{0.13\textwidth}@{}}", r"\toprule",
             r"样本 & 构造口径 & 时期 & 观测数 & 研究用途 \\ \midrule"]
    lines += [" & ".join(r) + r" \\" for r in rows]
    lines += [r"\bottomrule", r"\end{tabularx}"]
    write("sample_construction.tex", lines)


def main_results() -> None:
    d = pd.read_csv(ARCHIVE / "tables/paper_main_results.csv", encoding="utf-8-sig")
    english_to_cn = {
        "A only": "仅A股", "A + US (CORAL)": "A股+美股（CORAL）",
        "A + HK (CORAL)": "A股+港股（CORAL）",
        "A + HK + US (CORAL)": "A股+港股+美股（CORAL）",
        "A + HK + US (DANN)": "A股+港股+美股（DANN）",
    }
    lines = [r"\begin{tabular}{@{}lrrrrrr@{}}", r"\toprule",
             r"模型 & 案例最大分位 & Hit@5\% & AUPRC & AUROC & 案例--对照均值差 & Top 5\%占比差 \\ \midrule"]
    for r in d.itertuples(index=False):
        lines.append(f"{english_to_cn[r[0]]} & {fmt(r[2])} & {fmt(r[4])} & {fmt(r[5])} & {fmt(r[6])} & {fmt(r[8])} & {fmt(r[9])}" + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("main_results.tex", lines)


def incremental() -> None:
    d = pd.read_csv(ARCHIVE / "tables/paper_cross_market_incremental.csv", encoding="utf-8-sig")
    model = {
        "A + US (CORAL)": "A股+美股（CORAL）", "A + HK (CORAL)": "A股+港股（CORAL）",
        "A + HK + US (CORAL)": "三市场（CORAL）", "A + HK + US (DANN)": "三市场（DANN）",
    }
    metric = {"max_svdd_percentile": "期间最大分位", "mean_svdd_percentile": "期间平均分位",
              "share_top_1pct": "Top 1\\%交易日占比", "share_top_5pct": "Top 5\\%交易日占比"}
    lines = [r"\begin{table}[H]", r"\centering",
             r"\caption{相对于仅A股模型的三随机种子集成增量。星号标在增量估计量上：$^{*}p<0.10$，$^{**}p<0.05$，$^{***}p<0.01$。}\label{tab:incremental-full}",
             r"\scriptsize", r"\setlength{\tabcolsep}{3pt}", r"\resizebox{\textwidth}{!}{%",
             r"\begin{tabular}{@{}llrrrrr@{}}",
             r"\toprule 模型 & 指标 & 增量 & 95\%下限 & 95\%上限 & 单侧$p$ & FDR $q$ \\ \midrule"]
    for r in d.itertuples(index=False):
        lines.append(f"{model[r.model]} & {metric[r.metric]} & {coef(r.incremental_estimate, r.permutation_p_one_sided)} & {fmt(r.ci_low)} & {fmt(r.ci_high)} & {pplain(r.permutation_p_one_sided)} & {fmt(r.fdr_q_one_sided_within_metric)}" + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table}"]
    write("incremental_results.tex", lines)


def time_placebo() -> None:
    d = pd.read_csv(ARCHIVE / "tables/paper_time_placebo.csv", encoding="utf-8-sig")
    model = {"A only": "仅A股", "A + US (CORAL)": "A股+美股（CORAL）"}
    metric = {"max_svdd_percentile": "期间最大分位", "mean_svdd_percentile": "期间平均分位",
              "share_top_1pct": "Top 1\\%交易日占比", "share_top_5pct": "Top 5\\%交易日占比"}
    lines = [r"\begin{tabular}{@{}llrrrr@{}}", r"\toprule",
             r"模型 & 指标 & 真实期$-$安慰剂期 & 95\%下限 & 95\%上限 & 单侧$p$ \\ \midrule"]
    for r in d.itertuples(index=False):
        lines.append(f"{model[r.model]} & {metric[r.metric]} & {coef(r.actual_minus_placebo, r.permutation_p_one_sided)} & {fmt(r.ci_low)} & {fmt(r.ci_high)} & {pplain(r.permutation_p_one_sided)}" + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("time_placebo.tex", lines)


def windowshap() -> None:
    d = pd.read_csv(ARCHIVE / "tables/paper_windowshap_similarity.csv", encoding="utf-8-sig")
    model = {"A only": "仅A股", "A + US (CORAL)": "A股+美股（CORAL）",
             "A + HK (CORAL)": "A股+港股（CORAL）", "A + HK + US (CORAL)": "三市场（CORAL）"}
    basis = {"signed_phi": "有符号贡献", "absolute_share": "绝对贡献占比"}
    lines = [r"\begin{tabular}{@{}llrrrrrr@{}}", r"\toprule",
             r"模型 & 归因口径 & 案例相似度 & 对照相似度 & 差异 & 95\%下限 & 95\%上限 & 单侧$p$ \\ \midrule"]
    for r in d.itertuples(index=False):
        lines.append(f"{model[r.model]} & {basis[r.basis]} & {fmt(r.mean_case_similarity)} & {fmt(r.mean_control_similarity)} & {coef(r.paired_difference, r.permutation_p_one_sided)} & {fmt(r.ci_low)} & {fmt(r.ci_high)} & {pplain(r.permutation_p_one_sided)}" + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("windowshap_similarity.tex", lines)


def matching() -> None:
    d = pd.read_csv(ARCHIVE / "diagnostics/matching_balance.csv", encoding="utf-8-sig")
    wide = d.pivot(index="feature", columns="sample", values="smd")
    labels = {
        "cumulative_market_adjusted_return": "累计市场调整收益",
        "realized_return_volatility": "已实现收益波动率", "mean_turnover_shock": "平均成交额冲击",
        "mean_relative_volume": "平均相对成交量", "mean_intraday_range": "平均日内振幅",
        "log_median_circulating_market_value": "流通市值对数", "log_median_listing_age": "上市年龄对数",
        "mean_log_amihud": "平均Amihud非流动性对数", "extreme_return_share": "极端收益日占比",
        "mean_absolute_ma_deviation": "平均绝对均线偏离",
    }
    lines = [r"\begin{tabular}{@{}lrr@{}}", r"\toprule", r"协变量 & 匹配前SMD & 匹配后SMD \\ \midrule"]
    for key, label in labels.items():
        lines.append(f"{label} & {fmt(wide.loc[key, 'unmatched_candidate_pool'])} & {fmt(wide.loc[key, 'matched_controls'])}" + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("matching_balance.tex", lines)


def all_runs() -> None:
    d = pd.read_csv(ARCHIVE / "tables/all_15_runs.csv", encoding="utf-8-sig")
    lines = [r"\begin{table}[H]", r"\centering",
             r"\caption{全部15次运行结果。}\label{tab:all-runs}",
             r"\scriptsize", r"\setlength{\tabcolsep}{2.5pt}", r"\resizebox{\textwidth}{!}{%",
             r"\begin{tabular}{@{}llrrrrrrrr@{}}",
             r"\toprule 模型 & 种子 & 最优轮次 & 最大分位 & Hit@1\% & Hit@5\% & AUPRC & AUROC & 均值差 & Top5差 \\ \midrule"]
    for r in d.itertuples(index=False):
        lines.append(f"{MODEL[r.dataset]} & {r.seed} & {int(r.best_epoch)} & {fmt(r.mean_case_max_percentile)} & {fmt(r.case_hit_at_1pct)} & {fmt(r.case_hit_at_5pct)} & {fmt(r.matched_auprc_max_percentile)} & {fmt(r.matched_auroc_max_percentile)} & {fmt(r.paired_difference_mean_svdd_percentile)} & {fmt(r.paired_difference_share_top_5pct)}" + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table}"]
    write("all_15_runs.tex", lines)


def fixed_tables() -> None:
    features = [
        ("复权对数收益率", r"$\log(P_t^a/P_{t-1}^a)$", "当日可见"),
        ("市场调整收益率", r"$r_{it}-\bar r_{m(i),t}$", "同市场等权基准"),
        ("日内振幅", r"$\log(H_t/L_t)$", "原始OHLC比例"),
        ("滚动波动率", r"$\log\{\operatorname{sd}(r_{t-20:t-1})+\epsilon\}$", "滞后20日"),
        ("相对成交量", r"$\log V_t-\log\operatorname{median}(V_{t-60:t-1})$", "滞后60日基准"),
        ("成交额冲击", r"$\log A_t-\log\operatorname{median}(A_{t-60:t-1})$", "滞后60日基准"),
        ("Amihud非流动性", r"$\log\{\operatorname{mean}(|r|/A)_{t-20:t-1}+\epsilon\}$", "滞后20日"),
        ("价量相关性", r"$\operatorname{corr}(r,A)_{t-20:t-1}$", "滞后20日"),
        ("均线偏离", r"$\log P_t^a-\log\operatorname{median}(P^a_{t-20:t-1})$", "对数口径"),
        ("收盘位置", r"$(2C_t-H_t-L_t)/(H_t-L_t)$", "区间为零则缺失"),
        ("下行波动率", r"$\operatorname{sd}(r\mathbf{1}\{r<0\})_{t-20:t-1}$", "滞后20日"),
        ("尾部收益冲击", r"$r_t/\operatorname{sd}(r_{t-20:t-1})$", "当日收益、滞后尺度"),
    ]
    lines = [r"\begin{longtable}{@{}p{0.20\textwidth}p{0.50\textwidth}p{0.22\textwidth}@{}}",
             r"\caption{编码器输入变量。}\label{tab:variables}\\", r"\toprule 特征 & 构造 & 信息口径 \\ \midrule \endfirsthead",
             r"\toprule 特征 & 构造 & 信息口径 \\ \midrule \endhead"]
    lines += [" & ".join(r) + r" \\" for r in features]
    lines += [r"\bottomrule", r"\end{longtable}"]
    write("feature_definitions.tex", lines)

    arch = [
        ("窗口", r"60个交易日$\times$12个特征；训练步长5，评分步长1"),
        ("编码器", "6个膨胀残差TCN块；隐藏维64；膨胀率1、2、4、8、16、32"),
        ("表征", "末时点、均值、最大值池化后投影至128维并归一化"),
        ("自监督目标", "两组仅遮蔽增强；实例对比与多尺度时间对比"),
        ("市场抽样", r"A股占50\%；其余参考市场均分剩余批次"),
        ("CORAL", "将参考域表征协方差对齐A股；损失权重5,000"),
        ("DANN", "梯度反转；预热后最大系数0.10"),
        ("优化", "AdamW，学习率0.001，批次128，余弦退火，FP16，最多30轮"),
        ("模型选择", "仅依据2017年无标签对比学习验证损失，耐心值5"),
        ("单类边界", "25万个2011--2016年A股表征的坐标中位数与MAD尺度"),
        ("评价", "同日A股横截面分位；监管标签不参与拟合或选择"),
    ]
    lines = [r"\begin{tabularx}{\textwidth}{@{}lX@{}}", r"\toprule", r"模块 & 冻结设定 \\ \midrule"]
    lines += [" & ".join(r) + r" \\" for r in arch]
    lines += [r"\bottomrule", r"\end{tabularx}"]
    write("architecture.tex", lines)


def copy_assets() -> None:
    # 中文图由 plot_chinese_manuscript_figures.py 直接从冻结结果重绘；这里不以英文图覆盖。
    for src in [ARCHIVE / "tables/paper_main_results.csv", ARCHIVE / "tables/paper_cross_market_incremental.csv",
                ARCHIVE / "tables/paper_time_placebo.csv", ARCHIVE / "tables/paper_windowshap_similarity.csv",
                ARCHIVE / "tables/all_15_runs.csv", ARCHIVE / "diagnostics/matching_balance.csv"]:
        shutil.copy2(src, SOURCE / src.name)


def manifest() -> None:
    rows = []
    excluded = {"build", "tmp", "template_sources", "notes"}
    for f in sorted(
        x for x in PAPER.rglob("*")
        if x.is_file()
        and x.name != "asset_manifest.csv"
        and not (x.parent == PAPER and x.suffix.lower() == ".pdf")
        and not excluded.intersection(x.relative_to(PAPER).parts)
    ):
        rows.append({"path": str(f.relative_to(PAPER)), "bytes": f.stat().st_size,
                     "sha256": hashlib.sha256(f.read_bytes()).hexdigest()})
    pd.DataFrame(rows).to_csv(PAPER / "asset_manifest.csv", index=False)


def main() -> None:
    for d in (TABLES, FIGURES, SOURCE): d.mkdir(parents=True, exist_ok=True)
    sample_table(); main_results(); incremental(); time_placebo(); windowshap(); matching(); all_runs(); fixed_tables()
    copy_assets(); manifest()
    print(f"中文论文资产已生成：{PAPER}")


if __name__ == "__main__":
    main()
