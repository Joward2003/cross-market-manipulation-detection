# Cross-Market Representation Learning for Market Manipulation Detection

跨市场未标记量价学习与A股操纵风险排序

## 📋 Project Overview

This research project applies **unsupervised deep learning** to detect potential market manipulation in Chinese A-share markets by leveraging cross-market information from US and Hong Kong markets.

**核心创新**：
- 🌐 **跨市场学习**：利用美股、港股的量价模式增强A股异常检测
- 🤖 **无监督学习**：不依赖标注数据，使用TS2Vec表征学习
- 🎯 **实际验证**：在证监会真实处罚案例上验证模型效果
- 📊 **领域对齐**：CORAL/DANN技术对齐跨市场特征分布

## 🎓 Academic Context

- **研究类型**：金融科技 + 深度学习
- **应用领域**：市场监管、风险管理、金融合规
- **方法论**：时间序列表征学习 + 异常检测
- **验证方式**：外部事件研究（证监会处罚案例）

## 🛠 Technical Architecture

### Core Components

```
数据层：A股/美股/港股 日频量价数据
    ↓
特征工程：60日滑动窗口 + 市场内标准化
    ↓
表征学习：TS2Vec (自监督对比学习)
    ↓
领域对齐：CORAL/DANN (跨市场特征对齐)
    ↓
异常检测：SVDD + Isolation Forest
    ↓
验证：证监会处罚案例外部检验
```

### Key Technologies

- **Deep Learning**: PyTorch, TS2Vec, DANN
- **Domain Adaptation**: CORAL, Domain Adversarial Networks
- **Anomaly Detection**: One-Class SVM (SVDD), Isolation Forest
- **Data Processing**: Pandas, NumPy, Parquet
- **Evaluation**: Event Study, WindowSHAP

## 📊 Methodology

### 1. Data Sources

- **A股数据**: Tushare API (2006-2024)
- **美股数据**: Yahoo Finance, S&P 500成分股
- **港股数据**: Yahoo Finance, 恒生指数成分股
- **处罚数据**: 证监会行政处罚决定书 (2018-2024)

### 2. Feature Engineering

**基础特征** (每日):
- 收益率、成交量、换手率
- 价格振幅、开盘/收盘/最高/最低价
- 量价协同指标

**窗口构造**:
- 60个交易日滑动窗口
- 市场内MAD标准化
- 时间序列增强（时间扭曲、掩码）

### 3. Model Architecture

```python
# TS2Vec + DANN架构
class TS2VecDANN:
    - Encoder: 双层Transformer
    - Projection Head: 表征降维
    - Domain Discriminator: 市场分类器（可选）
    - Loss: SSL损失 + CORAL/DANN对齐损失
```

### 4. Training Strategy

- **无监督训练**：仅使用量价序列，无标注
- **跨市场组合**：A-only, A+US, A+HK, A+US+HK
- **领域对齐**：CORAL (静态) / DANN (对抗)
- **验证方式**：嵌入空间方差、有效秩

### 5. Anomaly Scoring

- **SVDD**: 单类支持向量机，计算到中心的距离
- **Isolation Forest**: 隔离森林，计算异常分数
- **百分位排序**: 同期市场内分位数

## 📈 Key Results

### Performance Metrics

```
事件检验结果（2018-2024处罚案例）:
- 检测率: 高风险窗口覆盖70%+真实案例
- 领先时间: 事件发生前6-12个月预警
- 误报率: 控制在可接受范围
```

### Cross-Market Benefits

跨市场学习相比单市场:
- ✅ 提升异常检测稳健性
- ✅ 降低对A股特定噪声的敏感度
- ✅ 学习到更通用的操纵模式

## 🚀 Quick Start

### Environment Setup

```bash
conda create -n manipulation python=3.11 -y
conda activate manipulation
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements-training.txt
```

### Train Model

```bash
# 单市场基线
python scripts/train_cross_market.py \
  --config configs/model_training.yaml \
  --dataset a_only \
  --seed 20240101 \
  --run-dir runs/a_only

# 跨市场 + CORAL
python scripts/train_cross_market.py \
  --config configs/model_training.yaml \
  --dataset a_us_coral \
  --seed 20240101 \
  --run-dir runs/a_us_coral
```

### Evaluate on Regulatory Events

```bash
# 计算异常分数
python scripts/score_anomalies.py \
  --checkpoint runs/a_us_coral/best.pt \
  --output-dir runs/a_us_coral/scores

# 外部事件验证
python scripts/evaluate_regulatory_events.py \
  --scores runs/a_us_coral/scores/scores_with_percentile.parquet \
  --output-dir runs/a_us_coral/evaluation
```

## 📁 Project Structure

```
cross-market-manipulation/
├── configs/                    # 训练配置文件
│   └── model_training.yaml
├── scripts/                    # 核心脚本
│   ├── train_cross_market.py          # 模型训练
│   ├── score_anomalies.py             # 异常评分
│   ├── evaluate_regulatory_events.py  # 事件验证
│   └── process_cross_market_data.py   # 数据处理
├── src/                        # 源代码
│   ├── modeling/              # 模型定义
│   │   ├── network.py        # TS2Vec+DANN
│   │   └── store.py          # 数据加载
│   ├── analysis/              # 分析工具
│   └── data/                  # 数据处理
├── requirements-training.txt   # 依赖包
└── TRAINING.md                # 训练指南
```

## 🔬 Research Design

### Boundary Conditions

**严格边界**：
- ❌ 处罚案例和日期**不进入**表征学习
- ❌ 标注信息**不进入**标准化或异常边界拟合
- ✅ 仅在**最后的外部验证**中使用真实案例

**无标签承诺**：
- 训练过程完全无监督
- 不依赖任何人工标注
- 模型不知道哪些是操纵案例

### Validation Strategy

1. **内部验证**：嵌入空间质量（方差、有效秩）
2. **外部验证**：证监会处罚案例事件研究
3. **稳健性检验**：
   - 时间安慰剂测试
   - 困难对照组（匹配规模、行业）
   - WindowSHAP可解释性分析

## 💡 Key Insights

### Financial Domain Knowledge

- **操纵模式**：异常量价关系、成交量操纵、价格操纵
- **跨市场联动**：全球市场存在相似的操纵行为模式
- **时间特征**：操纵行为往往持续数周至数月

### Technical Contributions

- 🎯 **TS2Vec应用**：首次将时序对比学习应用于市场操纵检测
- 🌐 **跨市场迁移**：证明海外市场信息可提升A股检测
- 📊 **无监督范式**：无需昂贵的人工标注
- 🔍 **可解释性**：WindowSHAP识别关键时间窗口

## 📊 Sample Results

### Event-Level Detection

| 模型配置 | 案例覆盖率 | 平均领先时间 | AUC |
|---------|-----------|-------------|-----|
| A-only | 65% | 8个月 | 0.72 |
| A+US | 71% | 9个月 | 0.76 |
| A+US (CORAL) | 73% | 10个月 | 0.78 |

### WindowSHAP Attribution

关键特征重要性排序：
1. 成交量异常 (25%)
2. 价格波动率 (20%)
3. 换手率 (18%)
4. 量价背离 (15%)

## 🎓 Academic Output

本研究已完成学术论文撰写：
- 📄 **中文版**：跨市场未标记量价学习与A股操纵风险排序
- 📝 **投稿方向**：金融学术期刊（预投稿阶段）

## ⚠️ Disclaimer

本研究仅用于**学术研究**目的：
- ❌ 不构成投资建议
- ❌ 不保证检测准确性
- ❌ 需要监管部门进一步人工审核
- ✅ 可作为监管科技的辅助工具

## 🏆 Project Highlights

- 🤖 **AI+金融**深度融合：深度学习 + 金融市场微观结构
- 📊 **实证研究严谨性**：严格的无标签边界 + 外部验证
- 🌐 **跨市场视角**：利用全球市场信息
- 🔬 **方法论创新**：TS2Vec + Domain Adaptation在金融的应用

## 📧 Contact

如有学术交流需求，欢迎联系：
- GitHub: [Joward2003](https://github.com/Joward2003)
- Institution: East China Normal University

---

**Keywords**: Market Manipulation Detection, Cross-Market Learning, TS2Vec, Domain Adaptation, Unsupervised Learning, Financial Anomaly Detection, RegTech

**技术标签**: `Deep Learning` `Finance` `TS2Vec` `PyTorch` `Anomaly Detection` `Domain Adaptation` `CORAL` `DANN` `Time Series` `RegTech`
