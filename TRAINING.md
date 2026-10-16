# CUDA 训练说明

这套管线遵守一个边界：处罚案例和事件日期不进入表征学习、标准化或异常边界拟合；它们只在最后的外部事件检验中使用。

## 1. 在 Ubuntu / WSL 的 Linux 本地盘准备项目

请把整个项目（至少 `data/processed/unlabeled_feature_datasets/`、`outputs/`、`configs/`、`scripts/`、`src/`）放在 Linux 文件系统，例如 `~/projects/a-share-manipulation`。不要直接从 `/mnt/c/...` 训练，mmap 和数据加载会明显变慢。

```bash
cd ~/projects/a-share-manipulation
conda create -n manipulation python=3.11 -y
conda activate manipulation
python -m pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements-training.txt
python scripts/verify_cuda.py
```

最后一行必须显示 RTX 4060 和 `CUDA runtime`。训练脚本默认在没有 CUDA 时直接报错，避免意外使用 CPU。

## 2. 首次构建紧凑的序列存储

```bash
python scripts/build_sequence_store.py
```

它只读取现有 `base_feature_panel`，生成 `data/analysis/sequence_store/`：

- 日频特征 mmap；
- 股票连续交易段 offset；
- 合法 60 日窗口端点索引。

它不复制四份数据集，也不展开为 60 日大矩阵。若构建中断，删除这个**生成目录**后重跑即可；已有原始 Parquet 不会被修改。

## 3. 先跑一个小型冒烟试验

将 `configs/model_training_memory_efficient.yaml` 临时复制为 `configs/pilot.yaml`，只改：

```yaml
sampler:
  steps_per_epoch: 30
optimization:
  epochs: 1
```

然后运行：

```bash
python scripts/train_cross_market.py --config configs/pilot.yaml --dataset a_us_coral --seed 20260727 --run-dir runs/pilot_a_us_coral
python scripts/score_anomalies.py --checkpoint runs/pilot_a_us/best.pt --output-dir runs/pilot_a_us/scores
python scripts/evaluate_regulatory_events.py --scores runs/pilot_a_us/scores/scores_with_percentile.parquet --output-dir runs/pilot_a_us/event_evaluation
```

确认 `best.pt`、`scores_with_percentile.parquet` 和 `event_summary.json` 都生成后，再删掉 pilot 输出并开始正式运行。

## 4. 正式运行

```bash
tmux new -s manipulation
bash scripts/run_gpu_training.sh |& tee training.log
```

默认脚本顺序运行 A-only、三个直接混合、三个 DANN 组合 × 3 个随机种子，每个组合完成后立即评分并输出事件级结果。CORAL 先以一个随机种子做检查，再通过环境变量加入，例如：

```bash
EXPERIMENTS="a_us_coral a_hk_coral a_hk_us_coral" SEEDS="20260727" bash scripts/run_gpu_training.sh
```

断线后重新连接：

```bash
tmux attach -t manipulation
```

`--resume` 会从每个目录的 `last.pt` 恢复训练；完成后评分使用 `best.pt`。监测 GPU：

```bash
watch -n 2 nvidia-smi
```

## 输出与解释

每个 `runs/<dataset>/seed_<seed>/` 下会有：

- `metrics.jsonl`：每轮的无监督验证损失、嵌入方差和有效秩；
- `best.pt` / `last.pt`：模型和仅由训练期拟合的市场内 MAD 标准化参数；
- `scores/scores_with_percentile.parquet`：2018–2025 年每个合法 A 股窗口的 SVDD 和 Isolation Forest 同日分位数；
- `event_evaluation/event_level_scores.csv`：对 2018 年起严格监管事件的外部检验。

主结论应比较三次随机种子的事件级指标。困难对照组、WindowSHAP 与点时标普500是下一层的评估/稳健性模块，不会影响这里无标签训练的完整性。
