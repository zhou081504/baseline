# 修复后的训练与准备流程

2026-10-22：原 CARROT、CogLTX 检查点均出现分类分数几乎恒定的现象。
所有方法的默认训练入口现在采用原始 BGE-M3 权重、分类头预热、编码器/分类头不同学习率、
64 条样本的有效批量、10% 学习率预热和验证集 AUPRC 选择检查点。
分类概率及损失在 FP32 中计算，编码器使用 BF16。
旧模型、旧 prepared 和原始预训练权重均保留。

## 当前数据的快速准备

当前 CSV 有 23,934 条记录、413 条风险标签。旧 prepared 中有 332 条风险标签。
将当前 CSV 的标签恢复成旧 prepared 的标签后，按原始格式重建文件，SHA256
与旧 prepare_config 中的源 CSV SHA256 完全一致。这证明仅 81 个标签变化，
文本、ID、顺序未变，因此可以复用 CARROT 与标签无关的检索和选句结果。

以下命令重新生成标签、分层划分、文本校验值和来源记录；一旦文本变化，命令会拒绝复用。
输出必须是新目录，避免混用数据版本。以后文本变化应重新执行原 prepare 流程。

```bash
.venv-table/bin/python -m table_baselines.refresh_prepared \
  --csv merged_contact.csv \
  --prepared Selection_Retrieval/CARROT/artifacts/prepared \
  --output-dir Selection_Retrieval/CARROT/artifacts/prepared_20261022_stable
```

## CARROT

```bash
.venv-table/bin/python -m table_baselines.stable_training \
  --prepared Selection_Retrieval/CARROT/artifacts/prepared_20261022_stable \
  --output-dir Selection_Retrieval/CARROT/artifacts/model_20261022_stable \
  --classifier-model /root/zzl/baseline/models/BAAI--bge-m3 \
  --epochs 2 --device cuda
```

先缓存原始编码器的 CLS 特征并训练分类头 8 轮，再全参数微调 2 轮。
编码器学习率 2e-6，分类头微调学习率 2e-5；自然采样，正例权重上限 20。
只在验证集选检查点和 F1 阈值；测试集仅在训练结束后评估。
原有 infer 入口可读取新模型，建议 `--batch-size 32`。

## CogLTX

```bash
.venv-table/bin/python -m table_baselines.stable_cogltx \
  --csv merged_contact.csv \
  --cache-dir Selection_Retrieval/CogLTX/artifacts/prepared_20261022_stable \
  --output-dir Selection_Retrieval/CogLTX/artifacts/model_20261022_stable \
  --classifier-model /root/zzl/baseline/models/BAAI--bge-m3 \
  --encoder /data/models/chinese-bert-wwm-ext \
  --epochs 2 --device cuda
```

缓存两套 tokenizer 的分块结果，用 CSV、tokenizer、长度预算及划分校验缓存。
默认原文预算/分类器预算均为 4096，内部块长 256。每个块独立打分但批量计算，
所有块都能放入预算且未设置 top-k 时直接保留全部块，结果与排序后装入完全相同。
删块监督仍来自训练样本的“删除块后无权重交叉熵损失增量”；不使用验证/测试标签训练 judge。
reasoner 和 judge 均参与联合训练，两者按同一验证轮次成对保存。
只对实际选中且进行删块评估的块训练 judge；按每条记录的有效块 token 数归一化。

## 产物与检查

- `run_config.json`：数据来源、参数、固定阈值和选中的验证轮次。
- `warmup_history.json`、`finetune_history.json` / `joint_history.json`：完整训练指标。
- `test_metrics.json`、`test_predictions.csv`：测试结果。
- `completion.json`：完成标记与分数离散程度检查。
- `runs/stable_20261022/`：任务日志、状态和修改前源码备份。

2026-10-23 起，Recent、Dense、SAGE、CARROT、LongLLMLingua、RECOMP 的 train.py 以及统一 train 命令直接使用稳定训练；CogLTX 的 train.py 和 train-cogltx 使用稳定联合训练。上面的 20261022 目录是已有成果，不能直接覆盖重跑。新运行默认输出 artifacts/model_stable，也可显式指定新的空目录。
比较旧/新模型时须注意：本次 CSV 标签与数据划分已更新，旧测试分数不能直接当成同一测试集上的差异。

```bash
.venv-table/bin/python -m unittest tests.test_default_stable tests.test_stable_training tests.test_long_context tests.test_pipeline -v
```

## 默认入口、参数迁移和失败保护

进入方法目录后运行 `../../.venv-table/bin/python train.py --device cuda`。普通方法须先有当前 CSV 对应的 prepared；省略 --prepared 会优先选择匹配当前 CSV 的已完成准备目录。CogLTX 自动建立可校验的分块缓存。SAGE 的分段器训练仍为独立辅助任务。

- 默认头预热 8 轮（head-lr=1e-3），分类微调 2 轮，encoder-lr=2e-6，finetune-head-lr=2e-5，有效批量 64。
- batch-size 是物理微批量上限（默认 16）；train-token-budget 默认 2048，按长度分组节省显存；effective-batch-size 控制一次更新覆盖的样本数，最后不足 64 的批次按实际数量归一化。
- 旧 --lr 和 --gradient-accumulation-steps 明确报错，避免照旧命令意外恢复高学习率、小有效批量。推理 amp=auto 优先沿用训练精度；CPU 用 FP32。
- 训练前校验 CSV 哈希、prepared 标签、划分、tokenizer 和长度预算；完整训练须覆盖全部 CSV 行。CSV 变化后不能静默使用旧 prepared。
- 预热和每轮验证均检查概率。如果概率完全一致，或标准差低于 1e-6，不允许其覆盖健康检查点；后续轮次坍塌就保留先前最佳结果。没有健康检查点则 failure.json 并退出。
- 最终重载最佳模型，重新验证，再固定验证集阈值评估测试集。测试输出异常只报告失败，不根据测试集更换检查点或重新调参。completion.json 仅在检查通过后写入。
- history 保存每轮指标；collapse_guard.json 记录中止及保留检查点；run_config.json 记录实际完成轮数。不同样本得到不同分数只是最低检查，不代表召回率或泛化能力一定好。
- infer.py 默认优先读取已完成的稳定模型；显式 --model-dir 始终优先。已有崩溃权重必须重新训练；修改代码不会修改旧权重。
