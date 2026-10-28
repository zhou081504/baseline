# 通话风险分类 Baselines

## 当前数据源（2026-10-27）

统一读取根目录 `merged_最终版.xlsx` 的第一个工作表。每个 `c` 是一个完整样本，同一 `c` 下的 `s` 按数字顺序拼接（s2 在 s10 前），段间添加换行，段内正文原样保留。合并单元格中的 `id`、`label` 按所属样本处理；只使用样本级 `label`，完全忽略 `turn_label`，不将其输入模型或用作监督。样本顺序保持 c 首次出现的顺序，不重新编号。

先汇总 c，再按样本级 label 分层划分 70%/15%/15%，seed=42。同一 c 的所有 s 始终在同一集合。缺失或冲突的样本标签、重复 s、缺失正文会报错。七个方法使用同一个读取与划分模块。

`--data` 与兼容参数 `--csv` 均支持此 Excel；旧的每行一个样本 CSV 仍可显式指定。新数据须重新 prepare/train；本次使用 `artifacts/prepared_final` 和 `artifacts/model_final`，旧结果保留。六个普通方法在 infer 时读取新模型对应的 prepared；CogLTX 在 train/infer 时直接读取 Excel。

七个方法的最终分类器统一为 **BGE-M3 + 二分类头**，默认分类输入上限 **4096 token**；原文截取上限由独立的 **max_length** 参数控制。任务是读完整通电话后预测风险，原始数据为根目录 `merged_最终版.xlsx`。

| 分类 | 方法与操作说明 |
|---|---|
| Selection_Retrieval | [Recent Truncation](Selection_Retrieval/Recent_Truncation/README.md) |
| Selection_Retrieval | [Dense-TopK (BGE-M3)](Selection_Retrieval/Dense_TopK_BGE_M3/README.md) |
| Selection_Retrieval | [CogLTX (adapted)](Selection_Retrieval/CogLTX/README.md) |
| Selection_Retrieval | [SAGE (adapted)](Selection_Retrieval/SAGE/README.md) |
| Selection_Retrieval | [CARROT (adapted)](Selection_Retrieval/CARROT/README.md) |
| Context_Compression | [LongLLMLingua](Context_Compression/LongLLMLingua/README.md) |
| Context_Compression | [RECOMP-Extractive](Context_Compression/RECOMP_Extractive/README.md) |

## 稳定训练默认开启

2026-10-23 起，七个方法的 train.py 均默认使用分类头预热、分组低学习率和有效批量 64，保存模型前检查恒定预测。本次新数据默认模型目录为 artifacts/model_final，旧结果保留。旧学习率/梯度累积命令需迁移，详见 [稳定训练与检查](STABLE_TRAINING.md)。

## 从哪里开始

进入需要运行的模型目录，按该目录 README 执行环境、准备、训练、推理命令。普通方法为 `prepare.py → train.py → infer.py`；CogLTX 使用联合训练入口；SAGE 正式运行前需要训练分段器。

```text
baseline/
├── Selection_Retrieval/      # Recent、Dense、CogLTX、SAGE、CARROT
├── Context_Compression/     # LongLLMLingua、RECOMP
├── table_baselines/         # 共用分类训练、长上下文注意力、预算和入口
├── common/                  # CSV/Excel 读取、划分和评估
├── models/                  # 官方权重；每个方法独立保存微调结果
├── data/                    # 公开辅助数据
├── runs/                    # 历史实验结果；历史记录，不是当前新实验结果
├── docs/                    # 环境、协议、来源、验证和清理记录
├── tests/                   # 自动检查；GPU验证须显式执行
├── merged_最终版.xlsx      # 当前原始数据
├── requirements-table.txt
├── run_table.py             # 可选统一命令入口
└── .venv-table/             # 已安装的独立环境
```

- [环境配置](docs/ENVIRONMENT.md)
- [实验协议与两个长度参数](docs/PROTOCOL.md)
- [验证记录](docs/VALIDATION.md)
- [清理记录](docs/CLEANUP.md)
- [论文与源码来源](docs/SOURCES.md)

旧的中文 BERT 下游分类入口已移出当前项目。CogLTX 内部选块器与 SAGE 分段器仍保留 BERT，这是用户确认保留的方法内部组件。BGE-M3 属于 XLM-R 架构，不能把依赖库或官方模型中所有含 “bert” 的名字当作废文件删除。

更换源数据文件、tokenizer 或最终分类器后必须重新准备和训练。原文max_length与classifier_max_length分别记录在配置中；旧版本max_length表示分类预算，不能沿用旧prepared缓存或旧CogLTX checkpoint。旧实验结果保留作历史记录，不与新配置混用。TurnRank 已暂停、EviCompose 不在本次范围。

## 两个长度参数

```text
一个c下所有s的完整merged_text
  → 按max_length保留末尾（默认4096正文token，可设32768等）
  → 各方法选择/压缩；CogLTX为分块、judge打分、选块并按原顺序拼接
  → 最多classifier_max_length（默认4096，含特殊token）
  → BGE-M3二分类器
```

例如普通方法在prepare时设置`--max_length 32768 --classifier_max_length 4096`；CogLTX在train时设置相同参数。训练、推理读取保存的两个预算。所有方法每个c对应一个样本，不按s拆成独立训练样本。
