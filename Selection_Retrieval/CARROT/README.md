# CARROT (adapted)

## 当前数据源（2026-10-27）

统一读取根目录 `merged_最终版.xlsx` 的第一个工作表。每个 `c` 是一个完整样本，同一 `c` 下的 `s` 按数字顺序拼接（s2 在 s10 前），段间添加换行，段内正文原样保留。合并单元格中的 `id`、`label` 按所属样本处理；只使用样本级 `label`，完全忽略 `turn_label`，不将其输入模型或用作监督。样本顺序保持 c 首次出现的顺序，不重新编号。

先汇总 c，再按样本级 label 分层划分 70%/15%/15%，seed=42。同一 c 的所有 s 始终在同一集合。缺失或冲突的样本标签、重复 s、缺失正文会报错。七个方法使用同一个读取与划分模块。

`--data` 与兼容参数 `--csv` 均支持此 Excel；旧的每行一个样本 CSV 仍可显式指定。新数据须重新 prepare/train；本次使用 `artifacts/prepared_final` 和 `artifacts/model_final`，旧结果保留。六个普通方法在 infer 时读取新模型对应的 prepared；CogLTX 在 train/infer 时直接读取 Excel。

按现有适配保留检索、重排序与搜索选择；prepare 阶段为每通电话挑选文本，train 阶段训练独立分类器。

## 分类器与两个长度参数

最终分类器统一为 **BAAI/bge-m3（XLM-R 架构）+ 二分类头**，tokenizer 统一为 BGE-M3，默认分类预算 `classifier_max_length=4096`，包含特殊 token。分类头从随机初始化开始，用通话训练集标签学习；BGE-M3 官方权重本身不提供信访风险预测。

BGE-M3 与 BGE reranker v2-m3。 所有官方模型保存在根目录 `models/`，无需重复下载。分类器微调结果写到本方法 `artifacts/model_final/`，不会修改官方权重。


| 参数 | 含义 | 默认值 |
|---|---|---|
| `--max_length` | 一条完整样本进入选块/压缩前的正文token上限；超出保留末尾 | 4096 |
| `--classifier_max_length` | 选块/压缩/拼接后送入最终分类器的总token上限，含特殊token | 4096 |

两种计数都使用最终BGE-M3 tokenizer，保证七个方法可比较。原文`max_length`可以设为32768等更大值，不与分类器的位置上限比较。分类预算可以自定义，但不能超过BGE-M3真实支持的8192；默认4K就是4096。

Excel 中一个 c 的全部 s 汇总为一个样本；分段不会增加样本数，turn_label 不参与模型输入或训练。

例如 `--max_length 32768 --classifier_max_length 4096`：先保留样本末尾32768个正文token，再选取/压缩成最多4096个分类token。Recent方法最终继续保留末尾；其它方法执行各自的选择/压缩。

## 环境

```bash
ROOT=/root/zzl/baseline
cd "$ROOT/Selection_Retrieval/CARROT"
PY="$ROOT/.venv-table/bin/python"
$PY --version
```

现有公共环境：Python 3.10、PyTorch 2.6.0+cu124、Transformers 4.44.2。无需重新安装。新机器的安装步骤见 [环境说明](../../docs/ENVIRONMENT.md)，本目录 `requirements.txt` 引用公共依赖。

## 准备、训练与推理

从上面的模型目录执行，使用当前 `merged_最终版.xlsx`：

```bash
$PY prepare.py --data "$ROOT/merged_最终版.xlsx" --max_length 4096 --classifier_max_length 4096 \
  --classifier-model "$ROOT/models/BAAI--bge-m3" --device cuda \
  --output-dir artifacts/prepared_final
$PY train.py --prepared artifacts/prepared_final --output-dir artifacts/model_final \
  --classifier-model "$ROOT/models/BAAI--bge-m3" --epochs 2 \
  --device cuda --batch-size 16 --effective-batch-size 64 \
  --gradient-checkpointing --amp bf16
$PY infer.py --model-dir artifacts/model_final \
  --split test --device cuda --batch-size 32 --amp bf16
```

输出目录必须不存在或为空。再次实验时使用新的目录，并保持 train/infer 路径对应。普通方法在 prepare 时设置长度，训练和推理读取保存的两个长度；CogLTX 在 train 时设置。输入与分类预算均为上限，短文本不需要填满。

## 配置与结果

- 最终分类器：`--classifier-model`。修改它或源数据文件 后必须重新准备数据、重新训练；不能复用旧 512-token 中文 BERT 缓存或分类权重。
- 参数同时支持下划线和连字符：`--max_length` / `--max-length`，`--classifier_max_length` / `--classifier-max-length`。设置后会记录到配置文件。
- 设备：`--device cuda` 使用 GPU；默认 CPU。`--amp auto` 在 GPU 自动选择 bf16/fp16，在 CPU 使用 float32。训练和推理都使用相同的注意力优化。
- `--gradient-checkpointing` 默认开启；`--no-gradient-checkpointing` 可关闭。训练物理批量默认上限 16，按 token 预算自动缩小；显存不足可降低 batch-size，有效批量仍默认 64。
- 数据划分：分层 70%/15%/15%，seed=42。所有对比实验使用相同数据源、划分、预算与分类器。
- 检索方法：`--query` 默认“从这通电话里找出与信访风险有关的话。”；`--top-k` 控制选取数量，`--retrieval-max-length` 控制检索编码器长度，均不同于最终分类预算。
- 新的完整数据/权重不会因改变输出目录名称而自动生效；查看 `run_config.json` 的实际配置。

主要结果在 `artifacts/model_final/`：`run_config.json`、`test_metrics.json`、`test_predictions.csv`。`infer.py` 对已保存划分做离线推理和评估，支持 `--split train/val/test/all`，不是无标签电话在线服务。最佳模型根据验证集 AUPRC 保存，分类阈值只由验证集决定，测试标签不参与选句。

`--smoke-test` 仅用于流程检查（每划分每类 2 条），结果不能当正式指标。验证范围见 [验证记录](../../docs/VALIDATION.md)。

## 代码阅读

入口：`train.py` / `infer.py`，普通方法另有 `prepare.py`。本方法关键逻辑：selector.py、model.py。
共享分类训练：`../../table_baselines/runtime.py`；长上下文和 SDPA：`../../table_baselines/long_context.py`；预算：`../../table_baselines/budget.py`。
`upstream/` 如存在，保存论文原始源码与许可证，作为来源参考；本任务使用外层入口。来源见 [SOURCES](../../docs/SOURCES.md)。

## 稳定训练默认开启（2026-10-23）

所有 train.py 已使用分类头预热 8 轮、编码器学习率 2e-6、分类头微调学习率 2e-5、有效批量 64 和 10% 学习率预热。分类微调默认 2 轮。`--encoder-lr`、`--finetune-head-lr` 可分别设置；旧 `--lr` 和 `--gradient-accumulation-steps` 会给出迁移提示并退出。

保存前检查验证概率：恒定或几乎恒定的轮次不能覆盖已有健康检查点；没有健康检查点则报错，不写完成标记。最终重载并检查验证/测试输出，测试异常只报告，不据此调参。该检查不能保证模型泛化效果，仍需看验证与测试指标。

普通方法省略 `--prepared` 时优先选择匹配当前数据源 的 prepared；训练前检查源文件哈希、标签与划分。省略 `--model-dir` 时优先选择已完成的稳定模型；指定目录则使用指定模型。旧权重不会自动变好，需用新目录重新训练。完整说明见 [稳定训练](../../STABLE_TRAINING.md)。
