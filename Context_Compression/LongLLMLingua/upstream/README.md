# LongLLMLingua 对比基线

> 本文件是原仓库说明。针对 `merged_contact_orders.csv` 的统一训练、推理和指标说明见 [README_adaptation.md](README_adaptation.md)。

从用户提供的 Microsoft LLMLingua 本地仓库提取，可脱离原仓库使用。
源提交：`e0e9d99beb94098bbd924aa53c2c112eac41c758`，文件摘要见 `provenance.json`。

## 提取范围

LongLLMLingua 并非独立类，而是共享 `PromptCompressor` 中的问题感知压缩路径。
为避免删改影响最终对比，`longllmlingua_baseline/_vendor/` 原样保留核心文件和工具依赖。
其中仍有 LLMLingua / LLMLingua-2 的共享或其他分支，但独立入口明确关闭 LLMLingua-2 / SecurityLingua，固定走 LongLLMLingua。
未复制训练、数据收集、图片和无关实验目录；保留上游 MIT 许可证。

- `longllmlingua_baseline/baseline.py`：专用 Python 入口及参数校验。
- `longllmlingua_baseline/_vendor/prompt_compressor.py`：上游原始算法。
- `run_baseline.py`：JSONL 批量压缩和实验记录。
- `examples/input.jsonl`：输入示例，answer 仅随 source 保存，不传给压缩器。

核心代码阅读顺序：`compress_prompt` → `control_context_budget` / `get_rank_results`（问题感知文档排序）→ `get_dynamic_compression_ratio`（预算分配）→ `iterative_compress_prompt` / `get_compressed_input`（迭代 token 压缩）。`get_ppl` / `get_condition_ppl` 提供评分。

## 安装和运行

在本目录运行，建议使用独立 Python 3.10/3.11 环境：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
python run_baseline.py --input examples/input.jsonl --output results/demo.jsonl --device cuda --rate 0.5
```

默认压缩模型沿用上游 `NousResearch/Llama-2-7b-hf`，首次运行需要可访问的模型权重和相应运行环境。可用 `--model /absolute/path/to/model` 指向本地权重。CPU 可用 `--device cpu`，7B 模型的内存和耗时开销较大。MPS 入口保留，但未验证。

按目标 token 数运行：

```bash
python run_baseline.py --input your_data.jsonl --output results/target500.jsonl --device cuda --target-token 500
pip freeze > results/environment.txt
```

`--rate` 与 `--target-token` 互斥。输出文件已存在时拒绝覆盖。
依赖范围是基于上游旧式 KV-cache 调用的兼容性约束，不是已验证的环境锁文件；首次真实运行验证后保存完整依赖版本。

## Python 调用

```python
from longllmlingua_baseline import LongLLMLingua

compressor = LongLLMLingua(model_name="NousResearch/Llama-2-7b-hf", device="cuda")
result = compressor.compress(
    context=["第一篇文档……", "第二篇文档……"],
    question="需要从文档中回答的具体问题是什么？",
    instruction="根据材料回答问题。",
    rate=0.5,
)
print(result["compressed_prompt"])
```

## 固定配置与实验口径

采用原仓库 LongLLMLingua 测试中的主要配置：

| 参数 | 值 |
|---|---|
| rank_method | longllmlingua |
| condition_in_question | after_condition |
| reorder_context | sort |
| dynamic_context_compression_ratio | 0.4 |
| condition_compare | True |
| context_budget | +100 |
| use_context_level_filter | True |
| use_token_level_filter | True |
| use_sentence_level_filter | False |
| concate_question | True |

注意原 DOCUMENT.md 示例写了 `ratio`，实际代码入参是 `rate`，本包按代码接口调用。
`rate=0.5` 表示期望保留比例，而非删除比例；`target_token` 是算法预算，并不保证最终输出严格等于这个 token 数。
上游返回的 origin_tokens / compressed_tokens 使用 GPT-3.5 对应的 tiktoken 编码，压缩内部预算使用压缩模型的 tokenizer，二者不能直接视为相同口径。
`compressed_prompt` 已包含 instruction 和 question，下游不要重复拼接。
上游 `saving` 是历史价格假设的字符串，不能作为当前成本指标。

输出每行保存原始记录、压缩结果、配置、依赖版本与耗时。耗时包含设备同步后的压缩过程，不含模型加载和下游回答；首条可能存在预热影响。
最终比较请固定输入文档划分、问题、压缩模型权重版本、下游回答模型及实际 token 预算，并用下游模型 tokenizer 重新统计实际输入量。
这个包完成压缩基线接入；尚不包含你们任务的下游生成或质量评测。LongLLMLingua 必须提供问题，弱信号发现任务需要事先统一问题表述，避免各方法输入条件不一致。

## 本次验证边界

已检查 Python 语法、CLI 帮助、包装入口参数透传和输入校验，并逐文件核验核心源码 SHA-256 与原仓库一致。
未下载模型权重、未运行真实模型推理，也未验证论文指标。此目录的独立封装不代表已完成实验复现。
