# 当前4K分类与独立输入预算验证（2026-10-20）

## 本轮检查

- 19项CPU回归检查全部通过（45.355秒）。覆盖原文预算大于编码器上限、七个方法先截断后选择、单行仍为单样本、Unicode与换行保留、judge高分选块与原顺序拼接、双模型训练/保存/推理一致性。
- 真实BGE-M3在4090上输入4096 token，bf16、SDPA、gradient checkpointing，完成两步前向/反向/AdamW更新。
- 同时驻留内部BERT及其优化器。峰值分配显存11.525 GiB，峰值保留12.713 GiB；保存重载后推理logits完全相同。
- 当前CSV有23934条样本、ID非空且唯一。用12条分层流程样本验证`max_length=32768`、`classifier_max_length=4096`，最大最终输入4096。该12条在32768上限下无需截断；强制recent截断由单独的短预算回归测试覆盖。
- 该GPU检查为合成输入容量验证，不是正式指标；本轮未启动七模型全量训练，也未重新执行所有官方选择器的全量预处理。

本轮记录：`validation/tests_4k.log`、`validation/gpu_4096.json`、`validation/current_csv_4k.json`、`validation/migration_4k.json`。

## 仍需训练

SAGE正式公开百科分段器仍需训练。各方法都需要按照新两个预算重新prepare/训练/评估。CogLTX选择流程已按用户要求简化为judge排序，不沿用旧memory replay检查点。旧prepared数据中max_length的含义不同，入口会明确拒绝混用。

## 历史记录

`validation/gpu_8192.json`及旧回归记录来自前一轮8192分类预算检查，保留作历史证据；当前默认分类预算为4096。原始数据未被改写。
