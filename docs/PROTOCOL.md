# 完整样本输入与4K分类协议

## 样本与数据划分

CSV每行的完整`merged_text`作为一个二分类样本，保留`id`与`label`。若一行已经汇总同一人的完整对话，则以这份完整对话作为样本；本代码不跨行按人合并。标签“正常”为0，“信访风险件”为1。分块/压缩不会生成额外样本。
默认分层70%/15%/15%、seed=42。仅训练标签用于学习，验证集决定模型与分类阈值，测试集最终评估。

## 两个独立预算

- `max_length`：完整原文的正文token上限，默认4096。超出执行recent truncation，保留末尾原文；不与最终编码器的位置上限比较，可以设置32768等值。
- `classifier_max_length`：选择/压缩/拼接后送入最终分类器的总token上限，默认4096，包含首尾及块分隔特殊token。该参数受模型真实长度限制，BGE-M3支持最高8192。
- 两种预算都用最终BGE-M3 tokenizer计数。输入截断用fast tokenizer的字符offset恢复原文后缀，保留换行、说话人和Unicode字符，并重新核对token数，不通过decode破坏对话结构。
- 准备结果记录原始/保留token数、是否截断、两个预算、tokenizer指纹和CSV哈希。`length_schema=input_and_classifier_v1`标识新语义；旧缓存拒绝混用。
- 所有方法的选择/压缩器都只看到截断后的文本，不接收样本标签。

## 分类器与方法

七个最终分类器统一为BGE-M3加二分类头，默认4K（4096）输入；各方法从同一官方权重分别训练自己的分类头和编码器。官方权重本身不是现成的风险分类器。
Recent最后继续保留末尾；Dense、SAGE、CARROT、LongLLMLingua、RECOMP保留已有选择/压缩机制。
CogLTX按用户要求采用一次judge逐块打分，高分优先装入分类预算，再恢复原文顺序拼接。内部judge保留BERT，每块受自身512上限约束；最终BGE-M3默认4096，不被judge的512限制。两个tokenizer按同一原文块对齐，不共享token IDs。
CogLTX不再使用memory replay及四类分类缓冲区。judge保留训练集上的删块损失监督，未评估块不自动赋负标签；验证/测试仅使用预测分数。它是任务适配版本，不声称与原论文全部机制相同。
SAGE内部BERT分段器继续用公开百科训练；其辅助train_segmenter.py的max_length是句子编码长度，与通话入口max_length分开。正式分段器尚需训练。论文梯度公式歧义按用户确认保留。

## 运行与复现

普通方法在prepare设置两个预算，train/infer读取保存的配置；CogLTX在train设置，infer读取。
保留原SDPA、gradient checkpointing与bf16/fp16配置，降低长序列显存。检索/生成模型的内部tokenizer和长度参数独立，不替代最终分类预算。
旧max_length=8192只代表旧分类预算，不代表做过原文recent截断。请重新prepare与训练，不混用旧数据缓存和CogLTX权重。

来源：[BGE-M3官方模型卡](https://huggingface.co/BAAI/bge-m3)、[PyTorch SDPA](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)。
