# 目录整理记录

执行前备份：`/root/zzl/baseline_backups/20261020_061730_before_8192`。

两个分类目录已去掉名称前的 `***`，使用真实目录，移除了旧兼容链接。七个启用模型保留独立入口和中文README；共享代码、测试和说明分别放在各自目录。

移出当前项目并保留可恢复备份：旧base/SOTA目录、暂停的TurnRank、旧BERT分类与风险训练入口、512-token历史运行、旧prepared/模型artifacts、临时检查权重、重复说明、旧依赖文件和无关论文模板。源码原状另存source_before.tgz。详细原路径见备份migration.json的archived字段。

删除的已验证冗余下载分片约 10.41 GiB。已完成的四套官方权重先重新校验SHA256，models/内文件保留。自动生成的源码字节码缓存和.DS_Store已清除。

原始CSV内容哈希在整理前后相同。公开辅助数据、官方源码许可证、现有虚拟环境和内部必要的BERT保留；没有修改SSH、网络或VS Code配置。

旧Recent train_8192_v2任务按用户要求停止；没有启动新全量训练。本次只进行了CPU回归和合成输入的GPU验证。

## 后续4K与独立原文预算更新

修改前源码备份：`/root/zzl/baseline_backups/20261020_065111_before_4k_inputs`。
七个分类器默认预算改为4096，max_length改为独立原文正文截取上限（默认8192，超出保留末尾）；新增classifier_max_length。CogLTX按用户要求改为逐块judge打分、按分数选块后恢复原顺序拼接，移除memory replay选择流程。旧GPU检查脚本重命名为tests/validate_gpu.py，默认验证4096。本轮没有改写原始数据、预训练权重或虚拟环境，没有启动全量训练。
