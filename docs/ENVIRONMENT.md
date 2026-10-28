# 环境

当前服务器直接使用 `/root/zzl/baseline/.venv-table/bin/python`，不需要修改系统 Python、CUDA、SSH 或 VS Code 设置。

已验证环境：Python 3.10、PyTorch 2.6.0+cu124、Transformers 4.44.2；精确安装版本见 `environment_versions.json`。

在新机器且网络可用时：

```bash
cd /root/zzl/baseline
python3.10 -m venv .venv-table
.venv-table/bin/python -m pip install -r requirements-table.txt
```

PyTorch CUDA 构建应按机器驱动选择；本机现有构建为 cu124。不要随意升级 Transformers：保留的 LongLLMLingua 上游代码依赖当前4.x接口。项目为 XLM-R 提供了本地 SDPA 适配，不改第三方 site-packages。

官方权重放在 `models/`，模型目录各有 `download_manifest.json` 与官方 SHA256。`--classifier-model BAAI/bge-m3` 自动解析到本地完整权重。离线运行可设置 `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`。

测试：

```bash
.venv-table/bin/python -m unittest discover -s tests -p 'test_*.py' -v
# 以下显式使用GPU和真实权重，执行合成4096输入验证，不产生正式实验指标：
CUDA_VISIBLE_DEVICES=0 .venv-table/bin/python tests/validate_gpu.py --sequence-length 4096
```
