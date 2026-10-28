# CogLTX Baseline 独立适配

## 模型

保留 CogLTX 的长文本关键块选择思想。BERT 对 token 编码，block score 层估计每个块的证据相关性，加权聚合后接二分类头：

```text
merged_text -> BERT token 编码 -> block selection -> 二分类头
```

## 文件说明

- `model.py`：CogLTX 风格 block selector + BERT 二分类头。
- `dataset.py`：整段文本和 block id 生成。
- `train.py`：训练入口。
- `infer.py`：推理入口。

## Linux / conda

```bash
conda create -n cogltx-baseline python=3.10 -y
conda activate cogltx-baseline
cd /path/to/baseline\ code/SOTA\ method/CogLTX
pip install -r ../../requirements-adaptation-gpu.txt
```

## 训练

```bash
python train.py \
  --csv /root/zzl/baseline/merged_contact_orders.csv \
  --output-dir outputs/cogltx \
  --model-name /data/models/chinese-bert-wwm-ext \
  --max-length 512 \
  --block-size 32 \
  --batch-size 2 \
  --epochs 2
```

## 推理

```bash
python infer.py \
  --csv /path/to/merged_contact_orders.csv \
  --output-dir outputs/cogltx \
  --split test
```

## 输出

`metrics.json` 和 `predictions.csv` 包含 AUPRC、Recall、F1 和逐样本概率。
