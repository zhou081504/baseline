# RECOMP Baseline 独立适配

## 模型

RECOMP 采用抽取式证据选择。BERT 对 token 编码，句子打分层评估每个句子的证据相关性，加权聚合后接二分类头：

```text
merged_text -> BERT token 编码 -> sentence selection -> 二分类头
```

## 文件说明

- `model.py`：sentence selection + BERT 二分类头。
- `dataset.py`：文本 tokenization 和 sentence id 生成。
- `train.py`：训练入口。
- `infer.py`：推理入口。

## Linux / conda

```bash
conda create -n recomp-baseline python=3.10 -y
conda activate recomp-baseline
cd /path/to/baseline\ code/SOTA\ method/recomp
pip install -r ../../requirements-adaptation-gpu.txt
```

## 训练

```bash
python train.py \
  --csv /path/to/merged_contact_orders.csv \
  --output-dir outputs/recomp \
  --model-name /data/models/chinese-bert-wwm-ext \
  --max-length 512 \
  --batch-size 16 \
  --epochs 4
```

## 推理

```bash
python infer.py \
  --csv /path/to/merged_contact_orders.csv \
  --output-dir outputs/recomp \
  --split test
```

## 输出

`metrics.json` 和 `predictions.csv` 包含 AUPRC、Recall、F1 和逐样本概率。
