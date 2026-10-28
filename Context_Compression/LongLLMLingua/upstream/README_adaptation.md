# LongLLMLingua Baseline 独立适配

## 模型

LongLLMLingua 原本没有风险分类头。本适配使用 BERT 编码，并加入 token-level importance gating，模拟问题感知压缩后接二分类头：

```text
merged_text -> BERT -> token importance gating -> 二分类头
```

## 文件说明

- `model.py`：token importance gating + BERT 二分类头。
- `train.py`：训练入口。
- `infer.py`：推理入口。

## Linux / conda

```bash
conda create -n longllmlingua-baseline python=3.10 -y
conda activate longllmlingua-baseline
cd /path/to/baseline\ code/SOTA\ method/longllmlingua
pip install -r ../../requirements-adaptation-gpu.txt
```

## 训练

```bash
python train.py \
  --csv /path/to/merged_contact_orders.csv \
  --output-dir outputs/longllmlingua \
  --model-name /data/models/chinese-bert-wwm-ext \
  --max-length 512 \
  --batch-size 16 \
  --epochs 4
```

## 推理

```bash
python infer.py \
  --csv /path/to/merged_contact_orders.csv \
  --output-dir outputs/longllmlingua \
  --split test
```

## 输出

`metrics.json` 和 `predictions.csv` 包含 AUPRC、ROC-AUC、Recall、F1。
