# 📄 Fine-Tuned BERT for Scientific Abstract Classification (ArXiv)

[![license](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](#)
[![python](https://img.shields.io/badge/python-3.8%2B-orange.svg)](#)
[![pytorch](https://img.shields.io/badge/pytorch-%3E%3D1.10-red.svg)](#)

> Fine-tune a Transformer model to classify ArXiv paper abstracts into subject categories (Computer Science, Mathematics, Physics, etc.). This repo is GitHub-ready and includes training, evaluation, a Colab notebook, and a SciBERT variant tuned for scientific text.

---

## 🚀 Quick Summary

This repository demonstrates a production-friendly pipeline to fine-tune a Transformer (default: **SciBERT** / `allenai/scibert_scivocab_uncased`) for **multi-class classification** on the ArXiv abstracts dataset. The project focuses on reproducibility, sensible defaults, and easy experimentation.

**Outcomes:**

* Strong baseline: TF-IDF + Logistic Regression
* Transformer baseline: SciBERT / `bert-base-uncased` fine-tuned for classification
* Metrics: Accuracy, Macro-F1, Weighted-F1, Confusion matrix

---

## ✅ Features

* Modular training script with CLI arguments
* Data preprocessing & label mapping utilities
* Model checkpointing, early stopping, and reproducible seeds
* Evaluation script that saves metrics & plots (confusion matrix)
* Optional use of `scibert` for scientific text
* Colab-ready notebook for quick experimentation
* Dockerfile for environment reproducibility
* Continuous Integration test (lightweight smoke test)

---

## 📁 Project Structure

```
├── README.md
├── LICENSE
├── requirements.txt
├── dataset/
│   ├── download_arxiv.py      # (optional) script to download & preprocess
│   └── processed/             # final TSV/JSONL splits
├── src/
│   ├── data.py                # Dataset class, tokenization, collators
│   ├── model.py               # Model wrapper (HF transformers)
│   ├── train.py               # Training entrypoint
│   ├── evaluate.py            # Evaluation entrypoint
│   ├── utils.py               # helpers: metrics, seed, IO
│   └── predict.py             # Inference script for single abstract
├── notebooks/
│   └── colab_experiment.ipynb # runnable Colab notebook
├── docker/
│   ├── Dockerfile
│   └── entrypoint.sh
├── .github/
│   └── workflows/ci.yml       # lightweight CI tests
└── saved_models/
    └── bert-arxiv/            # example checkpoint structure
```

---

## 🔧 Requirements

Prefer a conda environment (example `env.yml` included)

* Python 3.8+
* PyTorch (GPU recommended)
* `transformers` (Hugging Face)
* `datasets` (Hugging Face datasets)
* `scikit-learn`, `numpy`, `pandas`, `tqdm`, `matplotlib`

Example install:

```bash
python -m pip install -r requirements.txt
```

---

## 📥 Data (Download & Preprocess)

1. Download the [ArXiv dataset (Kaggle / Cornell)](#). Save as `dataset/raw/arxiv.csv`.
2. Run the preprocessing script to extract `abstract`, `primary_category`, and create splits:

```bash
python dataset/download_arxiv.py --input dataset/raw/arxiv.csv --out_dir dataset/processed --split_ratio 0.8 0.1 0.1
```

`download_arxiv.py` will:

* Normalize categories (e.g., map subcategories to top-level categories)
* Remove extremely short or empty abstracts
* Create `train.jsonl`, `val.jsonl`, `test.jsonl` with fields `{id, title, abstract, label, label_id}`

**Label mapping:** A `label_map.json` is saved with `{label_name: label_id}` to guarantee stable mapping between runs.

---

## 🏋️‍♀️ Training (example)

Default uses SciBERT. To train with SciBERT (recommended for scientific text):

```bash
python src/train.py \
  --train_file dataset/processed/train.jsonl \
  --val_file dataset/processed/val.jsonl \
  --model_name_or_path allenai/scibert_scivocab_uncased \
  --output_dir saved_models/bert-arxiv \
  --num_labels 10 \
  --max_length 256 \
  --batch_size 16 \
  --epochs 4 \
  --lr 2e-5 \
  --seed 42
```

Key behaviors implemented in the training script:

* Mixed-precision if supported (AMP)
* Gradient accumulation for large effective batch sizes
* Checkpointing on best validation Macro-F1
* Logging to `training.log` and saving a `training_config.json` alongside the model

---

## 📊 Evaluation

Run evaluation (generates metrics JSON and confusion matrix PNG):

```bash
python src/evaluate.py --model_dir saved_models/bert-arxiv --test_file dataset/processed/test.jsonl --out_dir results/
```

Outputs:

* `results/metrics.json` (accuracy, f1\_macro, f1\_weighted, precision, recall)
* `results/confusion_matrix.png`
* `results/predictions.csv` (id, true\_label, pred\_label, pred\_probs)

---

## 🧾 Example CLI Options (train.py)

* `--model_name_or_path` (str) — HF model name (default `allenai/scibert_scivocab_uncased`)
* `--max_length` (int) — tokenizer max length (default 256)
* `--batch_size` (int)
* `--epochs` (int)
* `--lr` (float)
* `--weight_decay` (float)
* `--warmup_ratio` (float)
* `--early_stopping_patience` (int)

---

## ♻️ Reproducibility

* Seed is set across Python, NumPy, and PyTorch
* Environment recorded in `requirements.txt` and `training_config.json`
* Model deterministic flags where possible (note: exact GPU nondeterminism may vary)

---

## 🧩 Tips & Ablations

* Try `roberta-base` and `allenai/scibert_scivocab_uncased` and compare Macro-F1
* Increase `max_length` to 512 for longer abstracts (slower/training cost up)
* Use class weighting or focal loss if label distribution is skewed
* Use `stratified` splits by label for stability in small classes

---

## 🧪 CI / Tests

A lightweight test runs a single forward pass on a tiny synthetic batch to validate dependencies and imports. This is used in `.github/workflows/ci.yml`.

---

## 📓 Colab Notebook

`notebooks/colab_experiment.ipynb` includes:

* Dataset preview (first 100 examples)
* One-click model training (small subset for demo)
* Evaluation and visualizations

A downloadable copy of the notebook is included so users can launch on Colab with the `Open in Colab` badge.

---

## 🐳 Docker

Dockerfile builds a reproducible image with GPU-capable base (nvidia/cuda) for training. `docker/entrypoint.sh` wraps the training command.

---

## ✍️ Contribution Guide

* Open an issue for feature requests or bugs
* Fork, create a feature branch, and open a PR with tests/docs
* Follow code style: `black` + `flake8`

---

## 📚 References & Acknowledgments

This project builds on top of Hugging Face Transformers and the ArXiv dataset. SciBERT is recommended for best results on scientific text.

---

## 🧾 License

Apache 2.0

---
