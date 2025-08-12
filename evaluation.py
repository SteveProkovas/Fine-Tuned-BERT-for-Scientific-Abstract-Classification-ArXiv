#!/usr/bin/env python3
"""
src/evaluate.py

Evaluate a saved checkpoint (HF-format or pytorch_model.bin) on a test set.

Example:
python src/evaluate.py \
  --test_file dataset/processed/test.jsonl \
  --checkpoint_dir saved_models/bert-arxiv/best \
  --model_name_or_path allenai/scibert_scivocab_uncased \
  --text_column abstract \
  --label_column label_id \
  --batch_size 16 \
  --max_length 256 \
  --output_dir results/
"""

import argparse
import json
import os
from pathlib import Path
from typing import Optional, Tuple, Dict

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate model checkpoint on test set (Windows/CPU-friendly).")
    p.add_argument("--test_file", type=str, required=True, help="Path to test file (csv/json/jsonl).")
    p.add_argument("--checkpoint_dir", type=str, required=True, help="Directory with checkpoint (HF format preferred) or pytorch_model.bin.")
    p.add_argument("--model_name_or_path", type=str, default=None, help="Fallback HF model id/path to construct model/tokenizer if needed.")
    p.add_argument("--text_column", type=str, default="abstract", help="Text column name in test file.")
    p.add_argument("--label_column", type=str, default="label_id", help="Label column name in test file (numeric or string).")
    p.add_argument("--label_map_file", type=str, default=None, help="Optional JSON mapping label->id or id->label.")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=0, help="DataLoader workers (Windows default=0).")
    p.add_argument("--device", type=str, default=None, help="cuda or cpu (auto-detected if not set).")
    p.add_argument("--output_dir", type=str, default="results", help="Where to save metrics/artifacts.")
    return p.parse_args()


def load_label_mapping(label_map_file: Optional[str], test_ds, label_column: str) -> Tuple[Optional[dict], Optional[dict], bool]:
    """
    Return (label2id, id2label, test_has_string_labels)
    """
    label2id = None
    id2label = None
    test_has_string_labels = False

    if label_map_file:
        with open(label_map_file, "r") as f:
            mapping = json.load(f)
        # detect mapping direction
        vals = list(mapping.values())
        if len(vals) > 0 and isinstance(vals[0], int):
            # label -> id
            label2id = mapping
            id2label = {v: k for k, v in mapping.items()}
        else:
            # id -> label (keys probably strings of ints)
            id2label = {int(k): v for k, v in mapping.items()}
            label2id = {v: k for k, v in id2label.items()}
        return label2id, id2label, False

    # no mapping file: inspect test dataset for string labels
    if label_column not in test_ds.column_names:
        raise ValueError(f"Label column '{label_column}' not found in test dataset.")
    # sample small set to determine type
    sample = test_ds.select(range(min(1000, len(test_ds))))[label_column]
    if len(sample) > 0 and isinstance(sample[0], str):
        test_has_string_labels = True
        uniques = sorted(list(set(sample)))
        label2id = {lab: i for i, lab in enumerate(uniques)}
        id2label = {i: lab for lab, i in label2id.items()}
    return label2id, id2label, test_has_string_labels


def prepare_test_dataloader(test_ds, tokenizer, text_column: str, label_column: str, batch_size: int, max_length: int, num_workers: int, device):
    def preprocess(examples):
        return tokenizer(examples[text_column], truncation=True, max_length=max_length)

    # keep just text & label (safe)
    remove_cols = [c for c in test_ds.column_names if c not in (text_column, label_column, "label")]
    test_tk = test_ds.map(preprocess, batched=True, remove_columns=remove_cols)
    # rename label -> labels (HF convention)
    if label_column in test_tk.column_names and "labels" not in test_tk.column_names:
        test_tk = test_tk.rename_column(label_column, "labels")
    test_tk.set_format(type="torch")
    collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="longest", return_tensors="pt")
    pin_memory = True if device.type == "cuda" else False
    loader = DataLoader(test_tk, batch_size=batch_size, shuffle=False, collate_fn=collator, num_workers=num_workers, pin_memory=pin_memory)
    return loader, test_tk


@torch.no_grad()
def run_inference(model, dataloader, device):
    model.eval()
    all_logits = []
    all_labels = []
    for batch in tqdm(dataloader, desc="Running inference"):
        labels = batch.pop("labels").to(device)
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        logits = outputs.logits.detach().cpu().numpy()
        all_logits.append(logits)
        all_labels.append(labels.detach().cpu().numpy())
    logits = np.concatenate(all_logits, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    return logits, labels


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def plot_confusion_matrix(cm: np.ndarray, labels: list, out_path: str):
    plt.figure(figsize=(max(6, 0.5 * len(labels)), max(5, 0.4 * len(labels))))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45, ha="right")
    plt.yticks(tick_marks, labels)
    thresh = cm.max() / 2.0 if cm.size else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(int(cm[i, j]), "d"), horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def instantiate_model_and_tokenizer(checkpoint_dir: Path, fallback_model_name: Optional[str], device: torch.device):
    """
    Try to load tokenizer/model directly from checkpoint_dir (HF format). If not possible,
    instantiate from fallback_model_name and load state_dict from pytorch_model.bin if present.
    """
    # Prefer HF format
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_dir), use_fast=True)
        model = AutoModelForSequenceClassification.from_pretrained(str(checkpoint_dir))
        return tokenizer, model
    except Exception:
        # fallback
        if fallback_model_name is None:
            raise ValueError("Failed to load HF-style checkpoint from checkpoint_dir and no --model_name_or_path provided as fallback.")
        tokenizer = AutoTokenizer.from_pretrained(fallback_model_name, use_fast=True)
        # attempt to infer num_labels from config or rely on model default
        model = AutoModelForSequenceClassification.from_pretrained(fallback_model_name)
        # if pytorch_model.bin exists in checkpoint_dir, try to load
        state_path = checkpoint_dir / "pytorch_model.bin"
        if state_path.exists():
            sd = torch.load(state_path, map_location="cpu")
            try:
                model.load_state_dict(sd)
            except Exception:
                # try stripping "module." prefix
                new_sd = {k.replace("module.", ""): v for k, v in sd.items()}
                model.load_state_dict(new_sd)
        return tokenizer, model


def main():
    args = parse_args()
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")

    # load test dataset (csv/json/jsonl supported)
    ext = args.test_file.split(".")[-1].lower()
    if ext not in ("csv", "json", "jsonl"):
        raise ValueError("Test file must be .csv, .json or .jsonl")
    dataset = load_dataset(ext, data_files={"test": args.test_file})
    test_ds = dataset["test"]

    # label mapping inference or load
    label2id, id2label, test_has_strings = load_label_mapping(args.label_map_file, test_ds, args.label_column)
    if label2id is not None:
        print(f"Using label mapping with {len(label2id)} labels.")
    elif test_has_strings:
        print("Detected string labels in test set and created mapping.")
    else:
        print("Expecting numeric labels in test set.")

    # if we inferred mapping (string labels), map test dataset now
    if test_has_strings and label2id:
        def map_fn(ex):
            ex[args.label_column] = int(label2id[ex[args.label_column]])
            return ex
        test_ds = test_ds.map(map_fn)

    # instantiate tokenizer & model
    ckpt = Path(args.checkpoint_dir)
    tokenizer, model = instantiate_model_and_tokenizer(ckpt, args.model_name_or_path, device)
    model.to(device)

    # prepare dataloader
    dataloader, test_processed = prepare_test_dataloader(test_ds, tokenizer, args.text_column, args.label_column, args.batch_size, args.max_length, args.num_workers, device)

    # inference
    logits, labels = run_inference(model, dataloader, device)
    preds = logits.argmax(axis=-1)
    probs = softmax(logits)

    # compute metrics
    acc = float(accuracy_score(labels, preds))
    macro_f1 = float(f1_score(labels, preds, average="macro"))
    weighted_f1 = float(f1_score(labels, preds, average="weighted"))
    precision, recall, f1_per_class, support = precision_recall_fscore_support(labels, preds, average=None, zero_division=0)

    # prepare id2label mapping for human-readable outputs
    if id2label is None:
        # try to get from model config if present
        cfg = getattr(model, "config", None)
        if cfg and hasattr(cfg, "id2label") and cfg.id2label:
            id2label = {int(k): v for k, v in cfg.id2label.items()}
        else:
            # fallback numeric labels
            id2label = {i: str(i) for i in range(max(1, len(precision)))}

    # aggregate metrics
    metrics = {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "num_examples": int(len(labels)),
    }
    per_class = {}
    for i in range(len(precision)):
        lab = id2label.get(i, str(i))
        per_class[lab] = {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1_per_class[i]),
            "support": int(support[i]),
        }
    metrics["per_class"] = per_class

    # save outputs
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # predictions.csv
    rows = []
    for i, (t, p, prob_row) in enumerate(zip(labels.tolist(), preds.tolist(), probs.tolist())):
        rows.append({"index": i, "true_label": id2label.get(int(t), int(t)), "pred_label": id2label.get(int(p), int(p)), "pred_probs": prob_row})
    pd.DataFrame(rows).to_csv(out / "predictions.csv", index=False)

    # per-class CSV
    pd.DataFrame.from_dict(per_class, orient="index").rename_axis("label").to_csv(out / "per_class_metrics.csv")

    # confusion matrix
    cm = confusion_matrix(labels, preds)
    plot_confusion_matrix(cm, [id2label[i] for i in sorted(id2label.keys())], str(out / "confusion_matrix.png"))

    # textual report
    report = classification_report(labels, preds, target_names=[id2label[i] for i in sorted(id2label.keys())], zero_division=0)
    print("=== Classification Report ===")
    print(report)
    print("Saved metrics & artifacts to:", str(out))


if __name__ == "__main__":
    main()
