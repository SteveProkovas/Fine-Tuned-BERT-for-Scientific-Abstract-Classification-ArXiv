"""
Train script for fine-tuning a Transformer on ArXiv abstracts — optimized for
CPU / Windows development machines (example config: AMD Ryzen 5 4600H,
8 GB RAM, 64-bit Windows).

Optimizations & sensible defaults for this environment:
- Default batch size smaller to fit limited RAM (default: 8)
- Gradient accumulation to emulate larger batch sizes when needed
- `num_workers` default 0 to avoid Windows multiprocessing overhead
- Automatic disabling of mixed-precision (fp16) when CUDA is not present
- Limit PyTorch intra-op / inter-op threads to CPU core count
- Pin memory only when CUDA available
- Recommendations in comments for using smaller models (DistilBERT) if CPU-only

Usage example (CPU/Windows-friendly):
python src/train.py \
  --train_file dataset/processed/train.jsonl \
  --val_file dataset/processed/val.jsonl \
  --model_name_or_path allenai/scibert_scivocab_uncased \
  --output_dir saved_models/bert-arxiv \
  --num_labels 10 \
  --max_length 256 \
  --batch_size 8 \
  --epochs 5 \
  --lr 2e-5 \
  --seed 42

Note: If you don't have a GPU, consider using a smaller model such as
`distilbert-base-uncased` or a lightweight SciBERT variant to reduce CPU
and memory usage.
"""

import argparse
import json
import logging
import math
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
)
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support


logger = logging.getLogger(__name__)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Make CuDNN deterministic where possible (may slow training)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Transformer for scientific abstract classification")

    # Data
    parser.add_argument("--train_file", type=str, required=True, help="Path to train jsonl")
    parser.add_argument("--val_file", type=str, required=True, help="Path to validation jsonl")
    parser.add_argument("--text_column", type=str, default="abstract", help="Column name for text")
    parser.add_argument("--label_column", type=str, default="label_id", help="Column name for label ids (int)")

    # Model / tokenizer
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="allenai/scibert_scivocab_uncased",
        help="HF model id or path (consider `distilbert-base-uncased` on CPU-only machines)",
    )
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--num_labels", type=int, required=True)

    # Training hyperparams (conservative defaults for 8 GB RAM)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.0)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    # DataLoader / system
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of DataLoader worker processes (Windows: default 0 to avoid spawn overhead)",
    )

    # Misc
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true", help="Use mixed precision (only when CUDA available)")
    parser.add_argument("--early_stopping_patience", type=int, default=3)
    parser.add_argument("--save_every_epoch", action="store_true", help="Save checkpoint every epoch in addition to best")
    parser.add_argument("--logging_steps", type=int, default=50)

    args = parser.parse_args()
    return args


def setup_logging(output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(output_dir, "training.log")),
        ],
    )


def load_and_prepare_datasets(
    train_file: str,
    val_file: str,
    text_column: str,
    label_column: str,
    tokenizer: AutoTokenizer,
    max_length: int,
):
    # expects newline-delimited JSON (jsonl) with fields including text_column and label_column (int)
    data_files = {"train": train_file, "validation": val_file}
    ds = load_dataset("json", data_files=data_files)

    def preprocess_function(examples):
        texts = examples[text_column]
        result = tokenizer(texts, truncation=True, max_length=max_length)
        result["labels"] = examples[label_column]
        return result

    # Use batched mapping to tokenize efficiently
    tokenized = ds.map(preprocess_function, batched=True, remove_columns=ds["train"].column_names)

    return tokenized["train"], tokenized["validation"]


def create_dataloaders(train_dataset, val_dataset, batch_size: int, data_collator, num_workers: int, device):
    pin_memory = True if device.type == "cuda" else False
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=data_collator, num_workers=num_workers, pin_memory=pin_memory
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, collate_fn=data_collator, num_workers=num_workers, pin_memory=pin_memory
    )
    return train_loader, val_loader


def evaluate(model, dataloader, device) -> Tuple[Dict[str, float], List[int], List[int]]:
    model.eval()
    preds_all = []
    labels_all = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1).detach().cpu().numpy().tolist()
            labels = batch["labels"].detach().cpu().numpy().tolist()
            preds_all.extend(preds)
            labels_all.extend(labels)

    acc = accuracy_score(labels_all, preds_all)
    macro_f1 = f1_score(labels_all, preds_all, average="macro")
    weighted_f1 = f1_score(labels_all, preds_all, average="weighted")
    precision, recall, f1_per_class, _ = precision_recall_fscore_support(labels_all, preds_all, average=None)

    metrics = {
        "accuracy": float(acc),
        "f1_macro": float(macro_f1),
        "f1_weighted": float(weighted_f1),
    }
    model.train()
    return metrics, labels_all, preds_all


def save_state(model, tokenizer, output_dir: str, epoch: int = None, prefix: str = "best"):
    dest = Path(output_dir) / f"{prefix}"
    dest.mkdir(parents=True, exist_ok=True)
    # save model & tokenizer in HF format
    model_to_save = model.module if hasattr(model, "module") else model
    model_to_save.save_pretrained(dest)
    tokenizer.save_pretrained(dest)
    logger.info(f"Saved checkpoint to {dest}")


def main():
    args = parse_args()
    setup_logging(args.output_dir)
    logger.info("Training configuration: %s", vars(args))

    # Limit PyTorch thread usage to match CPU cores (Ryzen 5 4600H: 6 physical cores / 12 threads)
    # Use a conservative number of threads to avoid oversubscription on 8GB machines.
    torch.set_num_threads(6)
    torch.set_num_interop_threads(6)

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # If no CUDA available, automatically ignore fp16 to avoid errors
    if not torch.cuda.is_available() and args.fp16:
        logger.warning("CUDA not available: disabling --fp16 to avoid errors on CPU-only machine.")
        args.fp16 = False

    # Load model config / tokenizer / model
    config = AutoConfig.from_pretrained(args.model_name_or_path, num_labels=args.num_labels)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path, config=config)
    model.to(device)

    # Data
    train_dataset, val_dataset = load_and_prepare_datasets(
        args.train_file, args.val_file, args.text_column, args.label_column, tokenizer, args.max_length
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    train_loader, val_loader = create_dataloaders(train_dataset, val_dataset, args.batch_size, data_collator, args.num_workers, device)

    # Optimizer & scheduler
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.lr)

    # Compute total training steps conservatively (avoid zero division)
    effective_train_batch_size = args.batch_size * max(1, args.gradient_accumulation_steps)
    steps_per_epoch = max(1, math.ceil(len(train_loader) / 1))
    t_total = (len(train_loader) // max(1, args.gradient_accumulation_steps)) * args.epochs
    num_warmup_steps = int(args.warmup_ratio * t_total)

    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=max(1, t_total))

    scaler = torch.cuda.amp.GradScaler() if args.fp16 and torch.cuda.is_available() else None

    best_macro_f1 = -1.0
    patience_counter = 0

    # Save config
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "training_config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    global_step = 0
    logger.info("Starting training")
    model.zero_grad()

    for epoch in range(1, args.epochs + 1):
        epoch_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False)
        for step, batch in enumerate(progress_bar, start=1):
            model.train()
            batch = {k: v.to(device) for k, v in batch.items()}

            # forward
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = model(**batch)
                    loss = outputs.loss
                    loss = loss / args.gradient_accumulation_steps
                scaler.scale(loss).backward()
            else:
                outputs = model(**batch)
                loss = outputs.loss
                loss = loss / args.gradient_accumulation_steps
                loss.backward()

            epoch_loss += loss.item() * args.gradient_accumulation_steps

            if step % args.gradient_accumulation_steps == 0:
                # clip gradients
                if scaler is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()

                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % args.logging_steps == 0:
                    progress_bar.set_postfix({"loss": f"{epoch_loss/global_step:.4f}"})

        # End epoch
        avg_epoch_loss = epoch_loss / max(1, len(train_loader))
        logger.info(f"Epoch {epoch} finished. avg_loss={avg_epoch_loss:.4f}")

        # Optionally save epoch checkpoint
        if args.save_every_epoch:
            save_state(model, tokenizer, args.output_dir, epoch=epoch, prefix=f"epoch-{epoch}")

        # Evaluate
        metrics, labels_all, preds_all = evaluate(model, val_loader, device)
        logger.info(f"Validation metrics at epoch {epoch}: {metrics}")

        # Save metrics JSON
        metrics_path = os.path.join(args.output_dir, f"val_metrics_epoch_{epoch}.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        current_macro = metrics.get("f1_macro", 0.0)
        if current_macro > best_macro_f1:
            best_macro_f1 = current_macro
            patience_counter = 0
            logger.info(f"New best Macro-F1: {best_macro_f1:.4f}. Saving model...")
            save_state(model, tokenizer, args.output_dir, epoch=epoch, prefix="best")
        else:
            patience_counter += 1
            logger.info(f"No improvement. Patience: {patience_counter}/{args.early_stopping_patience}")

        # Early stopping
        if patience_counter >= args.early_stopping_patience:
            logger.info("Early stopping triggered")
            break

    logger.info("Training complete. Best Macro-F1: %.4f", best_macro_f1)


if __name__ == "__main__":
    main()
