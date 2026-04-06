"""Train a seq2seq feedback model from the existing answer-quality dataset.

The dataset in this repo does not include human-written feedback text, so this
script creates synthetic targets with the same rubric used at inference time.
That makes the model reproducible, while still letting you swap in manually
annotated feedback later for higher quality.

Usage:
    python -m services.train_feedback_generator \
        --dataset data/upsc_gs_training_v2.json \
        --output-dir services/feedback_model \
        --base-model google/flan-t5-small
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import Dataset

from .feedback_generator import build_training_example


@dataclass
class TrainConfig:
    dataset_path: Path
    output_dir: Path
    base_model: str = "google/flan-t5-small"
    max_input_length: int = 512
    max_target_length: int = 256
    train_batch_size: int = 4
    eval_batch_size: int = 4
    num_train_epochs: int = 4
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    seed: int = 42


class FeedbackDataset(Dataset):
    def __init__(self, records: Sequence[Dict[str, Any]], tokenizer, max_input_length: int, max_target_length: int):
        self.records = list(records)
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.records[idx]
        input_enc = self.tokenizer(
            row["input_text"],
            max_length=self.max_input_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        target_enc = self.tokenizer(
            row["target_text"],
            max_length=self.max_target_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        labels = target_enc["input_ids"].squeeze(0)
        labels = labels.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": input_enc["input_ids"].squeeze(0),
            "attention_mask": input_enc["attention_mask"].squeeze(0),
            "labels": labels,
        }


def load_dataset(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Dataset must be a JSON list of training rows")
    return data


def build_records(samples: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for sample in samples:
        try:
            example = build_training_example(sample, force_template=True)
            records.append({
                "sample_id": sample.get("sample_id"),
                "question_id": sample.get("question_data", {}).get("id"),
                "module": sample.get("module"),
                **example,
            })
        except Exception:
            continue
    return records


def split_by_question(records: Sequence[Dict[str, Any]], seed: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    groups = [str(r.get("question_id") or r.get("sample_id")) for r in records]
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=seed)
    indices = np.arange(len(records))
    train_idx, eval_idx = next(splitter.split(indices, groups=groups))
    train_records = [records[i] for i in train_idx]
    eval_records = [records[i] for i in eval_idx]
    return train_records, eval_records


def train(config: TrainConfig) -> Dict[str, Any]:
    from transformers import (
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        set_seed,
    )

    set_seed(config.seed)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    raw_samples = load_dataset(config.dataset_path)
    records = build_records(raw_samples)
    if not records:
        raise ValueError("No usable records were built from the dataset")

    train_records, eval_records = split_by_question(records, config.seed)

    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    model = AutoModelForSeq2SeqLM.from_pretrained(config.base_model)

    train_ds = FeedbackDataset(train_records, tokenizer, config.max_input_length, config.max_target_length)
    eval_ds = FeedbackDataset(eval_records, tokenizer, config.max_input_length, config.max_target_length)

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(config.output_dir),
        per_device_train_batch_size=config.train_batch_size,
        per_device_eval_batch_size=config.eval_batch_size,
        gradient_accumulation_steps=1,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        num_train_epochs=config.num_train_epochs,
        predict_with_generate=True,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=25,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=False,
        report_to=[],
        seed=config.seed,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    train_output = trainer.train()
    eval_metrics = trainer.evaluate()

    trainer.save_model(str(config.output_dir))
    tokenizer.save_pretrained(str(config.output_dir))

    metadata = {
        "dataset_path": str(config.dataset_path),
        "base_model": config.base_model,
        "output_dir": str(config.output_dir),
        "train_records": len(train_records),
        "eval_records": len(eval_records),
        "train_loss": float(train_output.training_loss),
        "eval_loss": float(eval_metrics.get("eval_loss", 0.0)),
        "note": (
            "Synthetic targets were generated from the same rubric used at inference time. "
            "Replace or augment with human annotations for better style and accuracy."
        ),
    }

    with (config.output_dir / "training_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    return metadata


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train the feedback generation model")
    parser.add_argument("--dataset", required=True, help="Path to the training dataset JSON")
    parser.add_argument("--output-dir", default="services/feedback_model", help="Directory to save the model")
    parser.add_argument("--base-model", default="google/flan-t5-small", help="Hugging Face base model")
    parser.add_argument("--max-input-length", type=int, default=512)
    parser.add_argument("--max-target-length", type=int, default=256)
    parser.add_argument("--train-batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--num-train-epochs", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    return TrainConfig(
        dataset_path=Path(args.dataset),
        output_dir=Path(args.output_dir),
        base_model=args.base_model,
        max_input_length=args.max_input_length,
        max_target_length=args.max_target_length,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        seed=args.seed,
    )


if __name__ == "__main__":
    cfg = parse_args()
    result = train(cfg)
    print(json.dumps(result, indent=2, ensure_ascii=False))
