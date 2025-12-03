"""
CodeBERT fine-tuning script for SemEval 2026 Task 13 – Subtask A (binary).
"""

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

from data_utils_task_a import load_task_a_splits, basic_clean_task_a


MODEL_NAME = "microsoft/codebert-base"
OUTPUT_DIR = "models/task_a_codebert"


def to_hf_dataset(df: pd.DataFrame) -> Dataset:
    """Convert pandas DataFrame (code, label) to HuggingFace Dataset."""
    return Dataset.from_pandas(df[["code", "label"]], preserve_index=False)


def tokenize_function(examples, tokenizer):
    return tokenizer(
        examples["code"],
        padding="max_length",
        truncation=True,
        max_length=256,
    )


def compute_metrics(eval_pred) -> Dict[str, float]:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    acc = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro")

    return {"accuracy": acc, "macro_f1": macro_f1}


def main():
    print("Loading Task A splits...")
    train_df, val_df, test_df = load_task_a_splits(data_root="data")

    print("Cleaning data...")
    train_df = basic_clean_task_a(train_df)
    val_df   = basic_clean_task_a(val_df)
    test_df  = basic_clean_task_a(test_df)

    # Ensure labels are ints starting at 0
    if train_df["label"].dtype != int:
        label2id = {lbl: i for i, lbl in enumerate(sorted(train_df["label"].unique()))}
        id2label = {i: lbl for lbl, i in label2id.items()}

        train_df["label"] = train_df["label"].map(label2id)
        val_df["label"]   = val_df["label"].map(label2id)
    else:
        unique_labels = sorted(train_df["label"].unique())
        id2label = {i: str(i) for i in unique_labels}
        label2id = {str(i): i for i in unique_labels}

    num_labels = len(id2label)

    print(f"Num labels (Task A): {num_labels}")

    print("Converting to HF datasets...")
    train_ds = to_hf_dataset(train_df)
    val_ds   = to_hf_dataset(val_df)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenized_train = train_ds.map(lambda ex: tokenize_function(ex, tokenizer), batched=True)
    tokenized_val   = val_ds.map(lambda ex: tokenize_function(ex, tokenizer), batched=True)

    tokenized_train = tokenized_train.remove_columns(["code"])
    tokenized_val   = tokenized_val.remove_columns(["code"])

    tokenized_train.set_format("torch")
    tokenized_val.set_format("torch")

    print("Loading CodeBERT model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )

    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=100,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        learning_rate=2e-5,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    print("Starting training for Task A CodeBERT...")
    trainer.train()

    print("Evaluating on validation set...")
    metrics = trainer.evaluate()
    print("Validation metrics:", metrics)

    print("Saving model and tokenizer...")
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    print("Done. Saved CodeBERT Task A model to:", output_dir)


if __name__ == "__main__":
    main()
