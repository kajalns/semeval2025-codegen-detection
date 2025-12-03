"""
Fine-tune CodeBERT for SemEval 2026 Task 13 – Subtask B (authorship detection).

This script:
  1. Loads Task B train/validation/test splits from data/task_b.
  2. Applies basic cleaning: keep (code, label), drop missing code, strip whitespace.
  3. Tokenizes code with the CodeBERT tokenizer.
  4. Fine-tunes `microsoft/codebert-base` for multi-class classification.
  5. Prints validation accuracy and macro-F1.
  6. Saves the model and tokenizer under models/task_b_codebert.
"""

from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)

from data_utils_task_b import load_task_b_splits, basic_clean_task_b


def prepare_task_b_datasets(
    data_root: str = "data",
    max_length: int = 256,
) -> Dict[str, Dataset]:
    """
    Load and preprocess Task B data for CodeBERT fine-tuning.

    Args:
        data_root: Root directory where data/task_b is located.
        max_length: Maximum sequence length for tokenization.

    Returns:
        A dict with Hugging Face Datasets: {"train": train_ds, "val": val_ds, "test": test_ds}.
    """
    train_df, val_df, test_df = load_task_b_splits(data_root=data_root)

    # Clean train and validation (code + label)
    train_df = basic_clean_task_b(train_df)
    val_df   = basic_clean_task_b(val_df)

    # Test only has code (no labels)
    test_df = test_df[["code"]].copy()
    test_df["code"] = test_df["code"].astype(str).str.strip()

    # Convert to HF Dataset
    train_ds = Dataset.from_pandas(train_df[["code", "label"]], preserve_index=False)
    val_ds   = Dataset.from_pandas(val_df[["code", "label"]],   preserve_index=False)
    test_ds  = Dataset.from_pandas(test_df[["code"]],           preserve_index=False)

    # Rename column for Trainer compatibility
    train_ds = train_ds.rename_column("label", "labels")
    val_ds   = val_ds.rename_column("label", "labels")

    # Load tokenizer
    model_name = "microsoft/codebert-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        # Make sure pad token is set
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.cls_token

    def tok_fn(batch):
        return tokenizer(
            batch["code"],
            truncation=True,
            max_length=max_length,
        )

    train_tok = train_ds.map(tok_fn, batched=True)
    val_tok   = val_ds.map(tok_fn,   batched=True)
    test_tok  = test_ds.map(tok_fn,  batched=True)

    # Remove raw text column
    train_tok = train_tok.remove_columns(["code"])
    val_tok   = val_tok.remove_columns(["code"])
    test_tok  = test_tok.remove_columns(["code"])

    # Set torch format
    train_tok.set_format(type="torch")
    val_tok.set_format(type="torch")
    test_tok.set_format(type="torch")

    return {
        "train": train_tok,
        "val": val_tok,
        "test": test_tok,
    }


def compute_metrics(eval_pred: Any) -> Dict[str, float]:
    """
    Compute accuracy and macro-F1 for the validation set.
    """
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    macro_f1 = f1_score(labels, preds, average="macro")
    acc      = accuracy_score(labels, preds)

    return {"macro_f1": macro_f1, "accuracy": acc}


def train_codebert_task_b(
    data_root: str = "data",
    output_dir: str = "models/task_b_codebert",
    num_train_epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    max_length: int = 256,
):
    """
    Fine-tune CodeBERT for Subtask B.

    Args:
        data_root: Root directory where data/task_b lives.
        output_dir: Directory to save the fine-tuned model and tokenizer.
        num_train_epochs: Number of training epochs.
        batch_size: Per-device batch size for training and evaluation.
        learning_rate: Learning rate for AdamW.
        max_length: Maximum sequence length for tokenization.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Prepare datasets
    ds = prepare_task_b_datasets(data_root=data_root, max_length=max_length)
    train_tok = ds["train"]
    val_tok   = ds["val"]
    test_tok  = ds["test"]

    # Load model and tokenizer
    model_name = "microsoft/codebert-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.cls_token

    num_labels = len(pd.unique(train_tok["labels"].numpy()))
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
    )
    model.to(device)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=200,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        report_to=[],  # disable wandb etc.
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Train and evaluate
    trainer.train()
    eval_results = trainer.evaluate(eval_dataset=val_tok)
    print("CodeBERT Task B validation results:", eval_results)

    # Save model and tokenizer
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    trainer.save_model(output_path)
    tokenizer.save_pretrained(output_path)

    # Optional: get predictions on test set for later submission
    test_outputs = trainer.predict(test_tok)
    test_preds   = np.argmax(test_outputs.predictions, axis=-1).astype(int)
    np.save(output_path / "test_predictions.npy", test_preds)
    print("Saved test predictions to:", output_path / "test_predictions.npy")


if __name__ == "__main__":
    # Simple default run; you can later add argparse if needed.
    train_codebert_task_b()
