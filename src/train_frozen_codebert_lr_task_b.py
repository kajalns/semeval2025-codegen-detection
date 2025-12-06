"""
Frozen CodeBERT + Logistic Regression baseline for SemEval 2026 Task 13 – Subtask B.

Pipeline:
  1. Load Task B train/validation/test splits from data/task_b.
  2. Apply basic cleaning: keep (code, label), drop missing code.
  3. Tokenize code with CodeBERT tokenizer.
  4. Use a FROZEN CodeBERT encoder to extract CLS embeddings for each sample.
  5. Train a Logistic Regression classifier on the extracted features.
  6. Report validation accuracy and macro-F1.
  7. Save the LR model, vector features, and test predictions under models/task_b_frozen_codebert_lr.

Note:
  - The encoder is not fine-tuned here; only the LR head is trained.
"""

from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModel

from data_utils_task_b import load_task_b_splits, basic_clean_task_b


MODEL_NAME = "microsoft/codebert-base"  
MAX_LENGTH = 256
BATCH_SIZE = 16


def prepare_task_b_for_features(
    data_root: str = "data",
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Load and preprocess Task B data for frozen CodeBERT feature extraction.

    Args:
        data_root: Root directory where data/task_b is located.

    Returns:
        (train_tok, val_tok, test_tok): tokenized Hugging Face Datasets.
    """
    train_df, val_df, test_df = load_task_b_splits(data_root=data_root)

    # Clean train and validation using the helper
    train_df = basic_clean_task_b(train_df)
    val_df   = basic_clean_task_b(val_df)

    # Test only has code
    test_df = test_df[["code"]].copy()
    test_df["code"] = test_df["code"].astype(str).str.strip()

    # Convert to HF Datasets
    train_ds = Dataset.from_pandas(train_df[["code", "label"]], preserve_index=False)
    val_ds   = Dataset.from_pandas(val_df[["code", "label"]],   preserve_index=False)
    test_ds  = Dataset.from_pandas(test_df[["code"]],           preserve_index=False)

    train_ds = train_ds.rename_column("label", "labels")
    val_ds   = val_ds.rename_column("label", "labels")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.cls_token

    def tok_fn(batch):
        return tokenizer(
            batch["code"],
            truncation=True,
            max_length=MAX_LENGTH,
            padding="max_length",
        )

    train_tok = train_ds.map(tok_fn, batched=True, remove_columns=["code"])
    val_tok   = val_ds.map(tok_fn,   batched=True, remove_columns=["code"])
    test_tok  = test_ds.map(tok_fn,  batched=True, remove_columns=["code"])

    train_tok.set_format(type="torch")
    val_tok.set_format(type="torch")
    test_tok.set_format(type="torch")

    return train_tok, val_tok, test_tok


def extract_frozen_codebert_features(
    dataset: Dataset,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Tuple[np.ndarray, np.ndarray | None]:
    """
    Extract [CLS] embeddings from a frozen CodeBERT encoder.

    Args:
        dataset: Tokenized HF Dataset with 'input_ids' and 'attention_mask'.
                 Optionally has 'labels'.
        device: 'cuda' or 'cpu'.

    Returns:
        (features, labels) where:
          - features is a NumPy array of shape [N, hidden_size],
          - labels is a NumPy array of shape [N] or None if no labels in dataset.
    """
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = AutoModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    # Freeze encoder parameters (no gradients)
    for param in model.parameters():
        param.requires_grad = False

    all_feats = []
    all_labels = []

    with torch.inference_mode():
        for batch in tqdm(loader, desc="Extracting CodeBERT features"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            # CLS token embedding: [batch_size, hidden_size]
            cls_repr = outputs.last_hidden_state[:, 0, :]

            all_feats.append(cls_repr.cpu().numpy())

            if "labels" in batch:
                all_labels.append(batch["labels"].cpu().numpy())

    features = np.concatenate(all_feats, axis=0)
    if all_labels:
        labels = np.concatenate(all_labels, axis=0)
        return features, labels

    return features, None


def train_frozen_codebert_lr_task_b(
    data_root: str = "data",
    output_dir: str = "models/task_b_frozen_codebert_lr",
):
    """
    Train Logistic Regression on frozen CodeBERT features for Subtask B.

    Args:
        data_root: Root data directory where data/task_b lives.
        output_dir: Directory to save the LR model and predictions.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    #Prepare tokenized datasets
    train_tok, val_tok, test_tok = prepare_task_b_for_features(data_root=data_root)

    #Extract features
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Extracting features for train set...")
    X_train, y_train = extract_frozen_codebert_features(train_tok, device=device)

    print("Extracting features for validation set...")
    X_val, y_val = extract_frozen_codebert_features(val_tok, device=device)

    print("Extracting features for test set...")
    X_test, _ = extract_frozen_codebert_features(test_tok, device=device)

    print("Feature shapes:", X_train.shape, X_val.shape, X_test.shape)

    #Train Logistic Regression
    lr = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        n_jobs=-1,
        multi_class="auto",
    )
    lr.fit(X_train, y_train)
    print("Logistic Regression trained on frozen CodeBERT features.")

    #Evaluate on validation set
    val_preds = lr.predict(X_val)
    val_acc = accuracy_score(y_val, val_preds)
    val_f1  = f1_score(y_val, val_preds, average="macro")

    print(f"[Task B] Frozen CodeBERT + LR Dev Accuracy : {val_acc:.4f}")
    print(f"[Task B] Frozen CodeBERT + LR Dev Macro F1 : {val_f1:.4f}")

    print("\nConfusion matrix (val):")
    print(confusion_matrix(y_val, val_preds))

    print("\nClassification report (val):")
    print(classification_report(y_val, val_preds, digits=4))

    #Predict on test set and save everything
    test_preds = lr.predict(X_test).astype(int)
    np.save(output_path / "val_predictions.npy", val_preds)
    np.save(output_path / "test_predictions.npy", test_preds)

    joblib.dump(lr, output_path / "lr_model.joblib")

    print("Saved LR model and predictions to:", output_path)


if __name__ == "__main__":
    train_frozen_codebert_lr_task_b()
