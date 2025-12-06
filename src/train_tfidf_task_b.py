"""
Train a TF–IDF + Logistic Regression baseline for SemEval 2026 Task 13 – Subtask B.

This script:
  1. Loads the Task B parquet files from data/task_b.
  2. Applies simple cleaning to keep (code, label).
  3. Builds TF–IDF features over code.
  4. Trains a Logistic Regression classifier with class_weight="balanced".
  5. Prints validation accuracy and macro-F1.
  6. Saves the fitted vectorizer and classifier under experiments/tfidf_task_b.
"""

from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from data_utils_task_b import load_task_b_splits, basic_clean_task_b


def load_and_clean_task_b(data_root: str = "data"):
    """
    Load raw Subtask B splits and apply basic cleaning.

    Args:
        data_root: Root data directory (default: "data").

    Returns:
        (train_df, val_df, test_df) cleaned DataFrames.
    """
    train_df, val_df, test_df = load_task_b_splits(data_root=data_root)

    # Clean train/val using the shared helper
    train_df = basic_clean_task_b(train_df)
    val_df   = basic_clean_task_b(val_df)

    # Test has only code, no labels
    test_df = test_df[["code"]].copy()
    test_df["code"] = test_df["code"].astype(str).str.strip()

    return train_df, val_df, test_df


def train_tfidf_lr_task_b(
    data_root: str = "data",
    output_dir: str = "experiments/tfidf_task_b",
    max_features: int = 20000,
):
    """
    Train TF–IDF + Logistic Regression for Subtask B and save the model.

    Args:
        data_root: Root directory where data/task_b is located.
        output_dir: Directory to save the fitted vectorizer and classifier.
        max_features: Maximum vocabulary size for TF–IDF.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    train_df, val_df, test_df = load_and_clean_task_b(data_root=data_root)

    # Text and labels for training and validation
    X_train_text = train_df["code"].tolist()
    y_train = train_df["label"].astype(int).values

    X_val_text = val_df["code"].tolist()
    y_val = val_df["label"].astype(int).values

    # TF–IDF features with a code-aware token pattern
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),      # unigrams + bigrams
        max_features=max_features,
        token_pattern=r"\w+|\S", # treat identifiers and symbols as tokens
    )

    X_train = vectorizer.fit_transform(X_train_text)
    X_val   = vectorizer.transform(X_val_text)

    # Logistic Regression classifier (multi-class for Task B)
    clf = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    print("TF–IDF + LR model trained on Subtask B.")

    # Evaluate on validation set
    val_preds = clf.predict(X_val)
    val_acc = accuracy_score(y_val, val_preds)
    val_f1  = f1_score(y_val, val_preds, average="macro")

    print(f"[Task B] TF–IDF + LR Dev Accuracy : {val_acc:.4f}")
    print(f"[Task B] TF–IDF + LR Dev Macro F1 : {val_f1:.4f}")

    # Save model artifacts for reproducibility
    joblib.dump(vectorizer, output_path / "vectorizer.joblib")
    joblib.dump(clf,        output_path / "lr_model.joblib")
    np.save(output_path / "val_predictions.npy", val_preds)


if __name__ == "__main__":
    # Simple entry point; you can later add argparse if needed.
    train_tfidf_lr_task_b()
