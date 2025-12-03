# src/train_ensemble_task_a.py

"""
Weighted-probability ensemble for SemEval 2026 Task 13 – Subtask A (binary).

This script:
  1. Loads validation and test probabilities from individual models:
       - CodeBERT      → experiments/task_a_codebert/val_probs.npy, test_probs.npy
       - GraphCodeBERT → experiments/task_a_graphcodebert/val_probs.npy, test_probs.npy
       - UniXcoder     → experiments/task_a_unixcoder/val_probs.npy, test_probs.npy
  2. Loads the true validation labels from the Task A validation parquet file.
  3. Evaluates each model individually on validation (accuracy, macro-F1).
  4. Builds a weighted-probability ensemble:
       p_ens = w1 * p_codebert + w2 * p_graphcodebert + w3 * p_unixcoder
  5. Reports ensemble validation accuracy and macro-F1.
  6. Saves ensemble val/test predictions and a submission-style CSV under:
       experiments/task_a_ensemble/
"""

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from data_utils_task_a import load_task_a_splits, basic_clean_task_a


def load_probs(model_folder: str, experiments_root: Path) -> Dict[str, np.ndarray]:
    """
    Load validation and test probability arrays for a given model.

    Expects files:
      experiments_root / model_folder / "val_probs.npy"
      experiments_root / model_folder / "test_probs.npy"
    """
    model_dir = experiments_root / model_folder
    val_path = model_dir / "val_probs.npy"
    test_path = model_dir / "test_probs.npy"

    if not val_path.exists():
        raise FileNotFoundError(f"Missing validation probs for {model_folder}: {val_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Missing test probs for {model_folder}: {test_path}")

    val_probs = np.load(val_path)
    test_probs = np.load(test_path)

    return {"val": val_probs, "test": test_probs}


def weighted_probs_ensemble(
    probs_list: List[np.ndarray],
    weights: np.ndarray,
) -> np.ndarray:
    """
    Combine a list of [N, C] probability matrices using a weight vector.

    Args:
        probs_list: list of arrays, each shape [N, num_classes].
        weights: 1D array of shape [num_models].

    Returns:
        Ensemble probabilities of shape [N, num_classes].
    """
    if len(probs_list) != len(weights):
        raise ValueError("Number of probability matrices and weights must match.")

    stacked = np.stack(probs_list, axis=0)  # [M, N, C]
    w = weights.reshape(-1, 1, 1)           # [M, 1, 1]
    weighted = stacked * w                  # [M, N, C]
    ens_probs = weighted.sum(axis=0)        # [N, C]
    return ens_probs


def ensemble_task_a(
    data_root: str = "data",
    experiments_root: str = "experiments",
    output_dir: str = "experiments/task_a_ensemble",
    # change these to match your notebook weights if needed
    weight_codebert: float = 0.4,
    weight_graphcodebert: float = 0.3,
    weight_unixcoder: float = 0.3,
):
    """
    Build and evaluate a weighted-probability ensemble for Subtask A.

    Args:
        data_root: Root directory where data/task_a lives.
        experiments_root: Root directory containing experiment subfolders.
        output_dir: Directory to save ensemble predictions and submission file.
        weight_codebert: Weight for CodeBERT probabilities.
        weight_graphcodebert: Weight for GraphCodeBERT probabilities.
        weight_unixcoder: Weight for UniXcoder probabilities.
    """
    exp_root_path = Path(experiments_root)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load true validation labels
    _, val_df, test_df = load_task_a_splits(data_root=data_root)
    val_df = basic_clean_task_a(val_df)
    y_val = val_df["label"].astype(int).values

    # Load probability matrices from each model
    model_folders = {
        "codebert": "task_a_codebert",
        "graphcodebert": "task_a_graphcodebert",
        "unixcoder": "task_a_unixcoder",
    }

    val_probs_dict: Dict[str, np.ndarray] = {}
    test_probs_dict: Dict[str, np.ndarray] = {}

    for key, folder_name in model_folders.items():
        probs = load_probs(folder_name, exp_root_path)
        val_probs_dict[key] = probs["val"]
        test_probs_dict[key] = probs["test"]
        print(f"Loaded probabilities for {key} from {exp_root_path / folder_name}")

    # Sanity checks
    n_val = len(y_val)
    num_classes = val_probs_dict["codebert"].shape[1]
    for name, probs in val_probs_dict.items():
        if probs.shape[0] != n_val:
            raise ValueError(
                f"Validation length mismatch for {name}: {probs.shape[0]} vs {n_val} labels"
            )
        if probs.shape[1] != num_classes:
            raise ValueError(
                f"Number of classes mismatch for {name}: {probs.shape[1]} vs {num_classes}"
            )

    # Individual model performance
    print("\n=== Individual model performance on validation (Task A) ===")
    for name, probs in val_probs_dict.items():
        preds = probs.argmax(axis=-1)
        acc = accuracy_score(y_val, preds)
        f1 = f1_score(y_val, preds, average="macro")
        print(f"{name:12s}  Acc: {acc:.4f}  Macro-F1: {f1:.4f}")

    # Weighted ensemble on validation
    weights = np.array(
        [weight_codebert, weight_graphcodebert, weight_unixcoder],
        dtype=float,
    )
    weights = weights / weights.sum()

    val_list = [
        val_probs_dict["codebert"],
        val_probs_dict["graphcodebert"],
        val_probs_dict["unixcoder"],
    ]
    val_ens_probs = weighted_probs_ensemble(val_list, weights)
    val_ens_preds = val_ens_probs.argmax(axis=-1)

    ens_acc = accuracy_score(y_val, val_ens_preds)
    ens_f1 = f1_score(y_val, val_ens_preds, average="macro")

    print("\n=== Weighted ensemble performance on validation ===")
    print(f"Ensemble   Acc: {ens_acc:.4f}  Macro-F1: {ens_f1:.4f}")

    print("\nEnsemble confusion matrix (val):")
    print(confusion_matrix(y_val, val_ens_preds))

    print("\nEnsemble classification report (val):")
    print(classification_report(y_val, val_ens_preds, digits=4))

    # Weighted ensemble on test probabilities
    test_list = [
        test_probs_dict["codebert"],
        test_probs_dict["graphcodebert"],
        test_probs_dict["unixcoder"],
    ]
    test_ens_probs = weighted_probs_ensemble(test_list, weights)
    test_ens_preds = test_ens_probs.argmax(axis=-1).astype(int)

    np.save(output_path / "val_predictions.npy", val_ens_preds)
    np.save(output_path / "test_predictions.npy", test_ens_preds)
    print("\nSaved ensemble val/test predictions to:", output_path)

    # Submission-style CSV
    if "id" in test_df.columns:
        ids = test_df["id"].values
    else:
        ids = np.arange(len(test_df))

    sub_df = pd.DataFrame({"id": ids, "label": test_ens_preds})
    sub_path = output_path / "submission_ensemble_task_a.csv"
    sub_df.to_csv(sub_path, index=False)
    print("Saved ensemble submission CSV to:", sub_path)


if __name__ == "__main__":
    ensemble_task_a()
