from pathlib import Path
from typing import Tuple

import pandas as pd


def load_task_b_splits(data_root: str = "data") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load train, validation, and test splits for SemEval Task 13 Subtask B.

    This function assumes that the parquet files are stored under:

        data/task_b/task_b_training_set.parquet
        data/task_b/task_b_validation_set.parquet
        data/task_b/task_b_test_set_sample.parquet

    Args:
        data_root:
            Root directory where the 'task_b' folder is located.
            By default we use 'data', so the expected path is 'data/task_b'.

    Returns:
        A tuple (train_df, val_df, test_df) with the raw DataFrames.
    """
    base = Path(data_root) / "task_b"

    train_path = base / "task_b_training_set.parquet"
    val_path   = base / "task_b_validation_set.parquet"
    test_path  = base / "task_b_test_set_sample.parquet"

    if not train_path.exists():
        raise FileNotFoundError(f"Could not find training file at: {train_path}")
    if not val_path.exists():
        raise FileNotFoundError(f"Could not find validation file at: {val_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Could not find test file at: {test_path}")

    train_df = pd.read_parquet(train_path)
    val_df   = pd.read_parquet(val_path)
    test_df  = pd.read_parquet(test_path)

    return train_df, val_df, test_df


def basic_clean_task_b(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply basic cleaning for Subtask B:

      - keep only 'code' and 'label' columns,
      - drop rows where code is missing,
      - cast label to int,
      - strip leading/trailing whitespace from code.

    Args:
        df: Raw DataFrame with at least 'code' and 'label' columns.

    Returns:
        A cleaned DataFrame ready for feature extraction.
    """
    df = df[["code", "label"]].dropna(subset=["code"])
    df["label"] = df["label"].astype(int)
    df["code"] = df["code"].astype(str).str.strip()
    return df


if __name__ == "__main__":
    # Small sanity check when you run this file directly.
    # Example (from repo root):
    #   python -m src.data_utils_task_b
    train, val, test = load_task_b_splits(data_root="data")
    print("Task B shapes:")
    print("  Train:", train.shape)
    print("  Val  :", val.shape)
    print("  Test :", test.shape)
