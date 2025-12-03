from pathlib import Path
from typing import Tuple

import pandas as pd


def get_task_dir(data_root: str, task: str) -> Path:
    """
    Return the directory for a given SemEval task.

    Args:
        data_root: Root data directory (e.g. "data").
        task: "A" or "B".

    Returns:
        Path to the task folder, e.g. data/task_a or data/task_b.
    """
    root = Path(data_root)
    task = task.upper()
    if task == "A":
        return root / "task_a"
    elif task == "B":
        return root / "task_b"
    else:
        raise ValueError(f"Unknown task: {task}")


def load_splits(data_root: str, task: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load train / validation / test parquet files for Subtask A or B.

    Expects files:
      task_a_training_set_1.parquet  /  task_b_training_set_1.parquet
      task_a_validation_set.parquet  /  task_b_validation_set.parquet
      task_a_test_set_sample.parquet /  task_b_test_set_sample.parquet
    inside data/task_a or data/task_b.

    Args:
        data_root: Root data directory.
        task: "A" or "B".

    Returns:
        (train_df, val_df, test_df)
    """
    task_dir = get_task_dir(data_root, task)
    prefix = f"task_{task.lower()}"

    train_df = pd.read_parquet(task_dir / f"{prefix}_training_set_1.parquet")
    val_df   = pd.read_parquet(task_dir / f"{prefix}_validation_set.parquet")
    test_df  = pd.read_parquet(task_dir / f"{prefix}_test_set_sample.parquet")

    return train_df, val_df, test_df
