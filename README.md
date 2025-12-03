# semeval2026-task13-teamkajal_keerthana
# SemEval 2026 Task 13 – Detecting Machine-Generated Code

This repository contains our system for **SemEval 2026 Task 13**:

- **Subtask A:** binary detection (human vs machine-generated code).
- **Subtask B:** multi-class authorship detection (which generator produced the code).

We use a mix of classical ML and transformer-based models:

- TF–IDF + Logistic Regression baseline
- Fine-tuned transformers: **CodeBERT, GraphCodeBERT, UniXcoder**
- Frozen CodeBERT with:
  - Logistic Regression head
  - BiLSTM + linear head (hybrid)
- Weighted-probability ensemble of the three transformers

Most experiments were originally run in Google Colab notebooks.  
The code in `src/` is a clean, modular version of the same pipelines so that
the system can be reproduced offline.

---

## Repository structure

```text
configs/            # (optional) configuration files if needed
data/               # expected location of SemEval Task 13 parquet files (not committed)
experiments/        # small experiment artifacts: .npy predictions, result tables, ensemble outputs
notebooks/          # Colab notebooks for data analysis and training
scripts/            # (optional) shell scripts, if used
src/                # Python source files for data loading and model training
README.md
requirements.txt
```

-----
## Setup
We recommend using a virtual environment:
python -m venv venv\
source venv/bin/activate      # on Windows: venv\Scripts\activate\
pip install -r requirements.txt\

----
## Data
We use the public SemEval 2026 Task 13 dataset
(Kaggle slug: daniilor/semeval-2026-task13). Raw data is not committed.

The expected local layout is:
```text
data/
  task_a/
    task_a_training_set_1.parquet
    task_a_validation_set.parquet
    task_a_test_set_sample.parquet

  task_b/
    task_b_training_set.parquet
    task_b_validation_set.parquet
    task_b_test_set_sample.parquet
```
----
## Running Subtask B pipelines

From the project root, after installing dependencies and placing the data:
# TF–IDF + Logistic Regression baseline
python -m src.train_tfidf_task_b

# Fine-tuned transformers
python -m src.train_codebert_task_b
python -m src.train_graphcodebert_task_b
python -m src.train_unixcoder_task_b

# Frozen CodeBERT baselines
python -m src.train_frozen_codebert_lr_task_b
python -m src.train_frozen_codebert_bilstm_task_b

# Transformer ensemble (requires saved val/test probabilities)
python -m src.train_ensemble_task_b
