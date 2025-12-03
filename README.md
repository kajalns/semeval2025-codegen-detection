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
  data_utils_task_b.py
  train_tfidf_task_b.py
  train_codebert_task_b.py
  train_graphcodebert_task_b.py
  train_unixcoder_task_b.py
  train_frozen_codebert_lr_task_b.py
  train_frozen_codebert_bilstm_task_b.py
  train_ensemble_task_b.py
README.md
requirements.txt
```

-----
## Setup
python -m venv venv
source venv/bin/activate      # on Windows: venv\Scripts\activate
pip install -r requirements.txt



