# Experiments

This directory stores lightweight experiment artifacts for SemEval 2026 Task 13.

We do not commit large model checkpoints here, only small `.npy` arrays.

## Subtask B layout

- `task_b_tfidf/` – TF–IDF + LR predictions.
- `task_b_codebert/` – fine-tuned CodeBERT probabilities and predictions.
- `task_b_graphcodebert/` – fine-tuned GraphCodeBERT.
- `task_b_unixcoder/` – fine-tuned UniXcoder.
- `task_b_frozen_codebert_lr/` – frozen CodeBERT + LR head.
- `task_b_frozen_codebert_bilstm/` – frozen CodeBERT + BiLSTM head.
- `task_b_ensemble/` – weighted ensemble predictions and submission file.

## Subtask A layout
