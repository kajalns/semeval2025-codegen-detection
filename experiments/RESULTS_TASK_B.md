# Subtask B – Validation Results (SemEval 2026 Task 13)

All scores are computed on the official Subtask B dev set.

| Model                            | Accuracy | Macro F1 |
|----------------------------------|---------:|---------:|
| TF–IDF + LR (char 3–6g)          | 0.9113   | 0.3418   |
| Frozen CodeBERT + LR             | 0.9005   | 0.2860   |
| CodeBERT (fine-tuned)            | 0.9344   | 0.5309   |
| GraphCodeBERT (fine-tuned)       | 0.9374   | 0.5499   |
| UniXcoder (fine-tuned)           | 0.9430   | 0.5750   |
| CodeBERT + BiLSTM (hybrid)       | 0.9329   | 0.5303   |
| **Ensemble (3 transformers)**    | **0.9437** | **0.5770** |
