# Subtask A – Validation Results (SemEval 2026 Task 13)

All scores are computed on the official Subtask A dev set.

| Model                         | Accuracy | Macro F1 |
|------------------------------ |----------|----------|
| TF–IDF + LR                   | 0.9079   | 0.9079   |
| Frozen CodeBERT + LR          | 0.9641   | 0.9641   |
| CodeBERT (fine-tuned)         | 0.9938   | 0.9938   |
| GraphCodeBERT (fine-tuned)    | 0.9941   | 0.9940   |
| UniXcoder (fine-tuned)        | 0.9940   | 0.9940   |
| CodeBERT + BiLSTM (hybrid)    | 0.9938   | 0.9938   |
| **Ensemble (3 transformers)** | **0.9950** | **0.9950** |
