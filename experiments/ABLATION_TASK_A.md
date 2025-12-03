# Subtask A – Ablation Study (Binary Detection)

All scores are on the official Subtask A validation set.

---

## 1. CodeBERT encoder training (frozen vs hybrid vs fine-tuned)

| Model (CodeBERT family)   | Accuracy | Macro F1 |
|---------------------------|---------:|---------:|
| Frozen CodeBERT + LR      | 0.9640   | 0.9640   |
| Hybrid CodeBERT + BiLSTM  | 0.9940   | 0.9940   |
| Fully fine-tuned CodeBERT | 0.9940   | 0.9940   |

Freezing the encoder and training only a linear head is clearly weaker.
Adding a BiLSTM on top of frozen embeddings closes almost all of the gap,
and fully fine-tuning CodeBERT reaches the best performance.

---

## 2. UniXcoder ablations (length and language tag)

| Variant                     | Max length | Lang tag | Accuracy | Macro F1 |
|----------------------------|-----------:|:--------:|---------:|---------:|
| UniXcoder (baseline)       | 256        | no       | 0.9940   | 0.9940   |
| UniXcoder + `[LANG]` prefix| 256        | yes      | 0.9943   | 0.9942   |
| UniXcoder                  | 128        | no       | 0.9926   | 0.9926   |

Shorter sequences (128 tokens) slightly hurt performance.
Adding an explicit language tag as a prefix gives a very small
but consistent gain over the baseline.

---

## 3. Per-class metrics (CodeBERT)

| Label | Description | Precision | Recall | F1     | Support |
|------:|-------------|----------:|-------:|-------:|--------:|
| 0     | Human       | 0.9926    | 0.9945 | 0.9935 | 47,695  |
| 1     | Machine     | 0.9949    | 0.9933 | 0.9941 | 52,305  |

CodeBERT performs almost symmetrically on human and machine code, with
very high precision and recall for both labels.

---

## 4. Per-language performance (CodeBERT)

| Language | Accuracy | Macro F1 |
|----------|---------:|---------:|
| C++      | 0.9637   | 0.9636   |
| Java     | 0.9668   | 0.9668   |
| Python   | 0.9965   | 0.9965   |

Performance is strongest on Python, but accuracy and macro F1 remain
above 0.96 for all three languages, indicating good cross-language
generalization in the binary detection setting.
