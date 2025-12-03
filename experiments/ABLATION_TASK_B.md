# Subtask B – Ablation Study (11-way Authorship)

All scores are on the official Subtask B dev set.

---

## 1. CodeBERT encoder training (frozen vs hybrid vs fine-tuned)

| Model (CodeBERT family)   | Accuracy | Macro F1 |
|---------------------------|---------:|---------:|
| Frozen CodeBERT + LR      | 0.9005   | 0.2860   |
| Hybrid CodeBERT + BiLSTM  | 0.9330   | 0.5300   |
| Fully fine-tuned CodeBERT | 0.9340   | 0.5310   |

With many generator classes, a frozen encoder with a linear head is not
sufficient (very low macro F1). Adding a BiLSTM on top of frozen
embeddings recovers most of the gap, and full fine-tuning gives a small
additional improvement.

---

## 2. UniXcoder ablations (length and language tag)

| Variant                     | Max length | Lang tag | Accuracy | Macro F1 |
|----------------------------|-----------:|:--------:|---------:|---------:|
| UniXcoder (baseline)       | 256        | no       | 0.9430   | 0.5750   |
| UniXcoder + `[LANG]` prefix| 256        | yes      | 0.9435   | 0.5823   |
| UniXcoder                  | 128        | no       | 0.9330   | 0.5190   |

Reducing the maximum sequence length to 128 degrades macro F1
substantially. Adding an explicit language prefix gives a small but
consistent gain over the baseline (about +0.7 macro F1).

---

## 3. UniXcoder with and without class-weighted loss

| Model variant                       | Accuracy | Macro F1 |
|------------------------------------|---------:|---------:|
| UniXcoder (no class weights)       | 0.9430   | 0.5750   |
| UniXcoder (class-weighted loss)    | 0.9410   | 0.5700   |

Class-weighted loss slightly improves minority classes but hurts overall
macro F1, so we keep the unweighted loss in the final system.

---

## 4. Per-language performance (CodeBERT)

| Language   | Accuracy | Macro F1 |
|-----------|---------:|---------:|
| C         | 0.9458   | 0.4391   |
| C#        | 0.9442   | 0.4365   |
| C++       | 0.9155   | 0.5129   |
| Go        | 0.9366   | 0.4357   |
| Java      | 0.9466   | 0.5009   |
| JavaScript| 0.8896   | 0.4103   |
| PHP       | 0.9788   | 0.3717   |
| Python    | 0.9240   | 0.5578   |

Performance varies by language: PHP and JavaScript show weaker macro F1,
likely due to fewer examples and more heterogeneous coding styles.
Python and C++ have the strongest macro F1, indicating that the model
captures generator differences better for these languages.

---

## 5. Overall comparison with classical baseline and ensemble

For convenience, the main Subtask B dev results are summarized in
`experiments/RESULTS_TASK_B.md`. In short:

- TF–IDF + LR (char 3–6g) reaches **0.9113 accuracy / 0.3418 macro F1**.
- Single transformers (CodeBERT, GraphCodeBERT, UniXcoder) are much stronger.
- A weighted ensemble of the three transformers achieves
  **0.9437 accuracy / 0.5770 macro F1**, our best overall configuration.
