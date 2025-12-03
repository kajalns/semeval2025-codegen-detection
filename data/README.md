# Data layout – SemEval 2026 Task 13

We use the public SemEval 2026 Task 13 dataset (Kaggle slug
`daniilor/semeval-2026-task13`). The raw data is **not** committed to this
repository. Instead, we expect the following local layout:

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
