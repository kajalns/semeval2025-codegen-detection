"""
TF-IDF + Logistic Regression baseline for SemEval Task 13 – Subtask A
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from joblib import dump

from data_utils_task_a import load_task_a_splits, basic_clean_task_a


def run_tfidf_lr_task_a():
    """
    Train a TF-IDF + Logistic Regression model for Subtask A.
    """

    print("Loading Task A splits...")
    train_df, val_df, test_df = load_task_a_splits(data_root="data")

    print("Cleaning data...")
    train_df = basic_clean_task_a(train_df)
    val_df   = basic_clean_task_a(val_df)
    test_df  = basic_clean_task_a(test_df)

    print("Extracting TF-IDF features...")
    tfidf = TfidfVectorizer(
        max_features=30000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        dtype=float,
    )

    X_train = tfidf.fit_transform(train_df["text"])
    X_val   = tfidf.transform(val_df["text"])

    print("Training Logistic Regression...")
    clf = LogisticRegression(max_iter=300, n_jobs=-1)
    clf.fit(X_train, train_df["label"])

    print("Evaluating...")
    val_preds = clf.predict(X_val)

    acc  = accuracy_score(val_df["label"], val_preds)
    f1   = f1_score(val_df["label"], val_preds, average="macro")

    print(f"Validation Accuracy: {acc:.4f}")
    print(f"Validation Macro-F1: {f1:.4f}")

    print("Saving model + TF-IDF encoder...")
    dump(clf,  "models/task_a_tfidf_model.joblib")
    dump(tfidf, "models/task_a_tfidf_vectorizer.joblib")

    print("Done!")


if __name__ == "__main__":
    run_tfidf_lr_task_a()
