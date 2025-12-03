"""
Frozen CodeBERT + BiLSTM classifier for SemEval 2026 Task 13 – Subtask A.

Pipeline:
  1. Load Task A train/validation/test splits from data/task_a.
  2. Apply basic cleaning: keep (code, label), drop missing code, strip whitespace.
  3. Tokenize code with CodeBERT tokenizer.
  4. Pass token sequences through a FROZEN CodeBERT encoder.
  5. Feed the sequence of hidden states into a BiLSTM + linear layer.
  6. Train only the BiLSTM + classifier head (encoder is frozen).
  7. Report validation accuracy and macro-F1.
  8. Save the BiLSTM head weights and test predictions under
     models/task_a_frozen_codebert_bilstm.
"""

from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datasets import Dataset
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from transformers import (
    AutoTokenizer,
    AutoModel,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
from transformers.modeling_outputs import SequenceClassifierOutput

from data_utils_task_a import load_task_a_splits, basic_clean_task_a


MODEL_NAME = "microsoft/codebert-base"  # same base encoder as your notebook


def prepare_task_a_datasets_bilstm(
    data_root: str = "data",
    max_length: int = 256,
) -> Dict[str, Dataset]:
    """
    Load and preprocess Task A data for the frozen CodeBERT + BiLSTM model.

    Args:
        data_root: Root directory where data/task_a is located.
        max_length: Maximum sequence length for tokenization.

    Returns:
        A dict with Hugging Face Datasets: {"train", "val", "test"}.
    """
    train_df, val_df, test_df = load_task_a_splits(data_root=data_root)

    # Clean train and validation
    train_df = basic_clean_task_a(train_df)
    val_df   = basic_clean_task_a(val_df)

    # Test only has code
    test_df = test_df[["code"]].copy()
    test_df["code"] = test_df["code"].astype(str).str.strip()

    # Convert to HF Datasets
    train_ds = Dataset.from_pandas(train_df[["code", "label"]], preserve_index=False)
    val_ds   = Dataset.from_pandas(val_df[["code", "label"]],   preserve_index=False)
    test_ds  = Dataset.from_pandas(test_df[["code"]],           preserve_index=False)

    train_ds = train_ds.rename_column("label", "labels")
    val_ds   = val_ds.rename_column("label", "labels")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.cls_token

    def tok_fn(batch):
        return tokenizer(
            batch["code"],
            truncation=True,
            max_length=max_length,
        )

    train_tok = train_ds.map(tok_fn, batched=True)
    val_tok   = val_ds.map(tok_fn,   batched=True)
    test_tok  = test_ds.map(tok_fn,  batched=True)

    # remove raw text
    train_tok = train_tok.remove_columns(["code"])
    val_tok   = val_tok.remove_columns(["code"])
    test_tok  = test_tok.remove_columns(["code"])

    # set torch format
    train_tok.set_format(type="torch")
    val_tok.set_format(type="torch")
    test_tok.set_format(type="torch")

    return {"train": train_tok, "val": val_tok, "test": test_tok}


class CodeBertBiLSTMClassifier(nn.Module):
    """
    Hybrid model: frozen CodeBERT encoder + BiLSTM + linear classifier.
    """

    def __init__(
        self,
        encoder: AutoModel,
        num_labels: int,
        lstm_hidden_size: int = 256,
        lstm_num_layers: int = 1,
        bidirectional: bool = True,
        dropout: float = 0.1,
        freeze_encoder: bool = True,
    ):
        super().__init__()

        self.encoder = encoder
        self.num_labels = num_labels

        hidden_size = encoder.config.hidden_size  # 768 for base CodeBERT

        # Optionally freeze encoder parameters
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )

        lstm_out_dim = lstm_hidden_size * (2 if bidirectional else 1)

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(lstm_out_dim, num_labels)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        **kwargs,
    ):
        # Encoder outputs: [batch, seq_len, hidden]
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        last_hidden_state = encoder_outputs.last_hidden_state  # [B, L, H]

        # BiLSTM over token sequence
        lstm_out, _ = self.lstm(last_hidden_state)  # [B, L, H_lstm]
        # Use last time step
        last_hidden = lstm_out[:, -1, :]            # [B, H_lstm]

        logits = self.classifier(self.dropout(last_hidden))

        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )


def compute_metrics(eval_pred: Any) -> Dict[str, float]:
    """
    Compute accuracy and macro-F1 for validation.
    """
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    macro_f1 = f1_score(labels, preds, average="macro")
    acc      = accuracy_score(labels, preds)

    return {"macro_f1": macro_f1, "accuracy": acc}


def train_frozen_codebert_bilstm_task_a(
    data_root: str = "data",
    output_dir: str = "models/task_a_frozen_codebert_bilstm",
    num_train_epochs: int = 2,
    batch_size: int = 16,
    learning_rate: float = 2e-4,
    max_length: int = 256,
):
    """
    Train the frozen CodeBERT + BiLSTM classifier on Subtask A.

    Args:
        data_root: Root directory where data/task_a lives.
        output_dir: Directory to save the BiLSTM head weights and predictions.
        num_train_epochs: Number of training epochs.
        batch_size: Per-device batch size for training and evaluation.
        learning_rate: Learning rate for AdamW.
        max_length: Maximum sequence length for tokenization.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Prepare datasets
    ds = prepare_task_a_datasets_bilstm(data_root=data_root, max_length=max_length)
    train_tok = ds["train"]
    val_tok   = ds["val"]
    test_tok  = ds["test"]

    # Encoder + hybrid model (encoder frozen)
    encoder = AutoModel.from_pretrained(MODEL_NAME)
    num_labels = len(pd.unique(train_tok["labels"].numpy()))

    model = CodeBertBiLSTMClassifier(
        encoder=encoder,
        num_labels=num_labels,
        lstm_hidden_size=256,
        lstm_num_layers=1,
        bidirectional=True,
        dropout=0.1,
        freeze_encoder=True,
    )
    model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.cls_token

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="no",
        logging_steps=200,
        report_to=[],  # no W&B
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Train
    trainer.train()

    # Evaluate in more detail
    eval_outputs = trainer.predict(val_tok)
    val_logits   = eval_outputs.predictions
    val_preds    = np.argmax(val_logits, axis=-1)

    # True labels from validation set
    y_val_true = np.array(val_tok["labels"], dtype=int)

    val_acc = accuracy_score(y_val_true, val_preds)
    val_f1  = f1_score(y_val_true, val_preds, average="macro")

    print(f"[Task A] Frozen CodeBERT + BiLSTM Dev Accuracy : {val_acc:.4f}")
    print(f"[Task A] Frozen CodeBERT + BiLSTM Dev Macro F1 : {val_f1:.4f}")

    print("\nConfusion matrix (val):")
    print(confusion_matrix(y_val_true, val_preds))

    print("\nClassification report (val):")
    print(classification_report(y_val_true, val_preds, digits=4))

    # Predict on test set
    test_outputs = trainer.predict(test_tok)
    test_logits  = test_outputs.predictions
    test_preds   = np.argmax(test_logits, axis=-1).astype(int)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    np.save(output_path / "val_predictions.npy", val_preds)
    np.save(output_path / "test_predictions.npy", test_preds)

    # Save only the BiLSTM head + classifier (frozen encoder can be reloaded from MODEL_NAME)
    torch.save(model.state_dict(), output_path / "codebert_bilstm_head.pt")
    print("Saved BiLSTM head weights and predictions to:", output_path)


if __name__ == "__main__":
    train_frozen_codebert_bilstm_task_a()
