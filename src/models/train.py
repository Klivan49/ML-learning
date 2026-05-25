import os
import pickle
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)
from typing import Dict, List, Optional, Tuple

from configs.config import CONFIG
from src.features.features import get_feature_columns
from src.models.model import MODEL_REGISTRY


def load_dataset(csv_path: str) -> Tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(csv_path)
    cols = get_feature_columns()
    available = [c for c in cols if c in df.columns]
    X = df[available].fillna(0)
    y = df["target_class"]
    return X, y


def train_test_val_split(
    X: pd.DataFrame, y: pd.Series,
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int = 42,
) -> Tuple:
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y,
    )
    val_relative = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_relative, random_state=random_state, stratify=y_temp,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
    y_pred = model.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision_macro": precision_score(y_test, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_test, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "classification_report": classification_report(y_test, y_pred, zero_division=0),
    }


def train_and_evaluate(
    csv_path: str,
    models_to_train: Optional[List[str]] = None,
    output_dir: str = "models",
) -> Dict[str, Dict]:
    if models_to_train is None:
        models_to_train = list(MODEL_REGISTRY.keys())

    X, y = load_dataset(csv_path)
    print(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Classes: {y.unique()}")

    X_train, X_val, X_test, y_train, y_val, y_test = train_test_val_split(
        X, y,
        test_size=CONFIG.train_test_split,
        val_size=CONFIG.val_split,
    )
    print(f"Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")

    results = {}
    os.makedirs(output_dir, exist_ok=True)

    for name in models_to_train:
        print(f"\n{'='*60}")
        print(f"Training: {name}")
        print(f"{'='*60}")

        builder = MODEL_REGISTRY[name]
        model = builder()

        model.fit(X_train, y_train)

        val_metrics = evaluate_model(model, X_val, y_val)
        test_metrics = evaluate_model(model, X_test, y_test)

        print(f"Validation F1 (macro): {val_metrics['f1_macro']:.4f}")
        print(f"Test F1 (macro): {test_metrics['f1_macro']:.4f}")
        print(f"Test accuracy: {test_metrics['accuracy']:.4f}")
        print(f"\nClassification Report (Test):\n{test_metrics['classification_report']}")

        model_path = os.path.join(output_dir, f"{name}.pkl")
        joblib.dump(model, model_path)
        print(f"Model saved to: {model_path}")

        results[name] = {
            "val": val_metrics,
            "test": test_metrics,
            "model_path": model_path,
        }

    best_model = max(results, key=lambda k: results[k]["test"]["f1_macro"])
    print(f"\n{'='*60}")
    print(f"Best model: {best_model} (F1-macro: {results[best_model]['test']['f1_macro']:.4f})")
    print(f"{'='*60}")

    return results
