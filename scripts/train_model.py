#!/usr/bin/env python3
"""
Скрипт обучения модели классификации файлов.

Примеры:
  python scripts/train_model.py --data data/processed/dataset.csv
  python scripts/train_model.py --data data/processed/dataset.csv --models logistic_regression random_forest
  python scripts/train_model.py --data data/processed/dataset.csv --output-dir models
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import src._venv_setup  # noqa: F401

from configs.config import CONFIG
from src.models.train import train_and_evaluate
from src.models.model import MODEL_REGISTRY


def main():
    parser = argparse.ArgumentParser(
        description="Обучение модели классификации файлов"
    )
    parser.add_argument(
        "--profile", type=str, default="general", choices=["general", "education"],
        help="Профиль классификации (default: general)"
    )
    parser.add_argument(
        "--data", type=str, required=True,
        help="Путь к CSV-файлу с датасетом"
    )
    parser.add_argument(
        "--models", type=str, nargs="+",
        choices=list(MODEL_REGISTRY.keys()),
        default=list(MODEL_REGISTRY.keys()),
        help="Список моделей для обучения (default: все)"
    )
    parser.add_argument(
        "--output-dir", type=str, default="models",
        help="Директория для сохранения моделей (default: models)"
    )

    args = parser.parse_args()
    CONFIG.profile = args.profile
    CONFIG._apply_profile()

    results = train_and_evaluate(
        csv_path=args.data,
        models_to_train=args.models,
        output_dir=args.output_dir,
    )

    print("\nСводка результатов:")
    print(f"{'Model':<25} {'Val F1':<10} {'Test F1':<10} {'Test Acc':<10}")
    print("-" * 55)
    for name, res in results.items():
        vf = res["val"]["f1_macro"]
        tf = res["test"]["f1_macro"]
        ta = res["test"]["accuracy"]
        print(f"{name:<25} {vf:<10.4f} {tf:<10.4f} {ta:<10.4f}")


if __name__ == "__main__":
    main()
