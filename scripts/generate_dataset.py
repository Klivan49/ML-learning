#!/usr/bin/env python3
"""
Скрипт генерации датасета для классификации файлов.

Поддерживает два режима:
  1) --real PATH         — сбор признаков из реальных файлов
  2) --synthetic N       — генерация синтетического датасета (N примеров)

Примеры:
  python scripts/generate_dataset.py --real ~/Downloads --output data/processed/dataset_real.csv
  python scripts/generate_dataset.py --synthetic 5000 --output data/processed/dataset_synthetic.csv
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from configs.config import CONFIG
from src.data_prep.dataset import build_real_dataset, build_synthetic_dataset


def main():
    parser = argparse.ArgumentParser(
        description="Генерация датасета для классификации файлов"
    )
    parser.add_argument(
        "--profile", type=str, default="general", choices=["general", "education"],
        help="Профиль классификации (default: general)"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--real", type=str, metavar="PATH",
        help="Путь к корневой директории с файлами для сбора признаков"
    )
    group.add_argument(
        "--synthetic", type=int, metavar="N",
        help="Количество синтетических примеров для генерации"
    )

    parser.add_argument(
        "--output", type=str, default="data/processed/dataset.csv",
        help="Путь к выходному CSV-файлу (default: data/processed/dataset.csv)"
    )
    parser.add_argument(
        "--synthetic-dir", type=str, default="synthetic_data",
        help="Директория для временных синтетических файлов (default: synthetic_data)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Сид для воспроизводимости (default: 42)"
    )

    args = parser.parse_args()
    CONFIG.profile = args.profile
    CONFIG._apply_profile()

    if args.real:
        print(f"Collecting real files from: {args.real}")
        df = build_real_dataset(args.real, args.output)
        print(f"Dataset saved to {args.output}: {len(df)} samples, {len(df.columns)} columns")
        print(f"Class distribution:\n{df['target_class'].value_counts()}")
    else:
        print(f"Generating {args.synthetic} synthetic samples...")
        df = build_synthetic_dataset(
            num_samples=args.synthetic,
            output_csv=args.output,
            synthetic_dir=args.synthetic_dir,
            seed=args.seed,
        )
        print(f"Dataset saved to {args.output}: {len(df)} samples, {len(df.columns)} columns")
        print(f"Class distribution:\n{df['target_class'].value_counts()}")


if __name__ == "__main__":
    main()
