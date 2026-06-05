#!/usr/bin/env python3
"""
Скрипт автоматической сортировки файлов с использованием обученной модели.

Примеры:
  python scripts/sort_files.py --model models/random_forest.pkl --input ~/Downloads --output ~/Sorted
  python scripts/sort_files.py --model models/random_forest.pkl --input ~/file.pdf --output ~/Sorted
  python scripts/sort_files.py --model models/random_forest.pkl --input ~/Downloads --output ~/Sorted --dry-run
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import src._venv_setup  # noqa: F401

from configs.config import CONFIG
from src.inference.predict import FileClassifier


def main():
    parser = argparse.ArgumentParser(
        description="Автоматическая сортировка файлов с помощью ML"
    )
    parser.add_argument(
        "--profile", type=str, default="general", choices=["general", "education"],
        help="Профиль классификации (default: general)"
    )
    parser.add_argument(
        "--model", type=str, required=True,
        help="Путь к сохранённой модели (.pkl)"
    )
    parser.add_argument(
        "--input", type=str, required=True,
        help="Путь к файлу или директории для сортировки"
    )
    parser.add_argument(
        "--output", type=str, default=CONFIG.default_output_dir,
        help=f"Корневая директория для отсортированных файлов (default: {CONFIG.default_output_dir})"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Режим пробного запуска (без реального перемещения)"
    )
    parser.add_argument(
        "--no-recursive", action="store_true",
        help="Не обходить директории рекурсивно"
    )
    parser.add_argument(
        "--min-size", type=int, default=0,
        help="Минимальный размер файла в байтах (default: 0)"
    )
    parser.add_argument(
        "--max-size", type=int, default=0,
        help="Максимальный размер файла в байтах (default: без ограничений)"
    )
    parser.add_argument(
        "--extensions", type=str, nargs="*",
        help="Фильтр по расширениям (например, pdf txt jpg)"
    )

    args = parser.parse_args()

    CONFIG.profile = args.profile
    CONFIG._apply_profile()

    classifier = FileClassifier(args.model)
    input_path = os.path.abspath(args.input)
    output_path = os.path.abspath(args.output)

    if os.path.isfile(input_path):
        result = classifier.sort_file(input_path, output_path, dry_run=args.dry_run)
        print(result)
    elif os.path.isdir(input_path):
        results = classifier.sort_directory(
            input_path, output_path,
            recursive=not args.no_recursive,
            dry_run=args.dry_run,
            min_size=args.min_size,
            max_size=args.max_size,
            allowed_extensions=args.extensions,
        )
        for r in results:
            print(r)
        print(f"\nОбработано файлов: {len(results)}")
    else:
        print(f"Ошибка: {input_path} не существует")
        sys.exit(1)


if __name__ == "__main__":
    main()
