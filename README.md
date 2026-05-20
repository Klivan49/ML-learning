# ОМО -- проект: Автоматическая сортировка файлов

## Описание
Кроссплатформенное приложение для автоматической сортировки файлов в папке Downloads с использованием машинного обучения (TF-IDF + Logistic Regression).

## Установка
1. Клонируйте репозиторий:
   ```bash
   git clone <repo_url>
   cd ОМО\ \--\ проект
   ```
2. Создайте и активируйте виртуальное окружение:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
3. Установите зависимости:
   ```bash
   pip install -r requirements.txt
   ```

## Быстрый старт
1. Обучите модель (или используйте готовую):
   ```bash
   python scripts/generate_dataset.py
   python scripts/train.py
   ```
2. Запустите автосортировку (лайв-режим):
   ```bash
   python src/cli.py watch --config configs/sort_config.yaml
   ```
   Или отсортируйте все файлы в папке разово:
   ```bash
   python src/cli.py sort --path <путь_к_папке> --config configs/sort_config.yaml
   ```
3. Проверьте статистику:
   ```bash
   python src/cli.py stats --path ~/Downloads
   ```
4. Классифицируйте имя файла вручную:
   ```bash
   python src/cli.py predict "example_invoice_2024.pdf"
   ```
5. Откат последнего перемещения:
   ```bash
   python src/cli.py undo
   ```

## Конфиг
В файле `configs/sort_config.yaml` можно задать папки для каждой категории:
```yaml
categories:
  Documents: ~/Downloads/Documents
  Images: ~/Downloads/Images
  Archives: ~/Downloads/Archives
  Media: ~/Downloads/Media
  Invoices: ~/Downloads/Invoices
   Projects: ~/Downloads/Projects
   Presentations: ~/Downloads/Presentations
   Study: ~/Downloads/Учеба
   Study/Lab: ~/Downloads/Учеба/Лабораторные
   Study/Practice: ~/Downloads/Учеба/Практические
   Study/Report: ~/Downloads/Учеба/Отчеты
   Study/Coursework: ~/Downloads/Учеба/Курсовые
   Study/Test: ~/Downloads/Учеба/Контрольные
   Presentations: ~/Downloads/Presentations
   Others: ~/Downloads/Others
unknown: ~/Downloads/Others
```

## Категории
- Documents, Images, Archives, Media, Invoices, Projects, Presentations, Study (+подкатегории), Others

## Откат
- Все перемещения логируются в `~/Downloads/.sorter_log.json`.
- Для отката последнего действия: `python src/cli.py undo`

## Тесты
```bash
pytest tests/
```

## Сервис/демон
- Для Linux: создайте systemd unit, запускающий `python src/cli.py watch ...`
- Для Windows: используйте планировщик задач или NSSM.

## Лицензия
MIT
