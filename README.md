# File Sorter

ML-приложение для автоматической сортировки файлов по категориям на основе имени, размера, расширения и содержимого.

## Постановка задачи

Задача сведена к многоклассовой классификации: по вектору признаков файла `x ∈ ℝⁿ` модель предсказывает метку класса `y ∈ {documents, images, archives, code, media, other}`. Целевая метрика — F1-score (macro). Дополнительно отслеживаются accuracy, precision, recall и матрица ошибок.

## Архитектура

```
file_sorter/
├── configs/config.py            # Категории, расширения, keywords, параметры моделей
├── data/
│   ├── raw/                     # Реальные файлы (не отсортированные)
│   └── processed/               # Готовые датасеты (.csv)
├── scripts/
│   ├── generate_dataset.py      # CLI: генерация датасета
│   ├── train_model.py           # CLI: обучение модели
│   ├── sort_files.py            # CLI: сортировка файлов
│   └── gui.py                   # GUI (ttkbootstrap)
├── src/
│   ├── features/features.py     # Извлечение признаков из файлов
│   ├── data_prep/dataset.py     # Сбор реальных / генерация синтетических данных
│   ├── models/model.py          # Определение sklearn-пайплайнов
│   ├── models/train.py          # Train/val/test, обучение, оценка, сохранение
│   └── inference/predict.py     # Загрузка модели, предсказание, сортировка
├── models/                      # Сохранённые модели (.pkl)
├── synthetic_data/              # Временные синтетические файлы
├── requirements.txt
├── gui.sh                       # Linux-лаунчер (auto-venv)
└── README.md
```

## Система признаков (48 признаков)

| Группа | Признаки | Описание |
|---|---|---|
| **Имя файла** | `filename_len`, `filename_token_count`, `filename_digit_ratio`, `filename_word_count`, `filename_has_date` | Длина, число токенов, доля цифр, кол-во слов, наличие даты |
| | `kw_{documents,images,...}` (6 шт) | Бинарные флаги ключевых слов (invoice, photo, backup, main, ...) |
| **Размер** | `size_bytes`, `log_size`, `size_kb`, `size_mb` | Абсолютные и log-преобразованные значения |
| | `size_{tiny,small,medium,large,huge}` (5 шт) | One-hot классы размера |
| **Расширение** | `ext_group_{code,doc,img,arc,audio,video,other}` (7 шт) | One-hot группа расширения |
| | `ext_cat_{documents,images,...}` (6 шт) | One-hot категория расширения |
| **Содержимое (текст)** | `text_length`, `text_word_count`, `text_line_count`, `text_unique_word_ratio`, `text_avg_word_len` | Статистики текста (для txt, py, json, ...) |
| **Содержимое (бинарное)** | `binary_entropy`, `binary_first_bytes_hash`, `binary_zero_ratio`, `binary_printable_ratio` | Энтропия байтов, хеш первых 256 байт, доли нулевых/печатных байтов |

Признаки нормализуются внутри пайплайна (StandardScaler для LogisticRegression).

## Модели

| Модель | Параметры |
|---|---|
| LogisticRegression | C=1.0, max_iter=1000, StandardScaler |
| RandomForestClassifier | n_estimators=100, max_depth=None |
| GradientBoostingClassifier | n_estimators=100, lr=0.1, max_depth=3 |

Каждая модель обучается на train/val/test (60/20/20) со стратификацией. Logistic Regression выбрана базовой линией за счёт интерпретируемости коэффициентов. Random Forest — как ансамбль, устойчивый к переобучению. Gradient Boosting — для потенциально более высокого качества на реальных данных.

## Установка

**Linux / macOS:**
```bash
pip install -r requirements.txt
```

**Windows (cmd / PowerShell):**
```cmd
pip install -r requirements.txt
```

> **Примечание:** для GUI требуется `ttkbootstrap`. Он ставится автоматически из `requirements.txt`.
> Если tkinter не установлен — на Ubuntu/deb: `sudo apt install python3-tk`, на Arch: `sudo pacman -S tk`.

## Использование (GUI)

```bash
# Установка зависимостей
pip install -r requirements.txt

# Запуск
python scripts/gui.py

# Или через лаунчер (авто-venv)
./gui.sh
```

### Вкладки GUI

| Вкладка | Назначение |
|---|---|
| **Sort** | Выбор модели, входной/выходной директорий, фильтры, сортировка |
| **Dataset** | Генерация датасета: real (из каталога) или synthetic (N примеров) |
| **Train** | Обучение одной или нескольких моделей, таблица метрик |
| **Config** | Редактирование категорий, расширений и keywords |
| **About** | Информация о проекте |

### Workflow

1. **Dataset** → создать датасет (синтетический или из реальных файлов)
2. **Train** → обучить модель(и) на датасете
3. **Sort** → выбрать модель и отсортировать файлы
4. **Config** → настроить категории под свои задачи

Файлы перемещаются в `<output>/<predicted_class>/<filename>`. При совпадении имён добавляется суффикс `_1`, `_2` и т.д.

Доступные темы (меняются в `gui.py`, строка `tb.Window(themename=...)`):
`darkly`, `superhero`, `flatly`, `lumen`, `solar`, `cyborg`, `vapor`.

## Использование (CLI)

### 1. Генерация датасета

```bash
# Синтетический (N примеров, равномерно по классам)
python scripts/generate_dataset.py --synthetic 5000 --output data/processed/dataset.csv

# Из реальной директории (путь → класс из имени папки)
python scripts/generate_dataset.py --real ~/Downloads --output data/processed/dataset_real.csv
```

### 2. Обучение

```bash
# Все модели
python scripts/train_model.py --data data/processed/dataset.csv

# Выборочно
python scripts/train_model.py --data data/processed/dataset.csv \
    --models logistic_regression random_forest
```

На выходе — `.pkl` файлы в `models/` и сводка метрик по test-выборке.

### 3. Сортировка

```bash
# Один файл
python scripts/sort_files.py --model models/random_forest.pkl \
    --input ~/file.pdf --output ~/Sorted

# Вся директория рекурсивно
python scripts/sort_files.py --model models/random_forest.pkl \
    --input ~/Downloads --output ~/Sorted

# Пробный запуск (без перемещения)
python scripts/sort_files.py --model models/random_forest.pkl \
    --input ~/Downloads --output ~/Sorted --dry-run
```

### Education profile

```bash
python scripts/generate_dataset.py --profile education --synthetic 2000 --output data/processed/dataset_edu.csv
python scripts/train_model.py --profile education --data data/processed/dataset_edu.csv
python scripts/sort_files.py --profile education --model models/gradient_boosting.pkl --input ~/Studies --output ~/SortedStudies
```

Категории: лекции, лабораторные, пз, курсовые, методички, математика, физика,
программирование, информатика, химия.

## Категории

| Категория | Примеры расширений | Ключевые слова в имени |
|---|---|---|
| documents | pdf, docx, xlsx, txt, csv, rtf | invoice, report, letter, contract, memo |
| images | jpg, png, gif, bmp, svg, webp | img_, photo, screenshot, DSC |
| archives | zip, rar, tar, gz, 7z, bz2 | backup, archive, dump |
| code | py, js, ts, java, cpp, go, rs | main., index., app., test_, config. |
| media | mp3, wav, flac, mp4, avi, mkv | audio_, video_, recording, track_ |
| other | bin, dat, tmp, log, bak | misc_, temp_, untitled |



## Развитие

- **TF-IDF по имени файла** — если имя несёт много информации (счётчики, номера версий)
- **Byte-level Transformer** — для бинарных файлов (замена энтропии + хеша)
- **AutoML** — подбор гиперпараметров (GridSearchCV/RandomizedSearch)
- **Онлайн-обучение** — дообучение на новых файлах без перезапуска пайплайна
