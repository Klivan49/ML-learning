import os
import random
import csv
from typing import List

STUDY_SUBCATEGORIES = [
    "Lab",
    "Practice",
    "Report",
    "Coursework",
    "Test",
    "Exam",
    "Essay",
    "Assignment",
    "ЛР",
    "Лаб",
    "Лабораторная",
    "Практика",
    "Практическая",
    "Отчет",
    "Курсовая",
    "Контрольная",
    "Задание",
    "Реферат",
    "Тест",
    "Экзамен",
    "Часть",
    "Работа",
    "Домашняя",
    "Самостоятельная",
    "ЛР",
    "ЛР_",
    "ЛР-",
    "ЛР ",
    "ЛР№",
    "ЛР№",
    "ЛР1",
    "ЛР2",
    "ЛР3",
    "ЛР4",
    "ЛР5",
    "ЛР6",
    "ЛР7",
    "ЛР8",
    "ЛР9",
    "Лаб",
    "Лаб_",
    "Лаб-",
    "Лаб ",
    "Лаб№",
    "Лаб1",
    "Лаб2",
    "Лаб3",
    "Лаб4",
    "Лаб5",
    "Лаб6",
    "Лаб7",
    "Лаб8",
    "Лаб9",
    "Практикум",
    "Практическая",
    "Практическая работа",
    "Практическая_работа",
    "Практическая-работа",
    "Часть",
    "Часть_",
    "Часть-",
    "Часть ",
    "Часть1",
    "Часть2",
    "Часть3",
    "Часть4",
    "Часть5",
    "Курсовая",
    "Курсовая работа",
    "Курсовая_работа",
    "Курсовая-работа",
    "Контрольная",
    "Контрольная работа",
    "Контрольная_работа",
    "Контрольная-работа",
    "Задание",
    "Задание_",
    "Задание-",
    "Задание ",
    "Задание1",
    "Задание2",
    "Задание3",
    "Реферат",
    "Тест",
    "Экзамен",
    "Домашняя",
    "Самостоятельная",
]
BASE_CATEGORIES = ["Documents", "Study"]

# Шаблоны ТОЛЬКО для УЧЕБНЫХ файлов - четкие образовательные паттерны
STUDY_TEMPLATES = [
    "lab",
    "practice",
    "coursework",
    "test",
    "exam",
    "essay",
    "assignment",
    "lecture",
    "seminar",
    "ЛР",
    "Лаб",
    "Лабораторная",
    "Практика",
    "Практическая",
    "Практикум",
    "Отчет",
    "Курсовая",
    "Контрольная",
    "Задание",
    "Реферат",
    "Тест",
    "Экзамен",
    "Домашняя",
    "Самостоятельная",
    "Лекция",
    "Семинар",
    "Урок",
    "Часть",
    "Работа",
    "ЛР1",
    "ЛР2",
    "ЛР3",
    "ЛР4",
    "ЛР5",
    "ЛР6",
    "ЛР7",
    "ЛР8",
    "ЛР9",
    "ЛР10",
    "Лаб1",
    "Лаб2",
    "Лаб3",
    "Лаб4",
    "Лаб5",
    # Реалистичные примеры из Telegram Desktop
    "02ЛР Климович",
    "9_ЛР_МК_Мухин",
    "Лаба 1",
    "лаба 8",
    "Lab№0",
    "Исходные_данные_для_лабораторной",
    "КП-МК",
    "Курсчач МК",
    "Отчёт лабораторная",
    "ПЗ1",
    "ПЗ3",
    "ПЗ №5 ОУИС",
    "ПРАКТИЧЕСКОЕ ЗАДАНИЕ",
    "Вопросы на экзамен",
    "Реферат_Климович",
    "Титульный лист",
    "Тема 1",
    "Тема 10",
    "Тема 20",
    "МР по выполнению",
]

# Шаблоны для НЕ учебных файлов - чётко НЕ образовательные
NOT_STUDY_TEMPLATES = [
    "report_sales",
    "report_financial",
    "invoice",
    "bill",
    "payment",
    "receipt",
    "statement",
    "backup",
    "dump",
    "archive",
    "export",
    "database",
    "db",
    "photo",
    "img",
    "screenshot",
    "wallpaper",
    "picture",
    "scan",
    "image",
    "movie",
    "video",
    "song",
    "track",
    "audio",
    "clip",
    "music",
    "project",
    "draft",
    "sketch",
    "design",
    "prototype",
    "mockup",
    "license",
    "keygen",
    "crack",
    "patch",
    "update",
    "setup",
    "installer",
    "sales",
    "finance",
    "accounting",
    "hr_",
    "employees",
    "budget",
    "torrent",
    "rutracker",
    "download",
    "file",
    "data",
    "sample",
    "temp",
    "misc",
    "log",
    "config",
    "settings",
    "cache",
    "cookie",
    "session",
    # Реалистичные примеры из Telegram Desktop
    "Beautiful",
    "reversOC",
    "sklad",
    "Trendy-potrebleniya",
    "Vpsk",
    "Авторский договор",
    "Аналитика по предприятиям",
    "Государственное_регулирование",
    "Основные средства",
    "СТБ 1180",
    "Патентные исследования",
    "Приложение",
]

# Примеры шаблонов для генерации
TEMPLATES = {
    "Documents": [
        "summary",
        "notes",
        "plan",
        "doc",
        "presentation",
        "minutes",
        "manual",
        "guide",
    ],
    "Images": ["photo", "img", "screenshot", "wallpaper", "picture", "scan"],
    "Archives": ["backup", "archive", "dump", "snapshot", "export"],
    "Media": ["movie", "video", "song", "track", "audio", "clip"],
    "Invoices": ["invoice", "bill", "receipt", "payment", "statement"],
    "Projects": ["project", "draft", "sketch", "design", "prototype"],
    "Study": STUDY_TEMPLATES,
    "Others": ["misc", "temp", "file", "data", "sample"],
}

EXTENSIONS = {
    "Documents": [".pdf", ".docx", ".txt", ".pptx", ".xlsx"],
    "Images": [".jpg", ".png", ".jpeg", ".bmp", ".gif"],
    "Archives": [".zip", ".rar", ".7z", ".tar.gz"],
    "Media": [".mp4", ".mp3", ".avi", ".mov", ".wav"],
    "Invoices": [".pdf", ".xlsx"],
    "Projects": [".py", ".ipynb", ".cpp", ".js", ".java"],
    "Study": [".pdf", ".docx", ".txt", ".pptx", ".xlsx"],
    "Others": [".dat", ".bin", ".bak"],
}


def random_date():
    year = random.randint(2018, 2026)
    month = random.randint(1, 12)
    day = random.randint(1, 28)
    return f"{year}-{month:02d}-{day:02d}"


def generate_filename(category: str) -> str:
    base = random.choice(TEMPLATES[category])
    ext = random.choice(EXTENSIONS[category])
    date = random_date() if random.random() < 0.5 else ""
    version = f"_v{random.randint(1, 5)}" if random.random() < 0.3 else ""
    parts = [base, date, version]
    name = "_".join([p for p in parts if p])
    return name + ext


def generate_study_filename(base, ext):
    # Реалистичные паттерны: цифры, №, фамилии, скобки, пробелы, версии
    patterns = [
        "{base}{num}{ext}",
        "{base} {num}{ext}",
        "{base}({num}){ext}",
        "{base} №{num}{ext}",
        "{num}{base}{ext}",
        "{base}_{surname}{ext}",
        "{base} {surname}-{num}{ext}",
        "{base}_{num}_{surname}{ext}",
        "{base} {surname} {date}{ext}",
        "{base}{num}_v{ver}{ext}",
    ]
    surnames = ["Климович", "Иванов", "Петров", "Сидоров", "Smith", "Johnson", "Brown"]
    num = random.randint(1, 10)
    ver = random.randint(1, 5)
    date = random_date() if random.random() < 0.5 else ""
    surname = random.choice(surnames)
    pattern = random.choice(patterns)
    return pattern.format(
        base=base, num=num, ver=ver, date=date, surname=surname, ext=ext
    )


def generate_not_study_filename(base, ext):
    patterns = [
        "{base}{num}",
        "{base} {num}",
        "{base}({num})",
        "{base}_{date}",
        "{base}_{num}_{date}",
        "{base} - {date}",
        "{base}_{year}",
    ]
    num = random.randint(1, 100)
    date = random_date()
    year = random.randint(2020, 2026)
    pattern = random.choice(patterns)
    return pattern.format(base=base, num=num, date=date, year=year) + ext


def generate_dataset(n: int = 5000, out_path: str = None):
    # 50% Study, 50% Not Study (четкое разделение)
    study_share = 0.5
    n_study = int(n * study_share)
    n_not_study = n - n_study
    rows: List[List[str]] = []
    # Генерация Study - чёткие образовательные паттерны (много примеров)
    for _ in range(n_study):
        base = random.choice(STUDY_TEMPLATES)
        ext = random.choice(EXTENSIONS["Study"])
        name = generate_study_filename(base, ext)
        rows.append([name, "Study"])
        # Дублируем с разными расширениями для лучшего обучения
        if random.random() < 0.3:
            ext2 = random.choice([".docx", ".pdf", ".xlsx"])
            name2 = generate_study_filename(base, ext2)
            rows.append([name2, "Study"])
    # Генерация Not Study - чётко НЕ образовательные паттерны
    for _ in range(n_not_study):
        base = random.choice(NOT_STUDY_TEMPLATES)
        ext = random.choice(EXTENSIONS["Documents"])
        name = generate_not_study_filename(base, ext)
        rows.append([name, "NotStudy"])
    random.shuffle(rows)
    if out_path:
        with open(out_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "category"])
            writer.writerows(rows)
    return rows


if __name__ == "__main__":
    generate_dataset(
        2000, os.path.join(os.path.dirname(__file__), "../data/dataset.csv")
    )
    print("Synthetic dataset generated.")
