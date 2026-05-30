from dataclasses import dataclass, field
from typing import List


GENERAL = {
    "target_columns": ["documents", "images", "archives", "code", "media", "other"],
    "extension_to_category": {
        "documents": ["pdf", "doc", "docx", "xls", "xlsx", "ppt", "pptx", "odt", "ods", "odp", "rtf", "txt", "csv", "tsv"],
        "images":    ["jpg", "jpeg", "png", "gif", "bmp", "tiff", "tif", "svg", "webp", "ico", "heic", "raw"],
        "archives":  ["zip", "rar", "tar", "gz", "bz2", "xz", "7z", "zst"],
        "code":      ["py", "js", "ts", "java", "c", "cpp", "h", "hpp", "cs", "go", "rs", "rb", "php", "swift", "kt", "scala", "sh", "bash", "sql", "html", "css", "scss", "less", "xml", "json", "yaml", "yml", "toml", "ini", "cfg"],
        "media":     ["mp3", "wav", "flac", "aac", "ogg", "wma", "mp4", "avi", "mkv", "mov", "wmv", "flv", "webm"],
        "other":     ["bin", "dat", "tmp", "log", "bak", "old", "part"],
    },
    "category_keywords": {
        "documents": ["invoice", "report", "letter", "contract", "memo", "receipt", "statement", "resume", "cv", "presentation", "spreadsheet", "budget"],
        "images":    ["img_", "photo", "image", "screenshot", "pic_", "DSC", "IMG_"],
        "archives":  ["backup", "archive", "backup_", "archive_"],
        "code":      ["main.", "index.", "app.", "setup.", "utils.", "helper.", "test_", "config.", "module_"],
        "media":     ["audio_", "video_", "recording", "track_", "song_", "clip_"],
        "other":     ["misc_", "temp_", "random_", "untitled", "new_"],
    },
    "synthetic_filename_patterns": {
        "documents": ["invoice_{n}", "report_{date}", "letter_{name}", "contract_{n}", "memo_{n}", "receipt_{n}", "statement_{date}", "resume_{name}", "presentation_{topic}", "spreadsheet_{n}", "budget_{year}", "document_{n}", "notes_{date}", "agenda_{n}", "minutes_{date}"],
        "images":    ["IMG_{n}", "photo_{date}", "screenshot_{date}", "pic_{n}", "image_{n}", "DSC_{n}", "img_{date}", "snapshot_{date}", "photo_{n}", "picture_{n}"],
        "archives":  ["backup_{date}", "archive_{n}", "backup_{name}", "data_{date}", "dump_{date}", "snapshot_{date}", "backup_{n}"],
        "code":      ["main", "index", "app", "setup", "utils", "helper", "test_{n}", "config", "module_{name}", "controller", "model_{name}", "routes", "services", "api_{name}", "db_{name}"],
        "media":     ["track_{n}", "audio_{n}", "recording_{date}", "song_{name}", "clip_{n}", "video_{date}", "episode_{n}", "music_{name}", "sound_{n}", "media_{date}"],
        "other":     ["file_{n}", "data_{n}", "output_{n}", "temp_{date}", "untitled_{n}", "new_file_{n}", "misc_{n}"],
    },
    "size_ranges": {
        "documents": (1_000, 5_000_000), "images": (10_000, 20_000_000), "archives": (50_000, 500_000_000),
        "code": (50, 500_000), "media": (500_000, 200_000_000), "other": (10, 10_000_000),
    },
    "synthetic_corpora": {
        "documents": ["the", "company", "report", "annual", "meeting", "financial", "results", "quarter", "revenue", "profit", "loss", "statement", "summary", "analysis", "data", "project", "department"],
        "code": ["def", "import", "class", "return", "if", "else", "for", "while", "True", "False", "None", "lambda", "yield", "from", "as", "with"],
        "other": ["test", "data", "value", "entry", "record", "system", "info"],
    },
}

EDUCATION = {
    "target_columns": [
        "лабораторные", "практические", "методички", "курсовые", "код",
    ],
    "extension_to_category": {
        "лабораторные":  ["pdf", "doc", "docx", "xls", "xlsx", "py", "cpp", "c", "js", "ipynb", "md", "m", "pas"],
        "практические":  ["pdf", "doc", "docx", "xls", "xlsx", "txt", "md", "tex"],
        "методички":     ["pdf", "doc", "docx", "ppt", "pptx", "odp", "txt", "md"],
        "курсовые":      ["pdf", "doc", "docx", "ppt", "pptx", "zip", "rar"],
        "код":           ["py", "js", "ts", "java", "c", "cpp", "cs", "go", "rs", "rb", "php", "swift", "kt", "ipynb", "sql", "html", "css", "scss", "r"],
    },
    "category_keywords": {
        "лабораторные":  ["лаб", "laboratory", "lab_", "лабораторная", "labwork", "лр", "отчёт", "отчет", "ход_работы"],
        "практические":  ["пз", "практик", "практическая", "задани", "hw", "homework", "дз", "домашк", "семинар", "упражнен"],
        "методички":     ["метод", "method", "guide", "manual", "инструкц", "лекц", "lection", "lecture", "лекция", "тема", "теор", "пособи"],
        "курсовые":      ["курс", "coursework", "course_", "диплом", "diploma", "курсач", "курсовой"],
        "код":           ["programming", "program", "программир", "algorithm", "алгоритм", "coding", "code", "web", "database", "база_данных", "data_structure", "структур_данных", "task_", "project_"],
    },
    "synthetic_filename_patterns": {
        "лабораторные":  ["лаб_{n}_{topic}", "lab_{n}_{topic}", "лабораторная_{n}_{topic}", "labwork_{date}_{topic}", "лр_{n}", "отчёт_{n}"],
        "практические":  ["пз_{n}_{topic}", "hw_{n}_{topic}", "дз_{date}_{topic}", "практическая_{n}_{topic}", "семинар_{n}_{topic}", "задание_{n}"],
        "методички":     ["методичка_{topic}", "method_guide_{topic}", "manual_{topic}", "инструкция_{topic}", "лекция_{n}_{topic}"],
        "курсовые":      ["курсовая_{topic}", "coursework_{topic}", "диплом_{topic}", "course_project_{topic}", "курсач_{topic}"],
        "код":           ["prog_{n}", "task_{n}", "project_{topic}", "algorithm_{n}", "web_app_{n}", "database_{n}", "main", "app", "utils", "helper"],
    },
    "size_ranges": {
        "лабораторные": (5_000, 5_000_000), "практические": (5_000, 2_000_000),
        "методички": (50_000, 20_000_000), "курсовые": (100_000, 50_000_000),
        "код": (100, 1_000_000),
    },
    "synthetic_corpora": {
        "lab": ["лабораторная", "работа", "цель", "задание", "ход", "выполнения", "результат", "вывод", "отчет", "измерение", "расчет"],
        "practical": ["задание", "решение", "ответ", "вариант", "номер", "задача", "пример", "выполнил", "проверил", "оценка"],
        "method": ["лекция", "тема", "определение", "теорема", "формула", "пример", "доказательство", "свойство", "функция", "уравнение"],
        "coursework": ["курсовая", "работа", "введение", "глава", "заключение", "список", "источников", "реферат", "тема", "исследование"],
        "code": ["def", "import", "class", "return", "if", "else", "for", "while", "int", "float", "string", "print", "input", "function"],
    },
}


@dataclass
class FileSorterConfig:
    profile: str = "general"

    def __post_init__(self):
        self._apply_profile()

    def _apply_profile(self):
        data = GENERAL if self.profile == "general" else EDUCATION
        self.target_columns: List[str] = list(data["target_columns"])
        self.extension_to_category: dict = dict(data["extension_to_category"])
        self.category_keywords: dict = dict(data["category_keywords"])
        self.synthetic_filename_patterns: dict = dict(data["synthetic_filename_patterns"])
        self.size_ranges: dict = dict(data["size_ranges"])
        self.synthetic_corpora: dict = dict(data["synthetic_corpora"])

    extension_to_category: dict = field(default_factory=dict)
    category_keywords: dict = field(default_factory=dict)
    target_columns: List[str] = field(default_factory=list)
    synthetic_filename_patterns: dict = field(default_factory=dict)
    size_ranges: dict = field(default_factory=dict)
    synthetic_corpora: dict = field(default_factory=dict)

    extension_group_map: dict = field(default_factory=lambda: {
        "pdf": "doc", "doc": "doc", "docx": "doc", "txt": "doc", "csv": "doc", "rtf": "doc",
        "jpg": "img", "jpeg": "img", "png": "img", "gif": "img", "bmp": "img", "svg": "img",
        "zip": "arc", "rar": "arc", "tar": "arc", "gz": "arc", "7z": "arc",
        "py": "code", "js": "code", "ts": "code", "java": "code", "c": "code", "cpp": "code", "go": "code", "rs": "code",
        "mp3": "audio", "wav": "audio", "flac": "audio",
        "mp4": "video", "avi": "video", "mkv": "video", "mov": "video",
        "bin": "other", "dat": "other", "tmp": "other", "log": "other", "bak": "other",
    })

    text_extensions: List[str] = field(default_factory=lambda: [
        "txt", "csv", "tsv", "json", "xml", "yaml", "yml", "toml", "ini", "cfg",
        "py", "js", "ts", "java", "c", "cpp", "h", "hpp", "cs", "go", "rs", "rb",
        "php", "swift", "kt", "sh", "bash", "sql", "html", "css", "scss", "less",
        "md", "rst", "log", "env", "gitignore", "dockerfile",
    ])

    model_params: dict = field(default_factory=lambda: {
        "logistic_regression": {"C": 1.0, "max_iter": 1000, "random_state": 42},
        "random_forest": {"n_estimators": 100, "max_depth": None, "random_state": 42},
        "gradient_boosting": {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 3, "random_state": 42},
    })

    default_output_dir: str = "~/Sorted"

    train_test_split: float = 0.2
    val_split: float = 0.2


CONFIG = FileSorterConfig()
