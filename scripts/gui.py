#!/usr/bin/env python3
import os
import sys
import shutil
import threading

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import src._venv_setup  # noqa: F401

from configs.config import CONFIG
from src.inference.predict import FileClassifier
from src.data_prep.dataset import build_real_dataset, build_synthetic_dataset
from src.models.train import train_and_evaluate
from src.models.model import MODEL_REGISTRY

import tkinter as tk
from tkinter import filedialog, ttk

try:
    import ttkbootstrap as tb
    from ttkbootstrap.constants import *
    from ttkbootstrap.dialogs import Messagebox
    from ttkbootstrap.widgets.scrolled import ScrolledText
    THEMED = True
except ImportError:
    print("ttkbootstrap not installed. Run: pip install ttkbootstrap")
    print("Or install the full stack: pip install -r requirements.txt")
    sys.exit(1)


_LANG = "ru"

_TRANSLATIONS = {
    "📂  File Sorter — ML Classification": "📂  Сортировщик файлов — ML Классификация",
    "Sort": "Сортировка",
    "Dataset": "Датасет",
    "Train": "Обучение",
    "Config": "Настройки",
    "About": "О программе",
    "Model": "Модель",
    "Input": "Вход",
    "Output": "Выход",
    "Browse": "Обзор",
    "Options": "Опции",
    "Recursive": "Рекурсивно",
    "Dry Run": "Тестовый режим",
    "Copy": "Копировать",
    "Flag Suspicious": "Помечать подозрительные",
    "Profile:": "Профиль:",
    "Filters": "Фильтры",
    "Min size:": "Мин. размер:",
    "Max size:": "Макс. размер:",
    "Extensions:": "Расширения:",
    "space-sep": "через пробел",
    "▶  Run Sort": "▶  Сортировать",
    "⏹  Stop": "⏹  Стоп",
    "Generation Mode": "Режим генерации",
    "Real (from directory)": "Реальный (из папки)",
    "Synthetic": "Синтетический",
    "Input dir:": "Папка входа:",
    "Output CSV:": "CSV на выход:",
    "Samples:": "Примеров:",
    "Seed:": "Сид:",
    "📊  Generate Dataset": "📊  Сгенерировать датасет",
    "Dataset CSV:": "CSV датасета:",
    "Models": "Модели",
    "Output dir:": "Папка выхода:",
    "▶  Train Models": "▶  Обучить модели",
    "Results": "Результаты",
    "💾  Save": "💾  Сохранить",
    "↺  Reload": "↺  Перезагрузить",
    "Suspicious only": "Только подозрительные",
    "suspicious": "подозрительных",
    "Log": "Лог",
    "Idle": "Ожидание",
    "Sorting...": "Сортировка...",
    "Generating dataset...": "Генерация датасета...",
    "Training...": "Обучение...",
    "Error": "Ошибка",
    "Saved": "Сохранено",
    "Running": "Выполнение",
    "Select model": "Выберите модель",
    "Select input directory": "Выберите папку с файлами",
    "Select output directory": "Выберите папку для результатов",
    "Models output directory": "Папка для моделей",
    "Select dataset": "Выберите датасет",
    "Output CSV": "CSV на выход",
    "Model not found:": "Модель не найдена:",
    "Input not found:": "Вход не найден:",
    "Dataset not found:": "Датасет не найден:",
    "Output path required": "Укажите путь для выхода",
    "Valid input directory required": "Укажите существующую папку",
    "Select at least one model": "Выберите хотя бы одну модель",
    "Min/max size must be integers": "Мин./макс. размер — целые числа",
    "Configuration saved (runtime). Restart to persist.": "Конфигурация сохранена (в runtime). Перезапустите для сохранения.",
    "Operation in progress. Quit anyway?": "Выполняется операция. Выйти?",
    "Model:": "Модель:",
    "Input:": "Вход:",
    "Output:": "Выход:",
    "Filters:": "Фильтры:",
    "Mode:": "Режим:",
    "Starting…": "Запуск…",
    "Stop requested (after current file)": "Остановка (после текущего файла)",
    "Generating synthetic dataset…": "Генерация синтетического датасета…",
    "Extracting features from real files…": "Извлечение признаков из реальных файлов…",
    "copy": "копирование",
    "move": "перемещение",
    "dry-run": "тест",
    "live": "реальный",
    "recursive": "рекурсивно",
    "flat": "плоско",
    "all": "все",
    "[DRY RUN] Would copy": "[ТЕСТ] Будет скопирован",
    "Copied:": "Скопировано:",
    "Moved:": "Перемещён:",
    "Model Path": "Путь модели",
    "Val F1": "Val F1",
    "Test F1": "Test F1",
    "Test Acc": "Test Acc",
    "Profiles match dataset generation": "Профиль влияет на генерацию датасета",
    "Language:": "Язык:",
    "Language changed. Restart to apply fully.": "Язык изменён. Перезапустите для полного применения.",
    "File Sorter — ML Classification": "Сортировщик файлов — ML Классификация",
    "Categories:": "Категории:",
    "Models: Logistic Regression, Random Forest, Gradient Boosting": "Модели: Logistic Regression, Random Forest, Gradient Boosting",
    "Features: {} (numeric: {}, text: TF‑IDF)": "Признаков: {} (числовых: {}, текст: TF‑IDF)",
    "Workflow in GUI:": "Работа в GUI:",
    "  1. «Dataset» tab — create dataset (real files or synthetic)": "  1. Вкладка «Датасет» — создайте датасет (реальные файлы или синтетика)",
    "  2. «Train» tab — train one or more models": "  2. Вкладка «Обучение» — обучите одну или несколько моделей",
    "  3. «Sort» tab — select model and sort files": "  3. Вкладка «Сортировка» — выберите модель и сортируйте файлы",
    "  4. «Config» tab — configure categories and keywords": "  4. Вкладка «Настройки» — настройте категории и ключевые слова",
    "Also available via CLI:": "Также доступен CLI:",
    "  python scripts/generate_dataset.py --help": "  python scripts/generate_dataset.py --help",
    "  python scripts/train_model.py --help": "  python scripts/train_model.py --help",
    "  python scripts/sort_files.py --help": "  python scripts/sort_files.py --help",
    "Run GUI: python scripts/gui.py  or  ./gui.sh": "Запуск GUI: python scripts/gui.py  или  ./gui.sh",
}

def _(text):
    if _LANG != "ru":
        return text
    return _TRANSLATIONS.get(text, text)


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class FileSorterGUI:
    def __init__(self):
        self.root = tb.Window(themename="darkly") if THEMED else tk.Tk()
        self.root.title(_("\U0001F4C2  File Sorter — ML Classification"))
        self.root.geometry("950x700+100+100")
        self.root.minsize(850, 650)

        self._model_path = tb.StringVar(value=os.path.join(ROOT, "models/random_forest.pkl"))
        self._input_path = tb.StringVar()
        self._output_path = tb.StringVar(value=os.path.expanduser(CONFIG.default_output_dir))
        self._profile = tb.StringVar(value="general")
        self._recursive = tb.BooleanVar(value=True)
        self._dry_run = tb.BooleanVar(value=False)
        self._copy_mode = tb.BooleanVar(value=False)
        self._min_size = tb.StringVar(value="0")
        self._max_size = tb.StringVar(value="0")
        self._extension_var = tb.StringVar()
        self._flag_suspicious = tb.BooleanVar(value=True)

        self._ds_mode = tb.StringVar(value="real")
        self._ds_input_dir = tb.StringVar()
        self._ds_output_csv = tb.StringVar(value="data/processed/dataset.csv")
        self._ds_synthetic_count = tb.IntVar(value=2000)
        self._ds_seed = tb.IntVar(value=42)

        self._tr_csv_path = tb.StringVar(value="data/processed/dataset.csv")
        self._tr_output_dir = tb.StringVar(value="models")
        self._tr_models = {name: tb.BooleanVar(value=True) for name in MODEL_REGISTRY}

        self._log_lock = threading.Lock()
        self._running = False
        self._nav_buttons = {}
        self._log_buffer = []
        self._suspicious_only = tb.BooleanVar(value=False)
        self._suspicious_only.trace_add("write", lambda *_: self._reapply_log_filter())

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._show_page("sort")

    def _nav_click(self, page):
        for name, btn in self._nav_buttons.items():
            btn.configure(style="nav.TButton" if name != page else "nav-active.TButton")
        self._show_page(page)

    def _on_lang_change(self, *args):
        global _LANG
        lang = self._lang_var.get()
        _LANG = lang
        msg = _("Language changed. Restart to apply fully.")
        Messagebox.show_info(msg, parent=self.root)

    def _build_ui(self):
        outer = tb.Frame(self.root)
        outer.pack(fill=BOTH, expand=True)

        self._navbar = tb.Frame(outer, width=160, padding=10)
        self._navbar.pack(side=LEFT, fill=Y)
        self._navbar.pack_propagate(False)

        tb.Label(self._navbar, text=_("File Sorter"), font=("Segoe UI", 14, "bold")).pack(anchor=W, pady=(0, 20))

        pages = [
            ("sort",    "\U0001F500", _("Sort")),
            ("dataset", "\U0001F4CA", _("Dataset")),
            ("train",   "\U00002699", _("Train")),
            ("config",  "\U0001F6E0", _("Config")),
            ("about",   "\U00002139", _("About")),
        ]
        for name, icon, label in pages:
            btn = tb.Button(self._navbar, text=f"{icon}  {label}", style="nav.TButton",
                            command=lambda n=name: self._nav_click(n))
            btn.pack(fill=X, pady=2)
            self._nav_buttons[name] = btn

        if THEMED:
            self.root.style.configure("nav.TButton", font=("Segoe UI", 11), padding=8, anchor=W, borderwidth=0)
            self.root.style.configure("nav-active.TButton", font=("Segoe UI", 11, "bold"), padding=8, anchor=W)
            self.root.style.map("nav-active.TButton", background=[("active", self.root.style.colors.primary)])

        sep = tb.Separator(self._navbar)
        sep.pack(fill=X, pady=15)

        self._status_label = tb.Label(self._navbar, text=_("Idle"), font=("Segoe UI", 9))
        self._status_label.pack(anchor=W, pady=(0, 5))

        self._progress = tb.Progressbar(self._navbar, mode="indeterminate")
        self._progress.pack(fill=X)

        lang_frame = tb.Frame(self._navbar)
        lang_frame.pack(fill=X, pady=(15, 0))
        tb.Label(lang_frame, text=_("Language:"), font=("Segoe UI", 9)).pack(anchor=W)
        self._lang_var = tb.StringVar(value=_LANG)
        lang_combo = tb.Combobox(lang_frame, textvariable=self._lang_var,
                                 values=["en", "ru"], state="readonly", width=6)
        lang_combo.pack(anchor=W, pady=(2, 0))
        self._lang_var.trace_add("write", self._on_lang_change)

        self._container = tb.Frame(outer)
        self._container.pack(side=RIGHT, fill=BOTH, expand=True, padx=(0, 0), pady=(0, 0))

        frames = ["sort", "dataset", "train", "config", "about"]
        self._frames = {name: tb.Frame(self._container) for name in frames}
        self._log_frame = tb.LabelFrame(self._container, text=_("Log"))

        self._build_sort_tab()
        self._build_dataset_tab()
        self._build_train_tab()
        self._build_config_tab()
        self._build_about_tab()
        self._build_log()

    # ============================================================ SORT TAB
    def _build_sort_tab(self):
        f = self._frames["sort"]
        fields = [(self._model_path, "model"), (self._input_path, "input"), (self._output_path, "output")]
        for i, (txt, (w, key)) in enumerate([
            (_("Model"),  fields[0]),
            (_("Input"),  fields[1]),
            (_("Output"), fields[2]),
        ]):
            row = tb.Frame(f)
            row.pack(fill=X, pady=4)
            tb.Label(row, text=txt, width=max(7, len(txt)), anchor=E).pack(side=LEFT, padx=(0, 8))
            entry = tb.Entry(row, textvariable=w)
            entry.pack(side=LEFT, fill=X, expand=True, padx=(0, 6))
            tb.Button(row, text="Browse", command=lambda k=key: self._browse_sort(k), width=8).pack(side=LEFT)

        opt_frame = tb.LabelFrame(f, text=_("Options"))
        opt_frame.pack(fill=X, pady=(8, 0))

        row1 = tb.Frame(opt_frame)
        row1.pack(fill=X)
        tb.Checkbutton(row1, text=_("Recursive"),       variable=self._recursive,       bootstyle="round-toggle").pack(side=LEFT, padx=4)
        tb.Checkbutton(row1, text=_("Dry Run"),         variable=self._dry_run,         bootstyle="round-toggle").pack(side=LEFT, padx=4)
        tb.Checkbutton(row1, text=_("Copy"),            variable=self._copy_mode,       bootstyle="round-toggle").pack(side=LEFT, padx=4)
        tb.Checkbutton(row1, text=_("Flag Suspicious"), variable=self._flag_suspicious, bootstyle="round-toggle").pack(side=LEFT, padx=4)

        tb.Label(row1, text=_("Profile:"), font=("", 9, "bold")).pack(side=LEFT, padx=(20, 4))
        profile_combo = tb.Combobox(row1, textvariable=self._profile, values=["general", "education"],
                                     state="readonly", width=12)
        profile_combo.pack(side=LEFT)
        self._profile.trace_add("write", lambda *_: self._rebuild_config_tab())

        flt_frame = tb.LabelFrame(f, text=_("Filters"))
        flt_frame.pack(fill=X, pady=(8, 0))

        row2 = tb.Frame(flt_frame)
        row2.pack(fill=X, pady=2)
        tb.Label(row2, text=_("Min size:")).pack(side=LEFT)
        tb.Entry(row2, textvariable=self._min_size, width=10).pack(side=LEFT, padx=4)
        tb.Label(row2, text=_("Max size:")).pack(side=LEFT, padx=(15, 0))
        tb.Entry(row2, textvariable=self._max_size, width=10).pack(side=LEFT, padx=4)
        tb.Label(row2, text=_("Extensions:")).pack(side=LEFT, padx=(15, 0))
        tb.Entry(row2, textvariable=self._extension_var, width=20).pack(side=LEFT, fill=X, expand=True, padx=4)
        tb.Label(row2, text=_("space-sep"), font=("", 8)).pack(side=LEFT)

        btn_frame = tb.Frame(f)
        btn_frame.pack(fill=X, pady=(14, 0))
        tb.Button(btn_frame, text=_("\u25B6  Run Sort"), bootstyle="success", command=self._run_sort, width=18).pack(side=LEFT, padx=(0, 6))
        tb.Button(btn_frame, text=_("\u23F9  Stop"),     bootstyle="secondary", command=self._stop_sort, width=12).pack(side=LEFT)

    # ============================================================ DATASET TAB
    def _build_dataset_tab(self):
        f = self._frames["dataset"]

        mode_frame = tb.LabelFrame(f, text=_("Generation Mode"))
        mode_frame.pack(fill=X, pady=(0, 8))

        rb_frame = tb.Frame(mode_frame)
        rb_frame.pack(fill=X, pady=4)
        tb.Radiobutton(rb_frame, text=_("Real (from directory)"), variable=self._ds_mode, value="real",
                       command=self._toggle_ds_mode).pack(side=LEFT, padx=(0, 20))
        tb.Radiobutton(rb_frame, text=_("Synthetic"), variable=self._ds_mode, value="synthetic",
                       command=self._toggle_ds_mode).pack(side=LEFT)

        self._ds_real_frame = tb.Frame(mode_frame)
        self._ds_real_frame.pack(fill=X, pady=4)
        tb.Label(self._ds_real_frame, text=_("Input dir:"), width=10, anchor=E).pack(side=LEFT, padx=(0, 8))
        tb.Entry(self._ds_real_frame, textvariable=self._ds_input_dir).pack(side=LEFT, fill=X, expand=True, padx=(0, 6))
        tb.Button(self._ds_real_frame, text=_("Browse"), command=lambda: self._browse_ds("input"), width=8).pack(side=LEFT)

        self._ds_syn_frame = tb.Frame(mode_frame)
        row = tb.Frame(self._ds_syn_frame)
        row.pack(fill=X, pady=2)
        tb.Label(row, text=_("Samples:"), width=10, anchor=E).pack(side=LEFT, padx=(0, 8))
        tb.Spinbox(row, from_=100, to=100000, textvariable=self._ds_synthetic_count, width=10).pack(side=LEFT, padx=(0, 20))
        tb.Label(row, text=_("Seed:"), anchor=E).pack(side=LEFT, padx=(0, 4))
        tb.Entry(row, textvariable=self._ds_seed, width=8).pack(side=LEFT)

        profile_frame = tb.Frame(f)
        profile_frame.pack(fill=X, pady=4)
        tb.Label(profile_frame, text=_("Profile:"), font=("", 9, "bold")).pack(side=LEFT, padx=(0, 4))
        tb.Combobox(profile_frame, textvariable=self._profile, values=["general", "education"],
                     state="readonly", width=12).pack(side=LEFT)

        out_frame = tb.Frame(f)
        out_frame.pack(fill=X, pady=4)
        tb.Label(out_frame, text=_("Output CSV:"), width=10, anchor=E).pack(side=LEFT, padx=(0, 8))
        tb.Entry(out_frame, textvariable=self._ds_output_csv).pack(side=LEFT, fill=X, expand=True, padx=(0, 6))
        tb.Button(out_frame, text=_("Browse"), command=lambda: self._browse_ds("output"), width=8).pack(side=LEFT)

        btn_frame = tb.Frame(f)
        btn_frame.pack(fill=X, pady=(14, 0))
        tb.Button(btn_frame, text=_("\U0001F4CA  Generate Dataset"), bootstyle="success",
                  command=self._run_generate_dataset, width=22).pack(side=LEFT)

        self._toggle_ds_mode()

    def _toggle_ds_mode(self):
        mode = self._ds_mode.get()
        self._ds_real_frame.pack_forget()
        self._ds_syn_frame.pack_forget()
        if mode == "real":
            self._ds_real_frame.pack(fill=X, pady=4)
        else:
            self._ds_syn_frame.pack(fill=X, pady=4)

    # ============================================================ TRAIN TAB
    def _build_train_tab(self):
        f = self._frames["train"]

        row0 = tb.Frame(f)
        row0.pack(fill=X, pady=4)
        tb.Label(row0, text=_("Dataset CSV:"), width=12, anchor=E).pack(side=LEFT, padx=(0, 8))
        tb.Entry(row0, textvariable=self._tr_csv_path).pack(side=LEFT, fill=X, expand=True, padx=(0, 6))
        tb.Button(row0, text=_("Browse"), command=lambda: self._browse("tr_csv"), width=8).pack(side=LEFT)

        model_frame = tb.LabelFrame(f, text=_("Models"))
        model_frame.pack(fill=X, pady=8)
        row_m = tb.Frame(model_frame)
        row_m.pack(fill=X, pady=4)
        names = {
            "logistic_regression": "Logistic Regression",
            "random_forest": "Random Forest",
            "gradient_boosting": "Gradient Boosting",
        }
        for key, label in names.items():
            tb.Checkbutton(row_m, text=label, variable=self._tr_models[key],
                           bootstyle="round-toggle").pack(side=LEFT, padx=6)

        row1 = tb.Frame(f)
        row1.pack(fill=X, pady=4)
        tb.Label(row1, text=_("Output dir:"), width=12, anchor=E).pack(side=LEFT, padx=(0, 8))
        tb.Entry(row1, textvariable=self._tr_output_dir).pack(side=LEFT, fill=X, expand=True, padx=(0, 6))
        tb.Button(row1, text=_("Browse"), command=lambda: self._browse("tr_out"), width=8).pack(side=LEFT)

        btn_frame = tb.Frame(f)
        btn_frame.pack(fill=X, pady=(14, 0))
        tb.Button(btn_frame, text=_("\u25B6  Train Models"), bootstyle="success",
                  command=self._run_train_model, width=22).pack(side=LEFT)

        results_frame = tb.LabelFrame(f, text=_("Results"))
        results_frame.pack(fill=BOTH, expand=True, pady=(8, 0))

        columns = ("model", "val_f1", "test_f1", "test_acc", "path")
        self._tr_tree = ttk.Treeview(results_frame, columns=columns, show="headings", height=6)
        self._tr_tree.heading("model", text=_("Model"))
        self._tr_tree.heading("val_f1", text=_("Val F1"))
        self._tr_tree.heading("test_f1", text=_("Test F1"))
        self._tr_tree.heading("test_acc", text=_("Test Acc"))
        self._tr_tree.heading("path", text=_("Model Path"))
        self._tr_tree.column("model", width=180)
        self._tr_tree.column("val_f1", width=80)
        self._tr_tree.column("test_f1", width=80)
        self._tr_tree.column("test_acc", width=80)
        self._tr_tree.column("path", width=250)
        self._tr_tree.pack(fill=BOTH, expand=True)

    # ============================================================ CONFIG TAB
    def _build_config_tab(self):
        f = self._frames["config"]
        self._config_canvas = tb.Canvas(f, borderwidth=0, highlightthickness=0)
        self._config_scrollbar = tb.Scrollbar(f, orient=VERTICAL, command=self._config_canvas.yview)
        self._config_inner = tb.Frame(self._config_canvas)
        self._config_inner.bind("<Configure>", lambda e: self._config_canvas.configure(scrollregion=self._config_canvas.bbox("all")))
        self._config_canvas.create_window((0, 0), window=self._config_inner, anchor=NW)
        self._config_canvas.configure(yscrollcommand=self._config_scrollbar.set)
        self._config_canvas.pack(side=LEFT, fill=BOTH, expand=True)
        self._config_scrollbar.pack(side=RIGHT, fill=Y)
        self._bind_canvas_scroll(self._config_canvas)
        self._populate_config()

    @staticmethod
    def _bind_canvas_scroll(canvas):
        canvas.bind("<Enter>", lambda e: canvas.bind_all("<MouseWheel>", lambda e: canvas.yview_scroll(int(-1 * e.delta / 120), "units")))
        canvas.bind("<Leave>", lambda e: canvas.unbind_all("<MouseWheel>"))
        canvas.bind("<Button-4>", lambda e: canvas.yview_scroll(-2, "units"))
        canvas.bind("<Button-5>", lambda e: canvas.yview_scroll(2, "units"))

    def _rebuild_config_tab(self):
        profile = self._profile.get()
        CONFIG.profile = profile
        CONFIG._apply_profile()
        for w in self._config_inner.winfo_children():
            w.destroy()
        self._populate_config()

    def _populate_config(self):
        self._config_widgets = {}
        for i, cat in enumerate(CONFIG.target_columns):
            card = tb.LabelFrame(self._config_inner, text=f"  {cat}  ")
            card.grid(row=i, column=0, sticky=EW, pady=4, padx=2)
            self._config_inner.columnconfigure(0, weight=1)
            exts = ", ".join(CONFIG.extension_to_category.get(cat, []))
            kws = ", ".join(CONFIG.category_keywords.get(cat, []))
            tb.Label(card, text="Extensions:", font=("", 9, "bold")).grid(row=0, column=0, sticky=NW, padx=(0, 4))
            e_text = tb.Text(card, height=2, wrap=WORD, width=55)
            e_text.insert(END, exts)
            e_text.grid(row=0, column=1, sticky=EW, pady=1)
            card.columnconfigure(1, weight=1)
            tb.Label(card, text="Keywords:", font=("", 9, "bold")).grid(row=1, column=0, sticky=NW, padx=(0, 4))
            k_text = tb.Text(card, height=2, wrap=WORD, width=55)
            k_text.insert(END, kws)
            k_text.grid(row=1, column=1, sticky=EW, pady=1)
            self._config_widgets[cat] = (e_text, k_text)
        btn_row = tb.Frame(self._config_inner)
        btn_row.grid(row=len(CONFIG.target_columns), column=0, pady=12)
        tb.Button(btn_row, text=_("\U0001F4BE  Save"), bootstyle="success", command=self._save_config).pack(side=LEFT, padx=3)
        tb.Button(btn_row, text=_("\u21BA  Reload"), bootstyle="secondary", command=self._reload_config).pack(side=LEFT, padx=3)

    # ============================================================ ABOUT TAB
    def _build_about_tab(self):
        from src.features.features import FEATURE_COLUMNS, TEXT_FEATURE_COLUMNS
        n_feats = len(FEATURE_COLUMNS)
        n_text = len(TEXT_FEATURE_COLUMNS)
        categories = ", ".join(CONFIG.target_columns)
        lines = [
            "\U0001F4C2  " + _("File Sorter — ML Classification"),
            "",
            _("Categories:") + " " + categories,
            _("Models: Logistic Regression, Random Forest, Gradient Boosting"),
            _("Features: {} (numeric: {}, text: TF‑IDF)").format(n_feats, n_feats - n_text),
            "",
            _("Workflow in GUI:"),
            _("  1. «Dataset» tab — create dataset (real files or synthetic)"),
            _("  2. «Train» tab — train one or more models"),
            _("  3. «Sort» tab — select model and sort files"),
            _("  4. «Config» tab — configure categories and keywords"),
            "",
            _("Also available via CLI:"),
            _("  python scripts/generate_dataset.py --help"),
            _("  python scripts/train_model.py --help"),
            _("  python scripts/sort_files.py --help"),
            "",
            _("Run GUI: python scripts/gui.py  or  ./gui.sh"),
        ]
        txt = "\n".join(lines) + "\n"
        text = tb.Text(self._frames["about"], wrap=WORD, font=("Segoe UI", 11), padx=15, pady=15, state=NORMAL)
        text.insert(END, txt)
        text.configure(state=DISABLED)
        text.pack(fill=BOTH, expand=True)

    # ============================================================ LOG
    def _build_log(self):
        self._log_frame.pack(fill=BOTH, expand=True)
        toolbar = tb.Frame(self._log_frame)
        toolbar.pack(fill=X)
        tb.Checkbutton(toolbar, text=_("Suspicious only"), variable=self._suspicious_only,
                       bootstyle="round-toggle").pack(side=LEFT, padx=4)
        tb.Label(toolbar, text=f"  {len([m for m in self._log_buffer if 'ПОДОЗРИТЕЛЬНЫЙ' in m])} {_('suspicious')}",
                 font=("", 9), bootstyle="secondary").pack(side=LEFT, padx=2)
        self._sus_count_label = toolbar.winfo_children()[-1]
        st = ScrolledText(self._log_frame, height=8, wrap=WORD,
                          font=("Consolas", 9), autohide=True)
        st.pack(fill=BOTH, expand=True)
        st.text.configure(state=DISABLED)
        self._log_text = st

    def _show_page(self, name):
        for frame in self._frames.values():
            frame.pack_forget()
        self._log_frame.pack(fill=BOTH, expand=True, pady=(10, 0))
        self._frames[name].pack(fill=BOTH, expand=(name in ("train", "config", "about")))

    # ============================================================ BROWSE
    def _browse_sort(self, key):
        if key == "model":
            p = filedialog.askopenfilename(title=_("Select model"), filetypes=[("PKL", "*.pkl"), ("All", "*.*")])
            if p:
                self._model_path.set(p)
        elif key == "input":
            p = filedialog.askdirectory(title=_("Select input directory"))
            if p:
                self._input_path.set(p)
        else:
            p = filedialog.askdirectory(title=_("Select output directory"))
            if p:
                self._output_path.set(p)

    def _browse_ds(self, target):
        if target == "input":
            p = filedialog.askdirectory(title=_("Select input directory"))
            if p:
                self._ds_input_dir.set(p)
        else:
            p = filedialog.asksaveasfilename(title=_("Output CSV"), defaultextension=".csv",
                                             filetypes=[("CSV", "*.csv")])
            if p:
                self._ds_output_csv.set(p)

    def _browse(self, target):
        if target == "tr_csv":
            p = filedialog.askopenfilename(title=_("Select dataset"), filetypes=[("CSV", "*.csv")])
            if p:
                self._tr_csv_path.set(p)
        elif target == "tr_out":
            p = filedialog.askdirectory(title=_("Models output directory"))
            if p:
                self._tr_output_dir.set(p)

    # ============================================================ CONFIG SAVE/RELOAD
    def _save_config(self):
        try:
            for cat, (e_w, k_w) in self._config_widgets.items():
                exts = [x.strip().lower() for x in e_w.get("1.0", END).strip().split(",") if x.strip()]
                kws = [x.strip().lower() for x in k_w.get("1.0", END).strip().split(",") if x.strip()]
                CONFIG.extension_to_category[cat] = exts
                CONFIG.category_keywords[cat] = kws
            Messagebox.show_info(_("Configuration saved (runtime). Restart to persist."), _("Saved"), parent=self.root)
        except Exception as e:
            Messagebox.show_error(str(e), _("Error"), parent=self.root)

    def _reload_config(self):
        for cat, (e_w, k_w) in self._config_widgets.items():
            for w, src in [(e_w, CONFIG.extension_to_category), (k_w, CONFIG.category_keywords)]:
                w.delete("1.0", END)
                w.insert(END, ", ".join(src.get(cat, [])))

    # ============================================================ LOG
    def _log(self, msg):
        self._log_buffer.append(msg)
        self.root.after(0, self._append_log, msg)

    def _append_log(self, msg):
        if self._suspicious_only.get() and "ПОДОЗРИТЕЛЬНЫЙ" not in msg:
            return
        txt = self._log_text.text
        txt.configure(state=NORMAL)
        txt.insert(END, msg + "\n")
        txt.see(END)
        txt.configure(state=DISABLED)

    def _reapply_log_filter(self):
        txt = self._log_text.text
        txt.configure(state=NORMAL)
        txt.delete("1.0", END)
        show_all = not self._suspicious_only.get()
        for m in self._log_buffer:
            if show_all or "ПОДОЗРИТЕЛЬНЫЙ" in m:
                txt.insert(END, m + "\n")
        txt.see(END)
        txt.configure(state=DISABLED)
        sus_count = len([m for m in self._log_buffer if "ПОДОЗРИТЕЛЬНЫЙ" in m])
        self._sus_count_label.configure(text=f"  {sus_count} {_('suspicious')}")

    # ============================================================ SORT
    def _run_sort(self):
        if self._running:
            return
        mp = self._model_path.get().strip()
        ip = self._input_path.get().strip()
        op = self._output_path.get().strip()

        if not os.path.isfile(mp):
            Messagebox.show_error(_("Model not found:") + "\n" + mp, _("Error"), parent=self.root)
            return
        if not os.path.exists(ip):
            Messagebox.show_error(_("Input not found:") + "\n" + ip, _("Error"), parent=self.root)
            return
        if not op:
            Messagebox.show_error(_("Output path required"), _("Error"), parent=self.root)
            return

        try:
            mn, mx = int(self._min_size.get()), int(self._max_size.get())
        except ValueError:
            Messagebox.show_error(_("Min/max size must be integers"), _("Error"), parent=self.root)
            return

        ext_filter = None
        if self._extension_var.get().strip():
            ext_filter = [e.strip().lower().lstrip(".") for e in self._extension_var.get().strip().split()]

        CONFIG.profile = self._profile.get()
        CONFIG._apply_profile()

        self._running = True
        self._status_label.configure(text=_("Sorting..."))
        self._progress.start(10)

        self._log("\u2500" * 55)
        self._log(_("Model:") + f"   {mp}")
        self._log(_("Input:") + f"   {ip}")
        self._log(_("Output:") + f"  {op}")
        self._log(_("Profile:") + f" {CONFIG.profile}")
        self._log(_("Filters:") + f" min={mn}, max={mx}, ext={ext_filter or 'all'}")
        mode_parts = []
        mode_parts.append(_("copy") if self._copy_mode.get() else _("move"))
        mode_parts.append(_("dry-run") if self._dry_run.get() else _("live"))
        mode_parts.append(_("recursive") if self._recursive.get() else _("flat"))
        self._log(_("Mode:") + f"    {' | '.join(mode_parts)}")
        self._log(_("Starting\u2026"))

        def task():
            try:
                clf = FileClassifier(mp)
                if self._copy_mode.get():
                    clf.move_file = lambda fp, td, dr: self._copy_file(fp, td, dr)
                if os.path.isfile(ip):
                    self._log(clf.sort_file(ip, op, dry_run=self._dry_run.get()))
                else:
                    results = clf.sort_directory(ip, op,
                        recursive=self._recursive.get(),
                        dry_run=self._dry_run.get(),
                        min_size=mn, max_size=mx,
                        allowed_extensions=ext_filter,
                        flag_suspicious=self._flag_suspicious.get())
                    for r in results:
                        self._log(r)
                    self._log(f"\nГотово: {len(results)} файлов")
            except Exception as e:
                self._log(f"ОШИБКА: {e}")
            finally:
                self._running = False
                self.root.after(0, self._on_done)

        threading.Thread(target=task, daemon=True).start()

    def _on_done(self):
        self._progress.stop()
        self._status_label.configure(text=_("Idle"))

    def _copy_file(self, fp, td, dr):
        if dr:
            return _("[DRY RUN] Would copy") + f" {fp} -> {td}"
        os.makedirs(td, exist_ok=True)
        dest = os.path.join(td, os.path.basename(fp))
        if os.path.exists(dest):
            stem, ext = os.path.splitext(dest)
            c = 1
            while os.path.exists(f"{stem}_{c}{ext}"):
                c += 1
            dest = f"{stem}_{c}{ext}"
        shutil.copy2(fp, dest)
        return _("Copied:") + f" {fp} -> {dest}"

    def _stop_sort(self):
        self._log(_("Stop requested (after current file)"))
        self._running = False

    # ============================================================ DATASET GENERATION
    def _run_generate_dataset(self):
        if self._running:
            return
        CONFIG.profile = self._profile.get()
        CONFIG._apply_profile()

        output_csv = self._ds_output_csv.get().strip()
        if not output_csv:
            Messagebox.show_error(_("Output path required"), _("Error"), parent=self.root)
            return

        self._running = True
        self._status_label.configure(text=_("Generating dataset..."))
        self._progress.start(10)

        self._log("\u2500" * 55)
        self._log(_("Profile:") + f" {CONFIG.profile}")

        if self._ds_mode.get() == "real":
            input_dir = self._ds_input_dir.get().strip()
            if not input_dir or not os.path.isdir(input_dir):
                Messagebox.show_error(_("Valid input directory required"), _("Error"), parent=self.root)
                self._running = False
                return
            self._log(f"{_('Mode:')} real | {_('Input:')} {input_dir}")
            self._log(f"{_('Output:')} {output_csv}")
            self._log(_("Extracting features from real files\u2026"))

            def task():
                try:
                    df = build_real_dataset(input_dir, output_csv)
                    self._log(f"Готово: {len(df)} samples, {len(df.columns)} columns")
                    dist = df["target_class"].value_counts().to_string()
                    self._log(f"Распределение классов:\n{dist}")
                except Exception as e:
                    self._log(f"ОШИБКА: {e}")
                finally:
                    self._running = False
                    self.root.after(0, self._on_done)

            threading.Thread(target=task, daemon=True).start()
        else:
            count = self._ds_synthetic_count.get()
            seed = self._ds_seed.get()
            self._log(f"{_('Mode:')} synthetic | {_('Samples:')} {count} | {_('Seed:')} {seed}")
            self._log(f"{_('Output:')} {output_csv}")
            self._log(_("Generating synthetic dataset\u2026"))

            def task():
                try:
                    df = build_synthetic_dataset(count, output_csv, synthetic_dir="synthetic_data", seed=seed)
                    self._log(f"Готово: {len(df)} samples, {len(df.columns)} columns")
                    dist = df["target_class"].value_counts().to_string()
                    self._log(f"Распределение классов:\n{dist}")
                except Exception as e:
                    self._log(f"ОШИБКА: {e}")
                finally:
                    self._running = False
                    self.root.after(0, self._on_done)

            threading.Thread(target=task, daemon=True).start()

    # ============================================================ MODEL TRAINING
    def _run_train_model(self):
        if self._running:
            return
        csv_path = self._tr_csv_path.get().strip()
        output_dir = self._tr_output_dir.get().strip()

        if not os.path.isfile(csv_path):
            Messagebox.show_error(f"{_('Dataset not found:')}\n{csv_path}", _("Error"), parent=self.root)
            return

        models_to_train = [name for name, var in self._tr_models.items() if var.get()]
        if not models_to_train:
            Messagebox.show_error(_("Select at least one model"), _("Error"), parent=self.root)
            return

        CONFIG.profile = self._profile.get()
        CONFIG._apply_profile()

        self._running = True
        self._status_label.configure(text=_("Training..."))
        self._progress.start(10)

        for row in self._tr_tree.get_children():
            self._tr_tree.delete(row)

        self._log("\u2500" * 55)
        self._log(f"{_('Dataset CSV:').rstrip(':')}: {csv_path}")
        self._log(f"{_('Models')}: {', '.join(models_to_train)}")
        self._log(f"{_('Output:')} {output_dir}")
        self._log(f"{_('Profile:')} {CONFIG.profile}")
        self._log(_("Training\u2026"))

        def task():
            try:
                results = train_and_evaluate(csv_path, models_to_train, output_dir)
                for name, res in results.items():
                    vf = res["val"]["f1_macro"]
                    tf = res["test"]["f1_macro"]
                    ta = res["test"]["accuracy"]
                    path = res["model_path"]
                    self._log(f"{name}: Val F1={vf:.4f}, Test F1={tf:.4f}, Test Acc={ta:.4f}")
                    self.root.after(0, lambda n=name, v=vf, t=tf, a=ta, p=path: self._tr_tree.insert(
                        "", END, values=(n, f"{v:.4f}", f"{t:.4f}", f"{a:.4f}", p)))
                best = max(results, key=lambda k: results[k]["test"]["f1_macro"])
                self._log(f"\nЛучшая модель: {best}")
            except Exception as e:
                self._log(f"ОШИБКА: {e}")
                import traceback
                self._log(traceback.format_exc())
            finally:
                self._running = False
                self.root.after(0, self._on_done)

        threading.Thread(target=task, daemon=True).start()

    # ============================================================ CLOSE
    def _on_close(self):
        if self._running:
            if not Messagebox.yesno(_("Operation in progress. Quit anyway?"), _("Running"), parent=self.root):
                return
        self.root.destroy()

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    FileSorterGUI().run()
