#!/usr/bin/env python3
import os
import sys
import shutil
import threading

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from configs.config import CONFIG
from src.inference.predict import FileClassifier

import tkinter as tk
from tkinter import filedialog

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


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class FileSorterGUI:
    def __init__(self):
        self.root = tb.Window(themename="darkly") if THEMED else tk.Tk()
        self.root.title("File Sorter")
        self.root.geometry("900x620+100+100")
        self.root.minsize(800, 600)

        self.root.title("\U0001F4C2  File Sorter — ML Classification")

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

        self._log_lock = threading.Lock()
        self._running = False
        self._nav_buttons = {}

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._show_page("sort")

    def _nav_click(self, page):
        for name, btn in self._nav_buttons.items():
            btn.configure(style="nav.TButton" if name != page else "nav-active.TButton")
        self._show_page(page)

    def _build_ui(self):
        outer = tb.Frame(self.root)
        outer.pack(fill=BOTH, expand=True)

        self._navbar = tb.Frame(outer, width=160, padding=10)
        self._navbar.pack(side=LEFT, fill=Y)
        self._navbar.pack_propagate(False)

        tb.Label(self._navbar, text="File Sorter", font=("Segoe UI", 14, "bold")).pack(anchor=W, pady=(0, 20))

        pages = [
            ("sort",   "\U0001F500", "Sort"),
            ("config", "\U00002699", "Config"),
            ("about",  "\U00002139", "About"),
        ]
        for name, icon, label in pages:
            btn = tb.Button(self._navbar, text=f"{icon}  {label}", style="nav.TButton",
                            command=lambda n=name: self._nav_click(n))
            btn.pack(fill=X, pady=2)
            self._nav_buttons[name] = btn

        style_name = "nav.TButton"
        if THEMED:
            self.root.style.configure("nav.TButton", font=("Segoe UI", 11), padding=8, anchor=W, borderwidth=0)
            self.root.style.configure("nav-active.TButton", font=("Segoe UI", 11, "bold"), padding=8, anchor=W)
            self.root.style.map("nav-active.TButton", background=[("active", self.root.style.colors.primary)])

        sep = tb.Separator(self._navbar)
        sep.pack(fill=X, pady=15)

        self._status_label = tb.Label(self._navbar, text="Idle", font=("Segoe UI", 9))
        self._status_label.pack(anchor=W, pady=(0, 5))

        self._progress = tb.Progressbar(self._navbar, mode="indeterminate")
        self._progress.pack(fill=X)

        self._container = tb.Frame(outer)
        self._container.pack(side=RIGHT, fill=BOTH, expand=True, padx=(0, 0), pady=(0, 0))

        self._sort_frame = tb.Frame(self._container)
        self._config_frame = tb.Frame(self._container)
        self._about_frame = tb.Frame(self._container)
        self._log_frame = tb.LabelFrame(self._container, text="Log")

        self._build_sort_tab()
        self._build_config_tab()
        self._build_about_tab()
        self._build_log()

    def _build_sort_tab(self):
        for i, (txt, w) in enumerate([
            ("Model",  self._model_path),
            ("Input",  self._input_path),
            ("Output", self._output_path),
        ]):
            row = tb.Frame(self._sort_frame)
            row.pack(fill=X, pady=4)
            tb.Label(row, text=txt, width=7, anchor=E).pack(side=LEFT, padx=(0, 8))
            entry = tb.Entry(row, textvariable=w)
            entry.pack(side=LEFT, fill=X, expand=True, padx=(0, 6))
            tb.Button(row, text="Browse", command=lambda t=txt: self._browse(t), width=8).pack(side=LEFT)

        opt_frame = tb.LabelFrame(self._sort_frame, text="Options")
        opt_frame.pack(fill=X, pady=(8, 0))

        row1 = tb.Frame(opt_frame)
        row1.pack(fill=X)
        tb.Checkbutton(row1, text="Recursive", variable=self._recursive, bootstyle="round-toggle").pack(side=LEFT, padx=4)
        tb.Checkbutton(row1, text="Dry Run",   variable=self._dry_run,   bootstyle="round-toggle").pack(side=LEFT, padx=4)
        tb.Checkbutton(row1, text="Copy",      variable=self._copy_mode, bootstyle="round-toggle").pack(side=LEFT, padx=4)

        tb.Label(row1, text="Profile:", font=("", 9, "bold")).pack(side=LEFT, padx=(20, 4))
        profile_combo = tb.Combobox(row1, textvariable=self._profile, values=["general", "education"],
                                     state="readonly", width=12)
        profile_combo.pack(side=LEFT)
        self._profile.trace_add("write", lambda *_: self._rebuild_config_tab())

        flt_frame = tb.LabelFrame(self._sort_frame, text="Filters")
        flt_frame.pack(fill=X, pady=(8, 0))

        row2 = tb.Frame(flt_frame)
        row2.pack(fill=X, pady=2)
        tb.Label(row2, text="Min size:").pack(side=LEFT)
        tb.Entry(row2, textvariable=self._min_size, width=10).pack(side=LEFT, padx=4)
        tb.Label(row2, text="Max size:").pack(side=LEFT, padx=(15, 0))
        tb.Entry(row2, textvariable=self._max_size, width=10).pack(side=LEFT, padx=4)
        tb.Label(row2, text="Extensions:").pack(side=LEFT, padx=(15, 0))
        tb.Entry(row2, textvariable=self._extension_var, width=20).pack(side=LEFT, fill=X, expand=True, padx=4)
        tb.Label(row2, text="space-sep", font=("", 8)).pack(side=LEFT)

        btn_frame = tb.Frame(self._sort_frame)
        btn_frame.pack(fill=X, pady=(14, 0))
        tb.Button(btn_frame, text="\u25B6  Run Sort", bootstyle="success", command=self._run_sort, width=18).pack(side=LEFT, padx=(0, 6))
        tb.Button(btn_frame, text="\u23F9  Stop",     bootstyle="secondary", command=self._stop_sort, width=12).pack(side=LEFT)

    def _build_config_tab(self):
        self._config_canvas = tb.Canvas(self._config_frame, borderwidth=0, highlightthickness=0)
        self._config_scrollbar = tb.Scrollbar(self._config_frame, orient=VERTICAL, command=self._config_canvas.yview)
        self._config_inner = tb.Frame(self._config_canvas)
        self._config_inner.bind("<Configure>", lambda e: self._config_canvas.configure(scrollregion=self._config_canvas.bbox("all")))
        self._config_canvas.create_window((0, 0), window=self._config_inner, anchor=NW)
        self._config_canvas.configure(yscrollcommand=self._config_scrollbar.set)
        self._config_canvas.pack(side=LEFT, fill=BOTH, expand=True)
        self._config_scrollbar.pack(side=RIGHT, fill=Y)

        self._populate_config()

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
            kws  = ", ".join(CONFIG.category_keywords.get(cat, []))

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
        tb.Button(btn_row, text="\U0001F4BE  Save", bootstyle="success", command=self._save_config).pack(side=LEFT, padx=3)
        tb.Button(btn_row, text="\u21BA  Reload",  bootstyle="secondary", command=self._reload_config).pack(side=LEFT, padx=3)

    def _build_about_tab(self):
        c = (
            "\U0001F4C2  File Sorter — ML File Classification\n\n"
            "Categories: " + ", ".join(CONFIG.target_columns) + "\n"
            "Models: Logistic Regression, Random Forest, Gradient Boosting\n"
            f"Features: 48 (filename, size, extension, magic bytes, text stats)\n\n"
            "Workflow:\n"
            "  1. scripts/generate_dataset.py  — create dataset\n"
            "  2. scripts/train_model.py        — train model\n"
            "  3. scripts/sort_files.py         — CLI sorting\n"
            "  4. python scripts/gui.py         — this GUI\n"
        )
        text = tb.Text(self._about_frame, wrap=WORD, font=("Segoe UI", 11), padx=15, pady=15, state=NORMAL)
        text.insert(END, c)
        text.configure(state=DISABLED)
        text.pack(fill=BOTH, expand=True)

    def _build_log(self):
        self._log_frame.pack(fill=BOTH, expand=True)
        st = ScrolledText(self._log_frame, height=8, wrap=WORD,
                          font=("Consolas", 9), autohide=True)
        st.pack(fill=BOTH, expand=True)
        st.text.configure(state=DISABLED)
        self._log_text = st

    def _show_page(self, name):
        for f in (self._sort_frame, self._config_frame, self._about_frame, self._log_frame):
            f.pack_forget()
        self._log_frame.pack(fill=BOTH, expand=True, pady=(10, 0))
        if name == "sort":
            self._sort_frame.pack(fill=BOTH, expand=False)
        elif name == "config":
            self._config_frame.pack(fill=BOTH, expand=True)
        elif name == "about":
            self._about_frame.pack(fill=BOTH, expand=True)

    def _browse(self, target):
        if target == "Model":
            p = filedialog.askopenfilename(title="Select model", filetypes=[("PKL", "*.pkl"), ("All", "*.*")])
        elif target == "Input":
            p = filedialog.askdirectory(title="Select input directory")
        else:
            p = filedialog.askdirectory(title="Select output directory")
        if p:
            vars = {"Model": self._model_path, "Input": self._input_path, "Output": self._output_path}
            vars[target].set(p)

    def _save_config(self):
        try:
            for cat, (e_w, k_w) in self._config_widgets.items():
                exts = [x.strip().lower() for x in e_w.get("1.0", END).strip().split(",") if x.strip()]
                kws  = [x.strip().lower() for x in k_w.get("1.0", END).strip().split(",") if x.strip()]
                CONFIG.extension_to_category[cat] = exts
                CONFIG.category_keywords[cat] = kws
            Messagebox.show_info("Configuration saved (runtime). Restart to persist.", "Saved", parent=self.root)
        except Exception as e:
            Messagebox.show_error(str(e), "Error", parent=self.root)

    def _reload_config(self):
        for cat, (e_w, k_w) in self._config_widgets.items():
            for w, src in [(e_w, CONFIG.extension_to_category), (k_w, CONFIG.category_keywords)]:
                w.delete("1.0", END)
                w.insert(END, ", ".join(src.get(cat, [])))

    def _log(self, msg):
        with self._log_lock:
            txt = self._log_text.text
            txt.configure(state=NORMAL)
            txt.insert(END, msg + "\n")
            txt.see(END)
            txt.configure(state=DISABLED)
            self.root.update_idletasks()

    def _run_sort(self):
        if self._running:
            return
        mp = self._model_path.get().strip()
        ip = self._input_path.get().strip()
        op = self._output_path.get().strip()

        if not os.path.isfile(mp):
            Messagebox.show_error(f"Model not found:\n{mp}", "Error", parent=self.root)

        if not os.path.exists(ip):
            Messagebox.show_error(f"Input not found:\n{ip}", "Error", parent=self.root)
            return
        if not op:
            Messagebox.show_error("Output path required", "Error", parent=self.root)
            return

        try:
            mn, mx = int(self._min_size.get()), int(self._max_size.get())
        except ValueError:
            Messagebox.show_error("Min/max size must be integers", "Error", parent=self.root)
            return

        ext_filter = None
        if self._extension_var.get().strip():
            ext_filter = [e.strip().lower().lstrip(".") for e in self._extension_var.get().strip().split()]

        profile = self._profile.get()
        CONFIG.profile = profile
        CONFIG._apply_profile()

        self._running = True
        self._status_label.configure(text="Running...")
        self._progress.start(10)

        self._log("\u2500" * 55)
        self._log(f"Model:   {mp}")
        self._log(f"Input:   {ip}")
        self._log(f"Output:  {op}")
        self._log(f"Profile: {profile}")
        self._log(f"Filters: min={mn}, max={mx}, ext={ext_filter or 'all'}")
        self._log(f"Mode:    {'copy' if self._copy_mode.get() else 'move'} | "
                  f"{'dry-run' if self._dry_run.get() else 'live'} | "
                  f"{'recursive' if self._recursive.get() else 'flat'}")
        self._log("Starting\u2026")

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
                        allowed_extensions=ext_filter)
                    for r in results:
                        self._log(r)
                    self._log(f"\nDone: {len(results)} files")
            except Exception as e:
                self._log(f"ERROR: {e}")
            finally:
                self._running = False
                self.root.after(0, self._on_done)

        threading.Thread(target=task, daemon=True).start()

    def _on_done(self):
        self._progress.stop()
        self._status_label.configure(text="Idle")

    def _copy_file(self, fp, td, dr):
        if dr:
            return f"[DRY RUN] Would copy {fp} -> {td}"
        os.makedirs(td, exist_ok=True)
        dest = os.path.join(td, os.path.basename(fp))
        if os.path.exists(dest):
            stem, ext = os.path.splitext(dest)
            c = 1
            while os.path.exists(f"{stem}_{c}{ext}"):
                c += 1
            dest = f"{stem}_{c}{ext}"
        shutil.copy2(fp, dest)
        return f"Copied: {fp} -> {dest}"

    def _stop_sort(self):
        self._log("Stop requested (after current file)")
        self._running = False

    def _on_close(self):
        if self._running:
            if not Messagebox.yesno("Sorting in progress. Quit anyway?", "Running", parent=self.root):
                return
        self.root.destroy()

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    FileSorterGUI().run()
