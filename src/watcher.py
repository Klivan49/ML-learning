import json

import logging
import os
import shutil
import sys
import time
import yaml
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
from classifier import FileNameClassifier

DOWNLOADS_DIR = os.path.expanduser("~/Downloads")
CATEGORIES = [
    "Study",
    "Documents",
    "Images",
    "Archives",
    "Media",
    "Invoices",
    "Projects",
    "Presentations",
    "Others",
]


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


class DownloadEventHandler(FileSystemEventHandler):
    def log_move(self, src, dst, batch_id=None):
        log_path = os.path.expanduser(os.path.join(self.base_dir, ".sorter_log.json"))
        log = []
        if os.path.exists(log_path):
            with open(log_path, "r") as f:
                log = json.load(f)
        entry = {"src": src, "dst": dst}
        if batch_id is not None:
            entry["batch_id"] = batch_id
        log.append(entry)
        with open(log_path, "w") as f:
            json.dump(log, f)

    def __init__(self, base_dir=DOWNLOADS_DIR, model_path=None, config_path=None):
        super().__init__()
        self.base_dir = base_dir
        self.classifier = (
            FileNameClassifier(model_path) if model_path else FileNameClassifier()
        )
        self.config = self._load_config(config_path)

    def _load_config(self, config_path):
        if config_path and os.path.exists(config_path):
            with open(config_path, "r") as f:
                return yaml.safe_load(f)
        return None

    def on_created(self, event):
        if not event.is_directory:
            fname = os.path.basename(event.src_path)
            logging.info(f"New file detected: {fname}")
            category = self.classifier.predict(fname)
            logging.info(f"Predicted category: {category}")
            self.move_file(event.src_path, category)

    def move_file(self, src_path, category, batch_id=None):
        # Логируем категорию для отладки
        logging.info(f"Predicted category for {src_path}: '{category}'")
        # Если категория None — отправляем в unknown/Others
        if not category:
            norm_category = "Others"
        else:
            norm_category = category.strip()
        target_dir = None
        if self.config and "categories" in self.config:
            # Пробуем найти путь для полной категории
            if norm_category in self.config["categories"]:
                target_dir = os.path.expanduser(
                    self.config["categories"][norm_category]
                )
            else:
                # Пробуем родительскую категорию (до первого слэша)
                parent = norm_category.split("/")[0] if "/" in norm_category else None
                if parent and parent in self.config["categories"]:
                    target_dir = os.path.expanduser(self.config["categories"][parent])
                else:
                    target_dir = os.path.expanduser(
                        self.config.get(
                            "unknown", os.path.join(self.base_dir, "Others")
                        )
                    )
        else:
            target_dir = os.path.join(self.base_dir, norm_category)
        os.makedirs(target_dir, exist_ok=True)
        fname = os.path.basename(src_path)
        dst_path = os.path.join(target_dir, fname)
        # Обработка конфликта имён
        if os.path.exists(dst_path):
            base, ext = os.path.splitext(fname)
            i = 1
            while os.path.exists(dst_path):
                dst_path = os.path.join(target_dir, f"{base}_copy{i}{ext}")
                i += 1
        try:
            shutil.move(src_path, dst_path)
            logging.info(f"Moved {src_path} -> {dst_path}")
            self.log_move(src_path, dst_path, batch_id=batch_id)
        except Exception as e:
            logging.error(f"Failed to move {src_path}: {e}")

    def on_moved(self, event):
        if not event.is_directory:
            logging.info(f"File moved: {event.src_path} -> {event.dest_path}")


def start_watching(path=DOWNLOADS_DIR, model_path=None, config_path=None):
    event_handler = DownloadEventHandler(path, model_path, config_path)
    observer = Observer()
    observer.schedule(event_handler, path, recursive=False)
    observer.start()
    logging.info(f"Watching {path} for changes...")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


if __name__ == "__main__":
    start_watching()
