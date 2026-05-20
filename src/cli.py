def cmd_sort(args):
    """Пакетная сортировка всех файлов в папке"""
    import uuid
    from watcher import DownloadEventHandler
    handler = DownloadEventHandler(base_dir=args.path, config_path=args.config)
    batch_id = str(uuid.uuid4())
    for fname in os.listdir(args.path):
        fpath = os.path.join(args.path, fname)
        if os.path.isfile(fpath):
            category = handler.classifier.predict(fname)
            handler.move_file(fpath, category, batch_id=batch_id)
import argparse
import logging
import os
import sys
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from watcher import start_watching
from classifier import FileNameClassifier
from undo import undo_last

def cmd_undo(args):
    """Откат последнего перемещения файла"""
    undo_last()

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')


def cmd_watch(args):
    """Запуск наблюдения за папкой"""
    start_watching(args.path, args.model, args.config)

def cmd_predict(args):
    """Классификация имени файла"""
    clf = FileNameClassifier(args.model)
    category = clf.predict(args.filename)
    print(f"{args.filename} -> {category}")

def cmd_stats(args):
    """Показать статистику по категориям в папке"""
    base = args.path
    stats = {}
    for cat in os.listdir(base):
        cat_path = os.path.join(base, cat)
        if os.path.isdir(cat_path):
            stats[cat] = len([f for f in os.listdir(cat_path) if os.path.isfile(os.path.join(cat_path, f))])
    print("Категория : Кол-во файлов")
    for cat, count in stats.items():
        print(f"{cat:12}: {count}")

def main():

    parser = argparse.ArgumentParser(description="Автосортировка файлов в папке Downloads")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # sort
    p_sort = subparsers.add_parser("sort", help="Отсортировать все файлы в папке сразу")
    p_sort.add_argument("--path", default=os.path.expanduser("~/Downloads"), help="Папка для сортировки")
    p_sort.add_argument("--config", default=None, help="Путь к YAML-конфигу сортировки")
    p_sort.set_defaults(func=cmd_sort)

    # undo
    p_undo = subparsers.add_parser("undo", help="Откат последнего перемещения файла")
    p_undo.set_defaults(func=cmd_undo)

    # watch
    p_watch = subparsers.add_parser("watch", help="Запустить наблюдение за папкой")
    p_watch.add_argument("--path", default=os.path.expanduser("~/Downloads"), help="Папка для наблюдения")
    p_watch.add_argument("--model", default=None, help="Путь к модели (joblib)")
    p_watch.add_argument("--config", default=None, help="Путь к YAML-конфигу сортировки")
    p_watch.set_defaults(func=cmd_watch)

    # predict
    p_predict = subparsers.add_parser("predict", help="Классифицировать имя файла")
    p_predict.add_argument("filename", help="Имя файла для классификации")
    p_predict.add_argument("--model", default=None, help="Путь к модели (joblib)")
    p_predict.set_defaults(func=cmd_predict)

    # stats
    p_stats = subparsers.add_parser("stats", help="Показать статистику по категориям в папке")
    p_stats.add_argument("--path", default=os.path.expanduser("~/Downloads"), help="Папка для анализа")
    p_stats.set_defaults(func=cmd_stats)

    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
