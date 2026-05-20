# Откат последних перемещений файлов
# Для восстановления файлов после автосортировки

import os
import sys
import json
import logging
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from watcher import DOWNLOADS_DIR

LOG_PATH = os.path.expanduser(os.path.join(DOWNLOADS_DIR, ".sorter_log.json"))

def log_move(src, dst):
    log = []
    if os.path.exists(LOG_PATH):
        with open(LOG_PATH, 'r') as f:
            log = json.load(f)
    log.append({"src": src, "dst": dst})
    with open(LOG_PATH, 'w') as f:
        json.dump(log, f)

def undo_last():
    if not os.path.exists(LOG_PATH):
        print("Нет истории перемещений для отката.")
        return
    with open(LOG_PATH, 'r') as f:
        log = json.load(f)
    if not log:
        print("Нет действий для отката.")
        return
    # Найти последний batch_id (или None)
    last_batch_id = log[-1].get("batch_id") if "batch_id" in log[-1] else None
    batch_moves = []
    if last_batch_id is not None:
        # Откатываем все с этим batch_id
        while log and log[-1].get("batch_id") == last_batch_id:
            batch_moves.append(log.pop())
    else:
        # Откатываем все подряд без batch_id, пока не встретим запись с batch_id или пока не кончится лог
        while log and "batch_id" not in log[-1]:
            batch_moves.append(log.pop())
    if not batch_moves:
        print("Нет действий для отката.")
        return
    for move in reversed(batch_moves):
        src, dst = move["src"], move["dst"]
        if os.path.exists(dst):
            os.rename(dst, src)
            print(f"Откат: {dst} -> {src}")
        else:
            print(f"Файл {dst} не найден для отката.")
    with open(LOG_PATH, 'w') as f:
        json.dump(log, f)

if __name__ == "__main__":
    undo_last()
