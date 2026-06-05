"""Добавляет site-packages из venv/ в sys.path, если скрипт запущен не из venv."""

import sys
import os

def _setup():
    if sys.prefix != sys.base_prefix:
        return
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    venv = os.path.join(root, "venv", "lib")
    if not os.path.isdir(venv):
        return
    for d in os.listdir(venv):
        if d.startswith("python"):
            site = os.path.join(venv, d, "site-packages")
            if os.path.isdir(site) and site not in sys.path:
                sys.path.insert(0, site)

_setup()
