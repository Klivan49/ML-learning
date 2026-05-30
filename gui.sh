#!/usr/bin/env bash
set -e
ROOT="$(dirname "$(readlink -f "$0")")"

# check tkinter before anything else
python3 -c "import tkinter" 2>/dev/null || {
    echo "Ошибка: tkinter не установлен."
    if command -v pacman &>/dev/null; then
        echo "  sudo pacman -S tk"
    elif command -v apt &>/dev/null; then
        echo "  sudo apt install python3-tk"
    elif command -v dnf &>/dev/null; then
        echo "  sudo dnf install python3-tkinter"
    fi
    echo ""
    echo "После установки запусти скрипт снова."
    exit 1
}

# try existing venvs in order
for v in "$ROOT/build_venv" "$ROOT/venv" "/tmp/venv_file_sorter"; do
    if [ -f "$v/bin/python" ] && [ -f "$v/bin/pip" ]; then
        PYTHON="$v/bin/python"
        break
    fi
done

if [ -z "$PYTHON" ]; then
    echo "No venv found. Creating one..."
    python3 -m venv "$ROOT/venv"
    "$ROOT/venv/bin/pip" install -q -r "$ROOT/requirements.txt"
    PYTHON="$ROOT/venv/bin/python"
fi

exec "$PYTHON" "$ROOT/scripts/gui.py"
