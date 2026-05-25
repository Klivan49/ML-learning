#!/usr/bin/env bash
ROOT="$(dirname "$(readlink -f "$0")")"

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
