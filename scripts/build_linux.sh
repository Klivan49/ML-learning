#!/usr/bin/env bash
set -euo pipefail

ROOT="$(dirname "$(dirname "$(readlink -f "$0")")")"

echo "=== Linux standalone build ==="

if ! command -v python3 &>/dev/null; then
    echo "python3 required"; exit 1
fi

VENV="$ROOT/build_venv"
echo "Creating venv: $VENV"
python3 -m venv "$VENV"
source "$VENV/bin/activate"

pip install --upgrade pip
pip install -r "$ROOT/requirements.txt"
pip install pyinstaller

cd "$ROOT"
python3 scripts/build_exe.py

echo "=== Done ==="
echo "Binaries in: $ROOT/dist/"
ls -lh "$ROOT/dist/"
