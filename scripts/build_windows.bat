@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

echo === Windows standalone build ===

where python >nul 2>&1
if %errorlevel% neq 0 (
    echo Python not found. Install from https://www.python.org/downloads/
    pause
    exit /b 1
)

if not exist "build_venv" (
    echo Creating venv...
    python -m venv build_venv
)
call build_venv\Scripts\activate.bat

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install pyinstaller

python scripts\build_exe.py

echo === Done ===
echo Binaries in: dist\
dir /b dist\
pause
