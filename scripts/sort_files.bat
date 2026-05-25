@echo off
chcp 65001 >nul
python "%~dp0sort_files.py" %*
