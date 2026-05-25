@echo off
chcp 65001 >nul
python "%~dp0train_model.py" %*
