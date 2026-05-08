@echo off
cd /d "%~dp0"
python gui.py
if errorlevel 1 ( echo Erro ao abrir. Rode instalar.bat primeiro. & pause )
