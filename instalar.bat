@echo off
cd /d "%~dp0"
python --version
if errorlevel 1 ( echo ERRO: Python nao encontrado. & pause & exit /b 1 )
pip install -r requirements.txt
if errorlevel 1 ( pip install -r requirements.txt --trusted-host pypi.org --trusted-host files.pythonhosted.org )
if errorlevel 1 ( echo ERRO ao instalar dependencias. & pause & exit /b 1 )
echo.
echo Instalacao concluida com sucesso!
pause
