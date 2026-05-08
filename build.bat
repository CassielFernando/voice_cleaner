@echo off
cd /d "%~dp0"
echo Gerando icone...
python create_icon.py
if errorlevel 1 ( echo ERRO ao gerar icone. & pause & exit /b 1 )

echo Compilando executavel...
pyinstaller --onefile --windowed --name "VoiceCleaner" --icon "icon.ico" ^
  --add-data "icon.ico;." ^
  --collect-data customtkinter ^
  --collect-all noisereduce ^
  --collect-all librosa ^
  --collect-all audioread ^
  --collect-all soundfile ^
  gui.py

if errorlevel 1 ( echo ERRO na compilacao. & pause & exit /b 1 )

echo.
echo Build concluido!
powershell -Command "Get-Item dist\*.exe | Select-Object Length, Name | Format-Table"
pause
