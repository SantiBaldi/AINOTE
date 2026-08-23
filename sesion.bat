@echo off
REM Doble clic para una reunion: abre la ventana de notas y graba.
REM Al cerrarla, transcribe sola.

cd /d "%~dp0"

if not exist ".venv\Scripts\activate.bat" (
    echo   Falta el entorno. Abri ainote.bat para ver como crearlo.
    pause
    exit /b 1
)

call ".venv\Scripts\activate.bat"
python -m src sesion

echo.
pause
