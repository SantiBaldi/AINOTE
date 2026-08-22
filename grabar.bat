@echo off
REM Doble clic para grabar una reunion. Pide el nombre, graba, y al cortar con
REM Enter transcribe sola. Pensado para usarlo en la reunion sin tipear nada.

cd /d "%~dp0"

if not exist ".venv\Scripts\activate.bat" (
    echo   Falta el entorno. Abri ainote.bat para ver como crearlo.
    pause
    exit /b 1
)

call ".venv\Scripts\activate.bat"
python -m src grabar

echo.
pause
