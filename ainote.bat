@echo off
REM Abre una consola lista para usar AINOTE: entra al proyecto y activa el
REM entorno. Se puede ejecutar con doble clic desde el Explorador o crearle
REM un acceso directo en el escritorio.
REM
REM Existe porque una consola nueva arranca en la carpeta del usuario y sin el
REM venv, y ahi `python -m src` falla con "No module named src".

cd /d "%~dp0"

if not exist ".venv\Scripts\activate.bat" (
    echo.
    echo   No encuentro el entorno virtual en .venv
    echo   Crealo con:
    echo       python -m venv .venv
    echo       .venv\Scripts\activate
    echo       pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

call ".venv\Scripts\activate.bat"

echo.
echo   AINOTE listo. Comandos:
echo.
echo     python -m src grabar          graba y transcribe al cortar con Enter
echo     python -m src transcribir X   transcribe un WAV puntual
echo     python -m src vigilar         transcribe solo todo WAV nuevo
echo     python -m src dispositivos    lista microfonos
echo     python -m src gpu             VRAM libre y librerias de CUDA
echo.

cmd /k
