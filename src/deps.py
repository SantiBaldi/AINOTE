"""Importes de dependencias externas con mensajes en castellano.

`sounddevice` y `faster-whisper` se importan tarde y desde acá para que un
entorno a medio armar falle con una instrucción concreta en vez de un
`ModuleNotFoundError` pelado (CLAUDE.md §8: fallar temprano y en castellano).

`sounddevice` además puede importar bien y reventar después al cargar la DLL de
PortAudio, que es un `OSError`, no un `ImportError`. Se contemplan los dos.
"""

from __future__ import annotations

import os
from pathlib import Path


class DependenciaFaltante(RuntimeError):
    """Falta una dependencia o no se puede cargar."""


_INSTALAR = ("Activá el entorno y instalá las dependencias:\n"
             "    .venv\\Scripts\\activate\n"
             "    pip install -r requirements.txt")


def sounddevice():
    """Devuelve el módulo `sounddevice` listo para usar."""
    try:
        import sounddevice as sd
    except ImportError:
        raise DependenciaFaltante(
            f"Falta 'sounddevice', que es lo que habla con el micrófono.\n"
            f"  {_INSTALAR}") from None
    except OSError as error:
        raise DependenciaFaltante(
            f"'sounddevice' está instalado pero no pudo cargar PortAudio: {error}\n"
            f"  Reinstalalo con: pip install --force-reinstall sounddevice") from None
    return sd


def _carpetas_dll_cuda() -> list[Path]:
    """Carpetas con los DLLs de CUDA que pip deja dentro del venv.

    Los paquetes `nvidia-*-cu12` instalan sus binarios en
    `site-packages/nvidia/<libreria>/bin`.
    """
    try:
        import nvidia
    except ImportError:
        return []
    carpetas = []
    for raiz in nvidia.__path__:
        carpetas.extend(sorted(p for p in Path(raiz).glob("*/bin") if p.is_dir()))
    return carpetas


def registrar_dlls_cuda() -> int:
    """Agrega los DLLs de CUDA del venv a la ruta de búsqueda. Devuelve cuántos.

    Sólo hace algo en Windows. En Linux el enlazador los encuentra por RPATH,
    pero Windows no mira dentro de site-packages: sin esto, `ctranslate2` falla
    con "Library cublas64_12.dll is not found" aunque el paquete esté instalado.
    """
    if not hasattr(os, "add_dll_directory"):  # no es Windows
        return 0
    registradas = 0
    for carpeta in _carpetas_dll_cuda():
        try:
            os.add_dll_directory(str(carpeta))
            registradas += 1
        except OSError:
            continue
    return registradas


# Nombre del DLL -> paquete de pip que lo trae.
_PAQUETES_CUDA = {
    "cublas": "nvidia-cublas-cu12",
    "cudnn": "nvidia-cudnn-cu12",
    "cudart": "nvidia-cuda-runtime-cu12",
}


def traducir_error_de_dll(error: Exception) -> DependenciaFaltante | None:
    """Convierte un fallo de carga de DLL de CUDA en algo accionable.

    `ctranslate2` avisa con un RuntimeError en inglés que nombra el DLL pero no
    dice cómo conseguirlo, y sin CUDA Toolkit (que pide admin) la única vía es
    el paquete de pip equivalente.
    """
    texto = str(error)
    if ".dll" not in texto.lower() and ".so" not in texto.lower():
        return None

    paquete = next((p for clave, p in _PAQUETES_CUDA.items() if clave in texto.lower()),
                   None)
    if paquete is None:
        return None

    return DependenciaFaltante(
        f"Falta una librería de CUDA: {texto}\n"
        f"  Instalala en el entorno (no necesita permisos de administrador):\n"
        f"      pip install {paquete}\n"
        f"  Si después pide otra, instalá también:\n"
        f"      pip install " + " ".join(sorted(set(_PAQUETES_CUDA.values()))) + "\n"
        f"  Son varios cientos de MB: traen el CUDA que normalmente instalaría\n"
        f"  el Toolkit, que sí pide administrador.")


def whisper_model():
    """Devuelve la clase `WhisperModel` de faster-whisper."""
    registrar_dlls_cuda()
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        raise DependenciaFaltante(
            f"Falta 'faster-whisper', que es el motor de transcripción.\n"
            f"  {_INSTALAR}") from None
    return WhisperModel
