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


# DLLs que ctranslate2 carga en tiempo de ejecución, por librería.
_DLLS_ESPERADOS = {"cublas": "cublas64_12.dll", "cudnn": "cudnn*.dll"}


def diagnostico_cuda() -> list[str]:
    """Estado de las librerías de CUDA, para el comando `gpu`.

    Distingue los dos fallos que dan el mismo síntoma: que el paquete no esté
    instalado, y que esté pero Windows no encuentre sus DLLs.
    """
    lineas = []
    try:
        import nvidia
        raices = list(nvidia.__path__)
    except ImportError:
        raices = []

    if not raices:
        lineas.append("    No hay librerías de CUDA instaladas por pip.")
        lineas.append("    Si CUDA te funciona igual, tenés el Toolkit del sistema.")
        lineas.append("    Si no, instalalas con: pip install -r requirements.txt")
        return lineas

    carpetas = _carpetas_dll_cuda()
    if not carpetas:
        lineas.append(f"    El paquete 'nvidia' está en {raices[0]}")
        lineas.append("    pero no tiene ninguna carpeta */bin con DLLs adentro.")
        return lineas

    for carpeta in carpetas:
        dlls = sorted(p.name for p in carpeta.glob("*.dll"))
        sos = sorted(p.name for p in carpeta.parent.glob("lib/*.so*"))
        cuantos = len(dlls) or len(sos)
        lineas.append(f"    {carpeta.parent.name:<10} {cuantos} archivo(s) en {carpeta}")

    registradas = registrar_dlls_cuda()
    if hasattr(os, "add_dll_directory"):
        lineas.append(f"    {registradas} carpeta(s) agregadas a la ruta de búsqueda.")
        lineas.extend(_probar_carga())
    else:
        lineas.append("    (fuera de Windows no hace falta registrarlas)")
    return lineas


def _probar_carga() -> list[str]:
    """Intenta cargar de verdad los DLLs. Es la única prueba que vale."""
    import ctypes

    lineas = []
    for carpeta in _carpetas_dll_cuda():
        for dll in sorted(carpeta.glob("*.dll")):
            if not any(dll.name.startswith(p) for p in ("cublas64", "cudnn64", "cudnn_ops")):
                continue
            try:
                ctypes.WinDLL(str(dll))
                lineas.append(f"    OK    {dll.name}")
            except OSError as error:
                lineas.append(f"    FALLA {dll.name}: {error}")
    if not lineas:
        lineas.append("    No encontré cublas64_*.dll ni cudnn*.dll para probar.")
    return lineas
