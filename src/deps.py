"""Importes de dependencias externas con mensajes en castellano.

`sounddevice` y `faster-whisper` se importan tarde y desde acá para que un
entorno a medio armar falle con una instrucción concreta en vez de un
`ModuleNotFoundError` pelado (CLAUDE.md §8: fallar temprano y en castellano).

`sounddevice` además puede importar bien y reventar después al cargar la DLL de
PortAudio, que es un `OSError`, no un `ImportError`. Se contemplan los dos.
"""

from __future__ import annotations


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


def whisper_model():
    """Devuelve la clase `WhisperModel` de faster-whisper."""
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        raise DependenciaFaltante(
            f"Falta 'faster-whisper', que es el motor de transcripción.\n"
            f"  {_INSTALAR}") from None
    return WhisperModel
