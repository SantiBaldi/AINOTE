"""Cerrojo entre procesos para que nunca corran dos transcripciones a la vez.

La regla del proyecto es que sólo puede haber un modelo cargado en la GPU
(CLAUDE.md §2). Como cada transcripción corre en su propio proceso, la regla no
se puede sostener con un candado en memoria: hace falta uno en disco.

El escenario que esto evita es real: con `vigilar` corriendo en una consola y
`grabar` en otra, al cortar la grabación se lanzan dos transcripciones del mismo
audio —la encadenada y la del watcher— y las dos intentan cargar Whisper.

Se usa `O_CREAT | O_EXCL`, que es atómico en Windows y en POSIX, sin
dependencias ni servicios.
"""

from __future__ import annotations

import contextlib
import os
from pathlib import Path

from . import paths

# Un lock más viejo que esto quedó de un proceso que murió sin limpiarlo: una
# reunión larga puede tardar bastante, así que el margen es generoso.
VENCIMIENTO_S = 6 * 3600


class Ocupado(RuntimeError):
    """Ya hay otra transcripción corriendo."""


def _ruta() -> Path:
    return paths.RAIZ / ".ainote.lock"


def _esta_rancio(ruta: Path) -> bool:
    """True si el lock quedó de un proceso muerto.

    Se mide por antigüedad y no por PID a propósito: preguntarle al SO si un PID
    sigue vivo no es portable, y un PID reciclado daría un falso positivo.
    """
    try:
        import time
        return (time.time() - ruta.stat().st_mtime) > VENCIMIENTO_S
    except OSError:
        return False


@contextlib.contextmanager
def transcripcion():
    """Toma el cerrojo mientras dure el bloque. Levanta `Ocupado` si no puede."""
    ruta = _ruta()

    if ruta.exists() and _esta_rancio(ruta):
        print("  Encontré un cerrojo viejo de un proceso que murió; lo descarto.")
        ruta.unlink(missing_ok=True)

    try:
        descriptor = os.open(str(ruta), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise Ocupado(
            f"Ya hay otra transcripción corriendo: no arranco una segunda para "
            f"no quedarnos sin VRAM.\n"
            f"  Esperá a que termine. Si estás seguro de que no hay ninguna "
            f"(por ejemplo, si una se cortó a lo bruto), borrá el archivo:\n"
            f"    {paths.relativa(ruta)}") from None

    try:
        os.write(descriptor, str(os.getpid()).encode("ascii"))
        os.close(descriptor)
        yield ruta
    finally:
        ruta.unlink(missing_ok=True)
