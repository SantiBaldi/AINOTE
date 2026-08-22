"""Watcher de /audio/: transcribe solo cualquier WAV nuevo.

Polling con `os.scandir` en vez de `watchdog`: una dependencia menos, y en
Windows los eventos de filesystem llegan mientras el archivo todavía se está
escribiendo, con lo cual igual haría falta el chequeo de estabilidad de abajo.

Cada transcripción corre en un **subproceso** que muere al terminar. Es la
única forma verificable de devolver la VRAM al sistema (CLAUDE.md §2): el
watcher puede quedar vivo días, y un modelo cargado en su proceso nunca se
soltaría del todo.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

from . import paths

INTERVALO = 2.0  # segundos entre sondeos


def _wavs() -> list[Path]:
    if not paths.AUDIO.exists():
        return []
    return sorted(p for p in paths.AUDIO.glob("*.wav") if p.is_file())


def _pendientes() -> list[Path]:
    return [w for w in _wavs() if not paths.ruta_transcript(w.stem).exists()]


OCUPADO = 4  # el CLI devuelve esto cuando ya hay otra transcripción corriendo


def _transcribir_en_subproceso(wav: Path, extra: list[str]) -> int:
    orden = [sys.executable, "-m", "src", "transcribir", str(wav), *extra]
    print(f"\n  --> {wav.name}")
    return subprocess.run(orden, cwd=str(paths.RAIZ)).returncode


def vigilar(intervalo: float = INTERVALO, extra: list[str] | None = None) -> int:
    """Bucle de vigilancia. Corta con Ctrl+C."""
    extra = extra or []
    paths.asegurar_carpetas()

    tamanos: dict[Path, int] = {}
    fallidos: set[Path] = set()

    ya_estaban = _pendientes()
    print(f"  Vigilando {paths.relativa(paths.AUDIO)}/  (Ctrl+C para salir)")
    if ya_estaban:
        print(f"  Hay {len(ya_estaban)} audio(s) sin transcribir; arranco por ahí.")
    else:
        print("  Todo al día. Esperando audios nuevos...")

    try:
        while True:
            for wav in _pendientes():
                if wav in fallidos:
                    continue
                try:
                    tamano = wav.stat().st_size
                except OSError:
                    continue  # se movió o se borró entre el listado y el stat

                previo = tamanos.get(wav)
                tamanos[wav] = tamano

                # Estable = mismo tamaño en dos sondeos seguidos. Evita arrancar
                # sobre un WAV que todavía se está grabando.
                if previo is None or tamano != previo or tamano == 0:
                    continue

                codigo = _transcribir_en_subproceso(wav, extra)
                if codigo == OCUPADO:
                    # Otra transcripción tiene la GPU. No es un fallo del audio:
                    # se reintenta en el próximo sondeo.
                    tamanos.pop(wav, None)
                elif codigo != 0:
                    fallidos.add(wav)
                    print(f"  Falló la transcripción de {wav.name}. "
                          f"No lo reintento solo; corré "
                          f"'python -m src transcribir {wav.name}' para ver el error.",
                          file=sys.stderr)
            time.sleep(intervalo)
    except KeyboardInterrupt:
        print("\n  Listo, dejo de vigilar.")
        return 0
