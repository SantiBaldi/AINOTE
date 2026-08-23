"""Una sesión de reunión: grabar y tomar notas al mismo tiempo.

Coordina el grabador —que corre en su propio hilo— con el cuaderno de notas.
Lo único que comparten es el reloj: el segundo de audio efectivamente grabado,
que es lo que se sella en cada línea y lo que después permite saltar de la nota
al minuto exacto.

Ese reloj sale de los frames escritos al WAV, no de `time.monotonic()`. Si el
micrófono pierde bloques —desbordes de buffer, la notebook trabada un
segundo— el reloj de pared se adelantaría respecto del audio y los sellos
quedarían corridos. Contar frames mantiene las dos cosas atadas.

La transcripción va después, al cerrar, en su propio proceso: mientras se graba
no hay ningún modelo cargado en la GPU.
"""

from __future__ import annotations

import subprocess
import sys
import threading
from pathlib import Path

from . import notas, paths


class Sesion:
    """Grabación y notas de una reunión, con el reloj compartido."""

    def __init__(self, titulo: str, dispositivo=None, grabador=None):
        paths.asegurar_carpetas()
        self.nombre = paths.nombre_libre(titulo or "reunion")
        self.audio = paths.ruta_audio(self.nombre)
        self.cuaderno = notas.Cuaderno(nombre=self.nombre, audio=self.audio)
        self.dispositivo = dispositivo
        self.grabacion = None
        self.error = None

        self._grabador = grabador
        self._segundos = 0.0
        self._cortar = threading.Event()
        self._hilo: threading.Thread | None = None

    # -- ciclo de vida -----------------------------------------------------

    def iniciar(self) -> None:
        """Arranca la grabación en segundo plano. No bloquea."""
        if self._hilo is not None:
            raise RuntimeError("La sesión ya está grabando.")
        self.cuaderno.abrir()
        self._hilo = threading.Thread(target=self._grabar, daemon=True)
        self._hilo.start()

    def _grabar(self) -> None:
        grabador = self._grabador
        if grabador is None:
            from . import record

            grabador = record.grabar
        try:
            self.grabacion = grabador(
                self.audio, dispositivo=self.dispositivo,
                debe_cortar=self._cortar.is_set,
                al_avanzar=self._marcar)
        except Exception as error:  # se reporta en la interfaz, no en un traceback
            self.error = error
            self._cortar.set()

    def _marcar(self, segundos: float) -> None:
        self._segundos = segundos

    @property
    def grabando(self) -> bool:
        return self._hilo is not None and self._hilo.is_alive()

    def segundos(self) -> float:
        """Segundos de audio grabados hasta ahora."""
        return self._segundos

    def cerrar(self, espera: float = 10.0):
        """Corta la grabación y espera a que el WAV quede cerrado."""
        self._cortar.set()
        if self._hilo is not None:
            self._hilo.join(timeout=espera)
            self._hilo = None
        return self.grabacion

    # -- notas -------------------------------------------------------------

    def anotar(self, texto: str, segundos: float | None = None):
        """Sella una línea con el momento de la reunión que la disparó.

        `segundos` se pasa explícito cuando la interfaz registró el instante en
        que se empezó a tipear: entre eso y el Enter puede haber pasado medio
        minuto, y lo que interesa es el arranque.
        """
        cuando = self.segundos() if segundos is None else segundos
        return self.cuaderno.agregar(cuando, texto)

    # -- después de la reunión ---------------------------------------------

    def transcribir(self, extra: list[str] | None = None) -> int:
        """Lanza la transcripción en otro proceso y devuelve su código de salida.

        En otro proceso a propósito: es la única forma verificable de que la
        VRAM vuelva al sistema cuando termina (CLAUDE.md §2).
        """
        if not self.audio.exists():
            return 1
        orden = [sys.executable, "-m", "src", "transcribir", str(self.audio),
                 *(extra or [])]
        return subprocess.run(orden, cwd=str(paths.RAIZ)).returncode

    def resumen(self) -> str:
        """Una línea con lo que se juntó, para el pie de la pantalla."""
        cuenta = self.cuaderno.resumen()
        partes = []
        if cuenta["tarea"]:
            partes.append(f"{cuenta['tarea']} "
                          f"{'tarea' if cuenta['tarea'] == 1 else 'tareas'}")
        if cuenta["foco"]:
            partes.append(f"{cuenta['foco']} "
                          f"{'foco' if cuenta['foco'] == 1 else 'focos'}")
        if cuenta["seccion"]:
            partes.append(f"{cuenta['seccion']} "
                          f"{'sección' if cuenta['seccion'] == 1 else 'secciones'}")
        return " · ".join(partes) if partes else "sin marcadores todavía"
