"""Notas de la reunión con el timestamp del audio sellado en cada línea.

Es la mitad derecha de la pantalla partida y el reemplazo tipeado del "Smart
Pen" de la tablet: se escribe texto corrido, y los marcadores inline convierten
una línea en tarea, foco o encabezado sin salir de la nota.

El sello es lo que hace posible saltar de una línea al minuto exacto del audio.
Se toma cuando **se empieza** a escribir la línea, no cuando se termina: lo que
interesa es el momento de la reunión que la disparó, no cuánto tardó en
tipearse.

El archivo se escribe línea a línea. Si el proceso muere a los 50 minutos,
quedan 50 minutos de notas.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from . import paths

# Gramática de marcadores (CLAUDE.md §4). `@` y `!` son modificadores y pueden
# aparecer en cualquier posición de una línea de tarea o de foco.
SECCION = "#"
TAREA = "[]"
FOCO = "*"

PATRON_RESPONSABLE = re.compile(r"@([A-Za-zÁÉÍÓÚÜÑáéíóúüñ][\w.-]*)")
PATRON_VENCIMIENTO = re.compile(r"!(\d{1,2}-\d{1,2}(?:-\d{2,4})?)")


@dataclass
class Linea:
    """Una línea de la nota, con el momento del audio en que se empezó."""

    segundos: float
    texto: str

    @property
    def tipo(self) -> str:
        limpio = self.texto.lstrip()
        if limpio.startswith(TAREA):
            return "tarea"
        if limpio.startswith(SECCION):
            return "seccion"
        if limpio.startswith(FOCO):
            return "foco"
        return "libre"

    @property
    def responsables(self) -> list[str]:
        return PATRON_RESPONSABLE.findall(self.texto)

    @property
    def vencimiento(self) -> str | None:
        encontrado = PATRON_VENCIMIENTO.search(self.texto)
        return encontrado.group(1) if encontrado else None

    def formatear(self) -> str:
        return f"[{paths.hms(self.segundos)}] {self.texto}"


@dataclass
class Cuaderno:
    """Acumula las líneas de una sesión y las va guardando en `/notes/`.

    No modifica lo ya escrito: sólo agrega. `/notes/` es propiedad del usuario y
    el resto del sistema (tareas, minuta, índice) sólo lee de ahí.
    """

    nombre: str
    audio: Path | None = None
    lineas: list[Linea] = field(default_factory=list)
    _ruta: Path | None = None

    @property
    def ruta(self) -> Path:
        if self._ruta is None:
            self._ruta = paths.ruta_nota(self.nombre)
        return self._ruta

    def abrir(self) -> Path:
        """Crea el archivo con su encabezado. Idempotente dentro de una sesión."""
        self.ruta.parent.mkdir(parents=True, exist_ok=True)
        if not self.ruta.exists():
            self.ruta.write_text(self._encabezado(), encoding="utf-8", newline="\n")
        return self.ruta

    def _encabezado(self) -> str:
        audio = paths.relativa(self.audio) if self.audio else ""
        return ("---\n"
                f"reunion: {self.nombre}\n"
                f"audio: {audio}\n"
                f"iniciada: {datetime.now().isoformat(timespec='seconds')}\n"
                "---\n\n")

    def agregar(self, segundos: float, texto: str) -> Linea | None:
        """Sella y guarda una línea. Las vacías no se registran."""
        if not texto.strip():
            return None
        linea = Linea(segundos=max(0.0, float(segundos)), texto=texto.rstrip())
        self.lineas.append(linea)
        self.abrir()
        with self.ruta.open("a", encoding="utf-8", newline="\n") as archivo:
            archivo.write(linea.formatear() + "\n")
            archivo.flush()
        return linea

    def resumen(self) -> dict[str, int]:
        """Cuántas de cada tipo, para el pie de la pantalla."""
        cuenta = {"tarea": 0, "foco": 0, "seccion": 0, "libre": 0}
        for linea in self.lineas:
            cuenta[linea.tipo] += 1
        return cuenta
