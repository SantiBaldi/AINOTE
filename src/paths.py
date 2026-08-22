"""Rutas del proyecto y convención de nombres `AAAA-MM-DD_<slug>`.

Único módulo que sabe cómo se llaman las cosas. Nadie construye nombres de
sesión a mano: todo pasa por acá para que audio, transcripción, nota y minuta
de una misma reunión compartan exactamente el mismo identificador.
"""

from __future__ import annotations

import re
import unicodedata
from datetime import date
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent

AUDIO = RAIZ / "audio"
TRANSCRIPTS = RAIZ / "transcripts"
NOTES = RAIZ / "notes"
MINUTAS = RAIZ / "minutas"

TASKS = RAIZ / "tasks.json"
INDEX = RAIZ / "index.sqlite"
GLOSARIO = RAIZ / "glosario.txt"

CARPETAS = (AUDIO, TRANSCRIPTS, NOTES, MINUTAS)

# 2026-08-22_perdidas  /  2026-08-22_bajada-de-gerencia-2
PATRON_SESION = re.compile(r"^(\d{4}-\d{2}-\d{2})_([a-z0-9]+(?:-[a-z0-9]+)*)$")

# El grabador produce WAV, pero se aceptan los formatos que PyAV decodifica:
# las grabaciones viejas suelen venir en m4a o mp3 y reencodearlas sólo
# perdería calidad y tiempo.
EXTENSIONES_AUDIO = (".wav", ".m4a", ".mp3", ".ogg", ".flac", ".opus", ".webm",
                     ".mp4", ".aac", ".wma")

# Ocho dígitos de fecha dentro de un nombre de archivo: 20260721
PATRON_FECHA_SUELTA = re.compile(r"(20\d{2})(\d{2})(\d{2})")


def asegurar_carpetas() -> None:
    """Crea las carpetas de datos si no existen. Idempotente."""
    for carpeta in CARPETAS:
        carpeta.mkdir(parents=True, exist_ok=True)


LARGO_MAXIMO_SLUG = 50


def slug(texto: str, largo_max: int = LARGO_MAXIMO_SLUG) -> str:
    """Normaliza un texto libre a `[a-z0-9-]`.

    Saca tildes por descomposición NFKD, pasa a minúsculas, y colapsa todo lo
    que no sea alfanumérico en guiones simples.

        >>> slug("Reunión de Pérdidas — L4")
        'reunion-de-perdidas-l4'
    """
    plano = unicodedata.normalize("NFKD", texto)
    plano = "".join(c for c in plano if not unicodedata.combining(c))
    plano = plano.lower()
    plano = re.sub(r"[^a-z0-9]+", "-", plano)
    plano = plano.strip("-")

    # Los nombres que ponen las grabadoras pueden pasar los 100 caracteres.
    # Se corta en un guión para no partir una palabra al medio.
    if len(plano) > largo_max:
        plano = plano[:largo_max]
        if "-" in plano:
            plano = plano[:plano.rindex("-")]
        plano = plano.strip("-")

    return plano or "reunion"


def nombre_sesion(texto: str, dia: date | None = None) -> str:
    """Arma `AAAA-MM-DD_<slug>` a partir de un texto libre."""
    dia = dia or date.today()
    return f"{dia.isoformat()}_{slug(texto)}"


def es_nombre_sesion(nombre: str) -> bool:
    """True si el nombre (sin extensión) respeta la convención."""
    return bool(PATRON_SESION.match(nombre))


def nombre_libre(texto: str, dia: date | None = None) -> str:
    """Como `nombre_sesion`, pero evita pisar una reunión existente.

    Si ya hay un `2026-08-22_perdidas` (en cualquiera de las carpetas de datos),
    devuelve `2026-08-22_perdidas-2`, y así. El sufijo va pegado al slug, con lo
    cual el resultado sigue matcheando `PATRON_SESION`.
    """
    base = nombre_sesion(texto, dia)
    if not _ocupado(base):
        return base
    for n in range(2, 1000):
        candidato = f"{base}-{n}"
        if not _ocupado(candidato):
            return candidato
    raise RuntimeError(f"No hay nombre libre para {base}; hay demasiadas colisiones.")


def _ocupado(nombre: str) -> bool:
    # `any(glob(...))` sobre el generador daria siempre True: hay que consumirlo.
    return any(any(carpeta.glob(f"{nombre}.*")) for carpeta in CARPETAS)


def ruta_audio(nombre: str, extension: str = ".wav") -> Path:
    return AUDIO / f"{nombre}{extension}"


def buscar_audio(nombre: str) -> Path | None:
    """Encuentra el audio de una sesión, sea cual sea su formato."""
    for extension in EXTENSIONES_AUDIO:
        candidato = AUDIO / f"{nombre}{extension}"
        if candidato.exists():
            return candidato
    return None


def audios() -> list[Path]:
    """Todos los audios de /audio/, del más reciente al más viejo por nombre."""
    if not AUDIO.exists():
        return []
    return sorted((p for p in AUDIO.iterdir()
                   if p.is_file() and p.suffix.lower() in EXTENSIONES_AUDIO),
                  reverse=True)


def fecha_de(ruta: Path) -> date:
    """Fecha de una grabación: la del nombre si la trae, si no la del archivo.

    Las grabadoras suelen estampar `20260721` en el nombre, y para un audio
    viejo eso es más confiable que la fecha de modificación, que cambia al
    copiarlo de una carpeta a otra.
    """
    encontrada = PATRON_FECHA_SUELTA.search(ruta.stem)
    if encontrada:
        anio, mes, dia = (int(g) for g in encontrada.groups())
        try:
            return date(anio, mes, dia)
        except ValueError:
            pass
    try:
        return date.fromtimestamp(ruta.stat().st_mtime)
    except OSError:
        return date.today()


def ruta_transcript(nombre: str) -> Path:
    return TRANSCRIPTS / f"{nombre}.md"


def ruta_segmentos(nombre: str) -> Path:
    return TRANSCRIPTS / f"{nombre}.segments.json"


def relativa(ruta: Path) -> str:
    r"""Ruta relativa a la raíz del proyecto, con `/` — para el front-matter.

    Los dos lados se resuelven antes de compararlos. En Windows una misma
    carpeta puede escribirse de dos formas —el nombre corto 8.3
    (`C:\Users\SBALDI~1\...`) y el largo (`C:\Users\sbaldisones\...`)—, y
    `resolve()` devuelve siempre el largo. Comparar una ruta resuelta contra una
    que no lo está hacía fallar `relative_to`, y el front-matter terminaba con
    la ruta absoluta de la máquina en lugar de `audio/<reunion>.wav`.
    """
    try:
        return ruta.resolve().relative_to(RAIZ.resolve()).as_posix()
    except ValueError:
        return ruta.as_posix()


def hms(segundos: float) -> str:
    """Segundos a `HH:MM:SS`. Trunca, no redondea."""
    total = max(0, int(segundos))
    return f"{total // 3600:02d}:{total % 3600 // 60:02d}:{total % 60:02d}"


def ms(segundos: float) -> str:
    """Segundos a `MM:SS`, para el cronómetro de grabación."""
    total = max(0, int(segundos))
    return f"{total // 60:02d}:{total % 60:02d}"
