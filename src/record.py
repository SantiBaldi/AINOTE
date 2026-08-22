"""Grabador de micrófono a WAV, con cronómetro y corte por tecla.

Diseño (importante para la Fase 2): la captura y la escritura están separadas.
El callback de PortAudio sólo empuja bloques crudos a una cola y vuelve —nunca
hace I/O, nunca bloquea—; el hilo principal consume esa cola y escribe al WAV.
Para transcripción en vivo, la Fase 2 engancha un segundo consumidor de la
misma cola sin tocar este archivo.

El WAV se escribe bloque a bloque. Si el proceso muere a los 50 minutos de una
reunión de una hora, quedan 50 minutos de audio válido y reproducible.
"""

from __future__ import annotations

import queue
import sys
import time
import wave
from dataclasses import dataclass
from pathlib import Path

from .deps import DependenciaFaltante, sounddevice
from .paths import ms

try:  # sólo existe en Windows, que es el entorno de destino
    import msvcrt
except ImportError:  # pragma: no cover - desarrollo en Linux
    msvcrt = None

SAMPLE_RATE_DESEADO = 16000  # el que consume Whisper; evita un resampleo
CANALES_DESEADOS = 1         # mono: la mitad de disco, y a Whisper le da igual
ANCHO_MUESTRA = 2  # int16
BLOQUE = 1024


@dataclass
class Grabacion:
    ruta: Path
    duracion: float
    sample_rate: int
    canales: int
    desbordes: int


def listar_dispositivos() -> str:
    """Tabla de entradas de audio disponibles, para elegir el micrófono."""
    sd = sounddevice()

    lineas = ["Dispositivos de entrada disponibles:", ""]
    try:
        por_defecto = sd.default.device[0]
    except (TypeError, IndexError):
        por_defecto = None

    for indice, info in enumerate(sd.query_devices()):
        if info["max_input_channels"] < 1:
            continue
        marca = "*" if indice == por_defecto else " "
        lineas.append(
            f" {marca} [{indice:2d}] {info['name']}  "
            f"({info['max_input_channels']} canales, "
            f"{int(info['default_samplerate'])} Hz)"
        )
    lineas += ["", "  (*) el que se usa si no pasás --dispositivo",
               "  Elegí otro con: python -m src grabar --dispositivo N"]
    return "\n".join(lineas)


def _negociar_formato(dispositivo, deseado: int) -> tuple[int, int]:
    """Busca el mejor `(sample_rate, canales)` que el micrófono acepte.

    16 kHz mono es lo ideal: es lo que consume Whisper y ocupa la mitad. Pero
    hay entradas que no aceptan mono y otras que no aceptan 16 kHz, así que se
    prueban las cuatro combinaciones de menor a mayor costo en disco. Cualquiera
    sirve: `faster-whisper` resamplea y mezcla a mono al decodificar.

    Antes esto sólo negociaba el sample rate, y un micrófono que rechazara mono
    hacía fallar `InputStream` con un error de PortAudio en inglés.
    """
    sd = sounddevice()

    try:
        info = sd.query_devices(dispositivo, "input")
    except Exception as error:
        raise DependenciaFaltante(
            f"No puedo usar el dispositivo de entrada {dispositivo!r}: {error}\n"
            f"  Mirá cuáles hay con: python -m src dispositivos") from None

    nativo = int(info["default_samplerate"])
    maximo = int(info["max_input_channels"]) or 1
    candidatos = [(deseado, CANALES_DESEADOS), (nativo, CANALES_DESEADOS),
                  (deseado, maximo), (nativo, maximo)]

    for sample_rate, canales in candidatos:
        try:
            sd.check_input_settings(device=dispositivo, channels=canales,
                                    dtype="int16", samplerate=sample_rate)
            return sample_rate, canales
        except Exception:
            continue

    raise DependenciaFaltante(
        f"El micrófono '{info['name']}' no acepta ninguna combinación usable "
        f"(probé {deseado} y {nativo} Hz, 1 y {maximo} canales).\n"
        f"  Probá otro con: python -m src dispositivos")


def _tecla_de_corte() -> bool:
    """True si el usuario pidió cortar. No bloquea nunca."""
    if msvcrt is not None:
        while msvcrt.kbhit():
            tecla = msvcrt.getwch()
            if tecla in ("\x00", "\xe0"):  # tecla especial: consumir el scancode
                msvcrt.getwch()
                continue
            if tecla in ("\r", "\n", "q", "Q"):
                return True
        return False

    # Fallback fuera de Windows (desarrollo): Enter sobre una consola real.
    import select

    if not sys.stdin or not sys.stdin.isatty():
        return False
    listos, _, _ = select.select([sys.stdin], [], [], 0)
    if listos:
        sys.stdin.readline()
        return True
    return False


def _pintar_cronometro(transcurrido: float) -> None:
    sys.stdout.write(f"\r  [REC] {ms(transcurrido)}   "
                     f"Enter o 'q' para cortar    ")
    sys.stdout.flush()


def grabar(destino: Path, dispositivo=None,
           sample_rate: int = SAMPLE_RATE_DESEADO) -> Grabacion:
    """Graba del micrófono a `destino` hasta que se corte con una tecla."""
    sd = sounddevice()

    pedido = sample_rate
    sample_rate, canales = _negociar_formato(dispositivo, pedido)
    if (sample_rate, canales) != (pedido, CANALES_DESEADOS):
        print(f"  El micrófono no acepta {pedido} Hz mono; grabo a "
              f"{sample_rate} Hz / {canales} canal(es). Whisper lo convierte solo.")

    cola: queue.Queue[bytes] = queue.Queue()
    desbordes = 0

    def callback(indata, frames, tiempo, estado):
        nonlocal desbordes
        if estado:
            desbordes += 1
        # bytes() copia: el buffer de PortAudio se reutiliza en la próxima vuelta.
        cola.put(bytes(indata))

    # Se construye antes de abrir el WAV: si el dispositivo es inválido, la
    # excepción sale sin dejar un archivo vacío en /audio/ que el watcher
    # después intentaría transcribir.
    stream = sd.InputStream(samplerate=sample_rate, channels=canales,
                            dtype="int16", device=dispositivo,
                            blocksize=BLOQUE, callback=callback)

    destino.parent.mkdir(parents=True, exist_ok=True)
    frames_escritos = 0

    with wave.open(str(destino), "wb") as wav:
        wav.setnchannels(canales)
        wav.setsampwidth(ANCHO_MUESTRA)
        wav.setframerate(sample_rate)

        def volcar(datos: bytes) -> None:
            nonlocal frames_escritos
            wav.writeframes(datos)
            frames_escritos += len(datos) // (ANCHO_MUESTRA * canales)

        print(f"  Grabando en {destino.name}. Hablá tranquilo.")
        try:
            with stream:
                inicio = time.monotonic()
                ultimo_pintado = 0.0
                while True:
                    try:
                        volcar(cola.get(timeout=0.2))
                    except queue.Empty:
                        pass
                    ahora = time.monotonic()
                    if ahora - ultimo_pintado >= 0.25:
                        _pintar_cronometro(ahora - inicio)
                        ultimo_pintado = ahora
                    if _tecla_de_corte():
                        break
        except KeyboardInterrupt:
            print("\n  Corte con Ctrl+C.")
        finally:
            # El stream ya cerró: lo que quedó en la cola todavía es audio válido.
            while True:
                try:
                    volcar(cola.get_nowait())
                except queue.Empty:
                    break

    if frames_escritos == 0:
        destino.unlink(missing_ok=True)

    duracion = frames_escritos / sample_rate if sample_rate else 0.0
    sys.stdout.write("\r" + " " * 60 + "\r")
    sys.stdout.flush()
    return Grabacion(ruta=destino, duracion=duracion, sample_rate=sample_rate,
                     canales=canales, desbordes=desbordes)
