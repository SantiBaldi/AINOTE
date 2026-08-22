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

from .paths import ms

try:  # sólo existe en Windows, que es el entorno de destino
    import msvcrt
except ImportError:  # pragma: no cover - desarrollo en Linux
    msvcrt = None

SAMPLE_RATE_DESEADO = 16000  # el que consume Whisper; evita un resampleo
CANALES = 1
ANCHO_MUESTRA = 2  # int16
BLOQUE = 1024


@dataclass
class Grabacion:
    ruta: Path
    duracion: float
    sample_rate: int
    desbordes: int


def listar_dispositivos() -> str:
    """Tabla de entradas de audio disponibles, para elegir el micrófono."""
    import sounddevice as sd

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


def _resolver_sample_rate(dispositivo, deseado: int) -> tuple[int, bool]:
    """Devuelve `(sample_rate, hubo_fallback)`.

    Si el micrófono no acepta 16 kHz, se graba a su tasa nativa. No es un
    problema: `faster-whisper` resamplea al decodificar. Sólo ocupa más disco.
    """
    import sounddevice as sd

    try:
        sd.check_input_settings(device=dispositivo, channels=CANALES,
                                dtype="int16", samplerate=deseado)
        return deseado, False
    except Exception:
        info = sd.query_devices(dispositivo, "input")
        return int(info["default_samplerate"]), True


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
    import sounddevice as sd

    sample_rate, fallback = _resolver_sample_rate(dispositivo, sample_rate)
    if fallback:
        print(f"  El micrófono no acepta {SAMPLE_RATE_DESEADO} Hz; "
              f"grabo a {sample_rate} Hz (Whisper resamplea solo).")

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
    stream = sd.InputStream(samplerate=sample_rate, channels=CANALES,
                            dtype="int16", device=dispositivo,
                            blocksize=BLOQUE, callback=callback)

    destino.parent.mkdir(parents=True, exist_ok=True)
    frames_escritos = 0

    with wave.open(str(destino), "wb") as wav:
        wav.setnchannels(CANALES)
        wav.setsampwidth(ANCHO_MUESTRA)
        wav.setframerate(sample_rate)

        def volcar(datos: bytes) -> None:
            nonlocal frames_escritos
            wav.writeframes(datos)
            frames_escritos += len(datos) // (ANCHO_MUESTRA * CANALES)

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
    return Grabacion(ruta=destino, duracion=duracion,
                     sample_rate=sample_rate, desbordes=desbordes)
