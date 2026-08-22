"""Dobles de las dependencias que no existen en una máquina de desarrollo.

`sounddevice` necesita una placa de audio y `faster-whisper` una GPU con CUDA.
Ninguna de las dos está disponible fuera de la notebook, así que la lógica del
proyecto —bucle de grabación, formato del transcript, estabilidad del watcher—
se verifica contra estos dobles. Lo que sí queda sin cubrir, y sólo se puede
probar en la notebook, es que CUDA levante y que PortAudio vea el micrófono.
"""

from __future__ import annotations

import array
import shutil
import sys
import tempfile
import threading
import time
import types
import unittest
from pathlib import Path
from unittest import mock

from src import paths

SAMPLE_RATE = 16000


class CasoConCarpetas(unittest.TestCase):
    """Base que apunta las carpetas del proyecto a un directorio temporal.

    Sin esto los tests escribirían en el /audio/ y /transcripts/ reales, que es
    justo donde vive el material de planta.
    """

    def setUp(self) -> None:
        # .resolve() imita a paths.RAIZ, que sale de Path(__file__).resolve().
        self.tmp = Path(tempfile.mkdtemp(prefix="ainote-test-")).resolve()
        self.addCleanup(shutil.rmtree, self.tmp, True)

        carpetas = {n: self.tmp / n
                    for n in ("audio", "transcripts", "notes", "minutas")}
        parches = {
            "RAIZ": self.tmp,
            "AUDIO": carpetas["audio"],
            "TRANSCRIPTS": carpetas["transcripts"],
            "NOTES": carpetas["notes"],
            "MINUTAS": carpetas["minutas"],
            "CARPETAS": tuple(carpetas.values()),
            "GLOSARIO": self.tmp / "glosario.txt",
        }
        for nombre, valor in parches.items():
            parche = mock.patch.object(paths, nombre, valor)
            parche.start()
            self.addCleanup(parche.stop)
        paths.asegurar_carpetas()

    def escribir_wav(self, nombre: str, muestras: int = 1000) -> Path:
        """WAV mínimo pero válido, para los tests que no miran el contenido."""
        import wave

        ruta = paths.ruta_audio(nombre)
        with wave.open(str(ruta), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(SAMPLE_RATE)
            w.writeframes(array.array("h", [0] * muestras).tobytes())
        return ruta


# --------------------------------------------------------------------------
# sounddevice
# --------------------------------------------------------------------------

DISPOSITIVOS = [
    {"name": "Micrófono interno", "max_input_channels": 2, "default_samplerate": 48000.0},
    {"name": "Altavoces", "max_input_channels": 0, "default_samplerate": 48000.0},
    {"name": "Jabra Speak 510", "max_input_channels": 1, "default_samplerate": 16000.0},
]


class _InputStreamFalso:
    """Emula PortAudio: un hilo que empuja bloques al callback."""

    def __init__(self, samplerate, channels, dtype, device, blocksize, callback):
        self.sr, self.bloque, self.callback = samplerate, blocksize, callback
        self.estado = None  # se puede setear para simular un desborde
        self._parar = threading.Event()
        self._hilo = None

    def __enter__(self):
        def correr():
            while not self._parar.is_set():
                # Onda cuadrada: sirve para comprobar que lo escrito no es silencio.
                mitad = self.bloque // 2
                datos = array.array("h", [8000] * mitad + [-8000] * (self.bloque - mitad))
                self.callback(datos, self.bloque, None, self.estado)
                time.sleep(self.bloque / self.sr)
        self._hilo = threading.Thread(target=correr, daemon=True)
        self._hilo.start()
        return self

    def __exit__(self, *_):
        self._parar.set()
        if self._hilo:
            self._hilo.join(timeout=2)


def modulo_sounddevice(acepta_16k: bool = True,
                       acepta_mono: bool = True) -> types.ModuleType:
    sd = types.ModuleType("sounddevice")
    sd.default = types.SimpleNamespace(device=(0, 1))
    sd.InputStream = _InputStreamFalso

    def query_devices(dev=None, kind=None):
        # Misma semántica que el sounddevice real: sin argumentos devuelve la
        # lista; con un índice o un `kind`, el dict de un dispositivo puntual.
        if dev is None and kind is None:
            return DISPOSITIVOS
        if dev is None:
            return DISPOSITIVOS[0]  # el de entrada por defecto
        return DISPOSITIVOS[dev]

    def check_input_settings(**kw):
        if not acepta_16k and kw.get("samplerate") == 16000:
            raise ValueError("samplerate no soportado")
        if not acepta_mono and kw.get("channels") == 1:
            raise ValueError("el dispositivo no acepta mono")

    sd.query_devices = query_devices
    sd.check_input_settings = check_input_settings
    return sd


# --------------------------------------------------------------------------
# faster-whisper
# --------------------------------------------------------------------------

class _Palabra:
    def __init__(self, start, end, word):
        self.start, self.end, self.word = start, end, word


class _Segmento:
    def __init__(self, start, end, text, words):
        self.start, self.end, self.text, self.words = start, end, text, words


class _Info:
    def __init__(self, duration):
        self.duration = duration
        self.language = "es"


# (inicio, fin, texto). El tercero viene vacío a propósito: Whisper devuelve
# segmentos en blanco y no deben llegar al transcript.
SEGMENTOS = [
    (0.30, 4.10, " Buenas, arrancamos con las pérdidas de la semana."),
    (7.05, 12.90, " En L4 tuvimos scrap alto en la soldadora."),
    (61.20, 66.00, "   "),
    (95.80, 102.4, " Hay que revisar el seteo de la boquilla."),
    (3671.0, 3675.5, " Cerramos con el ACR del turno noche."),
]

DURACION = 3720.0


def _palabras_de(inicio: float, fin: float, texto: str) -> list:
    """Reparte el texto del segmento en palabras, como hace Whisper de verdad.

    Dos detalles imitan al modelo real y sostienen las pruebas:

    - La primera palabra arranca 0,25 s **antes** que el segmento. Así se
      verifica que la línea toma el timestamp de la palabra y no el del
      segmento, que es el que derrapa.
    - Las palabras quedan contiguas, con huecos muy por debajo del umbral de
      corte. Un Whisper real no deja segundos de silencio entre dos palabras
      del mismo segmento; el doble tampoco debe hacerlo.
    """
    tokens = texto.split()
    if not tokens:
        return []
    arranque = inicio - 0.25
    paso = (fin - arranque) / len(tokens)
    palabras = [_Palabra(arranque + i * paso, arranque + i * paso + paso * 0.9,
                         " " + token)
                for i, token in enumerate(tokens)]
    palabras[-1].end = fin
    return palabras


class WhisperModelFalso:
    """Registra con qué se lo llamó, para poder verificar los parámetros."""

    ultima_instancia = None

    def __init__(self, modelo, device=None, compute_type=None, **kw):
        self.args_init = {"modelo": modelo, "device": device,
                          "compute_type": compute_type}
        self.args_transcribe = None
        self.descargado = False
        self.model = types.SimpleNamespace(unload_model=self._unload)
        WhisperModelFalso.ultima_instancia = self

    def _unload(self):
        self.descargado = True

    def transcribe(self, ruta, **kw):
        self.args_transcribe = kw
        segmentos = [_Segmento(inicio, fin, texto, _palabras_de(inicio, fin, texto))
                     for inicio, fin, texto in SEGMENTOS]
        return iter(segmentos), _Info(DURACION)


class WhisperModelRoto(WhisperModelFalso):
    def transcribe(self, ruta, **kw):
        raise RuntimeError("CUDA se cayó a mitad de camino")


def instalar_sounddevice(caso: unittest.TestCase, acepta_16k: bool = True,
                         acepta_mono: bool = True):
    modulo = modulo_sounddevice(acepta_16k, acepta_mono)
    parche = mock.patch.dict(sys.modules, {"sounddevice": modulo})
    parche.start()
    caso.addCleanup(parche.stop)
    return modulo


def instalar_whisper(caso: unittest.TestCase, clase=WhisperModelFalso):
    modulo = types.ModuleType("faster_whisper")
    modulo.WhisperModel = clase
    parche = mock.patch.dict(sys.modules, {"faster_whisper": modulo})
    parche.start()
    caso.addCleanup(parche.stop)
    return modulo


import contextlib  # noqa: E402
import io  # noqa: E402


@contextlib.contextmanager
def silencio():
    """Traga la salida de consola y la devuelve, para poder aseverar sobre ella."""
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
        yield buffer
