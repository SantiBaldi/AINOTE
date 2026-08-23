"""Transcripción en vivo por ventanas, mientras se graba.

Whisper no transcribe en streaming: necesita un bloque de audio cerrado. Para
que aparezca texto durante la reunión se lo alimenta con ventanas de ~20 s.

El corte entre ventanas es el problema real: partir cada 20 s exactos cortaría
palabras al medio. En vez de eso, **la ventana siguiente arranca donde terminó
la última frase transcripta**. Si la ventana [0, 20] devuelve frases que
terminan en 18,4, esos 1,6 s finales vuelven al buffer y se transcriben junto
con lo que sigue. Nada se pierde en el borde y no hace falta deduplicar texto.

Lo que sale de acá es **provisional**: sirve para seguir la reunión y decidir
dónde poner un marcador. Al cerrar la sesión se transcribe el WAV completo de
una sola pasada, que es lo que queda en `/transcripts/`.
"""

from __future__ import annotations

import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path

from . import transcribe

VENTANA_S = 20.0        # cuánto audio juntar antes de transcribir
MINIMO_UTIL_S = 3.0     # menos que esto no vale la pena mandarlo al modelo

# Techo del sobrante que vuelve a la ventana siguiente. Sin este límite el
# buffer crece de a poco —una ventana de 20 s cuya última frase termina en 19,5
# deja medio segundo— y en una reunión de una hora llega a minutos: cada pasada
# procesaría más audio que la anterior y la latencia se iría acumulando.
SOBRANTE_MAXIMO_S = 3.0
ANCHO_MUESTRA = 2       # int16


@dataclass
class Parcial:
    """Una línea provisional de la transcripción en vivo."""

    inicio: float
    fin: float
    texto: str


class TranscriptorVivo:
    """Acumula audio y lo transcribe por ventanas.

    El modelo se inyecta ya cargado: durante la reunión se mantiene en la GPU
    (CLAUDE.md §2 lo contempla — mientras se graba corre únicamente Whisper) y
    recargarlo en cada ventana costaría más que transcribir.
    """

    def __init__(self, model, sample_rate: int, canales: int,
                 ventana_s: float = VENTANA_S,
                 hueco_max: float = transcribe.HUECO_MAXIMO_S,
                 hotwords: str = "", usar_vad: bool = transcribe.USAR_VAD):
        self.model = model
        self.sample_rate = sample_rate
        self.canales = canales
        self.ventana_s = ventana_s
        self.hueco_max = hueco_max
        self.hotwords = hotwords
        self.usar_vad = usar_vad

        self._buffer = bytearray()
        self._offset_s = 0.0  # segundo del audio donde arranca el buffer

    # -- entrada -----------------------------------------------------------

    def alimentar(self, bloque: bytes) -> None:
        """Suma audio crudo (int16) tal como sale del grabador."""
        self._buffer += bloque

    @property
    def segundos_en_buffer(self) -> float:
        marco = ANCHO_MUESTRA * self.canales
        return len(self._buffer) / marco / self.sample_rate if self.sample_rate else 0.0

    def hay_ventana(self) -> bool:
        return self.segundos_en_buffer >= self.ventana_s

    # -- salida ------------------------------------------------------------

    def procesar(self, forzar: bool = False) -> list[Parcial]:
        """Transcribe lo acumulado si alcanza. Devuelve las líneas nuevas.

        Con `forzar` procesa lo que haya, aunque no llegue a una ventana: es lo
        que se hace al cortar la grabación para no perder la última frase.
        """
        if not forzar and not self.hay_ventana():
            return []
        if self.segundos_en_buffer < MINIMO_UTIL_S:
            return []

        audio = bytes(self._buffer)
        arranque = self._offset_s

        with tempfile.TemporaryDirectory(prefix="ainote-vivo-") as carpeta:
            trozo = Path(carpeta) / "ventana.wav"
            self._escribir(trozo, audio)
            parciales = self._transcribir(trozo, arranque)

        self._recortar(parciales, arranque)
        return parciales

    def _escribir(self, ruta: Path, audio: bytes) -> None:
        """Vuelca la ventana a un WAV.

        Se pasa por disco a propósito: `faster-whisper` decodifica el archivo
        con PyAV, que resamplea a 16 kHz y mezcla a mono correctamente. Hacer
        esa conversión a mano exigiría un filtro antialias y sería peor.
        """
        with wave.open(str(ruta), "wb") as salida:
            salida.setnchannels(self.canales)
            salida.setsampwidth(ANCHO_MUESTRA)
            salida.setframerate(self.sample_rate)
            salida.writeframes(audio)

    def _transcribir(self, ruta: Path, arranque: float) -> list[Parcial]:
        segmentos, _ = self.model.transcribe(
            str(ruta),
            language=transcribe.IDIOMA,
            vad_filter=self.usar_vad,
            vad_parameters=None,
            word_timestamps=True,
            condition_on_previous_text=False,
            hotwords=self.hotwords or None,
            beam_size=1,  # en vivo importa la latencia, no el último punto de calidad
            no_speech_threshold=0.6,
            compression_ratio_threshold=2.4,
            temperature=0.0,
        )

        parciales = []
        for segmento in segmentos:
            for inicio, fin, texto in transcribe.partir_por_huecos(segmento,
                                                                   self.hueco_max):
                parciales.append(Parcial(inicio=arranque + inicio,
                                         fin=arranque + fin, texto=texto))
        return parciales

    def _recortar(self, parciales: list[Parcial], arranque: float) -> None:
        """Deja en el buffer el audio posterior a la última frase.

        Ese resto es lo que evita cortar una palabra al medio: vuelve a entrar
        en la próxima ventana en vez de perderse en el borde.
        """
        disponible = self.segundos_en_buffer
        if parciales:
            consumido = parciales[-1].fin - arranque
        else:
            # Silencio o ruido: no hay frase de dónde agarrarse, se descarta la
            # ventana entera.
            consumido = disponible

        # El sobrante tiene tope: ver SOBRANTE_MAXIMO_S.
        consumido = max(consumido, disponible - SOBRANTE_MAXIMO_S)

        marco = ANCHO_MUESTRA * self.canales
        bytes_consumidos = int(consumido * self.sample_rate) * marco
        bytes_consumidos = max(0, min(bytes_consumidos, len(self._buffer)))

        del self._buffer[:bytes_consumidos]
        self._offset_s = arranque + bytes_consumidos / marco / self.sample_rate
