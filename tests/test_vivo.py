"""Transcripción en vivo: ventanas, offsets y el corte entre ventanas."""

import array
import types
import unittest
import wave

from src import vivo
from tests.dobles import CasoConCarpetas

SR = 16000


class ModeloDeVentanas:
    """Devuelve frases fijas dentro de la ventana que le toca.

    Cada llamada consume la siguiente entrada de `guion`, que es una lista de
    listas de `(inicio, fin, texto)` **relativos a la ventana**, como haría
    Whisper sobre un WAV suelto.
    """

    def __init__(self, guion):
        self.guion = list(guion)
        self.llamadas = []

    def transcribe(self, ruta, **kw):
        with wave.open(str(ruta)) as w:
            duracion = w.getnframes() / w.getframerate()
        self.llamadas.append({"ruta": ruta, "duracion": duracion, "kw": kw})

        crudos = self.guion.pop(0) if self.guion else []
        segmentos = []
        for inicio, fin, texto in crudos:
            palabras = [types.SimpleNamespace(start=inicio, end=fin, word=" " + texto)]
            segmentos.append(types.SimpleNamespace(start=inicio, end=fin,
                                                   text=" " + texto, words=palabras))
        return iter(segmentos), types.SimpleNamespace(duration=duracion, language="es")


def audio(segundos, canales=1):
    muestras = int(SR * segundos) * canales
    return array.array("h", [1000] * muestras).tobytes()


class TestVentanas(unittest.TestCase):
    def _transcriptor(self, guion, **kw):
        self.modelo = ModeloDeVentanas(guion)
        return vivo.TranscriptorVivo(self.modelo, sample_rate=SR, canales=1, **kw)

    def test_no_transcribe_hasta_llenar_la_ventana(self):
        t = self._transcriptor([[(0.0, 5.0, "hola")]], ventana_s=20.0)
        t.alimentar(audio(10))
        self.assertFalse(t.hay_ventana())
        self.assertEqual(t.procesar(), [])
        self.assertEqual(self.modelo.llamadas, [])

    def test_transcribe_al_llenarse(self):
        t = self._transcriptor([[(1.0, 4.0, "arrancamos con las pérdidas")]],
                               ventana_s=20.0)
        t.alimentar(audio(20))
        parciales = t.procesar()
        self.assertEqual(len(parciales), 1)
        self.assertEqual(parciales[0].texto, "arrancamos con las pérdidas")

    def test_los_tiempos_son_absolutos_del_audio(self):
        # Una frase en el segundo 2 de la tercera ventana no es el segundo 2 de
        # la reunión: sin el offset, saltar al audio desde la nota caería mal.
        t = self._transcriptor([[(0.0, 19.0, "primera")],
                                [(1.0, 18.0, "segunda")]], ventana_s=20.0)
        t.alimentar(audio(20))
        primera = t.procesar()[0]
        t.alimentar(audio(20))
        segunda = t.procesar()[0]
        self.assertEqual(primera.inicio, 0.0)
        self.assertAlmostEqual(segunda.inicio, 20.0, places=2)

    def test_el_sobrante_vuelve_a_la_ventana_siguiente(self):
        # La ventana termina en 18: esos 2 s finales podrían tener media palabra,
        # así que se reprocesan en vez de perderse en el borde.
        t = self._transcriptor([[(0.0, 18.0, "primera")], []], ventana_s=20.0)
        t.alimentar(audio(20))
        t.procesar()
        self.assertAlmostEqual(t.segundos_en_buffer, 2.0, places=2)
        self.assertAlmostEqual(t._offset_s, 18.0, places=2)

    def test_una_ventana_sin_habla_se_descarta(self):
        # Si no hubo frase, no hay dónde cortar: guardar el audio haría crecer
        # el buffer sin límite durante los silencios de la reunión.
        t = self._transcriptor([[]], ventana_s=20.0)
        t.alimentar(audio(20))
        self.assertEqual(t.procesar(), [])
        self.assertAlmostEqual(t.segundos_en_buffer, 0.0, places=2)
        self.assertAlmostEqual(t._offset_s, 20.0, places=2)

    def test_forzar_procesa_la_cola_al_cortar(self):
        t = self._transcriptor([[(0.0, 4.0, "última frase")]], ventana_s=20.0)
        t.alimentar(audio(5))
        self.assertEqual(t.procesar(), [])          # no llega a la ventana
        parciales = t.procesar(forzar=True)          # al cortar, sí
        self.assertEqual(parciales[0].texto, "última frase")

    def test_forzar_con_muy_poco_audio_no_llama_al_modelo(self):
        t = self._transcriptor([[(0.0, 1.0, "eh")]], ventana_s=20.0)
        t.alimentar(audio(1))
        self.assertEqual(t.procesar(forzar=True), [])
        self.assertEqual(self.modelo.llamadas, [])

    def test_audio_estereo_no_desplaza_los_tiempos(self):
        # Con dos canales cada segundo ocupa el doble: si el cálculo usara sólo
        # el sample rate, los offsets saldrían al doble de lo real.
        self.modelo = ModeloDeVentanas([[(0.0, 19.0, "x")], []])
        t = vivo.TranscriptorVivo(self.modelo, sample_rate=SR, canales=2,
                                  ventana_s=20.0)
        t.alimentar(audio(20, canales=2))
        self.assertTrue(t.hay_ventana())
        t.procesar()
        self.assertAlmostEqual(t._offset_s, 19.0, places=2)

    def test_le_pasa_el_wav_con_el_formato_correcto(self):
        t = self._transcriptor([[(0.0, 19.0, "x")]], ventana_s=20.0)
        t.alimentar(audio(20))
        t.procesar()
        self.assertAlmostEqual(self.modelo.llamadas[0]["duracion"], 20.0, places=2)

    def test_usa_los_parametros_de_calidad_del_proyecto(self):
        t = self._transcriptor([[(0.0, 19.0, "x")]], ventana_s=20.0,
                               hotwords="scrap, ACR")
        t.alimentar(audio(20))
        t.procesar()
        kw = self.modelo.llamadas[0]["kw"]
        self.assertEqual(kw["language"], "es")
        self.assertFalse(kw["condition_on_previous_text"])
        self.assertTrue(kw["word_timestamps"])
        self.assertFalse(kw["vad_filter"], "el VAD se come frases; ver CLAUDE.md")
        self.assertEqual(kw["hotwords"], "scrap, ACR")

    def test_parte_las_lineas_por_huecos_igual_que_la_definitiva(self):
        largo = [(0.0, 1.0, "uno"), (15.0, 16.0, "dos")]
        t = self._transcriptor([largo], ventana_s=20.0)
        t.alimentar(audio(20))
        parciales = t.procesar()
        self.assertEqual([p.texto for p in parciales], ["uno", "dos"])
        self.assertAlmostEqual(parciales[1].inicio, 15.0, places=2)

    def test_no_deja_archivos_temporales(self):
        import tempfile, pathlib

        antes = set(pathlib.Path(tempfile.gettempdir()).glob("ainote-vivo-*"))
        t = self._transcriptor([[(0.0, 19.0, "x")]], ventana_s=20.0)
        t.alimentar(audio(20))
        t.procesar()
        despues = set(pathlib.Path(tempfile.gettempdir()).glob("ainote-vivo-*"))
        self.assertEqual(antes, despues)


class TestReunionLarga(unittest.TestCase):
    def test_el_buffer_no_crece_sin_limite(self):
        # Una hora de reunión: si el buffer creciera, la RAM se iría al diablo.
        guion = [[(0.0, 19.5, f"frase {i}")] for i in range(180)]
        modelo = ModeloDeVentanas(guion)
        t = vivo.TranscriptorVivo(modelo, sample_rate=SR, canales=1, ventana_s=20.0)

        maximo = 0.0
        for _ in range(180):
            t.alimentar(audio(20))
            t.procesar()
            maximo = max(maximo, t.segundos_en_buffer)

        self.assertLess(maximo, vivo.VENTANA_S + vivo.SOBRANTE_MAXIMO_S + 1,
                        f"el buffer llegó a {maximo:.0f} s")
        # Todo el audio está procesado o en el buffer: nada se pierde ni se
        # cuenta dos veces a lo largo de la hora.
        alimentado = 180 * 20.0
        self.assertAlmostEqual(t._offset_s + t.segundos_en_buffer, alimentado,
                               delta=0.1)


class TestSobrante(unittest.TestCase):
    def test_el_sobrante_tiene_techo(self):
        # Si la última frase termina muy al principio de la ventana, guardar
        # todo el resto significaría reprocesar casi la ventana entera.
        modelo = ModeloDeVentanas([[(0.0, 2.0, "corta")], []])
        t = vivo.TranscriptorVivo(modelo, sample_rate=SR, canales=1, ventana_s=20.0)
        t.alimentar(audio(20))
        t.procesar()
        self.assertLessEqual(t.segundos_en_buffer, vivo.SOBRANTE_MAXIMO_S + 0.1)

    def test_un_sobrante_chico_se_respeta(self):
        modelo = ModeloDeVentanas([[(0.0, 18.0, "casi toda")], []])
        t = vivo.TranscriptorVivo(modelo, sample_rate=SR, canales=1, ventana_s=20.0)
        t.alimentar(audio(20))
        t.procesar()
        self.assertAlmostEqual(t.segundos_en_buffer, 2.0, places=2)
