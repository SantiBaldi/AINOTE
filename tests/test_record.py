"""Bucle de grabación: escritura incremental, corte y drenaje de la cola."""

import array
import time
import wave

from src import record
from tests.dobles import CasoConCarpetas, instalar_sounddevice, silencio

SAMPLE_RATE = 16000


class TestGrabar(CasoConCarpetas):
    def setUp(self):
        super().setUp()
        self.sd = instalar_sounddevice(self)

    def _grabar(self, segundos=1.0, **kw):
        arranque = time.monotonic()
        parche = lambda: time.monotonic() - arranque > segundos  # noqa: E731
        original = record._tecla_de_corte
        record._tecla_de_corte = parche
        self.addCleanup(setattr, record, "_tecla_de_corte", original)
        with silencio():
            return record.grabar(self.tmp / "audio" / "prueba.wav", **kw)

    def test_produce_un_wav_valido(self):
        grabacion = self._grabar(1.0)
        with wave.open(str(grabacion.ruta)) as w:
            self.assertEqual(w.getnchannels(), 1)
            self.assertEqual(w.getsampwidth(), 2)
            self.assertEqual(w.getframerate(), SAMPLE_RATE)
            muestras = array.array("h", w.readframes(w.getnframes()))
        self.assertGreater(len(muestras), 0)
        self.assertGreater(max(abs(m) for m in muestras), 0,
                           "el WAV quedó en silencio: no se escribió lo capturado")

    def test_la_duracion_reportada_coincide_con_el_wav(self):
        grabacion = self._grabar(1.0)
        with wave.open(str(grabacion.ruta)) as w:
            real = w.getnframes() / w.getframerate()
        self.assertAlmostEqual(grabacion.duracion, real, places=3)

    def test_escribe_incremental_y_no_acumula_en_ram(self):
        # Un archivo que crece mientras se graba es la garantía de que morirse a
        # los 50 minutos deja 50 minutos utilizables.
        ruta = self.tmp / "audio" / "prueba.wav"
        tamanos = []
        arranque = time.monotonic()

        def cortar():
            if ruta.exists():
                tamanos.append(ruta.stat().st_size)
            return time.monotonic() - arranque > 1.2

        original = record._tecla_de_corte
        record._tecla_de_corte = cortar
        self.addCleanup(setattr, record, "_tecla_de_corte", original)
        with silencio():
            record.grabar(ruta)

        self.assertGreater(len(set(tamanos)), 1,
                           "el WAV no creció durante la grabación")

    def test_cae_a_la_tasa_nativa_si_el_mic_no_acepta_16k(self):
        instalar_sounddevice(self, acepta_16k=False)
        grabacion = self._grabar(0.5)
        self.assertEqual(grabacion.sample_rate, 48000)

    def test_dispositivo_invalido_no_deja_un_wav_vacio(self):
        def explotar(**kw):
            raise ValueError("dispositivo inexistente")
        self.sd.InputStream = explotar
        with self.assertRaises(ValueError), silencio():
            record.grabar(self.tmp / "audio" / "prueba.wav")
        self.assertFalse((self.tmp / "audio" / "prueba.wav").exists(),
                         "quedó un WAV de 44 bytes que el watcher intentaría transcribir")

    def test_cuenta_los_desbordes_del_buffer(self):
        original = self.sd.InputStream

        class ConDesborde(original):
            def __init__(self, *a, **kw):
                super().__init__(*a, **kw)
                self.estado = "input overflow"

        self.sd.InputStream = ConDesborde
        grabacion = self._grabar(0.5)
        self.assertGreater(grabacion.desbordes, 0)

    def test_ctrl_c_cierra_el_wav_igual(self):
        def explotar():
            raise KeyboardInterrupt
        original = record._tecla_de_corte
        record._tecla_de_corte = explotar
        self.addCleanup(setattr, record, "_tecla_de_corte", original)
        ruta = self.tmp / "audio" / "prueba.wav"
        with silencio():
            record.grabar(ruta)
        # El archivo tiene que quedar cerrado y legible, no corrupto.
        if ruta.exists():
            with wave.open(str(ruta)) as w:
                self.assertEqual(w.getframerate(), SAMPLE_RATE)


class TestListarDispositivos(CasoConCarpetas):
    def test_lista_solo_entradas(self):
        instalar_sounddevice(self)
        salida = record.listar_dispositivos()
        self.assertIn("Micrófono interno", salida)
        self.assertIn("Jabra Speak 510", salida)
        self.assertNotIn("Altavoces", salida)
