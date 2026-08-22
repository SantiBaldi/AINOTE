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


class TestNegociacionDeFormato(CasoConCarpetas):
    """16 kHz mono es lo ideal, pero no todo micrófono lo acepta."""

    def _grabar(self, **kw):
        import time as _t
        arranque = _t.monotonic()
        original = record._tecla_de_corte
        record._tecla_de_corte = lambda: _t.monotonic() - arranque > 0.4
        self.addCleanup(setattr, record, "_tecla_de_corte", original)
        with silencio() as salida:
            return record.grabar(self.tmp / "audio" / "prueba.wav", **kw), salida.getvalue()

    def test_el_caso_ideal_es_16k_mono(self):
        instalar_sounddevice(self)
        grabacion, _ = self._grabar()
        self.assertEqual((grabacion.sample_rate, grabacion.canales), (16000, 1))

    def test_mic_que_no_acepta_16k_cae_a_la_tasa_nativa_pero_sigue_mono(self):
        instalar_sounddevice(self, acepta_16k=False)
        grabacion, salida = self._grabar()
        self.assertEqual((grabacion.sample_rate, grabacion.canales), (48000, 1))
        self.assertIn("48000", salida)

    def test_mic_que_no_acepta_mono_graba_en_estereo_a_16k(self):
        # Antes esto reventaba con un error de PortAudio: se negociaba sólo el
        # sample rate y los canales quedaban clavados en 1.
        instalar_sounddevice(self, acepta_mono=False)
        grabacion, salida = self._grabar()
        self.assertEqual((grabacion.sample_rate, grabacion.canales), (16000, 2))
        self.assertIn("canal", salida)

    def test_mic_que_no_acepta_ni_una_cosa_ni_la_otra(self):
        instalar_sounddevice(self, acepta_16k=False, acepta_mono=False)
        grabacion, _ = self._grabar()
        self.assertEqual((grabacion.sample_rate, grabacion.canales), (48000, 2))

    def test_el_wav_declara_los_canales_reales(self):
        import wave
        instalar_sounddevice(self, acepta_mono=False)
        grabacion, _ = self._grabar()
        with wave.open(str(grabacion.ruta)) as w:
            self.assertEqual(w.getnchannels(), 2)
        # Y la duración tiene que contemplar que cada frame trae dos muestras.
        with wave.open(str(grabacion.ruta)) as w:
            self.assertAlmostEqual(grabacion.duracion,
                                   w.getnframes() / w.getframerate(), places=3)

    def test_dispositivo_inexistente_da_un_mensaje_en_castellano(self):
        from src.deps import DependenciaFaltante
        sd = instalar_sounddevice(self)

        def explotar(dev=None, kind=None):
            raise ValueError("error querying device -7")
        sd.query_devices = explotar

        with self.assertRaises(DependenciaFaltante) as caso, silencio():
            record.grabar(self.tmp / "audio" / "prueba.wav", dispositivo=99)
        self.assertIn("dispositivos", str(caso.exception))

    def test_ningun_formato_aceptado_da_un_mensaje_en_castellano(self):
        from src.deps import DependenciaFaltante
        sd = instalar_sounddevice(self)

        def rechazar_todo(**kw):
            raise ValueError("nope")
        sd.check_input_settings = rechazar_todo

        with self.assertRaises(DependenciaFaltante) as caso, silencio():
            record.grabar(self.tmp / "audio" / "prueba.wav")
        self.assertIn("micrófono", str(caso.exception))
