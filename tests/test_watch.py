"""Watcher: estabilidad del archivo, subproceso por audio y no reintentar en bucle."""

from src import paths, watch
from tests.dobles import CasoConCarpetas, silencio


class TestVigilar(CasoConCarpetas):
    def setUp(self):
        super().setUp()
        self.llamadas = []
        self.ciclos = 0
        self.exito = True
        self.acciones = {}

        original_sleep = watch.time.sleep
        original_sub = watch._transcribir_en_subproceso
        watch.time.sleep = self._sleep
        watch._transcribir_en_subproceso = self._subproceso
        self.addCleanup(setattr, watch.time, "sleep", original_sleep)
        self.addCleanup(setattr, watch, "_transcribir_en_subproceso", original_sub)

    def _sleep(self, _):
        self.ciclos += 1
        accion = self.acciones.get(self.ciclos)
        if accion:
            accion()
        if self.ciclos >= self.max_ciclos:
            raise KeyboardInterrupt

    def _subproceso(self, wav, extra):
        self.llamadas.append((wav.name, self.ciclos, tuple(extra)))
        if self.exito:
            paths.ruta_transcript(wav.stem).write_text("ok", encoding="utf-8")
        return self.exito

    def _correr(self, max_ciclos=5, extra=None):
        self.max_ciclos = max_ciclos
        with silencio():
            watch.vigilar(intervalo=0, extra=extra)

    def test_espera_a_que_el_wav_deje_de_crecer(self):
        # Arrancar sobre un WAV a medio grabar daría una transcripción trunca.
        ruta = paths.ruta_audio("2026-08-22_perdidas")
        ruta.write_bytes(b"\0" * 100)
        self.acciones = {1: lambda: ruta.write_bytes(b"\0" * 200),
                         2: lambda: ruta.write_bytes(b"\0" * 300)}
        self._correr(max_ciclos=5)
        self.assertEqual(len(self.llamadas), 1)
        self.assertEqual(self.llamadas[0][1], 3,
                         "debió esperar dos sondeos con el mismo tamaño")

    def test_transcribe_una_sola_vez(self):
        self.escribir_wav("2026-08-22_perdidas")
        self._correr(max_ciclos=6)
        self.assertEqual(len(self.llamadas), 1)

    def test_ignora_los_que_ya_tienen_transcripcion(self):
        self.escribir_wav("2026-08-22_perdidas")
        paths.ruta_transcript("2026-08-22_perdidas").write_text("ya", encoding="utf-8")
        self._correr(max_ciclos=4)
        self.assertEqual(self.llamadas, [])

    def test_un_fallo_no_se_reintenta_en_bucle(self):
        # Sin esto, un WAV corrupto haría girar el watcher para siempre.
        self.exito = False
        self.escribir_wav("2026-08-22_perdidas")
        self._correr(max_ciclos=8)
        self.assertEqual(len(self.llamadas), 1)

    def test_un_fallo_no_frena_a_los_demas(self):
        self.escribir_wav("2026-08-22_perdidas")
        self.escribir_wav("2026-08-22_acr-l3")
        self.exito = False
        self._correr(max_ciclos=6)
        procesados = {nombre for nombre, _, _ in self.llamadas}
        self.assertEqual(len(procesados), 2)

    def test_toma_los_audios_que_ya_estaban(self):
        self.escribir_wav("2026-08-20_perdidas")
        self.escribir_wav("2026-08-21_acr-l3")
        self._correr(max_ciclos=6)
        self.assertEqual(len(self.llamadas), 2)

    def test_ignora_un_wav_de_cero_bytes(self):
        paths.ruta_audio("2026-08-22_vacio").write_bytes(b"")
        self._correr(max_ciclos=5)
        self.assertEqual(self.llamadas, [])

    def test_propaga_los_flags_al_subproceso(self):
        self.escribir_wav("2026-08-22_perdidas")
        self._correr(max_ciclos=5, extra=["--compute-type", "int8_float16"])
        self.assertEqual(self.llamadas[0][2], ("--compute-type", "int8_float16"))
