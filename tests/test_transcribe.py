"""Transcripción: formato del .md, timestamps y descarga del modelo."""

import json

from src import paths, transcribe
from tests.dobles import (DURACION, CasoConCarpetas, WhisperModelFalso,
                          WhisperModelRoto, instalar_whisper, silencio)


class TestTranscribir(CasoConCarpetas):
    def setUp(self):
        super().setUp()
        instalar_whisper(self)
        self.wav = self.escribir_wav("2026-08-22_perdidas")

    def _transcribir(self, **kw):
        with silencio() as salida:
            destino = transcribe.transcribir(self.wav, device="cpu", **kw)
        return destino, salida.getvalue()

    def test_genera_el_md_con_front_matter(self):
        destino, _ = self._transcribir()
        texto = destino.read_text(encoding="utf-8")
        self.assertTrue(texto.startswith("---\n"))
        self.assertIn("reunion: 2026-08-22_perdidas", texto)
        self.assertIn("audio: audio/2026-08-22_perdidas.wav", texto)
        self.assertIn(f"duracion: {paths.hms(DURACION)}", texto)
        self.assertIn("idioma: es", texto)
        self.assertIn("modelo: large-v3-turbo", texto)

    def test_el_front_matter_se_parsea_sin_pyyaml(self):
        # La Fase 5 lo va a leer con un regex; tiene que ser clave: valor plano.
        destino, _ = self._transcribir()
        bloque = destino.read_text(encoding="utf-8").split("---")[1]
        campos = dict(linea.split(": ", 1)
                      for linea in bloque.strip().splitlines())
        self.assertEqual(campos["reunion"], "2026-08-22_perdidas")
        self.assertEqual(campos["idioma"], "es")

    def test_usa_el_timestamp_de_la_primera_palabra(self):
        # El doble pone la primera palabra 0,25 s antes que el segmento. Si el
        # código usara seg.start, el primer segmento (0.30) daría 00:00:00
        # igual, pero el segundo (7.05 -> 6.80) daría 00:00:07 en vez de 06.
        destino, _ = self._transcribir()
        lineas = [l for l in destino.read_text(encoding="utf-8").splitlines()
                  if l.startswith("[")]
        self.assertEqual(lineas[1][:10], "[00:00:06]")
        self.assertEqual(lineas[3][:10], "[01:01:10]")

    def test_todas_las_lineas_tienen_timestamp_hhmmss(self):
        import re
        destino, _ = self._transcribir()
        cuerpo = destino.read_text(encoding="utf-8").split("---\n", 2)[2]
        for linea in filter(None, cuerpo.splitlines()):
            self.assertRegex(linea, r"^\[\d{2}:\d{2}:\d{2}\] \S")

    def test_descarta_los_segmentos_vacios(self):
        destino, _ = self._transcribir()
        lineas = [l for l in destino.read_text(encoding="utf-8").splitlines()
                  if l.startswith("[")]
        self.assertEqual(len(lineas), 4)  # 5 del doble, uno en blanco

    def test_pasa_los_parametros_que_exige_el_diseno(self):
        self._transcribir()
        kw = WhisperModelFalso.ultima_instancia.args_transcribe
        self.assertEqual(kw["language"], "es")
        self.assertTrue(kw["word_timestamps"])
        self.assertFalse(kw["condition_on_previous_text"])
        self.assertTrue(kw["vad_filter"])
        self.assertEqual(kw["vad_parameters"]["max_speech_duration_s"], 25.0)
        self.assertGreater(kw["vad_parameters"]["speech_pad_ms"], 0)

    def test_inyecta_el_glosario_como_hotwords(self):
        paths.GLOSARIO.write_text("# comentario\nscrap\nACR\n\nboquilla  # inline\n",
                                  encoding="utf-8")
        self._transcribir()
        hotwords = WhisperModelFalso.ultima_instancia.args_transcribe["hotwords"]
        self.assertIn("scrap", hotwords)
        self.assertIn("ACR", hotwords)
        self.assertIn("boquilla", hotwords)
        self.assertNotIn("comentario", hotwords)
        self.assertNotIn("inline", hotwords)

    def test_sin_glosario_transcribe_igual(self):
        destino, _ = self._transcribir()
        self.assertIsNone(
            WhisperModelFalso.ultima_instancia.args_transcribe["hotwords"])
        self.assertTrue(destino.exists())

    def test_glosario_largo_se_recorta(self):
        # El prompt de Whisper tiene tope; pasarse lo trunca o lo hace fallar.
        paths.GLOSARIO.write_text("\n".join(f"termino{i}" for i in range(300)),
                                  encoding="utf-8")
        _, salida = self._transcribir()
        hotwords = WhisperModelFalso.ultima_instancia.args_transcribe["hotwords"]
        self.assertLessEqual(len(hotwords), transcribe.LIMITE_HOTWORDS)
        self.assertIn("glosario", salida.lower())

    def test_genera_el_indice_de_segmentos(self):
        self._transcribir()
        datos = json.loads(
            paths.ruta_segmentos("2026-08-22_perdidas").read_text(encoding="utf-8"))
        self.assertEqual(datos["reunion"], "2026-08-22_perdidas")
        self.assertEqual(len(datos["segmentos"]), 4)
        primero = datos["segmentos"][0]
        self.assertEqual(primero["inicio"], 0.05)
        self.assertLess(primero["inicio"], primero["fin"])

    def test_sin_json_no_lo_genera(self):
        self._transcribir(con_json=False)
        self.assertFalse(paths.ruta_segmentos("2026-08-22_perdidas").exists())

    def test_descarga_el_modelo_al_terminar(self):
        self._transcribir()
        self.assertTrue(WhisperModelFalso.ultima_instancia.descargado)

    def test_no_queda_el_archivo_parcial(self):
        destino, _ = self._transcribir()
        self.assertFalse(destino.with_suffix(".md.tmp").exists())

    def test_avisa_si_el_nombre_no_sigue_la_convencion(self):
        wav = self.escribir_wav("grabacion_random")
        with silencio() as salida:
            transcribe.transcribir(wav, device="cpu")
        self.assertIn("AAAA-MM-DD", salida.getvalue())

    def test_audio_inexistente(self):
        with self.assertRaises(FileNotFoundError):
            transcribe.transcribir(paths.AUDIO / "no-existe.wav", device="cpu")


class TestFallas(CasoConCarpetas):
    def setUp(self):
        super().setUp()
        instalar_whisper(self, WhisperModelRoto)
        self.wav = self.escribir_wav("2026-08-22_perdidas")

    def test_un_error_no_deja_un_md_a_medio_escribir(self):
        # El watcher usa la existencia del .md como marca de "ya procesado":
        # un .md incompleto sería un falso positivo permanente.
        with self.assertRaises(RuntimeError), silencio():
            transcribe.transcribir(self.wav, device="cpu")
        destino = paths.ruta_transcript("2026-08-22_perdidas")
        self.assertFalse(destino.exists())
        self.assertFalse(destino.with_suffix(".md.tmp").exists())

    def test_un_error_igual_descarga_el_modelo(self):
        with self.assertRaises(RuntimeError), silencio():
            transcribe.transcribir(self.wav, device="cpu")
        self.assertTrue(WhisperModelFalso.ultima_instancia.descargado)


class TestGlosario(CasoConCarpetas):
    def test_ignora_comentarios_y_lineas_vacias(self):
        paths.GLOSARIO.write_text("# encabezado\n\nscrap\n  \nOEE  # al costado\n",
                                  encoding="utf-8")
        self.assertEqual(transcribe.cargar_glosario(), "scrap, OEE")

    def test_archivo_ausente_da_vacio(self):
        self.assertEqual(transcribe.cargar_glosario(), "")


class TestErroresDeCuda(CasoConCarpetas):
    """Un DLL de CUDA ausente tiene que explicar cómo conseguirlo."""

    def _correr(self, clase):
        instalar_whisper(self, clase)
        wav = self.escribir_wav("2026-08-22_perdidas")
        with silencio():
            return transcribe.transcribir(wav, device="cuda")

    def test_dll_ausente_al_transcribir_se_traduce(self):
        from src.deps import DependenciaFaltante

        class SinCublas(WhisperModelFalso):
            def transcribe(self, ruta, **kw):
                raise RuntimeError(
                    "Library cublas64_12.dll is not found or cannot be loaded")

        with self.assertRaises(DependenciaFaltante) as caso:
            self._correr(SinCublas)
        self.assertIn("nvidia-cublas-cu12", str(caso.exception))

    def test_dll_ausente_al_cargar_el_modelo_se_traduce(self):
        from src.deps import DependenciaFaltante

        class NoCarga(WhisperModelFalso):
            def __init__(self, *a, **kw):
                raise RuntimeError("Library cudnn_ops64_9.dll is not found")

        with self.assertRaises(DependenciaFaltante) as caso:
            self._correr(NoCarga)
        self.assertIn("nvidia-cudnn-cu12", str(caso.exception))

    def test_un_oom_no_se_disfraza_de_dependencia_faltante(self):
        class SinMemoria(WhisperModelFalso):
            def transcribe(self, ruta, **kw):
                raise RuntimeError("CUDA failed with error out of memory")

        with self.assertRaises(RuntimeError) as caso:
            self._correr(SinMemoria)
        self.assertIn("out of memory", str(caso.exception))

    def test_el_dll_ausente_igual_descarga_el_modelo(self):
        from src.deps import DependenciaFaltante

        class SinCublas(WhisperModelFalso):
            def transcribe(self, ruta, **kw):
                raise RuntimeError("Library cublas64_12.dll is not found")

        with self.assertRaises(DependenciaFaltante):
            self._correr(SinCublas)
        self.assertTrue(WhisperModelFalso.ultima_instancia.descargado)
