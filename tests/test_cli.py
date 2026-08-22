"""CLI: resolución de rutas, códigos de salida y flags que sobreviven al subproceso."""

import unittest
from unittest import mock

from src import cli, gpu, paths
from src.deps import DependenciaFaltante
from tests.dobles import CasoConCarpetas, silencio


class TestResolverWav(CasoConCarpetas):
    def setUp(self):
        super().setUp()
        self.wav = self.escribir_wav("2026-08-22_perdidas")

    def test_acepta_las_tres_formas(self):
        for referencia in ("2026-08-22_perdidas", "2026-08-22_perdidas.wav",
                           str(self.wav)):
            self.assertEqual(cli._resolver_wav(referencia), self.wav, referencia)

    def test_audio_inexistente_lista_los_que_hay(self):
        # Equivocarse de fecha es facilísimo; el error tiene que mostrar los
        # nombres reales en vez de mandar a adivinar.
        self.escribir_wav("2026-08-20_acr-l3")
        with self.assertRaises(SystemExit) as caso:
            cli._resolver_wav("2026-08-25_perdidas")
        mensaje = str(caso.exception)
        self.assertIn("2026-08-25_perdidas", mensaje)
        self.assertIn("2026-08-22_perdidas", mensaje)
        self.assertIn("2026-08-20_acr-l3", mensaje)

    def test_sin_audios_lo_dice(self):
        self.wav.unlink()
        with self.assertRaises(SystemExit) as caso:
            cli._resolver_wav("cualquiera")
        self.assertIn("vacía", str(caso.exception))


class TestFlags(unittest.TestCase):
    def test_sobreviven_el_ida_y_vuelta_al_subproceso(self):
        # `grabar` y `vigilar` relanzan `transcribir` en otro proceso: si los
        # flags no se reconstruyen bien, la transcripción corre con otra config.
        args = cli.construir_parser().parse_args(
            ["vigilar", "--compute-type", "int8_float16", "--umbral-vad", "0.7",
             "--sin-json", "--device", "cpu"])
        extra = cli._extra_transcripcion(args)
        vuelta = cli.construir_parser().parse_args(["transcribir", "x.wav", *extra])
        self.assertEqual(vuelta.compute_type, "int8_float16")
        self.assertEqual(vuelta.umbral_vad, 0.7)
        self.assertEqual(vuelta.device, "cpu")
        self.assertTrue(vuelta.sin_json)

    def test_los_defaults_tambien_sobreviven(self):
        args = cli.construir_parser().parse_args(["vigilar"])
        extra = cli._extra_transcripcion(args)
        vuelta = cli.construir_parser().parse_args(["transcribir", "x.wav", *extra])
        self.assertEqual(vuelta.compute_type, args.compute_type)
        self.assertEqual(vuelta.modelo, args.modelo)
        self.assertFalse(vuelta.sin_json)

    def test_rechaza_un_compute_type_invalido(self):
        with self.assertRaises(SystemExit), silencio():
            cli.construir_parser().parse_args(["transcribir", "x", "--compute-type", "fp8"])


class TestCodigosDeSalida(CasoConCarpetas):
    def test_no_rehace_un_md_existente(self):
        self.escribir_wav("2026-08-22_perdidas")
        paths.ruta_transcript("2026-08-22_perdidas").write_text("ya", encoding="utf-8")
        with silencio() as salida:
            self.assertEqual(cli.main(["transcribir", "2026-08-22_perdidas"]), 0)
        self.assertIn("--forzar", salida.getvalue())

    def test_vram_insuficiente_sale_con_2(self):
        self.escribir_wav("2026-08-22_perdidas")
        with mock.patch.object(cli.transcribe, "transcribir",
                               side_effect=gpu.VramInsuficiente("faltan 1700 MB")):
            with silencio() as salida:
                codigo = cli.main(["transcribir", "2026-08-22_perdidas"])
        self.assertEqual(codigo, 2)
        self.assertIn("1700", salida.getvalue())

    def test_dependencia_faltante_sale_con_3(self):
        with mock.patch.object(cli, "cmd_gpu",
                               side_effect=DependenciaFaltante("falta sounddevice")):
            with silencio() as salida:
                codigo = cli.main(["gpu"])
        self.assertEqual(codigo, 3)
        self.assertIn("sounddevice", salida.getvalue())

    def test_ctrl_c_sale_con_130(self):
        with mock.patch.object(cli, "cmd_gpu", side_effect=KeyboardInterrupt):
            with silencio():
                self.assertEqual(cli.main(["gpu"]), 130)

    def test_gpu_sin_nvidia_smi_sale_con_1(self):
        with mock.patch.object(gpu, "consultar", return_value=None):
            with silencio():
                self.assertEqual(cli.main(["gpu"]), 1)

    def test_gpu_informa_que_modelo_entra(self):
        with mock.patch.object(gpu, "consultar", return_value=(2000, 8188)):
            with silencio() as salida:
                cli.main(["gpu"])
        texto = salida.getvalue()
        self.assertIn("float16", texto)
        self.assertIn("NO entra", texto)   # 2000 < 2600
        self.assertIn("entra", texto)      # int8_float16 sí


class TestGrabar(CasoConCarpetas):
    def test_encadena_la_transcripcion_en_un_subproceso(self):
        # Correr Whisper en el mismo proceso que PortAudio no devolvería la VRAM.
        from src import record

        grabacion = record.Grabacion(ruta=paths.ruta_audio("2026-08-22_perdidas"),
                                     duracion=180.0, sample_rate=16000,
                                     canales=1, desbordes=0)
        with mock.patch.object(record, "grabar", return_value=grabacion), \
             mock.patch.object(cli.subprocess, "run") as correr:
            correr.return_value = mock.Mock(returncode=0)
            with silencio():
                cli.main(["grabar", "perdidas"])
        orden = correr.call_args[0][0]
        self.assertIn("transcribir", orden)
        self.assertIn("-m", orden)

    def test_sin_transcribir_no_lanza_nada(self):
        from src import record

        grabacion = record.Grabacion(ruta=paths.ruta_audio("2026-08-22_perdidas"),
                                     duracion=180.0, sample_rate=16000,
                                     canales=1, desbordes=0)
        with mock.patch.object(record, "grabar", return_value=grabacion), \
             mock.patch.object(cli.subprocess, "run") as correr:
            with silencio():
                self.assertEqual(cli.main(["grabar", "perdidas", "--sin-transcribir"]), 0)
        correr.assert_not_called()

    def test_una_grabacion_vacia_no_encadena_y_avisa(self):
        from src import record

        grabacion = record.Grabacion(ruta=paths.ruta_audio("2026-08-22_perdidas"),
                                     duracion=0.0, sample_rate=16000,
                                     canales=1, desbordes=0)
        with mock.patch.object(record, "grabar", return_value=grabacion), \
             mock.patch.object(cli.subprocess, "run") as correr:
            with silencio() as salida:
                self.assertEqual(cli.main(["grabar", "perdidas"]), 1)
        correr.assert_not_called()
        self.assertIn("dispositivos", salida.getvalue())

    def test_el_nombre_sale_de_la_convencion(self):
        from src import record

        capturado = {}

        def falso_grabar(destino, **kw):
            capturado["destino"] = destino
            return record.Grabacion(ruta=destino, duracion=10.0,
                                    sample_rate=16000, canales=1, desbordes=0)

        with mock.patch.object(record, "grabar", side_effect=falso_grabar), \
             mock.patch.object(cli.subprocess, "run",
                               return_value=mock.Mock(returncode=0)):
            with silencio():
                cli.main(["grabar", "Reunión de Pérdidas — L4"])
        nombre = capturado["destino"].stem
        self.assertTrue(paths.es_nombre_sesion(nombre), nombre)
        self.assertTrue(nombre.endswith("_reunion-de-perdidas-l4"), nombre)
