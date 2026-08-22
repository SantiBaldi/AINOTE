"""Verificación de VRAM: tiene que fallar con un mensaje, no con un OOM."""

import subprocess
import unittest
from unittest import mock

from src import gpu


def _salida(texto):
    return mock.patch.object(
        gpu.subprocess, "run",
        return_value=subprocess.CompletedProcess([], 0, stdout=texto, stderr=""))


class TestConsultar(unittest.TestCase):
    def setUp(self):
        parche = mock.patch.object(gpu, "_binario", return_value="nvidia-smi")
        parche.start()
        self.addCleanup(parche.stop)

    def test_parsea_la_salida(self):
        with _salida("5432, 8188\n"):
            self.assertEqual(gpu.consultar(), (5432, 8188))

    def test_toma_la_primera_gpu(self):
        with _salida("5432, 8188\n1024, 4096\n"):
            self.assertEqual(gpu.consultar(), (5432, 8188))

    def test_tolera_decimales(self):
        with _salida("5432.0, 8188.0\n"):
            self.assertEqual(gpu.consultar(), (5432, 8188))

    def test_salida_ilegible_no_revienta(self):
        with _salida("no soy una medición\n"):
            self.assertIsNone(gpu.consultar())

    def test_nvidia_smi_que_falla_no_revienta(self):
        with mock.patch.object(gpu.subprocess, "run",
                               side_effect=subprocess.TimeoutExpired("nvidia-smi", 15)):
            self.assertIsNone(gpu.consultar())

    def test_sin_nvidia_smi_devuelve_none(self):
        with mock.patch.object(gpu, "_binario", return_value=None):
            self.assertIsNone(gpu.consultar())


class TestExigirVram(unittest.TestCase):
    def test_deja_pasar_si_alcanza(self):
        with mock.patch.object(gpu, "vram_libre_mb", return_value=6500):
            gpu.exigir_vram("float16")  # no debe levantar

    def test_aborta_si_no_alcanza(self):
        with mock.patch.object(gpu, "vram_libre_mb", return_value=900):
            with self.assertRaises(gpu.VramInsuficiente) as caso:
                gpu.exigir_vram("float16")
        mensaje = str(caso.exception)
        # El mensaje tiene que decir cuánto falta y qué hacer, no ser un OOM.
        self.assertIn("900", mensaje)
        self.assertIn("2600", mensaje)
        self.assertIn("int8_float16", mensaje)

    def test_int8_exige_menos_que_float16(self):
        self.assertLess(gpu.VRAM_MINIMA_MB["int8_float16"],
                        gpu.VRAM_MINIMA_MB["float16"])
        with mock.patch.object(gpu, "vram_libre_mb", return_value=2000):
            gpu.exigir_vram("int8_float16")
            with self.assertRaises(gpu.VramInsuficiente):
                gpu.exigir_vram("float16")

    def test_si_no_se_puede_medir_deja_seguir(self):
        # No poder medir no es lo mismo que no tener VRAM: bloquear el flujo
        # por no poder leer nvidia-smi sería peor que intentar cargar.
        with mock.patch.object(gpu, "vram_libre_mb", return_value=None):
            gpu.exigir_vram("float16")

    def test_compute_type_desconocido_usa_un_piso_razonable(self):
        with mock.patch.object(gpu, "vram_libre_mb", return_value=100):
            with self.assertRaises(gpu.VramInsuficiente):
                gpu.exigir_vram("bfloat16")
