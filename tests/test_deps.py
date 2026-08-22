"""Dependencias ausentes: mensaje accionable, no un traceback pelado."""

import builtins
import unittest
from unittest import mock

from src import deps


def _sin_modulo(nombre, error=ImportError):
    real = builtins.__import__

    def falso(name, *a, **kw):
        if name == nombre:
            raise error(f"No module named '{nombre}'")
        return real(name, *a, **kw)
    return mock.patch.object(builtins, "__import__", falso)


class TestDependencias(unittest.TestCase):
    def test_sin_sounddevice_explica_como_instalarlo(self):
        with _sin_modulo("sounddevice"):
            with self.assertRaises(deps.DependenciaFaltante) as caso:
                deps.sounddevice()
        mensaje = str(caso.exception)
        self.assertIn("sounddevice", mensaje)
        self.assertIn("requirements.txt", mensaje)

    def test_portaudio_que_no_carga_da_otro_mensaje(self):
        # `sounddevice` puede importar bien y reventar al abrir la DLL: es un
        # OSError, y la solución es distinta (reinstalar, no instalar).
        with _sin_modulo("sounddevice", OSError):
            with self.assertRaises(deps.DependenciaFaltante) as caso:
                deps.sounddevice()
        self.assertIn("reinstal", str(caso.exception).lower())

    def test_sin_faster_whisper_explica_como_instalarlo(self):
        with _sin_modulo("faster_whisper"):
            with self.assertRaises(deps.DependenciaFaltante) as caso:
                deps.whisper_model()
        self.assertIn("requirements.txt", str(caso.exception))


class TestDllsDeCuda(unittest.TestCase):
    """En Windows los DLLs de CUDA viven en site-packages, fuera de la ruta de búsqueda."""

    def test_sin_el_paquete_nvidia_no_hay_carpetas(self):
        with _sin_modulo("nvidia"):
            self.assertEqual(deps._carpetas_dll_cuda(), [])

    def test_encuentra_las_carpetas_bin(self):
        import sys, tempfile, types
        from pathlib import Path

        raiz = Path(tempfile.mkdtemp(prefix="nvidia-falso-"))
        for libreria in ("cublas", "cudnn"):
            (raiz / libreria / "bin").mkdir(parents=True)
        (raiz / "suelto.txt").write_text("no es una carpeta bin", encoding="utf-8")

        falso = types.ModuleType("nvidia")
        falso.__path__ = [str(raiz)]
        with mock.patch.dict(sys.modules, {"nvidia": falso}):
            carpetas = deps._carpetas_dll_cuda()
        self.assertEqual([c.parent.name for c in carpetas], ["cublas", "cudnn"])

    def test_fuera_de_windows_no_registra_nada(self):
        if not hasattr(deps.os, "add_dll_directory"):
            self.assertEqual(deps.registrar_dlls_cuda(), 0)

    def test_traduce_el_error_de_cublas(self):
        error = RuntimeError("Library cublas64_12.dll is not found or cannot be loaded")
        traducido = deps.traducir_error_de_dll(error)
        self.assertIsInstance(traducido, deps.DependenciaFaltante)
        self.assertIn("nvidia-cublas-cu12", str(traducido))
        self.assertIn("administrador", str(traducido))

    def test_traduce_el_error_de_cudnn(self):
        error = RuntimeError("Library cudnn_ops64_9.dll is not found")
        self.assertIn("nvidia-cudnn-cu12", str(deps.traducir_error_de_dll(error)))

    def test_no_se_mete_con_errores_ajenos(self):
        # Un OOM de CUDA o un modelo corrupto no se arreglan instalando nada.
        for ajeno in ("CUDA out of memory", "invalid model file", "no kernel image"):
            self.assertIsNone(deps.traducir_error_de_dll(RuntimeError(ajeno)))


class TestDiagnostico(unittest.TestCase):
    def test_sin_librerias_dice_como_instalarlas(self):
        with _sin_modulo("nvidia"):
            texto = "\n".join(deps.diagnostico_cuda())
        self.assertIn("requirements.txt", texto)

    def test_con_librerias_lista_las_carpetas(self):
        import sys, tempfile, types
        from pathlib import Path

        raiz = Path(tempfile.mkdtemp(prefix="nvidia-falso-"))
        (raiz / "cublas" / "bin").mkdir(parents=True)
        (raiz / "cublas" / "bin" / "cublas64_12.dll").write_bytes(b"")

        falso = types.ModuleType("nvidia")
        falso.__path__ = [str(raiz)]
        with mock.patch.dict(sys.modules, {"nvidia": falso}):
            texto = "\n".join(deps.diagnostico_cuda())
        self.assertIn("cublas", texto)
        self.assertNotIn("requirements.txt", texto)

    def test_paquete_sin_dlls_lo_dice(self):
        import sys, tempfile, types
        from pathlib import Path

        raiz = Path(tempfile.mkdtemp(prefix="nvidia-vacio-"))
        falso = types.ModuleType("nvidia")
        falso.__path__ = [str(raiz)]
        with mock.patch.dict(sys.modules, {"nvidia": falso}):
            texto = "\n".join(deps.diagnostico_cuda())
        self.assertIn("no tiene ninguna carpeta", texto)
