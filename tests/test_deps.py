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
