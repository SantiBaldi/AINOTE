"""Cerrojo entre procesos: nunca dos transcripciones a la vez."""

import os
import time
from unittest import mock

from src import lock, paths
from tests.dobles import CasoConCarpetas, silencio


class TestCerrojo(CasoConCarpetas):
    def test_se_toma_y_se_suelta(self):
        ruta = paths.RAIZ / ".ainote.lock"
        with lock.transcripcion():
            self.assertTrue(ruta.exists())
        self.assertFalse(ruta.exists(), "el cerrojo quedó tomado")

    def test_guarda_el_pid(self):
        with lock.transcripcion() as ruta:
            self.assertEqual(ruta.read_text(encoding="ascii"), str(os.getpid()))

    def test_el_segundo_no_entra(self):
        # Este es el escenario que importa: `grabar` encadena su transcripción
        # mientras el watcher lanza la suya sobre el mismo audio.
        with lock.transcripcion():
            with self.assertRaises(lock.Ocupado) as caso:
                with lock.transcripcion():
                    self.fail("entraron dos transcripciones a la vez")
        self.assertIn("VRAM", str(caso.exception))

    def test_se_suelta_aunque_reviente(self):
        with self.assertRaises(RuntimeError):
            with lock.transcripcion():
                raise RuntimeError("CUDA se cayó")
        self.assertFalse((paths.RAIZ / ".ainote.lock").exists())
        with lock.transcripcion():  # y el siguiente puede entrar
            pass

    def test_un_cerrojo_viejo_no_bloquea_para_siempre(self):
        # Si un proceso muere a lo bruto deja el archivo. Pasado el vencimiento
        # se descarta solo, en vez de exigir intervención manual.
        ruta = paths.RAIZ / ".ainote.lock"
        ruta.write_text("99999", encoding="ascii")
        viejo = time.time() - lock.VENCIMIENTO_S - 60
        os.utime(ruta, (viejo, viejo))
        with silencio() as salida, lock.transcripcion():
            pass
        self.assertIn("viejo", salida.getvalue())

    def test_un_cerrojo_reciente_si_bloquea(self):
        ruta = paths.RAIZ / ".ainote.lock"
        ruta.write_text("99999", encoding="ascii")
        with self.assertRaises(lock.Ocupado):
            with lock.transcripcion():
                pass
        # y no lo borra: el otro proceso lo sigue necesitando
        self.assertTrue(ruta.exists())


class TestIntegracionConTranscribir(CasoConCarpetas):
    def test_transcribir_respeta_el_cerrojo(self):
        from tests.dobles import instalar_whisper

        instalar_whisper(self)
        wav = self.escribir_wav("2026-08-22_perdidas")
        with lock.transcripcion():
            from src import transcribe
            with self.assertRaises(lock.Ocupado), silencio():
                transcribe.transcribir(wav, device="cpu")

    def test_el_cli_devuelve_4_si_esta_ocupado(self):
        from src import cli

        self.escribir_wav("2026-08-22_perdidas")
        with mock.patch.object(cli.transcribe, "transcribir",
                               side_effect=lock.Ocupado("hay otra corriendo")):
            with silencio() as salida:
                codigo = cli.main(["transcribir", "2026-08-22_perdidas"])
        self.assertEqual(codigo, 4)
        self.assertIn("otra", salida.getvalue())
