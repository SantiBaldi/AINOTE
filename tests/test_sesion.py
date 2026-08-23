"""Sesión: coordinación entre el grabador y el cuaderno."""

import threading
import time

from src import paths, record, sesion
from tests.dobles import CasoConCarpetas


class GrabadorFalso:
    """Imita a `record.grabar`: bloquea hasta que le pidan cortar."""

    def __init__(self, velocidad=50.0, explota=None):
        self.velocidad = velocidad   # segundos de audio por segundo real
        self.explota = explota
        self.arrancado = threading.Event()

    def __call__(self, destino, dispositivo=None, debe_cortar=None, al_avanzar=None):
        if self.explota:
            raise self.explota
        destino.write_bytes(b"RIFF" + b"\0" * 100)
        self.arrancado.set()
        inicio = time.monotonic()
        while not debe_cortar():
            if al_avanzar:
                al_avanzar((time.monotonic() - inicio) * self.velocidad)
            time.sleep(0.01)
        transcurrido = (time.monotonic() - inicio) * self.velocidad
        return record.Grabacion(ruta=destino, duracion=transcurrido,
                                sample_rate=16000, canales=1, desbordes=0)


class TestSesion(CasoConCarpetas):
    def _sesion(self, **kw):
        self.grabador = GrabadorFalso(**kw)
        return sesion.Sesion("perdidas", grabador=self.grabador)

    def test_el_nombre_sigue_la_convencion(self):
        s = self._sesion()
        self.assertTrue(paths.es_nombre_sesion(s.nombre), s.nombre)
        self.assertTrue(s.nombre.endswith("_perdidas"))

    def test_graba_en_segundo_plano_sin_bloquear(self):
        s = self._sesion()
        s.iniciar()
        self.assertTrue(self.grabador.arrancado.wait(2), "no arrancó el grabador")
        self.assertTrue(s.grabando)
        s.cerrar()
        self.assertFalse(s.grabando)

    def test_el_reloj_avanza(self):
        s = self._sesion()
        s.iniciar()
        self.grabador.arrancado.wait(2)
        time.sleep(0.1)
        self.assertGreater(s.segundos(), 0.0)
        s.cerrar()

    def test_la_nota_se_puede_escribir_mientras_graba(self):
        s = self._sesion()
        s.iniciar()
        self.grabador.arrancado.wait(2)
        time.sleep(0.05)
        linea = s.anotar("[] revisar boquilla @Torres")
        s.cerrar()
        self.assertGreater(linea.segundos, 0.0)
        self.assertIn("[] revisar boquilla @Torres",
                      s.cuaderno.ruta.read_text(encoding="utf-8"))

    def test_se_puede_sellar_con_un_momento_anterior(self):
        # Entre que se empieza a tipear y se aprieta Enter puede pasar medio
        # minuto; lo que importa es el arranque.
        s = self._sesion()
        s.iniciar()
        self.grabador.arrancado.wait(2)
        linea = s.anotar("* scrap alto", segundos=42.0)
        s.cerrar()
        self.assertEqual(linea.segundos, 42.0)
        self.assertIn("[00:00:42] * scrap alto",
                      s.cuaderno.ruta.read_text(encoding="utf-8"))

    def test_la_nota_existe_aunque_no_se_escriba_nada(self):
        s = self._sesion()
        s.iniciar()
        self.grabador.arrancado.wait(2)
        s.cerrar()
        self.assertTrue(s.cuaderno.ruta.exists())

    def test_el_audio_y_la_nota_comparten_nombre(self):
        s = self._sesion()
        self.assertEqual(s.audio.stem, s.cuaderno.ruta.stem)

    def test_un_fallo_del_microfono_no_cuelga_la_sesion(self):
        s = self._sesion(explota=ValueError("dispositivo inexistente"))
        s.iniciar()
        s.cerrar()
        self.assertIsInstance(s.error, ValueError)
        self.assertFalse(s.grabando)

    def test_no_se_puede_iniciar_dos_veces(self):
        s = self._sesion()
        s.iniciar()
        with self.assertRaises(RuntimeError):
            s.iniciar()
        s.cerrar()

    def test_cerrar_devuelve_la_grabacion(self):
        s = self._sesion()
        s.iniciar()
        self.grabador.arrancado.wait(2)
        time.sleep(0.05)
        grabacion = s.cerrar()
        self.assertIsNotNone(grabacion)
        self.assertGreater(grabacion.duracion, 0.0)

    def test_resumen(self):
        s = self._sesion()
        self.assertEqual(s.resumen(), "sin marcadores todavía")
        s.anotar("[] una", segundos=1)
        self.assertEqual(s.resumen(), "1 tarea")
        s.anotar("[] otra", segundos=2)
        s.anotar("* foco", segundos=3)
        s.anotar("# sección", segundos=4)
        self.assertEqual(s.resumen(), "2 tareas · 1 foco · 1 sección")

    def test_transcribir_corre_en_otro_proceso(self):
        from unittest import mock

        s = self._sesion()
        s.audio.write_bytes(b"x")
        with mock.patch.object(sesion.subprocess, "run") as correr:
            correr.return_value = mock.Mock(returncode=0)
            s.transcribir(["--sin-json"])
        orden = correr.call_args[0][0]
        self.assertIn("transcribir", orden)
        self.assertIn("--sin-json", orden)

    def test_transcribir_sin_audio_no_hace_nada(self):
        s = self._sesion()
        self.assertEqual(s.transcribir(), 1)
