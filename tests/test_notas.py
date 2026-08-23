"""Notas con timestamp sellado y gramática de marcadores."""

import unittest

from src import notas, paths
from tests.dobles import CasoConCarpetas


class TestGramatica(unittest.TestCase):
    def _tipo(self, texto):
        return notas.Linea(0.0, texto).tipo

    def test_reconoce_los_cuatro_tipos(self):
        self.assertEqual(self._tipo("# Cambio de formato L4"), "seccion")
        self.assertEqual(self._tipo("[] revisar seteo de boquilla"), "tarea")
        self.assertEqual(self._tipo("* scrap alto en soldadora"), "foco")
        self.assertEqual(self._tipo("se habló del turno noche"), "libre")

    def test_tolera_espacios_al_principio(self):
        self.assertEqual(self._tipo("   [] tarea indentada"), "tarea")

    def test_un_marcador_al_medio_no_cuenta(self):
        # Sólo el arranque de la línea define el tipo.
        self.assertEqual(self._tipo("hablamos de [] y de *"), "libre")

    def test_responsables_en_cualquier_posicion(self):
        linea = notas.Linea(0.0, "[] @Torres revisa la boquilla con @Gomez")
        self.assertEqual(linea.responsables, ["Torres", "Gomez"])

    def test_responsable_con_tilde(self):
        self.assertEqual(notas.Linea(0.0, "[] avisar a @Núñez").responsables,
                         ["Núñez"])

    def test_vencimiento(self):
        self.assertEqual(notas.Linea(0.0, "[] x !05-09").vencimiento, "05-09")
        self.assertEqual(notas.Linea(0.0, "[] x !5-9-2026").vencimiento, "5-9-2026")
        self.assertIsNone(notas.Linea(0.0, "[] x").vencimiento)

    def test_modificadores_en_una_linea_de_foco(self):
        linea = notas.Linea(0.0, "* scrap alto @Torres !05-09")
        self.assertEqual(linea.tipo, "foco")
        self.assertEqual(linea.responsables, ["Torres"])
        self.assertEqual(linea.vencimiento, "05-09")

    def test_un_mail_no_es_un_responsable_del_dominio(self):
        # Caso incómodo pero real: se prefiere capturar de más y filtrar después.
        self.assertEqual(notas.Linea(0.0, "escribir a juan@planta.com").responsables,
                         ["planta.com"])

    def test_formato_de_linea(self):
        self.assertEqual(notas.Linea(192.0, "[] x").formatear(), "[00:03:12] [] x")


class TestCuaderno(CasoConCarpetas):
    def setUp(self):
        super().setUp()
        self.cuaderno = notas.Cuaderno(nombre="2026-08-22_perdidas",
                                       audio=paths.ruta_audio("2026-08-22_perdidas"))

    def test_escribe_encabezado_y_lineas(self):
        self.cuaderno.agregar(12.0, "# Pérdidas semana 34")
        self.cuaderno.agregar(45.5, "[] revisar boquilla @Torres")
        texto = self.cuaderno.ruta.read_text(encoding="utf-8")
        self.assertIn("reunion: 2026-08-22_perdidas", texto)
        self.assertIn("audio: audio/2026-08-22_perdidas.wav", texto)
        self.assertIn("[00:00:12] # Pérdidas semana 34", texto)
        self.assertIn("[00:00:45] [] revisar boquilla @Torres", texto)

    def test_escribe_incremental(self):
        # Si el proceso muere a mitad de reunión, lo escrito tiene que estar.
        self.cuaderno.agregar(1.0, "primera")
        primero = self.cuaderno.ruta.read_text(encoding="utf-8")
        self.cuaderno.agregar(2.0, "segunda")
        segundo = self.cuaderno.ruta.read_text(encoding="utf-8")
        self.assertIn("primera", primero)
        self.assertNotIn("segunda", primero)
        self.assertIn("segunda", segundo)

    def test_no_registra_lineas_vacias(self):
        self.assertIsNone(self.cuaderno.agregar(1.0, "   "))
        self.assertEqual(self.cuaderno.lineas, [])

    def test_nunca_reescribe_lo_anterior(self):
        # /notes/ es del usuario: sólo se agrega al final.
        self.cuaderno.agregar(1.0, "primera")
        self.cuaderno.agregar(2.0, "segunda")
        lineas = [l for l in self.cuaderno.ruta.read_text(encoding="utf-8").splitlines()
                  if l.startswith("[")]
        self.assertEqual(lineas, ["[00:00:01] primera", "[00:00:02] segunda"])

    def test_segundos_negativos_no_rompen_el_formato(self):
        linea = self.cuaderno.agregar(-5.0, "x")
        self.assertEqual(linea.formatear(), "[00:00:00] x")

    def test_resumen_para_el_pie_de_pantalla(self):
        self.cuaderno.agregar(1.0, "[] una tarea")
        self.cuaderno.agregar(2.0, "[] otra tarea")
        self.cuaderno.agregar(3.0, "* un foco")
        self.cuaderno.agregar(4.0, "texto")
        self.assertEqual(self.cuaderno.resumen(),
                         {"tarea": 2, "foco": 1, "seccion": 0, "libre": 1})

    def test_el_encabezado_no_se_duplica(self):
        self.cuaderno.agregar(1.0, "a")
        self.cuaderno.agregar(2.0, "b")
        self.assertEqual(self.cuaderno.ruta.read_text(encoding="utf-8").count("---"), 2)
