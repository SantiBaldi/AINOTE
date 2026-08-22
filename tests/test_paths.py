"""Convención de nombres AAAA-MM-DD_<slug>."""

import unittest
from datetime import date

from src import paths
from tests.dobles import CasoConCarpetas


class TestSlug(unittest.TestCase):
    def test_saca_tildes_y_normaliza(self):
        self.assertEqual(paths.slug("Reunión de Pérdidas — L4"),
                         "reunion-de-perdidas-l4")
        self.assertEqual(paths.slug("ACR  Línea 3"), "acr-linea-3")
        self.assertEqual(paths.slug("Ñandú"), "nandu")

    def test_colapsa_separadores_y_recorta(self):
        self.assertEqual(paths.slug("  ---perdidas___semana!!  "),
                         "perdidas-semana")

    def test_texto_sin_nada_utilizable_no_da_vacio(self):
        # Un nombre vacío rompería la convención y dejaría archivos como
        # "2026-08-22_.wav", así que cae a un default.
        self.assertEqual(paths.slug("¿¿??"), "reunion")
        self.assertEqual(paths.slug(""), "reunion")

    def test_el_resultado_siempre_respeta_el_patron(self):
        for entrada in ("Reunión de Pérdidas — L4", "  ---x___y!!  ", "¿¿??",
                        "100% OEE", "Ñandú/Torres"):
            nombre = paths.nombre_sesion(entrada, date(2026, 8, 22))
            self.assertTrue(paths.es_nombre_sesion(nombre), nombre)


class TestNombreSesion(unittest.TestCase):
    def test_formato(self):
        self.assertEqual(paths.nombre_sesion("perdidas", date(2026, 8, 22)),
                         "2026-08-22_perdidas")

    def test_reconoce_nombres_validos(self):
        self.assertTrue(paths.es_nombre_sesion("2026-08-22_perdidas"))
        self.assertTrue(paths.es_nombre_sesion("2026-08-22_bajada-de-gerencia-2"))

    def test_rechaza_nombres_invalidos(self):
        for malo in ("perdidas", "2026-8-22_perdidas", "2026-08-22_Perdidas",
                     "2026-08-22_perdidas_x", "2026-08-22_", "2026-08-22perdidas"):
            self.assertFalse(paths.es_nombre_sesion(malo), malo)


class TestColisiones(CasoConCarpetas):
    def test_sin_colision_devuelve_el_nombre_base(self):
        self.assertEqual(paths.nombre_libre("perdidas", date(2026, 8, 22)),
                         "2026-08-22_perdidas")

    def test_dos_reuniones_el_mismo_dia_no_se_pisan(self):
        self.escribir_wav("2026-08-22_perdidas")
        self.assertEqual(paths.nombre_libre("perdidas", date(2026, 8, 22)),
                         "2026-08-22_perdidas-2")

    def test_la_colision_se_detecta_en_cualquier_carpeta(self):
        # El WAV puede haberse borrado pero la transcripción seguir ahí.
        paths.ruta_transcript("2026-08-22_perdidas").write_text("x", encoding="utf-8")
        self.assertEqual(paths.nombre_libre("perdidas", date(2026, 8, 22)),
                         "2026-08-22_perdidas-2")

    def test_el_sufijo_sigue_respetando_el_patron(self):
        self.escribir_wav("2026-08-22_perdidas")
        self.assertTrue(paths.es_nombre_sesion(
            paths.nombre_libre("perdidas", date(2026, 8, 22))))


class TestFormatoDeTiempo(unittest.TestCase):
    def test_hms_trunca(self):
        self.assertEqual(paths.hms(0), "00:00:00")
        self.assertEqual(paths.hms(6.99), "00:00:06")
        self.assertEqual(paths.hms(3671.0), "01:01:11")

    def test_no_se_rompe_con_negativos(self):
        self.assertEqual(paths.hms(-3), "00:00:00")
        self.assertEqual(paths.ms(-3), "00:00")

    def test_ms(self):
        self.assertEqual(paths.ms(75), "01:15")
        self.assertEqual(paths.ms(3600), "60:00")


class TestRutaRelativa(CasoConCarpetas):
    def test_deja_la_ruta_relativa_a_la_raiz(self):
        self.assertEqual(paths.relativa(paths.ruta_audio("2026-08-22_perdidas")),
                         "audio/2026-08-22_perdidas.wav")

    def test_usa_barras_normales_aunque_windows_use_contrabarras(self):
        self.assertNotIn("\\", paths.relativa(paths.ruta_transcript("2026-08-22_x")))

    def test_tolera_que_la_raiz_no_este_resuelta(self):
        # En Windows la misma carpeta se puede escribir con el nombre corto 8.3
        # o con el largo. Si sólo se resuelve un lado, relative_to falla y el
        # front-matter se llena con la ruta absoluta de la máquina.
        import unittest.mock
        with unittest.mock.patch.object(paths, "RAIZ", self.tmp / "." / ""):
            self.assertEqual(paths.relativa(paths.ruta_audio("2026-08-22_perdidas")),
                             "audio/2026-08-22_perdidas.wav")

    def test_una_ruta_de_afuera_queda_absoluta(self):
        from pathlib import Path
        ajena = Path("/otro/disco/audio.wav")
        self.assertEqual(paths.relativa(ajena), ajena.as_posix())
