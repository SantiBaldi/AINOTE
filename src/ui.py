"""Ventana de sesión: grabar y tomar notas en la misma pantalla.

Tkinter viene con Python, así que no agrega dependencias ni pide permisos de
administrador. No da la terminación de una tablet, pero sí lo que importa:
tipografía grande, marcadores que se distinguen de un vistazo, y el cronómetro
del audio siempre a la vista.

La lógica vive en `sesion.py` y `notas.py`; acá sólo se dibuja. Es a propósito:
esto no se puede probar en un entorno sin pantalla, así que cuanto menos
decida, mejor.
"""

from __future__ import annotations

from . import notas, paths

# Paleta: papel claro, como la tablet. Los acentos distinguen el tipo de línea
# sin necesidad de leerla.
PAPEL = "#F7F5F0"
TINTA = "#2B2B2B"
BARRA = "#1F1D1A"
BARRA_TEXTO = "#F0EDE6"
GRIS = "#9A9689"
BORDE = "#E2DED4"
ROJO = "#C0392B"

COLOR_TIPO = {
    "seccion": "#1F1D1A",
    "tarea": "#1F6F4A",
    "foco": "#B8860B",
    "libre": "#4A4640",
}

VINETA = {"seccion": "", "tarea": "▢ ", "foco": "● ", "libre": ""}

AYUDA = "[]  tarea      *  foco      #  sección      @Torres      !05-09"


def _fuente(candidatas, tamanio, peso="normal"):
    """Primera tipografía disponible de la lista. Windows y Linux difieren."""
    from tkinter import font as tkfont

    disponibles = {f.lower() for f in tkfont.families()}
    for nombre in candidatas:
        if nombre.lower() in disponibles:
            return (nombre, tamanio, peso)
    return ("TkDefaultFont", tamanio, peso)


class Ventana:
    def __init__(self, sesion):
        import tkinter as tk

        self.tk = tk
        self.sesion = sesion
        self.terminada = False
        self._inicio_linea: float | None = None

        self.raiz = tk.Tk()
        self.raiz.title(f"AINOTE — {sesion.nombre}")
        self.raiz.configure(bg=PAPEL)
        self.raiz.geometry("980x680")
        self.raiz.minsize(640, 420)

        self.f_titulo = _fuente(["Segoe UI Semibold", "Segoe UI", "DejaVu Sans"], 13)
        self.f_reloj = _fuente(["Consolas", "DejaVu Sans Mono", "Courier New"], 20, "bold")
        self.f_nota = _fuente(["Segoe UI", "DejaVu Sans", "Helvetica"], 14)
        self.f_sello = _fuente(["Consolas", "DejaVu Sans Mono", "Courier New"], 11)
        self.f_pie = _fuente(["Segoe UI", "DejaVu Sans"], 10)

        # El orden de armado no es el orden visual: se packean primero los
        # elementos de abajo, para que tengan el espacio asegurado. La hoja va
        # última porque es la que absorbe el sobrante — si se packeara antes,
        # con `expand=True` se comería el campo de escritura y el pie.
        self._cabecera()
        self._pie()
        self._entrada()
        self._hoja()

        self.raiz.protocol("WM_DELETE_WINDOW", self.terminar)
        self.campo.focus_set()
        self._latir()

    # -- construcción ------------------------------------------------------

    def _cabecera(self):
        tk = self.tk
        barra = tk.Frame(self.raiz, bg=BARRA, height=64)
        barra.pack(fill="x")
        barra.pack_propagate(False)

        izquierda = tk.Frame(barra, bg=BARRA)
        izquierda.pack(side="left", padx=20)
        self.punto = tk.Label(izquierda, text="●", bg=BARRA, fg=ROJO,
                              font=self.f_titulo)
        self.punto.pack(side="left", padx=(0, 10))
        self.reloj = tk.Label(izquierda, text="00:00:00", bg=BARRA,
                              fg=BARRA_TEXTO, font=self.f_reloj)
        self.reloj.pack(side="left")

        tk.Label(barra, text=self.sesion.nombre, bg=BARRA, fg=GRIS,
                 font=self.f_titulo).pack(side="right", padx=20)

    def _hoja(self):
        tk = self.tk
        marco = tk.Frame(self.raiz, bg=PAPEL)
        marco.pack(fill="both", expand=True, padx=28, pady=(22, 8))

        self.hoja = tk.Text(marco, wrap="word", bg=PAPEL, fg=TINTA,
                            font=self.f_nota, relief="flat", padx=8, pady=8,
                            spacing1=5, spacing3=9, cursor="arrow",
                            highlightthickness=0)
        barra = tk.Scrollbar(marco, command=self.hoja.yview, relief="flat",
                             bg=PAPEL, troughcolor=PAPEL, width=10)
        self.hoja.configure(yscrollcommand=barra.set)
        barra.pack(side="right", fill="y")
        self.hoja.pack(side="left", fill="both", expand=True)

        self.hoja.tag_configure("sello", foreground=GRIS, font=self.f_sello)
        for tipo, color in COLOR_TIPO.items():
            negrita = self.f_nota[:2] + ("bold",)
            self.hoja.tag_configure(
                tipo, foreground=color,
                font=negrita if tipo == "seccion" else self.f_nota,
                lmargin1=14, lmargin2=88)
        self.hoja.configure(state="disabled")

    def _entrada(self):
        tk = self.tk
        fila = tk.Frame(self.raiz, bg=PAPEL)
        fila.pack(side="bottom", fill="x", padx=28, pady=(10, 4))

        linea = tk.Frame(self.raiz, bg=BORDE, height=1)
        linea.pack(side="bottom", fill="x", padx=28)

        tk.Label(fila, text="›", bg=PAPEL, fg=GRIS,
                 font=self.f_nota).pack(side="left", padx=(8, 8))
        self.campo = tk.Entry(fila, bg=PAPEL, fg=TINTA, font=self.f_nota,
                              relief="flat", insertbackground=TINTA,
                              highlightthickness=0)
        self.campo.pack(side="left", fill="x", expand=True, ipady=6)
        self.campo.bind("<Return>", self._sellar)
        self.campo.bind("<Key>", self._primera_tecla)
        self.campo.bind("<Escape>", lambda _: self.terminar())

        tk.Button(fila, text="Terminar", command=self.terminar, relief="flat",
                  bg=BARRA, fg=BARRA_TEXTO, font=self.f_pie, padx=18, pady=6,
                  activebackground=ROJO, activeforeground=BARRA_TEXTO,
                  cursor="hand2").pack(side="right", padx=(12, 0))

    def _pie(self):
        tk = self.tk
        fila = tk.Frame(self.raiz, bg=PAPEL)
        fila.pack(side="bottom", fill="x", padx=36, pady=(0, 14))
        self.marcador = tk.Label(fila, text="sin marcadores todavía", bg=PAPEL,
                                 fg=GRIS, font=self.f_pie)
        self.marcador.pack(side="left")
        tk.Label(fila, text=AYUDA, bg=PAPEL, fg=GRIS,
                 font=self.f_pie).pack(side="right")

    # -- comportamiento ----------------------------------------------------

    def _primera_tecla(self, evento):
        """Registra cuándo se empezó la línea, no cuándo se la terminó.

        Entre el primer carácter y el Enter puede pasar medio minuto: el
        momento de la reunión que importa es el que disparó la anotación.
        """
        if self._inicio_linea is None and evento.char and evento.char.isprintable():
            self._inicio_linea = self.sesion.segundos()

    def _sellar(self, _=None):
        texto = self.campo.get()
        linea = self.sesion.anotar(texto, segundos=self._inicio_linea)
        self.campo.delete(0, "end")
        self._inicio_linea = None
        if linea is not None:
            self._pintar(linea)
            self.marcador.configure(text=self.sesion.resumen())
        return "break"

    def _pintar(self, linea):
        cuerpo = linea.texto.lstrip()
        for marca in (notas.TAREA, notas.SECCION, notas.FOCO):
            if cuerpo.startswith(marca):
                cuerpo = cuerpo[len(marca):].lstrip()
                break

        self.hoja.configure(state="normal")
        self.hoja.insert("end", f"{paths.hms(linea.segundos)}  ", "sello")
        self.hoja.insert("end", VINETA[linea.tipo] + cuerpo + "\n", linea.tipo)
        self.hoja.configure(state="disabled")
        self.hoja.see("end")

    def _latir(self):
        """Refresca el reloj y el punto de grabación cuatro veces por segundo."""
        self.reloj.configure(text=paths.hms(self.sesion.segundos()))
        if self.sesion.error is not None:
            self.punto.configure(fg=GRIS, text="■")
            self.reloj.configure(text="sin audio")
        elif self.sesion.grabando:
            visible = int(self.sesion.segundos() * 2) % 2 == 0
            self.punto.configure(fg=ROJO if visible else BARRA)
        if not self.terminada:
            self.raiz.after(250, self._latir)

    def terminar(self):
        if self.terminada:
            return
        # Lo que quedó tipeado sin Enter también es una nota.
        if self.campo.get().strip():
            self._sellar()
        self.terminada = True
        self.raiz.destroy()

    def correr(self):
        self.raiz.mainloop()


def abrir(sesion) -> None:
    """Abre la ventana y devuelve el control cuando el usuario la cierra."""
    Ventana(sesion).correr()
