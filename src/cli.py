"""Interfaz de línea de comandos.

    python -m src grabar          graba, corta con Enter, transcribe solo
    python -m src transcribir X   transcribe un WAV puntual
    python -m src vigilar         watcher sobre /audio/
    python -m src dispositivos    lista micrófonos (diagnóstico)
    python -m src gpu             VRAM libre (diagnóstico)
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from . import gpu, lock, paths, transcribe
from . import deps
from .deps import DependenciaFaltante

COMPUTE_TYPES = ("float16", "int8_float16", "int8", "float32")


def _opciones_transcripcion(sub: argparse.ArgumentParser) -> None:
    sub.add_argument("--modelo", default=transcribe.MODELO,
                     help=f"modelo de Whisper (por defecto: {transcribe.MODELO})")
    sub.add_argument("--compute-type", default=transcribe.COMPUTE_TYPE,
                     choices=COMPUTE_TYPES,
                     help="precisión; int8_float16 usa menos VRAM y algo menos de calidad")
    sub.add_argument("--device", default=transcribe.DEVICE, choices=("cuda", "cpu"),
                     help="cpu sirve para probar sin GPU, pero es lentísimo")
    sub.add_argument("--umbral-vad", type=float, default=transcribe.UMBRAL_VAD,
                     help="0.5 por defecto; subilo si el ruido de planta entra como voz")
    sub.add_argument("--hueco-maximo", type=float, default=transcribe.HUECO_MAXIMO_S,
                     help="segundos de silencio entre palabras que cortan la línea "
                          "(1.5 por defecto; bajalo para líneas más cortas)")
    sub.add_argument("--con-vad", action="store_true",
                     help="filtrar los silencios con el VAD antes de transcribir. "
                          "Más rápido, pero medido en la notebook: se come frases "
                          "enteras y desplaza timestamps. No lo uses para una "
                          "reunión que importe")
    sub.add_argument("--sin-json", action="store_true",
                     help="no generar el .segments.json")


def _extra_transcripcion(args: argparse.Namespace) -> list[str]:
    """Reconstruye los flags para pasárselos a un subproceso."""
    extra = ["--modelo", args.modelo, "--compute-type", args.compute_type,
             "--device", args.device, "--umbral-vad", str(args.umbral_vad),
             "--hueco-maximo", str(args.hueco_maximo)]
    if args.sin_json:
        extra.append("--sin-json")
    if args.con_vad:
        extra.append("--con-vad")
    return extra


def _resolver_wav(referencia: str) -> Path:
    """Acepta una ruta, un nombre de archivo o un nombre de sesión pelado."""
    candidato = Path(referencia)
    if candidato.exists():
        return candidato
    if (paths.AUDIO / referencia).exists():
        return paths.AUDIO / referencia
    encontrado = paths.buscar_audio(referencia)
    if encontrado:
        return encontrado

    # Listar lo que sí hay: equivocarse de fecha es fácil, y adivinar el nombre
    # exacto para reintentar es una pérdida de tiempo evitable.
    mensaje = [f"No encuentro el audio '{referencia}'."]
    disponibles = paths.audios()
    if disponibles:
        mensaje.append(f"  En {paths.relativa(paths.AUDIO)}/ tenés:")
        mensaje += [f"      {w.stem}" for w in disponibles[:10]]
        if len(disponibles) > 10:
            mensaje.append(f"      ... y {len(disponibles) - 10} más")
    else:
        mensaje.append(f"  La carpeta {paths.relativa(paths.AUDIO)}/ está vacía. "
                       f"Grabá algo con: python -m src grabar")
    raise SystemExit("\n".join(mensaje))


def cmd_grabar(args: argparse.Namespace) -> int:
    from . import record

    paths.asegurar_carpetas()

    titulo = args.titulo or input("  Nombre de la reunión (ej: perdidas): ").strip()
    nombre = paths.nombre_libre(titulo or "reunion")
    destino = paths.ruta_audio(nombre)

    grabacion = record.grabar(destino, dispositivo=args.dispositivo,
                              sample_rate=args.sample_rate)

    if grabacion.duracion <= 0:
        print("  No entró audio. Revisá el micrófono con: python -m src dispositivos",
              file=sys.stderr)
        return 1

    print(f"  Grabado: {paths.relativa(destino)}  ({paths.hms(grabacion.duracion)})")
    if grabacion.desbordes:
        print(f"  Aviso: {grabacion.desbordes} desborde(s) del buffer de audio; "
              f"puede haber microcortes.", file=sys.stderr)

    if args.sin_transcribir:
        print(f"  Para transcribir después: python -m src transcribir {nombre}")
        return 0

    # En subproceso a propósito: PortAudio ya cerró, y Whisper muere con el
    # proceso devolviendo toda la VRAM (CLAUDE.md §2).
    print()
    orden = [sys.executable, "-m", "src", "transcribir", str(destino),
             *_extra_transcripcion(args)]
    return subprocess.run(orden, cwd=str(paths.RAIZ)).returncode


def cmd_importar(args: argparse.Namespace) -> int:
    """Trae una grabación de afuera a /audio/ con el nombre de la convención.

    Copia, no mueve: el original queda donde estaba. No reencodea —`faster-whisper`
    decodifica m4a, mp3 y compañía— así que no se pierde calidad ni tiempo.
    """
    import shutil

    paths.asegurar_carpetas()
    origen = Path(args.archivo)
    if not origen.exists():
        print(f"  No encuentro el archivo:\n    {origen}", file=sys.stderr)
        return 2
    if origen.suffix.lower() not in paths.EXTENSIONES_AUDIO:
        print(f"  No sé leer '{origen.suffix}'. Formatos soportados: "
              f"{', '.join(paths.EXTENSIONES_AUDIO)}", file=sys.stderr)
        return 2

    dia = args.fecha or paths.fecha_de(origen)
    nombre = paths.nombre_libre(args.titulo or origen.stem, dia)
    destino = paths.ruta_audio(nombre, origen.suffix.lower())

    print(f"  Copiando a {paths.relativa(destino)} ...")
    shutil.copy2(origen, destino)
    print(f"  Listo. El original quedó donde estaba.")

    if args.sin_transcribir:
        print(f"  Para transcribir: python -m src transcribir {nombre}")
        return 0

    print()
    orden = [sys.executable, "-m", "src", "transcribir", str(destino),
             *_extra_transcripcion(args)]
    return subprocess.run(orden, cwd=str(paths.RAIZ)).returncode


def _fecha(texto: str):
    from datetime import datetime

    try:
        return datetime.strptime(texto, "%Y-%m-%d").date()
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"'{texto}' no es una fecha AAAA-MM-DD (ej: 2026-07-21)")


def cmd_transcribir(args: argparse.Namespace) -> int:
    paths.asegurar_carpetas()
    wav = _resolver_wav(args.audio)
    destino = paths.ruta_transcript(wav.stem)

    if destino.exists() and not args.forzar:
        print(f"  Ya existe {paths.relativa(destino)}. Usá --forzar para rehacerlo.")
        return 0

    try:
        transcribe.transcribir(wav, modelo=args.modelo,
                               compute_type=args.compute_type, device=args.device,
                               con_json=not args.sin_json,
                               umbral_vad=args.umbral_vad,
                               hueco_max=args.hueco_maximo,
                               usar_vad=args.con_vad)
    except lock.Ocupado as error:
        # Código propio: para el watcher esto no es un fallo del audio, sino un
        # "volvé más tarde".
        print(f"\n  {error}", file=sys.stderr)
        return 4
    except gpu.VramInsuficiente as error:
        print(f"\n  VRAM insuficiente.\n  {error}", file=sys.stderr)
        return 2
    except FileNotFoundError as error:
        print(f"\n  {error}", file=sys.stderr)
        return 2
    return 0


def cmd_vigilar(args: argparse.Namespace) -> int:
    from . import watch

    return watch.vigilar(intervalo=args.intervalo, extra=_extra_transcripcion(args))


def cmd_dispositivos(args: argparse.Namespace) -> int:
    from . import record

    print(record.listar_dispositivos())
    return 0


def cmd_gpu(args: argparse.Namespace) -> int:
    """Diagnóstico: VRAM y librerías de CUDA. Lo segundo se informa siempre,
    aunque no haya `nvidia-smi`: son dos fallas independientes."""
    medicion = gpu.consultar()
    if medicion is None:
        print("  No pude leer la VRAM: no encuentro nvidia-smi.", file=sys.stderr)
    else:
        libre, total = medicion
        print(f"  VRAM: {libre} MB libres de {total} MB ({total - libre} MB en uso)")
        for tipo, minimo in gpu.VRAM_MINIMA_MB.items():
            estado = "entra" if libre >= minimo else "NO entra"
            print(f"    Whisper {tipo:<13} necesita {minimo:>5} MB  ->  {estado}")

    print("\n  Librerías de CUDA:")
    for linea in deps.diagnostico_cuda():
        print(linea)
    return 0 if medicion else 1


def construir_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src",
        description="AINOTE - captura de reuniones, 100% local. Fase 1: "
                    "grabar y transcribir.")
    subs = parser.add_subparsers(dest="comando", required=True)

    grabar = subs.add_parser("grabar", help="grabar del micrófono y transcribir al cortar")
    grabar.add_argument("titulo", nargs="?",
                        help="nombre de la reunión; se te pregunta si no lo pasás")
    grabar.add_argument("--dispositivo", type=int, default=None,
                        help="índice del micrófono (ver: dispositivos)")
    grabar.add_argument("--sample-rate", type=int, default=16000,
                        help="16000 por defecto; cae a la tasa nativa si el mic no la acepta")
    grabar.add_argument("--sin-transcribir", action="store_true",
                        help="grabar nomás, sin encadenar la transcripción")
    _opciones_transcripcion(grabar)
    grabar.set_defaults(func=cmd_grabar)

    transcribir = subs.add_parser("transcribir", help="transcribir un WAV")
    transcribir.add_argument("audio", help="ruta al WAV o nombre de la sesión")
    transcribir.add_argument("--forzar", action="store_true",
                             help="rehacer aunque ya exista el .md")
    _opciones_transcripcion(transcribir)
    transcribir.set_defaults(func=cmd_transcribir)

    vigilar = subs.add_parser("vigilar", help="transcribir solo todo WAV nuevo en /audio/")
    vigilar.add_argument("--intervalo", type=float, default=2.0,
                         help="segundos entre sondeos")
    _opciones_transcripcion(vigilar)
    vigilar.set_defaults(func=cmd_vigilar)

    importar = subs.add_parser(
        "importar", help="traer una grabación de afuera y transcribirla")
    importar.add_argument("archivo", help="ruta al audio (wav, m4a, mp3, ...)")
    importar.add_argument("--titulo", help="nombre de la reunión; si no, sale "
                                           "del nombre del archivo")
    importar.add_argument("--fecha", type=_fecha,
                          help="AAAA-MM-DD; si no, se busca en el nombre del "
                               "archivo y si no está se usa la del archivo")
    importar.add_argument("--sin-transcribir", action="store_true",
                          help="sólo copiar, sin transcribir")
    _opciones_transcripcion(importar)
    importar.set_defaults(func=cmd_importar)

    dispositivos = subs.add_parser("dispositivos", help="listar micrófonos")
    dispositivos.set_defaults(func=cmd_dispositivos)

    estado_gpu = subs.add_parser("gpu", help="VRAM libre y qué modelo entra")
    estado_gpu.set_defaults(func=cmd_gpu)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = construir_parser().parse_args(argv)
    try:
        return args.func(args)
    except DependenciaFaltante as error:
        print(f"\n  {error}", file=sys.stderr)
        return 3
    except KeyboardInterrupt:
        print("\n  Cancelado.", file=sys.stderr)
        return 130
