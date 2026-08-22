"""WAV -> transcripción Markdown con timestamps, usando faster-whisper.

Parámetros y por qué (ver también CLAUDE.md §5):

- `vad_filter` con Silero y `max_speech_duration_s=25`: los cortes caen en
  silencios, no al medio de una palabra. Reemplaza al chunking por ventana fija
  con solape, que `faster-whisper` no expone y que obligaría a deduplicar texto
  entre ventanas (Whisper duplica o come frases al hacerlo, y ensucia los
  timestamps justo cuando más precisión hace falta).
- `word_timestamps=True`: el timestamp de cada línea sale del inicio de su
  primera palabra, no del segmento. El de segmento derrapa y rompe el objetivo
  de menos de 2 s de error.
- `condition_on_previous_text=False`: sin esto Whisper entra en bucles de
  repetición con audio ruidoso, que es exactamente el caso de una sala de planta.
- `hotwords` en lugar de `initial_prompt`: `initial_prompt` no sobrevive al
  parámetro anterior; `hotwords` se aplica en todos los bloques.

Este módulo corre siempre en su propio proceso, que muere al terminar. Es la
única forma de garantizar que la VRAM vuelve al sistema (CLAUDE.md §2).
"""

from __future__ import annotations

import gc
import json
import os
import sys
from datetime import datetime
from pathlib import Path

from . import gpu, lock, paths
from .deps import traducir_error_de_dll, whisper_model

MODELO = "large-v3-turbo"
COMPUTE_TYPE = "float16"
DEVICE = "cuda"
IDIOMA = "es"

MAX_BLOQUE_S = 25.0     # techo por bloque de voz
PADDING_MS = 400        # margen alrededor de cada bloque, para no comer arranques
SILENCIO_MIN_MS = 500   # silencio mínimo para considerar un corte
UMBRAL_VAD = 0.5

# Hueco entre palabras a partir del cual se corta la línea. El VAD saca los
# silencios antes de que Whisper vea el audio, así que un segmento puede juntar
# frases separadas por decenas de segundos de reloj: el timestamp de inicio
# queda bien, pero la línea abarca todo ese rango y saltar al audio desde ella
# no sirve. Se parte con las palabras, que sí conservan el tiempo real.
HUECO_MAXIMO_S = 1.5

LIMITE_HOTWORDS = 400   # el prompt de Whisper tope 224 tokens; no lo llenamos


def cargar_glosario(ruta: Path | None = None) -> str:
    """Junta `glosario.txt` en el string que espera `hotwords`.

    Una entrada por línea; `#` comenta. Si no existe el archivo, se transcribe
    igual sin jerga inyectada.
    """
    ruta = ruta or paths.GLOSARIO
    if not ruta.exists():
        return ""
    terminos = []
    for linea in ruta.read_text(encoding="utf-8").splitlines():
        linea = linea.split("#", 1)[0].strip()
        if linea:
            terminos.append(linea)

    texto = ", ".join(terminos)
    if len(texto) > LIMITE_HOTWORDS:
        recortado = []
        for termino in terminos:
            if len(", ".join(recortado + [termino])) > LIMITE_HOTWORDS:
                break
            recortado.append(termino)
        print(f"  Aviso: el glosario tiene {len(terminos)} términos y no entran "
              f"todos en el prompt; uso los primeros {len(recortado)}. "
              f"Poné los más importantes arriba en glosario.txt.", file=sys.stderr)
        texto = ", ".join(recortado)
    return texto


def partir_por_huecos(segmento, hueco_max: float = HUECO_MAXIMO_S
                      ) -> list[tuple[float, float, str]]:
    """Parte un segmento en donde haya un silencio largo entre sus palabras.

    Devuelve `[(inicio, fin, texto), ...]`. Sin palabras —o sin huecos— sale un
    solo trozo, equivalente al segmento original.

    Los tiempos salen siempre de las palabras: el `start` del segmento derrapa
    y es justo lo que rompe el objetivo de menos de 2 s de error.
    """
    palabras = [p for p in (getattr(segmento, "words", None) or []) if p.word.strip()]
    if not palabras:
        texto = segmento.text.strip()
        return [(float(segmento.start), float(segmento.end), texto)] if texto else []

    grupos = [[palabras[0]]]
    for anterior, palabra in zip(palabras, palabras[1:]):
        if float(palabra.start) - float(anterior.end) > hueco_max:
            grupos.append([])
        grupos[-1].append(palabra)

    trozos = []
    for grupo in grupos:
        texto = "".join(p.word for p in grupo).strip()
        if texto:
            trozos.append((float(grupo[0].start), float(grupo[-1].end), texto))
    return trozos


def _front_matter(nombre: str, wav: Path, duracion: float,
                  modelo: str, compute_type: str) -> str:
    return (
        "---\n"
        f"reunion: {nombre}\n"
        f"audio: {paths.relativa(wav)}\n"
        f"duracion: {paths.hms(duracion)}\n"
        f"modelo: {modelo}\n"
        f"compute_type: {compute_type}\n"
        f"idioma: {IDIOMA}\n"
        f"generado: {datetime.now().isoformat(timespec='seconds')}\n"
        "---\n\n"
    )


def transcribir(wav: Path, modelo: str = MODELO, compute_type: str = COMPUTE_TYPE,
                device: str = DEVICE, con_json: bool = True,
                umbral_vad: float = UMBRAL_VAD,
                hueco_max: float = HUECO_MAXIMO_S) -> Path:
    """Transcribe `wav` y devuelve la ruta del `.md` generado."""
    if not wav.exists():
        raise FileNotFoundError(f"No encuentro el audio: {wav}")

    nombre = wav.stem
    if not paths.es_nombre_sesion(nombre):
        print(f"  Aviso: '{nombre}' no sigue el patrón AAAA-MM-DD_<slug>. "
              f"Transcribo igual, pero las fases siguientes lo van a ignorar.",
              file=sys.stderr)

    destino = paths.ruta_transcript(nombre)
    parcial = destino.with_suffix(".md.tmp")
    destino.parent.mkdir(parents=True, exist_ok=True)

    # El cerrojo se toma antes de medir la VRAM: si hay otra transcripción en
    # curso, la medición de este proceso no significaría nada.
    with lock.transcripcion():
        return _transcribir_con_cerrojo(wav, nombre, destino, parcial, modelo,
                                        compute_type, device, con_json, umbral_vad,
                                        hueco_max)


def _transcribir_con_cerrojo(wav, nombre, destino, parcial, modelo, compute_type,
                             device, con_json, umbral_vad, hueco_max):
    if device == "cuda":
        gpu.exigir_vram(compute_type, f"Whisper {modelo}")
    libre_inicial = gpu.reporte("antes de cargar Whisper")

    # Import tardío a propósito: `--help` no tiene por qué cargar CUDA.
    WhisperModel = whisper_model()

    print(f"  Cargando {modelo} ({compute_type}) en {device}...")
    try:
        model = WhisperModel(modelo, device=device, compute_type=compute_type)
    except RuntimeError as error:
        raise traducir_error_de_dll(error) or error from None

    # A partir de acá hay pesos en la GPU: todo va dentro del try/finally para
    # que ningún camino de error se saltee la descarga.
    segmentos_json = []
    try:
        gpu.reporte("con Whisper cargado")
        hotwords = cargar_glosario()
        if hotwords:
            print(f"  Glosario de planta activo ({hotwords.count(',') + 1} términos).")

        segmentos, info = model.transcribe(
            str(wav),
            language=IDIOMA,
            task="transcribe",
            vad_filter=True,
            vad_parameters={
                "threshold": umbral_vad,
                "max_speech_duration_s": MAX_BLOQUE_S,
                "min_silence_duration_ms": SILENCIO_MIN_MS,
                "speech_pad_ms": PADDING_MS,
            },
            word_timestamps=True,
            condition_on_previous_text=False,
            hotwords=hotwords or None,
            beam_size=5,
            no_speech_threshold=0.6,
            compression_ratio_threshold=2.4,
            temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        )

        duracion = float(getattr(info, "duration", 0.0) or 0.0)
        print(f"  Audio: {paths.hms(duracion)}. Transcribiendo...")

        # Se escribe a .md.tmp y recién al final se renombra: el watcher usa la
        # existencia del .md como marca de "ya procesado", y un .md a medio
        # escribir sería un falso positivo.
        with parcial.open("w", encoding="utf-8", newline="\n") as salida:
            salida.write(_front_matter(nombre, wav, duracion, modelo, compute_type))
            contador = 0
            for segmento in segmentos:
                for inicio, fin, texto in partir_por_huecos(segmento, hueco_max):
                    salida.write(f"[{paths.hms(inicio)}] {texto}\n")
                    salida.flush()
                    segmentos_json.append({"i": contador, "inicio": round(inicio, 3),
                                           "fin": round(fin, 3), "texto": texto})
                    contador += 1
                    _avance(fin, duracion)

        sys.stdout.write("\r" + " " * 70 + "\r")
        os.replace(parcial, destino)
        plural = "segmento" if contador == 1 else "segmentos"
        print(f"  Transcripción lista: {paths.relativa(destino)} "
              f"({contador} {plural})")

        if con_json:
            ruta_json = paths.ruta_segmentos(nombre)
            ruta_json.write_text(
                json.dumps({"reunion": nombre,
                            "audio": paths.relativa(wav),
                            "duracion": round(duracion, 3),
                            "segmentos": segmentos_json},
                           ensure_ascii=False, indent=1),
                encoding="utf-8")
            print(f"  Índice de segmentos: {paths.relativa(ruta_json)}")

    except RuntimeError as error:
        # ctranslate2 carga cuBLAS y cuDNN de forma perezosa, al codificar el
        # primer bloque: el DLL ausente recién se nota acá, no al construir.
        raise traducir_error_de_dll(error) or error from None
    finally:
        parcial.unlink(missing_ok=True)
        _descargar(model)
        del model
        gc.collect()
        libre_final = gpu.reporte("después de descargar Whisper")
        _verificar_descarga(libre_inicial, libre_final)

    return destino


def _avance(posicion: float, total: float) -> None:
    if total > 0:
        pct = min(100, int(posicion / total * 100))
        sys.stdout.write(f"\r  {paths.hms(posicion)} / {paths.hms(total)}  ({pct} %)")
    else:
        sys.stdout.write(f"\r  {paths.hms(posicion)}")
    sys.stdout.flush()


def _descargar(model) -> None:
    """Pide a ctranslate2 que suelte los pesos. Best-effort a propósito.

    La garantía dura de liberación es que este proceso termine; esto sólo
    adelanta la devolución de memoria.
    """
    try:
        model.model.unload_model()
    except Exception:
        pass


def _verificar_descarga(inicial: int | None, final: int | None) -> None:
    if inicial is None or final is None:
        return
    if final < inicial - 500:
        print(f"  Nota: quedan ~{inicial - final} MB sin devolver al sistema. "
              f"El contexto CUDA los libera cuando este proceso termine; "
              f"verificalo con nvidia-smi en unos segundos.", file=sys.stderr)
