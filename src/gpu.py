"""Control de VRAM. Falla con un mensaje claro, nunca con un OOM de CUDA.

La regla arquitectónica del proyecto es que ASR y LLM jamás están cargados al
mismo tiempo (ver CLAUDE.md §2). Este módulo es el que la hace cumplir del lado
del ASR: antes de instanciar Whisper se consulta cuánta VRAM hay libre y, si no
alcanza, se aborta con una explicación en castellano.

Se consulta `nvidia-smi` por subprocess a propósito: viene con el driver, está
en el PATH de cualquier Windows con GPU NVIDIA y no agrega una dependencia.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys

# Piso de VRAM libre exigido antes de cargar Whisper, por `compute_type`.
# Incluye el peso del modelo mas el overhead del contexto CUDA (~400 MB).
VRAM_MINIMA_MB = {
    "float16": 2600,
    "float32": 4200,
    "int8_float16": 1800,
    "int8": 1500,
}

# Candidatas fuera del PATH, para instalaciones donde nvidia-smi no quedo expuesto.
_RUTAS_NVIDIA_SMI = (
    r"C:\Windows\System32\nvidia-smi.exe",
    r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe",
    "/usr/bin/nvidia-smi",
)


class VramInsuficiente(RuntimeError):
    """No hay VRAM libre para cargar el modelo pedido."""


def _binario() -> str | None:
    encontrado = shutil.which("nvidia-smi")
    if encontrado:
        return encontrado
    for ruta in _RUTAS_NVIDIA_SMI:
        if os.path.isfile(ruta):
            return ruta
    return None


def consultar() -> tuple[int, int] | None:
    """Devuelve `(libre_mb, total_mb)` de la primera GPU, o None si no se puede.

    None significa "no pude medir" (sin driver NVIDIA, sin nvidia-smi, timeout),
    no "no hay VRAM". Los llamadores degradan a advertencia y siguen: es mejor
    intentar cargar el modelo que bloquear el flujo por no poder medir.
    """
    binario = _binario()
    if binario is None:
        return None
    try:
        salida = subprocess.run(
            [binario, "--query-gpu=memory.free,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15, check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None

    for linea in salida.splitlines():
        partes = [p.strip() for p in linea.split(",")]
        if len(partes) >= 2:
            try:
                return int(float(partes[0])), int(float(partes[1]))
            except ValueError:
                continue
    return None


def vram_libre_mb() -> int | None:
    medicion = consultar()
    return medicion[0] if medicion else None


def reporte(etiqueta: str, archivo=sys.stderr) -> int | None:
    """Imprime el estado de la VRAM. Es la evidencia del antes/después.

    Sirve para verificar a ojo que el modelo se descargó: el `libre` del
    reporte final tiene que volver aproximadamente al valor del inicial.
    """
    medicion = consultar()
    if medicion is None:
        print(f"  VRAM [{etiqueta}]: no se pudo medir (nvidia-smi no disponible)",
              file=archivo)
        return None
    libre, total = medicion
    usada = total - libre
    print(f"  VRAM [{etiqueta}]: {libre} MB libres de {total} MB "
          f"({usada} MB en uso)", file=archivo)
    return libre


def exigir_vram(compute_type: str, componente: str = "Whisper") -> None:
    """Aborta si no hay VRAM suficiente para cargar `componente`.

    Levanta `VramInsuficiente` con un mensaje accionable. Si no se puede medir,
    deja pasar con una advertencia.
    """
    minimo = VRAM_MINIMA_MB.get(compute_type, 2600)
    libre = vram_libre_mb()

    if libre is None:
        print("  Aviso: no puedo medir la VRAM libre, sigo igual. Si esto "
              "revienta con un OOM de CUDA, revisá que nvidia-smi ande.",
              file=sys.stderr)
        return

    if libre < minimo:
        raise VramInsuficiente(
            f"Necesito {minimo} MB de VRAM libre para cargar {componente} "
            f"({compute_type}) y hay {libre} MB.\n"
            f"  Probá una de estas:\n"
            f"    - Cerrá el navegador o lo que esté usando la GPU.\n"
            f"    - Si tenés Ollama corriendo, bajalo: ollama stop --all\n"
            f"      (y acordate de OLLAMA_KEEP_ALIVE=0).\n"
            f"    - Esperá a que termine otra transcripción: nunca corren dos a la vez.\n"
            f"    - Como último recurso, --compute-type int8_float16 "
            f"(baja a ~1800 MB, algo menos de calidad)."
        )
