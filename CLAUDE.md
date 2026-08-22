# AINOTE — sistema local de captura de reuniones

Sistema personal para reuniones de planta metalúrgica (reuniones semanales de
pérdidas, ACR, bajadas de gerencia). Cierra el ciclo completo:
**audio → transcripción → notas → minuta → tareas → búsqueda histórica.**

Referente conceptual: tablet iFLYTEK AINOTE 2, **sin la parte de escritura a
mano**. Las tres ideas que se replican son: transcribir mientras se toma nota en
la misma pantalla, crear tareas y marcar focos sin salir de la nota, y saltar
desde una línea de la nota al minuto exacto del audio.

---

## 1. Entorno de ejecución — restricciones duras

Estas no se negocian. Cualquier propuesta que las viole está mal.

| | |
|---|---|
| SO | Windows 11 Pro 23H2, **sin permisos de administrador** |
| GPU | RTX 5050 Laptop, **8 GB VRAM compartida con el display** |
| CPU / RAM | i7-13620H / 32 GB |
| Red | **100 % local.** Cero APIs en la nube, cero telemetría, cero servicios externos |
| Idioma | **Español rioplatense** en interfaz, prompts al modelo y salidas |

- Todo se instala a nivel usuario: `python -m venv .venv` o `pip install --user`.
  Nada que pida elevación, toque `Program Files` o escriba en el registro.
- El material es **confidencial de planta**. No sale de la máquina. Ni a una API,
  ni a un servicio de transcripción, ni a un repo remoto.
- Una vez bajado el modelo, todo el flujo corre **sin conexión a internet**.

### El repo está en GitHub — los datos no

`santibaldi/ainote` es un remoto público de código. `.gitignore` excluye
`audio/`, `transcripts/`, `notes/`, `minutas/`, `tasks.json` e `index.sqlite`.
**Nunca quitar esas reglas ni forzar el add de esos archivos.** Al remoto va el
código; el contenido de las reuniones se queda en la notebook.

---

## 2. Regla de VRAM — el constraint que manda sobre el diseño

| Componente | VRAM aprox. |
|---|---|
| Whisper large-v3-turbo (fp16) | 1,8 GB |
| LLM 9B Q4_K_M | 5,5 GB |
| KV cache a 16k de contexto | 1–1,5 GB |

**Los dos juntos no entran en 8 GB.**

> ### ASR y LLM nunca están cargados al mismo tiempo.

Consecuencias operativas, todas obligatorias:

1. Durante la grabación corre **únicamente** Whisper.
2. Al cerrar la sesión, Whisper se descarga de la GPU **antes** de invocar a Ollama.
3. Ollama se configura con `OLLAMA_KEEP_ALIVE=0` para que no quede residente.
4. El código **verifica VRAM libre antes de cargar cualquier modelo** y falla con
   un mensaje claro en castellano, no con un OOM de CUDA.
5. **Un cerrojo en disco (`.ainote.lock`) impide dos transcripciones a la vez.**
   Como cada una corre en su propio proceso, un candado en memoria no alcanza.
   El escenario que evita es real: con `vigilar` en una consola y `grabar` en
   otra, al cortar la grabación se lanzan dos transcripciones del mismo audio.
   `src/lock.py` usa `O_CREAT | O_EXCL`, atómico en Windows y en POSIX. El CLI
   sale con código 4 cuando está ocupado, y el watcher lo reintenta después en
   vez de marcar el audio como fallido.

### Presupuesto real: ~6,5 GB, no 8

La GPU es compartida con el display. Windows, el compositor y un navegador se
comen entre 1 y 1,5 GB de forma permanente. **El presupuesto efectivo es de
~6,5 GB libres**, y hay que dimensionar contra ese número.

Consecuencia para la Fase 4: un 9B Q4_K_M (5,5 GB) más KV a 16k (1–1,5 GB) da
~7 GB y **no entra**, aunque Whisper ya esté descargado. Cuando se llegue a esa
fase hay que bajar a un 8B Q4, recortar el contexto a 8k, o ambas. Está anotado
acá para no descubrirlo con un OOM.

### Descargar VRAM de verdad: matar el proceso

`faster-whisper` no usa PyTorch, así que no hay `torch.cuda.empty_cache()`.
`ctranslate2` expone `unload_model()`, pero **el contexto CUDA del proceso puede
retener memoria igual** y `nvidia-smi` la va a seguir mostrando ocupada.

La única descarga 100 % verificable es la muerte del proceso: el SO libera todo.
Por eso **toda transcripción corre en su propio proceso**, que termina cuando
termina el trabajo. El watcher lanza un subproceso por archivo; `grabar` lanza
un subproceso al cortar. Se paga recargar el modelo (~10–20 s por archivo) a
cambio de una garantía dura, y para reuniones ese costo es irrelevante.

### Map-reduce obligatorio para transcripciones largas

Una reunión de 1 h son ~14k tokens de transcripción. **Nunca pasarla en un solo
prompt.** Bloques de 10 minutos → resumen parcial de cada uno → fusión final.
Aplica desde la Fase 4 en adelante.

---

## 3. Estructura de archivos

Todo en texto plano, para leerlo, grepearlo y versionarlo sin herramientas
propietarias.

```
/audio/2026-08-22_perdidas.wav          PCM 16-bit (16 kHz mono si el mic lo acepta)
/transcripts/2026-08-22_perdidas.md     líneas [HH:MM:SS] + front-matter
/transcripts/2026-08-22_perdidas.segments.json   índice inicio/fin por segmento
/notes/2026-08-22_perdidas.md           texto libre + marcadores (Fase 2)
/minutas/2026-08-22_perdidas.md         salida del LLM (Fase 4)
/tasks.json                             tareas extraídas (Fase 3)
/index.sqlite                           índice FTS5 (Fase 5)
/glosario.txt                           jerga de planta para hotwords
/src/                                   código
```

Reglas:

- **Nada de bases de datos propietarias ni formatos binarios propios.**
- `index.sqlite` es **sólo índice de búsqueda**: debe poder reconstruirse por
  completo desde los archivos de texto. Si se borra, no se pierde información.
- Convención de nombres: **`AAAA-MM-DD_<slug>`**, con el slug en minúsculas,
  sin tildes, y `[a-z0-9-]` únicamente. Colisiones del mismo día resuelven con
  sufijo `-2`, `-3`. Un solo módulo decide esto: `src/paths.py`. Nadie más
  construye nombres a mano.

### Formato del transcript

```markdown
---
reunion: 2026-08-22_perdidas
audio: audio/2026-08-22_perdidas.wav
duracion: 00:58:12
modelo: large-v3-turbo
compute_type: float16
idioma: es
generado: 2026-08-22T15:04:11
---

[00:00:00] Buenas, arrancamos con las pérdidas de la semana.
[00:00:07] En L4 tuvimos scrap alto en la soldadora.
```

Front-matter de `clave: valor` planos — se parsea con un regex, no requiere
PyYAML. Una línea por segmento, timestamp al inicio, siempre `[HH:MM:SS]`.

---

## 4. Gramática de marcadores inline

Es el reemplazo tipeado del "Smart Pen" de la tablet. Se parsea con expresiones
regulares sobre las notas.

```
# Cambio de formato L4        → sección / encabezado
[] revisar seteo de boquilla  → tarea
* scrap alto en soldadora     → punto clave (foco)
@Torres                       → responsable
!05-09                        → vencimiento
```

Reglas:

- `@` y `!` son **modificadores**: pueden aparecer en cualquier posición de una
  línea `[]` o `*`.
- Una línea sin marcador es **texto libre**.
- **El parser nunca modifica el archivo de notas.** Sólo lee y produce
  derivados (`tasks.json`, minuta, índice). `/notes/` es propiedad exclusiva del
  usuario.

---

## 5. Decisiones cerradas

No re-litigar en sesiones futuras:

- **Sin diarización ni identificación de hablantes.** Decisión tomada: las
  transcripciones de planta son ruidosas y atribuir hablantes induce a error.
- **Sin nube, sin APIs pagas.** Ni como alternativa, ni como fallback.
- **Sin frameworks web en la Fase 1.**
- **Sin dependencias pesadas** más allá de lo estrictamente necesario.
- **Chunking por VAD, no por ventana fija.** `faster-whisper` no expone
  solape entre chunks. El intent original ("25 s con 2 s de solape") era evitar
  cortar palabras al medio; con el VAD de Silero los cortes caen en silencios y
  el problema no existe. Se traduce a `max_speech_duration_s=25` (techo por
  bloque) + `speech_pad_ms` (padding alrededor de cada bloque de voz).
  Implementar el solape literal exigiría deduplicar texto entre ventanas, que
  con Whisper duplica o come frases y ensucia los timestamps.
- **Timestamps desde palabras, no desde segmentos.** `word_timestamps=True` y
  se usa el inicio de la primera palabra. El timestamp de segmento derrapa y
  rompe el objetivo de <2 s de error. Verificado en la notebook: diciendo un
  número por segundo conocido, el error fue de **0,06 s**.
- **Las líneas se parten por huecos entre palabras** (`partir_por_huecos`,
  1,5 s por defecto). El VAD elimina los silencios *antes* de que Whisper vea
  el audio, así que Whisper puede juntar en un segmento frases separadas por
  decenas de segundos de reloj: el inicio queda bien, pero la línea abarca todo
  ese rango. Medido en la notebook: cinco números dichos con 10 s de silencio
  entre medio salieron como **una sola línea de 10,06 a 50,74**. Como
  `word_timestamps` ya está activo, partir por los huecos reales cuesta nada y
  es lo que hace utilizable el salto al audio de la Fase 2.
- **El remapeo del VAD puede errar, y `partir_por_huecos` no lo puede arreglar.**
  Medido en la notebook con cinco números dichos cada 10 s: "10", "40" y "50"
  quedaron con 0,06 s de error, pero el "30" salió con timestamp ~21 en vez de
  ~30 — **9 segundos de error**. `faster-whisper` transcribe el audio
  concatenado por el VAD y después remapea; ese remapeo falló para una palabra.
  Partir por huecos no ayuda: no hay hueco que ver donde Whisper no lo reporta.
  Por eso existe `--sin-vad`, que es más lento pero toma los timestamps
  directos del audio, sin remapeo. El front-matter guarda `vad: si|no` para que
  dos transcripciones del mismo audio se puedan comparar.

### Medición de ambos modos (mismo audio, notebook, 22-08-2026)

Cinco números dichos mirando un cronómetro, uno cada 10 s, con silencio entre
medio. Error respecto del momento real (negativo = el timestamp cae antes):

| Dicho a los | Con VAD | Sin VAD |
|---|---|---|
| 10 s | +0,06 | −0,84 |
| 20 s | +0,28 | −1,02 |
| 30 s | **−9** | −1,96 |
| 40 s | −0,11 | −0,72 |
| 50 s | +0,16 | −0,66 |

Lo que dicen estos números:

- **Con VAD**: ±0,3 s cuando el remapeo funciona, catastrófico cuando falla.
  El modo de falla es el peor posible para el objetivo del proyecto.
- **Sin VAD**: nunca falla feo, pero tiene un sesgo sistemático de ~1 s hacia
  atrás. Los cinco segmentos duraron exactamente 1,40 s, cuando decir "diez"
  lleva medio segundo: Whisper estira el arranque hacia atrás. **Ese sesgo es
  benigno para saltar al audio**: se cae un segundo antes y se escucha la frase
  entera.
- Sin VAD **no alucinó nada** en 43 s de silencio, que era el riesgo principal
  de sacarlo. Pero era silencio limpio de oficina, no ruido de planta.

**Sigue siendo una decisión abierta**, y lo que falta medir sólo sale de una
reunión real: cuántas frases inventa sin VAD con ruido de máquinas de fondo, y
cuánto más tarda en una hora de audio. La evidencia hasta acá favorece
`--sin-vad`; el default no se cambió porque falta ese dato.
- **`condition_on_previous_text=False`.** Whisper entra en bucles de repetición
  con audio ruidoso; deshabilitar el arrastre de contexto lo corta.

---

## 6. Fases

| Fase | Qué hace | Estado |
|---|---|---|
| 1 | Grabador + transcriptor con timestamps, watcher | **Hecha** |
| 2 | Nota en vivo: pantalla partida, timestamp sellado por línea | Pendiente |
| 3 | Parser de marcadores → tareas y focos | Pendiente |
| 4 | Minuta por map-reduce con Ollama, salida JSON estructurada | Pendiente |
| 5 | Búsqueda global con SQLite FTS5 sobre todo el histórico | Pendiente |

**No implementar fases futuras ni stubs elaborados antes de que se pidan.**

### Gancho dejado para la Fase 2

`src/record.py` separa la captura (callback de PortAudio que produce bloques
hacia una cola) de la escritura del WAV. Para transcripción en vivo, la Fase 2
engancha un segundo consumidor de esa cola sin reescribir el grabador.

---

## 7. Comandos

```bash
python -m src grabar          # graba, corta con Enter, transcribe solo
python -m src transcribir X   # transcribe un WAV puntual
python -m src vigilar         # watcher sobre /audio/
python -m src dispositivos    # lista micrófonos (diagnóstico)
python -m src gpu             # VRAM libre (diagnóstico)
```

---

## 8. Notas para quien programe acá

- **El código no se puede probar del todo en el entorno de desarrollo remoto**:
  no hay GPU ni placa de audio. Por eso: fallar temprano, con mensajes en
  castellano (`src/deps.py` para dependencias ausentes), y mantener
  `dispositivos` y `gpu` como comandos de diagnóstico aislados.
- **Lo que sí se puede probar, se prueba.** `tests/` reemplaza `sounddevice` y
  `faster-whisper` por dobles (`tests/dobles.py`) y verifica toda la lógica sin
  GPU ni micrófono: `python -m unittest discover -s tests -t .`. Al tocar
  cualquier cosa de `src/`, correrlos antes de dar nada por hecho. Queda fuera
  de su alcance, y sólo se valida en la notebook: que CUDA levante el modelo,
  que PortAudio vea el micrófono, que `msvcrt` corte con Enter, y el error real
  de los timestamps.
- **Los dobles imitan la API real, no una versión conveniente.** Si un test
  falla, primero descartar que el error esté en el doble: `query_devices(None,
  'input')` devuelve el dispositivo por defecto, no la lista — ese fue un bug
  del doble que se hizo pasar por un bug del código.
- **El formato de captura se negocia, no se asume.** 16 kHz mono es lo ideal,
  pero hay entradas que no aceptan mono y otras que no aceptan 16 kHz.
  `record._negociar_formato` prueba las cuatro combinaciones y usa la primera
  que entre; `faster-whisper` resamplea y mezcla a mono al decodificar, así que
  cualquiera sirve. No volver a clavar los canales en 1: un micrófono
  estéreo-only hacía fallar `InputStream` con un error de PortAudio en inglés.
- **Escritura incremental siempre.** El WAV se escribe bloque a bloque y el
  transcript línea a línea. Si el proceso muere a los 50 minutos, quedan 50
  minutos de audio válido. Nunca acumular una reunión entera en RAM.
- **El transcript se escribe a `.md.tmp` y se renombra al final.** El watcher
  usa la existencia del `.md` como marca de "ya procesado"; un `.md` a medio
  escribir sería un falso positivo.
- **`hotwords` sobre `initial_prompt`.** `initial_prompt` no sobrevive a
  `condition_on_previous_text=False`; `hotwords` se aplica en todos los bloques.
  La jerga vive en `glosario.txt`, editable, no hardcodeada.
- **CUDA sin Toolkit.** `ctranslate2` necesita cuBLAS y cuDNN. El CUDA Toolkit
  pide administrador, así que van por pip (`nvidia-cublas-cu12`,
  `nvidia-cudnn-cu12`). En Windows **no alcanza con instalarlos**: los DLLs
  quedan en `site-packages/nvidia/*/bin`, que no está en la ruta de búsqueda, y
  `ctranslate2` falla con "Library cublas64_12.dll is not found" igual.
  `deps.registrar_dlls_cuda()` los pone al alcance antes de importar
  `faster-whisper`, y hace **dos** cosas porque una sola no alcanza:
  `os.add_dll_directory` (para módulos de extensión de Python) y agregar las
  carpetas al `PATH` del proceso. Lo segundo es lo que realmente arregla el
  problema: `ctranslate2` carga cuBLAS desde su código C++ con un
  `LoadLibrary` sin flags, y ese camino **no** consulta los directorios de
  `AddDllDirectory` —sólo participan si quien carga pasa
  `LOAD_LIBRARY_SEARCH_USER_DIRS`—. En Linux es un no-op: ahí los ubica el RPATH.
- **Probar que un DLL carga: por nombre, nunca por ruta absoluta.**
  `ctypes.WinDLL(r"C:\...\cublas64_12.dll")` da OK con que el archivo exista, y
  la pregunta es otra: si Windows lo encuentra solo. `ctypes.WinDLL("cublas64_12.dll")`
  es la prueba que vale. El diagnóstico de `gpu` daba falsos OK por esto.
- **Modelos offline.** Primera descarga a `HF_HOME`. Después de la primera
  corrida, `HF_HUB_OFFLINE=1` garantiza que nunca más toque la red.
