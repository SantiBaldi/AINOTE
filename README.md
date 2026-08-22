# AINOTE

Captura de reuniones **100 % local**: audio → transcripción → notas → minuta →
tareas → búsqueda. Sin nube, sin APIs, sin telemetría.

**Estado: Fase 1** — grabador y transcriptor. Las fases 2 a 5 (nota en vivo,
marcadores, minuta, búsqueda) todavía no están. Ver `CLAUDE.md`.

---

## Instalación (sin permisos de administrador)

```bat
cd C:\ruta\a\AINOTE
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Un venv vive en tu carpeta de usuario: no toca `Program Files` ni el registro.
Si preferís no usar venv, `pip install --user -r requirements.txt` instala en
`%APPDATA%\Python`.

### Modelo y modo offline

La primera transcripción baja `large-v3-turbo` (~1,6 GB) desde Hugging Face.
Conviene fijar dónde queda antes de esa primera corrida:

```bat
setx HF_HOME "%USERPROFILE%\.cache\huggingface"
```

Una vez bajado, para garantizar que nunca más toque la red:

```bat
setx HF_HUB_OFFLINE 1
```

El VAD de Silero viene empaquetado dentro de `faster-whisper`: no descarga nada
aparte.

---

## Uso

Doble clic en **`grabar.bat`**: pide el nombre de la reunión, graba, y al cortar
con Enter transcribe sola. Es todo lo que hace falta para una reunión.

Para la consola, doble clic en **`ainote.bat`** abre una ya ubicada en el
proyecto y con el entorno activado. Sin eso hay que hacer los dos pasos a mano
cada vez, porque una consola nueva arranca en la carpeta del usuario:

```bat
cd %USERPROFILE%\Documents\AINOTE
.venv\Scripts\activate
```

Después:

```bat
python -m src grabar
```

Te pide el nombre de la reunión, graba del micrófono mostrando el cronómetro, y
al cortar con **Enter** (o `q`) transcribe sola. Salen dos archivos:

```
audio/2026-08-22_perdidas.wav
transcripts/2026-08-22_perdidas.md
```

Otros comandos:

```bat
python -m src grabar perdidas --dispositivo 2   :: sin preguntar, con mic elegido
python -m src transcribir 2026-08-22_perdidas   :: transcribir un audio puntual
python -m src importar "C:\ruta\grabacion.m4a"  :: traer una grabación de afuera
python -m src vigilar                           :: transcribe solo todo WAV nuevo
python -m src dispositivos                      :: listar micrófonos
python -m src gpu                               :: VRAM libre y qué modelo entra
```

El VAD (filtro de silencios) va **apagado**: medido sobre una reunión real, se
comía intervenciones enteras. `--con-vad` lo activa si alguna vez la velocidad
importa más que no perder nada.

`vigilar` sirve si grabás con otra cosa: dejalo corriendo y tirá los WAV en
`audio/` con el nombre `AAAA-MM-DD_<slug>.wav`.

Podés dejar `vigilar` corriendo y usar `grabar` al mismo tiempo: un cerrojo en
disco garantiza que nunca se carguen dos Whisper a la vez, que es lo que haría
reventar la VRAM. El que llega segundo espera y reintenta solo.

---

## Verificar que anda

**Flujo completo.** `python -m src grabar prueba`, hablá tres minutos, cortá con
Enter. Sin tocar nada más tiene que aparecer `transcripts/AAAA-MM-DD_prueba.md`.

**Precisión de los timestamps (< 2 s).** Grabá mirando un cronómetro y decí en
voz alta el minuto cada 30 segundos ("treinta", "uno cero cero", "uno treinta").
Después comparás cada frase con el `[HH:MM:SS]` de su línea.

**VRAM liberada.** En otra consola, `nvidia-smi` antes, durante y después. El
"después" tiene que volver al valor del "antes". Los reportes que imprime la
transcripción te dan la misma información sin salir de la consola.

**Sin internet.** Con `HF_HUB_OFFLINE=1` y el modelo ya bajado, desconectá el
wifi y transcribí. Tiene que andar igual.

---

## Correr las pruebas

```bat
python -m unittest discover -s tests -t .
```

No necesitan GPU, micrófono ni internet: `sounddevice` y `faster-whisper` se
reemplazan por dobles. Corren en cualquier máquina y tardan unos segundos.

Cubren el bucle de grabación, el formato del transcript, la convención de
nombres, la verificación de VRAM y la lógica del watcher. **No cubren** —y sólo
se puede probar en la notebook— que CUDA levante el modelo, que PortAudio vea el
micrófono, y el error real de los timestamps.

---

## Si algo falla

| Síntoma | Qué mirar |
|---|---|
| `VRAM insuficiente` | `python -m src gpu`. Cerrá el navegador; si tenés Ollama, bajalo. Último recurso: `--compute-type int8_float16` |
| No entra audio | `python -m src dispositivos` y elegí con `--dispositivo N` |
| Se cuela ruido de máquina como voz | subí `--umbral-vad 0.6` o `0.7` |
| Las líneas abarcan demasiado tiempo | bajá `--hueco-maximo 1.0` o `0.8` |
| Las tildes se ven raras con `type` | es la consola, no el archivo. Los `.md` son UTF-8: abrilos en un editor, o corré `chcp 65001` antes |
| Whisper repite una frase en loop | ya está mitigado; si pasa igual, subí el umbral de VAD |
| Palabras de planta mal transcriptas | agregalas arriba de todo en `glosario.txt` |
| `Ya hay otra transcripción corriendo` | es a propósito: nunca corren dos a la vez. Esperá. Si ninguna está corriendo de verdad, borrá `.ainote.lock` |
| `El micrófono no acepta 16000 Hz mono` | no es un error: avisa que graba en otro formato y sigue |

---

## Privacidad

El material de estas reuniones es confidencial. `.gitignore` excluye `audio/`,
`transcripts/`, `notes/`, `minutas/`, `tasks.json` e `index.sqlite`. **Al repo
va el código; el contenido se queda en la notebook.** No quites esas reglas.
