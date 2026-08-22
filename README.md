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
python -m src transcribir 2026-08-22_perdidas   :: transcribir un WAV puntual
python -m src vigilar                           :: transcribe solo todo WAV nuevo
python -m src dispositivos                      :: listar micrófonos
python -m src gpu                               :: VRAM libre y qué modelo entra
```

`vigilar` sirve si grabás con otra cosa: dejalo corriendo y tirá los WAV en
`audio/` con el nombre `AAAA-MM-DD_<slug>.wav`.

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

## Si algo falla

| Síntoma | Qué mirar |
|---|---|
| `VRAM insuficiente` | `python -m src gpu`. Cerrá el navegador; si tenés Ollama, bajalo. Último recurso: `--compute-type int8_float16` |
| No entra audio | `python -m src dispositivos` y elegí con `--dispositivo N` |
| Se cuela ruido de máquina como voz | subí `--umbral-vad 0.6` o `0.7` |
| Whisper repite una frase en loop | ya está mitigado; si pasa igual, subí el umbral de VAD |
| Palabras de planta mal transcriptas | agregalas arriba de todo en `glosario.txt` |

---

## Privacidad

El material de estas reuniones es confidencial. `.gitignore` excluye `audio/`,
`transcripts/`, `notes/`, `minutas/`, `tasks.json` e `index.sqlite`. **Al repo
va el código; el contenido se queda en la notebook.** No quites esas reglas.
