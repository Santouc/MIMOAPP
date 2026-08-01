# Anexo técnico avanzado — T.L.S

Este anexo reúne detalles de implementación para responder preguntas de arquitectura, visión por computador, machine learning, datos, rendimiento y validación.

**Fuente de verdad:** la ruta usada por la aplicación es `desktop_app.py` → `app/` → `services/` → `core/`/`ml/`. Los scripts `ml/train.py` y `ml/train_dynamic.py`, además de `utils/config.py`, contienen rutas auxiliares o parámetros históricos que no siempre coinciden con la pantalla activa.

---

## 1. Contrato técnico de extremo a extremo

```text
BGR frame
  shape aproximado: (480, 640, 3)
       ↓ ImageProcessor
BGR frame preprocesado
       ↓ HandDetector / MediaPipe Tasks
lista de manos: [mano_0, mano_1, ...]
mano_0: (21, 3)
       ↓ estabilización
landmarks válidos: (21, 3)
       ├──────────────────────────────┐
       │                              │
       ▼                              ▼
normalize_landmarks            ventana de 20 frames
(21, 3)                         (20, 21, 3)
       │                              │
       ▼                              ▼
MLP estático                   BiLSTM dinámico
       └──────────────┬───────────────┘
                      ▼
              etiqueta + confianza
                      ▼
          consenso y compuertas temporales
                      ▼
               token de transcripción
                      ▼
        raw_text → interpretación → salida
```

### Contratos de forma

| Objeto | Forma | Tipo | Significado |
|---|---:|---|---|
| Frame | `(480, 640, 3)` | `uint8` | Imagen BGR de cámara, aproximadamente |
| Mano cruda | `(21, 3)` | `float` | 21 puntos `(x, y, z)` |
| Pose estática batch | `(N, 21, 3)` | `float32` | N ejemplos |
| Secuencia dinámica | `(T, 21, 3)` | `float32` | T frames de una muestra |
| Batch dinámico | `(N, T, 21, 3)` | `float32` | N secuencias |
| Etiquetas | `(C,)` | `list[str]` | Nombre de cada clase |
| Targets | `(N,)` | `int32` | Índice de label para cada muestra |
| Predicción | `(C,)` | `float` | Softmax por clase |

### Invariantes

1. Una mano válida debe tener exactamente 21 landmarks.
2. Cada landmark debe tener tres coordenadas.
3. `len(X)` debe coincidir con `len(y)`.
4. Cada índice de `y` debe poder resolverse en `labels`.
5. Una clase nueva requiere ejemplos y reentrenamiento.
6. Las labels exportadas deben conservar el mismo orden que los índices del dataset.
7. Una secuencia dinámica se lleva a 20 frames antes de inferencia.

---

## 2. Arquitectura de ejecución

### Inicio

`desktop_app.py` crea `QApplication`, crea `MainWindow`, muestra la ventana y entra en `application.exec()`.

`MainWindow` construye un `QStackedWidget` con siete vistas principales:

1. `HomeScreen`
2. `TeachSignScreen`
3. `ManageSignsScreen`
4. `TranslateScreen`
5. `ExtensionsScreen`
6. `DocumentScreen` de créditos
7. `DocumentScreen` de manual

`AppContext` inicializa ocho servicios explícitos compartidos:

- `PathService`
- `SignService`
- `CaptureService`
- `DocumentService`
- `TrainingService`
- `LibraryService`
- `TranscriptionService`
- `ExtensionService`

La visualización y los previews usan módulos auxiliares cuando se solicitan; no son un noveno servicio creado por `AppContext`.

### Ciclo de vida de recursos

```text
_iniciar_cámara
  → VideoCapture(0)
  → configurar 640×480 y 30 FPS
  → crear HandDetector(max_hands=1)
  → iniciar QTimer cada 30 ms

_procesar_frame
  → leer frame
  → procesar imagen
  → detectar landmarks
  → estabilizar
  → clasificar
  → transcribir
  → actualizar UI

_detener_cámara / closeEvent
  → detener timer
  → release de OpenCV
  → cleanup de MediaPipe
  → shutdown de extensiones
```

La cámara no se procesa en un bucle infinito independiente: el evento `QTimer` dispara `_process_frame`. Esto simplifica la integración con Qt, aunque el entrenamiento se ejecuta de forma modal y puede bloquear la interfaz cuando el dataset crece.

---

## 3. Preprocesamiento de imagen

`ImageProcessor.preprocess` realiza estas operaciones:

1. Redimensiona a `(640, 480)` si es necesario.
2. Aplica `GaussianBlur` con kernel `(5, 5)`.
3. Convierte a LAB y aplica CLAHE al canal de luminancia:
   - `clipLimit = 2.0`
   - `tileGridSize = (8, 8)`
4. Convierte a escala de grises para estimar iluminación.
5. Calcula un fondo suavizado con kernel `(101, 101)`.
6. Divide la imagen por la iluminación estimada y aplica el resultado a los tres canales.

También existen funciones auxiliares para región de piel, ROI y bordes. No todas son necesarias en la ruta principal porque MediaPipe recibe el frame completo preprocesado.

### MediaPipe

`HandDetector` usa la API `mp.tasks.vision.HandLandmarker` con:

- asset: `data/models/hand_landmarker.task`
- modo: `IMAGE`
- manos activas desde la pantalla: `1`
- confianza mínima de detección: `0.5` por defecto
- confianza mínima de presencia: `0.5`
- confianza mínima de tracking: `0.5`

El detector convierte BGR a RGB, crea un `mp.Image` y devuelve coordenadas normalizadas.

---

## 4. Normalización matemática

### Estática

Sea `X ∈ R^(N×21×3)` el batch de manos. Para cada muestra se resta la muñeca:

```text
X_centrada[n, i, :] = X[n, i, :] - X[n, 0, :]
```

Después se calcula una escala basada en la distancia entre la muñeca `0` y el punto `9`:

```text
s_n = ||X_centrada[n, 9, :]||_2
X_normalizada[n, i, :] = X_centrada[n, i, :] / max(s_n, 1e-6)
```

La constante `1e-6` evita división por cero.

### Dinámica diseñada

Para una secuencia `S ∈ R^(T×21×3)`:

```text
s = mediana_t( ||S[t, 9, :] - S[t, 0, :]||_2 )
S_centrada[t, i, :] = S[t, i, :] - S[0, 0, :]
S_normalizada = S_centrada / max(s, 1e-6)
```

La diferencia clave es que se resta la primera muñeca a toda la secuencia, no una muñeca distinta a cada frame. Eso mantiene la trayectoria global.

### Riesgo actual

`TeachSignScreen._record_dynamic_frame` guarda cada frame usando `normalize_single_hand`, que centra individualmente la mano. Luego el `TrainingService` aplica la normalización dinámica. La segunda normalización ya no puede recuperar un desplazamiento eliminado previamente.

**Conclusión técnica:** la función dinámica es trayectoria-aware, pero la ruta de captura debe almacenar landmarks crudos o una representación que preserve la posición global antes de entrenar trayectorias reales.

---

## 5. Estabilización y selección de predicción

### Parámetros activos de `TranslateScreen`

| Parámetro | Valor | Función |
|---|---:|---|
| `sign_buffer.maxlen` | 10 | Historial de predicciones estáticas |
| `dynamic_sequence.maxlen` | 20 | Ventana temporal |
| `dynamic_buffer.maxlen` | 5 | Consenso de predicciones dinámicas |
| `confidence_threshold` | 0.7 | Confianza mínima del consenso |
| `max_missed_frames` | 4 | Frames reutilizados al perder tracking |
| `smoothing_alpha` | 0.65 | Peso del landmark actual |
| `DYNAMIC_MOTION_THRESHOLD` | 0.012 | Límite para considerar poco movimiento |
| `DYNAMIC_STILL_FRAMES_RESET` | 6 | Frames quietos que limpian secuencia dinámica |
| `STATIC_MIN_STILL_FRAMES` | 6 | Frames quietos antes de aceptar pose estática |

El suavizado usa:

```text
suavizado = 0.65 · actual + 0.35 · anterior
```

### Predicción estática

1. El clasificador devuelve `(label, confidence)`.
2. La pareja se agrega al buffer de longitud 10.
3. Si hay menos de 6 frames quietos, no se acepta.
4. Se descartan predicciones debajo de `0.7` y `unknown`.
5. Se devuelve la etiqueta más frecuente.

### Predicción dinámica

1. Si el movimiento es insuficiente durante 6 frames, se limpia la cola.
2. Se acumulan hasta 20 landmarks.
3. Al completar 20, se ejecuta `classify_sequence`.
4. Las últimas 5 predicciones forman un buffer de consenso.
5. Se selecciona la clase más frecuente sobre `0.7`.
6. La etiqueta dinámica tiene prioridad sobre la estática.

### Máquina de estados conceptual

```text
PERDIDO
  ├─ landmarks válidos → OK
  └─ sin landmarks → PERDIDO

OK
  ├─ frame perdido, contador < 4 → RECUPERANDO
  ├─ frame perdido, contador >= 4 → PERDIDO
  └─ landmarks válidos → OK

OK + poco movimiento durante 6 frames → limpiar dinámica
OK + pose estable durante 6 frames → candidato estático
candidato + consenso/confianza → token aceptado
```

---

## 6. Arquitectura y conteo de parámetros

Los conteos siguientes se derivan de las arquitecturas construidas por `TrainingService`; sirven para explicar la complejidad de la red y no sustituyen a `model.summary()`.

### Modelo estático

```text
(21, 3)
→ Dense(64)
→ BatchNormalization
→ Dense(64)
→ Flatten
→ Dense(256)
→ Dropout(0.4)
→ Dense(128)
→ Dropout(0.3)
→ Dense(C)
```

Parámetros entrenables:

```text
Dense(64) por landmark:       3×64 + 64       = 256
BatchNorm(64):                                = 128
Dense(64):                    64×64 + 64      = 4.160
Dense(256):                   1.344×256 + 256 = 344.320
Dense(128):                   256×128 + 128   = 32.896
Salida:                       128×C + C       = 129C
Total:                                        = 381.760 + 129C
```

Con `C = 22` labels:

```text
381.760 + 129×22 = 384.598 parámetros entrenables aproximadamente
```

`Dropout` y `Flatten` no agregan parámetros.

### Modelo dinámico

```text
(20, 21, 3)
→ TimeDistributed(Dense(64))
→ TimeDistributed(Flatten)
→ BiLSTM(64, return_sequences=True)
→ Dropout(0.3)
→ BiLSTM(32)
→ Dense(64)
→ Dropout(0.3)
→ Dense(C)
```

Conteo aproximado:

```text
TimeDistributed(Dense(64)):                       = 256
BiLSTM(64), dos direcciones: 2×4×(1.344+64+1)×64 = 721.408
BiLSTM(32), dos direcciones: 2×4×(128+32+1)×32    = 41.216
Dense(64):                                         = 4.160
Salida:                                             = 65C
Total:                                              = 767.040 + 65C
```

Con `C = 22`:

```text
767.040 + 65×22 = 768.470 parámetros entrenables aproximadamente
```

Si el entrenamiento agrega `NO_SENA`, `C = 23` y el total aumenta en `65` parámetros.

### Interpretación

El modelo dinámico tiene aproximadamente el doble de parámetros que el estático para 22 clases. Por eso necesita más muestras, más tiempo de entrenamiento y una evaluación más cuidadosa.

---

## 7. Entrenamiento y generación de datos

### Entrenamiento estático activo

- Épocas: `50`.
- Batch: `min(8, número_de_muestras)`.
- Validación: `20%` si hay al menos `10` muestras procesadas; si no, `0%`.
- Orden aleatorio: generador NumPy con semilla `42`.
- Optimizador: Adam con `learning_rate = 0.001`.
- Pérdida: `sparse_categorical_crossentropy`.
- Salida: `data/models/model.h5` y `data/models/labels.json`.

### Entrenamiento dinámico activo

- Ventana esperada: `20` frames.
- Épocas: `80`.
- Batch: `min(8, número_de_secuencias)`.
- Validación: `20%` desde `10` secuencias.
- Se normaliza el batch conservando la trayectoria según la función del servicio.
- Se generan negativos si existe dataset estático.
- Salida: `data/models/model_dynamic.h5` y `data/models/labels_dynamic.json`.

### Negativos dinámicos

El número de negativos se calcula como:

```text
N_negativos = clip(round(max_class_count × 1.2), mínimo=6, máximo=60)
```

Se alternan tres tipos:

1. **Hold:** mano casi quieta con ruido y pequeña deriva.
2. **Transition:** interpolación entre dos poses distintas.
3. **Drift:** camino aleatorio con desplazamiento limitado.

La semilla usada es `7` para esta generación. Su objetivo es que el modelo no convierta cualquier transición o movimiento casual en una seña.

### Problema de validación

`validation_split` separa ejemplos del mismo conjunto, pero no garantiza separación por usuario. Si las muestras de entrenamiento y validación provienen de la misma sesión, la validación puede ser optimista. Una prueba sólida debe separar por:

- usuario;
- sesión de captura;
- iluminación/ángulo;
- seña y tipo estático/dinámico.

---

## 8. Persistencia y consistencia de índices

### Registro de una seña

Cada entrada contiene:

```json
{
  "id": "a",
  "name": "A",
  "types": ["static"],
  "created_at": "...",
  "updated_at": "...",
  "static_samples": 50,
  "dynamic_samples": 0
}
```

El ID se construye normalizando el nombre: se quitan tildes, se reemplazan caracteres no alfanuméricos y se pasa a minúsculas.

### Eliminación

Si se elimina la label con índice `k`:

1. Se eliminan sus muestras.
2. Se crea `new_labels` sin esa label.
3. Cada índice posterior se remapea a su nueva posición.
4. Se reescribe `X`, `y`, `labels` y metadata.
5. Se exportan labels nuevas.
6. Se invalidan los modelos antiguos.
7. La pantalla intenta reentrenar estático y dinámico.

### Reset

El reset elimina datasets, labels, modelos, pendientes, previews y registro, pero conserva `hand_landmarker.task` porque es un recurso base, no un dato aprendido por la aplicación.

---

## 9. Transcripción como problema de optimización

El texto bruto es una cadena de tokens, por ejemplo:

```text
raw_text = HOLASOYSANTI
```

El servicio construye un vocabulario con:

- hasta `50.000` palabras de `wordfreq`;
- `MIN_ZIPF = 3.0`;
- longitud máxima de palabra `24`;
- mapas definidos por el usuario;
- frases guardadas en memoria.

Para cada posición `i` de la cadena se prueban candidatos `key[i:j]` hasta `j = i + 24`. La programación dinámica conserva el mejor puntaje acumulado:

```text
score_palabra = score_previo + (frecuencia - 9.0) - 3.0
score_desconocido = score_previo - 2.2 - penalización_de_inicio
```

Después se reconstruyen los segmentos desde el final de la cadena. Las partes desconocidas se pueden reparar si tienen al menos `5` caracteres y distancia de Levenshtein `1` respecto de una palabra frecuente.

### Memoria aproximada

La memoria busca coincidencia exacta y luego la mejor distancia, aceptando una tolerancia dependiente del largo:

```text
aceptar si distancia ≤ max(1, largo_clave / 5)
```

### Idiomas

El selector contiene `6` códigos:

```text
es, en, pt, fr, it, de
```

La configuración actual persistida es `es`.

---

## 10. Extensión de voz y concurrencia

`voz` usa:

- demora automática: `2.0` segundos;
- velocidad: `165`;
- volumen: `1.0`;
- cola de texto: `queue.Queue`;
- un hilo daemon para el motor `pyttsx3`;
- un `threading.Timer` por pausa detectada.

Cada cambio de transcripción cancela el timer anterior. Si la cadena ya fue hablada, no vuelve a programarla. Al cerrar la aplicación se cancela el timer, se vacía la cola y se solicita el cierre del worker.

La interfaz sigue siendo Qt; la voz no debe ejecutarse en el hilo GUI porque `engine.runAndWait()` puede bloquearlo.

---

## 11. Complejidad y memoria aproximadas

### Representación

Con `float32`:

- pose estática: `21 × 3 × 4 = 252` bytes sin overhead;
- secuencia dinámica: `20 × 21 × 3 × 4 = 5.040` bytes sin overhead;
- batch estático de `8`: `8 × 252 = 2.016` bytes;
- batch dinámico de `8`: `8 × 5.040 = 40.320` bytes.

JSON ocupa más porque representa cada número como texto y agrega corchetes, comas, claves y espacios.

### Inferencia por frame

Para una mano, la extracción y suavizado recorren aproximadamente `21` puntos. La rama estática hace una inferencia por pose estable. La rama dinámica necesita acumular `20` frames y luego puede inferir sobre una ventana deslizante.

### Latencia conceptual

- La cámara se configura a `30 FPS`, cuyo intervalo ideal es `33,3 ms`.
- El `QTimer` se programa cada `30 ms`.
- La rama estática puede decidir después de al menos `6` frames quietos: aproximadamente `200 ms` a 30 FPS, más inferencia y consenso.
- La dinámica necesita `20` frames: aproximadamente `667 ms` a 30 FPS antes de la primera ventana completa, más `5` predicciones para consenso si se reciben de forma consecutiva.

Estos valores son límites teóricos; la latencia real depende de cámara, MediaPipe, CPU/GPU, TensorFlow y carga del sistema.

---

## 12. Configuración activa versus configuración histórica

| Tema | Ruta activa | Valor histórico/auxiliar | Cómo responder |
|---|---|---|---|
| FPS | `TranslateScreen` y `TeachSignScreen` | `30` | Es el valor usado por las pantallas actuales |
| FPS | `utils/config.py` | `15` | Configuración antigua/general no conectada al flujo actual |
| Máximo de manos | pantallas crean detector con `1` | `Config` declara `2` | El flujo actual prioriza una mano |
| Clases | labels JSON | `Config.num_classes = 8` | La fuente actual es el registro/labels exportado |
| Threshold consenso | pantalla `0.7` | config histórico `0.8` | Para explicar live se usa `0.7` |
| Normalización dinámica | servicio conserva trayectoria | script independiente normaliza por frame | `TrainingService` describe la ruta de app, pero la captura UI debe alinearse |

Reconocer estas diferencias demuestra que se revisó el código real y evita atribuir parámetros de archivos no usados a la demo.

---

## 13. Observabilidad y fallos

### Estados visibles

- Estado de modelos: ambos, solo estático, solo dinámico o ninguno.
- Tracking: `OK`, `RECUPERANDO`, `PERDIDO`.
- FPS calculado en la pantalla.
- Estado de transcripción: esperando, detectando, manteniendo, registrado, repetido.
- Estado de extensión: activa, habilitada no cargada, desactivada o error.

### Fallos controlados

- Cámara no disponible: mensaje crítico y liberación de recurso.
- Modelo inexistente: clasificador devuelve `unknown`.
- Landmark inválido: no se clasifica la pose.
- Dataset vacío o de una sola clase: entrenamiento detenido con mensaje.
- Extensión defectuosa: error aislado en `ExtensionService`.
- Dependencia `pyttsx3` ausente: voz no disponible, aplicación sigue.

### Fallos que requieren mejora

- Entrenamiento modal puede bloquear la GUI.
- El log de archivo está desactivado por defecto en `utils/logger.py`.
- JSON no ofrece transacciones ni versionado de dataset.
- No hay métrica automática de test por usuario.
- El modelo dinámico no existe en el estado actual.

---

## 14. Plan de validación recomendado

### Nivel 1 — Integridad de datos

- validar shapes;
- comprobar `len(X) == len(y)`;
- verificar índices dentro de `[0, C-1]`;
- comprobar que cada clase activa tenga muestras;
- comparar conteos del registro contra `y`.

### Nivel 2 — Clasificación offline

Separar train/validation/test por sesión y usuario. Reportar:

- accuracy;
- balanced accuracy;
- precision macro;
- recall macro;
- F1 macro;
- matriz de confusión;
- porcentaje de `unknown`;
- latencia p50 y p95.

### Nivel 3 — Pipeline temporal

Probar:

- pose mantenida;
- transición entre letras;
- mano quieta;
- movimiento casual;
- pérdida temporal de tracking;
- repetición de una misma letra;
- dos señas consecutivas;
- dinámica con distinta velocidad.

### Nivel 4 — Prueba de usuario

Medir por usuario nuevo y condiciones distintas:

- iluminación;
- fondo;
- distancia;
- orientación;
- mano dominante;
- velocidad del gesto.

El script `_test_deteccion.py` es un punto de partida para integración sintética, no un reemplazo de esta evaluación.

---

## 15. Respuestas técnicas de 15 segundos

- **¿Por qué 21 puntos?** Porque representan la topología anatómica de la mano y reducen la entrada a 63 coordenadas por pose.
- **¿Por qué BiLSTM?** Porque una seña dinámica depende del orden temporal de 20 frames, no solo de una forma.
- **¿Cómo se evita repetir?** Retención mínima, cooldown, buffers y consenso mayoritario.
- **¿Cuántos parámetros tiene?** Para 22 labels, aproximadamente 384.598 entrenables en estático y 768.470 en dinámico según las arquitecturas del servicio.
- **¿Cuál es la mayor limitación?** Datos y generalización: el modelo aprende el dataset y el dinámico aún no tiene artefacto entrenado en el snapshot.
- **¿Qué mejora primero?** Preservar trayectoria desde captura, crear test independiente y reportar métricas por usuario.
