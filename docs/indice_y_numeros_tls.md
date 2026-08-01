# Índice y números de T.L.S

Documento de consulta rápida para encontrar cualquier sección de la presentación y responder preguntas cuantitativas durante la defensa.

> **Fecha del snapshot numérico:** 31-07-2026.  
> Los valores marcados como **activo** salen de la ruta de la aplicación. Los valores **históricos/auxiliares** aparecen en archivos que no necesariamente controlan la demo actual.

---

# Índice de documentos

| Documento | Para qué sirve |
|---|---|
| [`presentacion_tls.md`](presentacion_tls.md) | Guion completo de exposición y diapositivas |
| [`guia_defensa_tls.md`](guia_defensa_tls.md) | Preguntas y respuestas para el jurado |
| [`anexo_tecnico_avanzado_tls.md`](anexo_tecnico_avanzado_tls.md) | Arquitectura, fórmulas, parámetros, complejidad y validación |
| [`infografia_tls.svg`](infografia_tls.svg) | Diagrama visual del sistema |
| [`anexo_tecnico.md`](anexo_tecnico.md) | Anexo técnico base de la aplicación |
| [`conceptos.md`](conceptos.md) | Explicación conceptual de landmarks y modos |
| [`manual_uso.md`](manual_uso.md) | Instalación, controles y flujo de usuario |
| [`../README.md`](../README.md) | Descripción general, tecnologías y estructura |
| [`../extensions/README.md`](../extensions/README.md) | Contrato para crear extensiones |

## Índice por tema

- **Problema y objetivos:** presentación, diapositivas 1–2.
- **Funciones de usuario:** presentación, diapositiva 3.
- **Arquitectura:** presentación, diapositiva 4; anexo avanzado, secciones 1–2.
- **Cámara y MediaPipe:** presentación, diapositivas 5–6; anexo, sección 3.
- **Normalización:** presentación, diapositiva 6; anexo, sección 4.
- **Buffers y estados:** presentación, diapositiva 7; anexo, sección 5.
- **Modelos TensorFlow:** presentación, diapositiva 8; anexo, sección 6.
- **Entrenamiento:** presentación, diapositiva 9; anexo, sección 7.
- **Transcripción:** presentación, diapositivas 10–11; anexo, sección 9.
- **Extensiones y voz:** presentación, diapositiva 12; anexo, sección 10.
- **Persistencia:** presentación, diapositiva 13; anexo, sección 8.
- **Estado actual:** presentación, diapositiva 15; guía, preguntas de estado.
- **Limitaciones y futuro:** presentación, diapositiva 17; anexo, secciones 12–14.

---

# 1. Números del estado actual

## 1.1 Dataset y labels

| Dato | Valor |
|---|---:|
| Labels registradas | **22** |
| Letras A–Z posibles | **26** |
| Letras faltantes en el snapshot | **4**: G, J, S, X |
| Clases con muestras estáticas | **10** |
| Clases sin muestras estáticas | **12** |
| Muestras estáticas totales | **632** |
| Secuencias dinámicas actuales | **0** |
| Clases declaradas en dataset dinámico | **22** |
| Modelo estático presente | **1**: `model.h5` |
| Modelo dinámico presente | **0**: no existe `model_dynamic.h5` |
| Archivos de labels presentes | **2**: estático y dinámico |
| Frases actuales en memoria | **2** |
| Extensiones detectadas en el repositorio | **1**: `voz` |
| Extensiones desactivadas en configuración | **0** |
| Integraciones Arduino en alcance | **0** |

## 1.2 Distribución exacta de las 632 muestras estáticas

| Clase | Muestras | Porcentaje del total |
|---|---:|---:|
| A | 50 | 7,91% |
| B | 34 | 5,38% |
| C | 38 | 6,01% |
| D | 42 | 6,65% |
| E | 44 | 6,96% |
| F | 100 | 15,82% |
| H | 57 | 9,02% |
| I | 80 | 12,66% |
| L | 66 | 10,44% |
| O | 121 | 19,15% |
| **Total** | **632** | **100%** |

Las otras 12 labels del registro tienen cero muestras: `K, M, N, P, Q, R, T, U, V, W, Y, Z`.

## 1.3 Archivos de datos observados

| Archivo | Valor cuantitativo observado |
|---|---:|
| `dataset_static.json` | 632 muestras, 22 labels, shape conceptual `(632, 21, 3)` |
| `dataset_dynamic.json` | 0 secuencias, 22 labels |
| `labels.json` | 22 nombres |
| `labels_dynamic.json` | 22 nombres |
| `signs.json` | 22 registros |
| `memory.json` | 2 entradas |
| `rules.json` | idioma `es` y mapas base |
| `extensions.json` | lista `disabled` con 0 elementos |

### Tamaños de archivo del snapshot, en bytes

Estos tamaños son informativos y pueden cambiar al guardar datos/modelos:

| Archivo | Bytes |
|---|---:|
| `data/models/hand_landmarker.task` | 7.819.105 |
| `data/models/model.h5` | 4.666.224 |
| `data/datasets/dataset_static.json` | 1.428.032 |
| `data/datasets/dataset_dynamic.json` | 485 |
| `data/models/labels.json` | 179 |
| `data/models/labels_dynamic.json` | 179 |
| `data/signs/signs.json` | 5.085 |
| `data/transcription/rules.json` | 379 |
| `data/transcription/memory.json` | 77 |
| `extensions/voz/extension.py` | 5.203 |

---

# 2. Números de cámara y procesamiento

## 2.1 Cámara activa

| Parámetro | Valor |
|---|---:|
| Dispositivo OpenCV | `0` |
| Ancho | **640 px** |
| Alto | **480 px** |
| Relación aproximada | **4:3** |
| FPS solicitado por la pantalla | **30** |
| Intervalo ideal a 30 FPS | **33,3 ms** |
| Intervalo del `QTimer` | **30 ms** |
| Manos usadas en pantallas | **1** |
| Confianza de detección por defecto | **0,5** |
| Confianza de presencia por defecto | **0,5** |
| Confianza de tracking por defecto | **0,5** |

## 2.2 Preprocesamiento de imagen

| Operación | Número |
|---|---:|
| Canales de imagen BGR | **3** |
| Kernel Gaussian Blur | **5 × 5** |
| CLAHE `clipLimit` | **2,0** |
| CLAHE `tileGridSize` | **8 × 8** |
| Kernel para iluminación | **101 × 101** |
| Canales corregidos después de la máscara | **3** |
| Límite HSV inferior usado por máscara de piel | **[0, 20, 70]** |
| Límite HSV superior usado por máscara de piel | **[20, 255, 255]** |
| Kernel morfológico de piel | **5 × 5** |

La máscara de piel y la detección de bordes son funciones auxiliares; la ruta principal usa el frame preprocesado y el detector MediaPipe.

---

# 3. Números de landmarks y tensores

| Concepto | Cálculo | Resultado |
|---|---:|---:|
| Landmarks por mano | estándar MediaPipe | **21** |
| Coordenadas por landmark | `x, y, z` | **3** |
| Valores de una pose | `21 × 3` | **63** |
| Bytes float32 de una pose | `21 × 3 × 4` | **252** |
| Frames de una secuencia dinámica | ventana activa | **20** |
| Valores de secuencia dinámica | `20 × 21 × 3` | **1.260** |
| Bytes float32 de secuencia | `20 × 21 × 3 × 4` | **5.040** |
| Bytes de batch estático de 8 | `8 × 252` | **2.016** |
| Bytes de batch dinámico de 8 | `8 × 5.040` | **40.320** |
| Coordenadas normalizadas | `x, y, z` | aproximadamente 3 por punto |
| Punto de referencia de escala | landmark | **9** |
| Punto de traslación | landmark muñeca | **0** |
| Ejes de proyección visual | `x, y` | **2** |

## Numeración anatómica

| Región | IDs |
|---|---|
| Muñeca | 0 |
| Pulgar | 1–4: **4** puntos |
| Índice | 5–8: **4** puntos |
| Medio | 9–12: **4** puntos |
| Anular | 13–16: **4** puntos |
| Meñique | 17–20: **4** puntos |
| Total | **1 + 5×4 = 21** |

---

# 4. Números de estabilización y lógica temporal

| Parámetro | Valor activo |
|---|---:|
| Buffer estático | **10** predicciones |
| Ventana dinámica | **20** frames |
| Buffer dinámico | **5** predicciones |
| Confianza mínima del consenso UI | **0,7** |
| Confianza mínima del clasificador para no devolver `unknown` | **0,5** |
| Frames quietos mínimos para estática | **6** |
| Frames quietos para reset dinámico | **6** |
| Frames perdidos tolerados en traducción | **4** |
| Alpha de suavizado | **0,65** |
| Peso de landmarks anteriores | **0,35** |
| Umbral de movimiento | **0,012** |
| Retención mínima de una seña | **0,75 s** |
| Cooldown para repetir la misma seña | **0,9 s** |
| Frames mínimos para conservar una secuencia dinámica capturada | **5** |
| Frames objetivo después de remuestrear una secuencia | **20** |
| Intervalo de animación del preview | **120 ms** |

## Conversiones útiles a 30 FPS

| Evento | Frames | Tiempo teórico |
|---|---:|---:|
| 6 frames quietos | 6 | **200 ms** |
| 20 frames dinámicos | 20 | **667 ms** |
| 5 predicciones dinámicas consecutivas | 5 ventanas | depende de ventana deslizante |
| 4 frames perdidos | 4 | **133 ms** |
| Captura dinámica mínima | 5 | **167 ms** |

La retención de `0,75 s` es una regla temporal del servicio de transcripción y no equivale a los 6 frames de quietud del clasificador.

---

# 5. Números de los modelos

## 5.1 Estático activo

| Capa | Forma o configuración |
|---|---|
| Entrada | `(21, 3)` |
| Dense 1 | `64` unidades, ReLU |
| BatchNorm | `64` características |
| Dense 2 | `64` unidades, ReLU |
| Flatten | `21 × 64 = 1.344` valores |
| Dense 3 | `256` unidades, ReLU |
| Dropout 1 | `0,4` |
| Dense 4 | `128` unidades, ReLU |
| Dropout 2 | `0,3` |
| Salida | `C` clases, Softmax |
| Épocas | **50** |
| Batch máximo | **8** |
| Validación | **20%** desde 10 muestras |
| Learning rate | **0,001** |
| Semilla de shuffle | **42** |

### Conteo para 22 clases

```text
Dense(64):       256
BatchNorm:       128 parámetros entrenables
Dense(64):       4.160
Dense(256):      344.320
Dense(128):      32.896
Salida 22:       2.838
Total:           384.598 parámetros entrenables aprox.
```

## 5.2 Dinámico activo

| Capa | Forma o configuración |
|---|---|
| Entrada | `(20, 21, 3)` |
| TimeDistributed Dense | `64` unidades, ReLU |
| TimeDistributed Flatten | `21 × 64 = 1.344` por frame |
| BiLSTM 1 | `64` por dirección, `return_sequences=True` |
| Dropout 1 | `0,3` |
| BiLSTM 2 | `32` por dirección |
| Dense | `64` unidades, ReLU |
| Dropout 2 | `0,3` |
| Salida | `C` clases, Softmax |
| Épocas | **80** |
| Batch máximo | **8** |
| Validación | **20%** desde 10 secuencias |
| Learning rate | **0,001** |
| Negativos | entre **6** y **60** |
| Factor de negativos | **1,2 ×** máxima clase |
| Semilla de negativos | **7** |

### Conteo para 22 clases

```text
TimeDistributed Dense: 256
BiLSTM(64) bidireccional: 721.408
BiLSTM(32) bidireccional: 41.216
Dense(64): 4.160
Salida 22: 1.430
Total: 768.470 parámetros entrenables aprox.
```

Con `NO_SENA` como clase 23, la salida agrega **65** parámetros y el total aproximado es **768.535**.

---

# 6. Números de entrenamiento y negativos

| Parámetro | Valor |
|---|---:|
| Épocas estáticas | **50** |
| Épocas dinámicas | **80** |
| Batch máximo | **8** |
| Split de validación | **0,2** |
| Mínimo de muestras para activar validación | **10** |
| Mínimo de clases para entrenar | **2** |
| Adam learning rate | **0,001** |
| Semilla shuffle estático | **42** |
| Semilla generación negativos | **7** |
| Negativos mínimos | **6** |
| Negativos máximos | **60** |
| Factor `max_class_count` | **1,2** |
| Tipos de negativos | **3** |

Tipos de negativos:

1. Hold/quietud.
2. Transition/transición entre poses.
3. Drift/deriva aleatoria.

## Script de integración `_test_deteccion.py`

| Parámetro de prueba | Valor |
|---|---:|
| Semilla RNG global | **42** |
| FPS simulado | **30,0** |
| Frames quietos para cada pose sintética | **40** |
| Frames de transición estática | **12** |
| Frames de secuencia dinámica | **20** |
| Muestras de círculo sintético | **10** |
| Muestras de línea sintética | **10** |
| Muestras de deriva | **10** |
| Repeticiones de cada par de transición | **2** |
| Pares base de transición | **5** |
| Épocas del baseline de comparación | **80** |
| Batch del baseline | **8** |

Este script genera datos sintéticos para integración; no equivale a una evaluación oficial con usuarios reales.

---

# 7. Números de transcripción

| Constante | Valor |
|---|---:|
| Tamaño máximo del léxico `wordfreq` | **50.000** palabras |
| Frecuencia Zipf mínima | **3,0** |
| Largo máximo de palabra | **24** caracteres |
| Penalización de inserción de palabra | **3,0** |
| Costo de carácter desconocido | **2,2** |
| Distancia máxima de fuzzy repair | **1** |
| Largo mínimo para fuzzy repair | **5** caracteres |
| Retención mínima de token | **0,75 s** |
| Cooldown del mismo token | **0,9 s** |
| Idiomas disponibles | **6** |
| Códigos de idioma | `es`, `en`, `pt`, `fr`, `it`, `de` |
| Idioma persistido actual | **1**: `es` |

### Letras individuales permitidas por idioma

| Idioma | Letras de un solo carácter aceptadas | Cantidad |
|---|---|---:|
| Español | A, E, O, U, Y | **5** |
| Inglés | A, I | **2** |
| Portugués | A, E, O | **3** |
| Francés | A, Y | **2** |
| Italiano | A, E, O | **3** |
| Alemán | ninguna | **0** |
| **Total de entradas** |  | **15** |

### Tokens de control base

| Token | Acción |
|---|---|
| `ESPACIO` / `SPACE` | agrega espacio |
| `BORRAR` / `DELETE` | elimina último token |
| `LIMPIAR` / `CLEAR` | limpia toda la transcripción |

### Memoria actual

| Clave | Frase |
|---|---|
| `HOLASOYSANTI` | Hola, soy Santi. |
| `TENGOHAMBRE` | Tengo hambre. |

---

# 8. Números de voz y extensiones

| Parámetro | Valor |
|---|---:|
| Extensiones incluidas | **1** |
| Extensión incluida | `voz` |
| Pausa automática | **2,0 s** |
| Velocidad pyttsx3 | **165** |
| Volumen pyttsx3 | **1,0** |
| Hilos principales de voz | **1** worker daemon |
| Timer automático | **1** timer por cambio |
| Atajo reservado para corregir | `C` |
| Atajo reservado en entrenamiento | `T` |
| Atajo reservado para terminar entrenamiento | `Q` |
| Atajo de voz | `V` |
| Métodos opcionales de extensión documentados | **4** |

Métodos:

1. `setup(context)`.
2. `translate_actions(screen)`.
3. `transcription_changed(state)`.
4. `shutdown()`.

---

# 9. Números de interfaz

| Elemento | Valor |
|---|---:|
| Tamaño inicial de ventana | **980 × 680 px** |
| Altura mínima video traducción | **430 px** |
| Altura mínima cámara entrenamiento | **420 px** |
| Ancho mínimo diálogo corrección | **520 px** |
| Alto mínimo texto de corrección | **90 px** |
| Alto mínimo preview landmarks | **260 px** |
| Tamaño fuente título | **34 px** |
| Tamaño fuente subtítulo | **17 px** |
| Tamaño fuente cuerpo | **17 px** |
| Tamaño fuente traducción | **44 px** |
| Alto mínimo botón | **42 px** |
| Radio visual de botones | **10 px** |
| Radio visual de traducción | **14 px** |
| Radio visual cámara | **12 px** |

---

# 10. Números de archivos y estructura

## Carpetas activas principales

| Carpeta | Rol |
|---|---|
| `app/` | interfaz |
| `services/` | lógica de negocio |
| `core/` | visión y datos |
| `ml/` | machine learning |
| `data/` | artefactos generados |
| `extensions/` | plugins |
| `visualization/` | previews |
| `utils/` | configuración y logging |
| `docs/` | documentación |
| `archive/` | código histórico |

## Archivos clave

| Grupo | Cantidad/archivos |
|---|---|
| Pantallas principales | **7** dentro de `MainWindow` |
| Widgets reutilizables | **2** principales |
| Servicios importados en `AppContext` | **8** clases explícitas |
| Clasificadores activos | **2** tipos |
| Modelos presentes hoy | **1** entrenado utilizable |
| Datasets | **2**: estático y dinámico |
| Labels | **2**: estático y dinámico |
| Documentos base previos | **3**: manual, conceptos, anexo |
| Documentos de defensa agregados | **4** contando presentación, guía, anexo avanzado e índice |
| Extensiones incluidas | **1** |
| Código legado archivado | **3** grupos: API, consola y ML extras |

## Precisión temporal de persistencia

- Timestamps de sesiones y registro: ISO con precisión de **segundos** mediante `timespec="seconds"`.
- Identificador de sesión: fecha/hora hasta segundos más **8** caracteres hexadecimales de UUID.
- ID de colisión de una seña: sufijo UUID de **8** caracteres hexadecimales.

---

# 11. Parámetros históricos que no debes mezclar

| Archivo | Parámetro | Valor | Estado |
|---|---|---:|---|
| `utils/config.py` | FPS cámara | **15** | histórico/auxiliar |
| `utils/config.py` | tiempo máximo por frame | **66 ms** | histórico/auxiliar |
| `utils/config.py` | manos máximas | **2** | histórico/auxiliar |
| `utils/config.py` | threshold TensorFlow | **0,7** | general, no controla toda la UI |
| `utils/config.py` | clases configuradas | **8** | antiguo, no coincide con 22 labels |
| `utils/config.py` | buffer | **10** | coincide con traducción activa |
| `utils/config.py` | consensus threshold | **0,8** | antiguo; pantalla usa 0,7 |
| `utils/config.py` | stabilization frames | **5** | auxiliar |
| `utils/config.py` | hold time | **1.000 ms** | auxiliar |
| `utils/config.py` | transition delay | **500 ms** | auxiliar |
| `utils/config.py` | máximo de señas por secuencia | **20** | auxiliar |

**Regla de defensa:** cuando pregunten por el comportamiento de la aplicación actual, citar `TranslateScreen`, `TeachSignScreen`, `TrainingService` y los JSON de `data/`. Cuando pregunten por configurabilidad o evolución, mencionar `utils/config.py` como configuración histórica/centralizable.

---

# 12. Números que conviene decir de memoria

```text
21 landmarks
3 coordenadas por landmark
63 valores por pose
20 frames por secuencia dinámica
10 predicciones en buffer estático
5 predicciones en buffer dinámico
6 frames quietos para compuertas
4 frames perdidos tolerados
0,65 de suavizado
0,7 de consenso UI
0,5 para unknown del clasificador
0,75 s de retención
0,9 s de cooldown
640×480 a 30 FPS
50 épocas estáticas
80 épocas dinámicas
batch máximo 8
20% de validación desde 10 muestras
50.000 palabras de léxico
6 idiomas
2 segundos para voz automática
22 labels actuales
632 muestras estáticas
10 clases con datos
0 secuencias dinámicas actuales
1 modelo estático presente
0 modelos dinámicos presentes
```

## Respuesta numérica modelo

> La pose estática entra como `(21, 3)`, es decir, 63 coordenadas. La secuencia dinámica usa `(20, 21, 3)`, 1.260 valores por muestra. La red estática tiene aproximadamente 384.598 parámetros para 22 clases y la dinámica aproximadamente 768.470. El entrenamiento usa 50 épocas para estático, 80 para dinámico, batch máximo 8 y validación del 20% desde 10 muestras.

---

# 13. Checklist antes de exponer

- [ ] Abrir `presentacion_tls.md`.
- [ ] Abrir `guia_defensa_tls.md`.
- [ ] Abrir `anexo_tecnico_avanzado_tls.md`.
- [ ] Tener visible `infografia_tls.svg`.
- [ ] Recordar que el snapshot tiene **632** muestras estáticas.
- [ ] Recordar que hay **22** labels, pero solo **10** clases con muestras.
- [ ] No afirmar que el modelo dinámico está listo: hay **0** secuencias y **0** modelos dinámicos presentes.
- [ ] Explicar la diferencia entre confianza y accuracy.
- [ ] Explicar la diferencia entre memoria de frases y reentrenamiento visual.
- [ ] Mencionar la limitación de normalización dinámica en captura.
- [ ] Ejecutar `py desktop_app.py` desde la raíz.
- [ ] Usar iluminación buena, fondo simple y una mano.
- [ ] No activar `Resetear todo` durante la demo.

---

# 14. Fuentes de cada número

- Cámara, buffers, tracking y thresholds: `app/screens/translate_screen.py`.
- Captura y remuestreo: `app/screens/teach_sign_screen.py`.
- Hiperparámetros y modelos activos: `services/training_service.py`.
- Preprocesamiento: `core/image_processor.py` y `core/preprocessing.py`.
- Detección: `core/hand_detector.py`.
- Transcripción: `services/transcription_service.py`.
- Voz: `extensions/voz/extension.py`.
- Estado del dataset: `data/datasets/dataset_static.json`, `data/datasets/dataset_dynamic.json`, `data/signs/signs.json`.
- Parámetros históricos: `utils/config.py`.
- Pruebas sintéticas: `_test_deteccion.py`.

> Si un número no aparece en este índice, no lo presentes como dato oficial sin verificar primero el archivo fuente correspondiente.
