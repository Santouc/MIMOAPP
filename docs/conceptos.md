# Conceptos del sistema: reconocimiento de señas con 21 landmarks

Este documento explica, de forma conceptual y sin necesidad de leer el código, cómo funciona el proyecto: qué es un landmark, en qué se diferencian los modos estático y dinámico, cómo fluye la información desde la cámara hasta la predicción, y cuáles son las limitaciones y buenas prácticas.

## 1) Visión general
El sistema reconoce letras o gestos de mano a partir de la imagen de la cámara. Para ello:
- Detecta la mano en cada cuadro (frame) del video.
- Extrae 21 puntos clave (landmarks) de la mano, con coordenadas 3D normalizadas.
- Preprocesa y estabiliza esos puntos.
- Pasa los datos por uno o dos modelos de aprendizaje automático:
  - Modelo **estático** (predice por cuadro).
  - Modelo **dinámico** (predice a partir de una secuencia de cuadros en el tiempo).

## 2) Componentes principales
- **Detección y tracking de mano**: Se usa un modelo de visión (p. ej. MediaPipe Hands) que localiza la mano y calcula los 21 landmarks por frame.
- **Landmarks (21 puntos)**: Son posiciones anatómicas de la mano: muñeca y articulaciones/puntas de los dedos. Cada landmark incluye `(x, y, z)`:
  - `x` y `y` están normalizados respecto al tamaño de la imagen (entre 0 y 1 aprox.).
  - `z` es una profundidad relativa (negativa suele indicar cercanía a la cámara); se usa de forma cualitativa.
- **Preprocesamiento**:
  - Verificación de que existan exactamente 21 landmarks válidos.
  - Suavizado (smoothing) para reducir el ruido cuadro a cuadro.
  - Tolerancia a cuadros perdidos (si un frame no detecta, puede reusar el último válido por un corto intervalo).
- **Clasificación**:
  - **Estática**: toma un solo frame (21×3 valores) y produce una etiqueta con probabilidad. Es ideal para letras o poses fijas.
  - **Dinámica**: toma una **secuencia** de frames (p. ej., 20) y modela el movimiento en el tiempo (p. ej., con capas recurrentes tipo BiLSTM). Es ideal para gestos o letras con trazo.

## 3) Los 21 landmarks de la mano
Numeración y significado (referencia común en visión por computadora):

| ID | Punto |
|----|-------|
| 0  | Muñeca (Wrist) |
| 1  | Pulgar CMC |
| 2  | Pulgar MCP |
| 3  | Pulgar IP |
| 4  | Punta del pulgar |
| 5  | Índice MCP |
| 6  | Índice PIP |
| 7  | Índice DIP |
| 8  | Punta del índice |
| 9  | Medio MCP |
| 10 | Medio PIP |
| 11 | Medio DIP |
| 12 | Punta del medio |
| 13 | Anular MCP |
| 14 | Anular PIP |
| 15 | Anular DIP |
| 16 | Punta del anular |
| 17 | Meñique MCP |
| 18 | Meñique PIP |
| 19 | Meñique DIP |
| 20 | Punta del meñique |

Notas:
- MCP/PIP/DIP son articulaciones (metacarpo-falángica, interfalángica proximal y distal).
- Los nombres ayudan a entender qué parte de la mano cambia entre letras/gestos.

## 4) Modos de operación
- **Modo estático**
  - Predice por cuadro, sin mirar el movimiento. Es rápido y suficiente para letras “fijas”.
  - Requiere que la forma de la mano sea distintiva en un instante.
  - Ventaja: baja latencia, simple de entrenar.
  - Limitación: no distingue gestos que dependen del movimiento.

- **Modo dinámico**
  - Observa una ventana temporal (p. ej., 20 frames). Aprende patrones de cambio en el tiempo.
  - Ventaja: reconoce gestos/“trazos” y desambiguaciones por movimiento.
  - Limitación: un poco más de latencia (hay que acumular frames) y más datos para entrenar.

## 5) Flujo de datos (de la cámara a la predicción)
1. La cámara entrega frames (p. ej., 30 FPS, 320×240 para mejor rendimiento).
2. El detector calcula los 21 landmarks por frame.
3. Se aplica estabilización para reducir saltos (suavizado exponencial) y se toleran algunos frames perdidos antes de declarar "tracking perdido".
4. Para el modelo estático:
   - Se forma un vector 21×3 y se infiere una etiqueta por frame.
   - Puede usarse un pequeño buffer para estabilizar la etiqueta mostrada y evitar parpadeos.
5. Para el modelo dinámico:
   - Se mantiene una cola (ventana) de N frames (p. ej., 20) y se infiere cuando hay suficientes.

## 6) Datos y entrenamiento (conceptual)
- Los ejemplos capturados se guardan en archivos JSON, separando estático y dinámico.
- El entrenamiento aprende un mapeo de landmarks → etiqueta.
- Si agregas una etiqueta nueva o cambias significativamente la definición de una existente, debes **reentrenar** el modelo correspondiente (estático y/o dinámico).
- Balance y diversidad del dataset (ángulos, distancias, iluminación) mejoran la generalización.

## 7) Indicadores y estados del sistema
- **Tracking:** indica si la mano está bien detectada (OK), recuperándose o perdida.
- **Buffers:** reducen parpadeos en las predicciones.
- **Mensajes de captura:** confirman guardados de muestras/secuencias y conteos del dataset.

## 8) Limitaciones habituales
- **Oclusiones:** si parte de la mano no es visible, los 21 landmarks pueden ser imprecisos o incompletos.
- **Iluminación:** condiciones pobres degradan la detección.
- **Variabilidad entre usuarios:** tamaños de mano, forma de los dedos y postura; mitígalo con más datos.
- **Una mano a la vez:** el sistema asume una mano prioritaria.
- **Cambios de fondo/cámara:** pueden afectar la estabilidad si son extremos.

## 9) Rendimiento y recursos (orientativo)
- Ejecución en vivo: ~700–1,200 MB de RAM según hardware y dependencias.
- Entrenamiento: puede requerir varios GB de RAM; depende del tamaño del dataset.
- Resolución y FPS influyen en la fluidez y la carga de cómputo.

## 10) Extensibilidad
- Agregar letras/gestos: captura nuevas muestras y reentrena.
- Ajustar etiquetas visibles: se puede cambiar el archivo de etiquetas para los nombres mostrados, pero si cambias el conjunto de clases debes reentrenar.
- Migración de modelos: es posible explorar formatos más ligeros (p. ej., TFLite/ONNX) en fases futuras.

## 11) Glosario rápido
- **Landmark:** punto anatómico clave de la mano con coordenadas (x,y,z).
- **Frame:** imagen individual de un video.
- **Buffer:** memoria temporal de las últimas predicciones/frames para estabilizar.
- **Secuencia:** conjunto ordenado de frames usados por el modelo dinámico.

Referencias: MediaPipe Hands (Google) para detección/landmarks de mano.
