# Guía de defensa técnica — T.L.S

Esta guía está pensada para responder preguntas del profesor/jurado sin exagerar el estado real del prototipo.

**Para profundizar:** [Anexo técnico avanzado](anexo_tecnico_avanzado_tls.md) · [Índice y números](indice_y_numeros_tls.md)

## 1. Respuesta inicial recomendada

> T.L.S es una aplicación de escritorio local que reconoce señas de mano mediante 21 landmarks de MediaPipe. El usuario puede registrar una seña, capturar ejemplos, revisarlos y reentrenar un clasificador TensorFlow. En traducción en vivo, la predicción pasa por estabilización temporal y luego por una capa de transcripción que transforma letras o tokens en texto natural. La voz está implementada como extensión local con pyttsx3. El snapshot actual tiene completo el flujo estático con 632 muestras y el modelo estático; el pipeline dinámico está implementado, pero necesita secuencias reales y `model_dynamic.h5` para demostrarse de punta a punta.

## 2. Preguntas de producto y propósito

### ¿Qué problema resuelve?

Reduce la barrera de comunicación entre una persona que usa señas y una persona que no las conoce, mostrando texto y opcionalmente voz.

### ¿Quién usa el sistema?

La persona que realiza las señas, su interlocutor y quien administra/enseña el vocabulario al sistema.

### ¿Cuál es la propuesta diferencial?

Es local, entrenable por el usuario, modular y separa reconocimiento de visión, transcripción y voz. No depende de una API de pago.

### ¿Por qué es una aplicación de escritorio?

Permite acceder directamente a la cámara, modelos y archivos locales con baja dependencia de red. PySide6 también permite construir una interfaz de eventos adecuada para captura continua.

### ¿Es un traductor universal?

No debe presentarse como universal en sentido lingüístico. El objetivo es que pueda aprender distintos vocabularios si se le entregan muestras, pero la salida actual trabaja principalmente con etiquetas, letras, reglas y un vocabulario configurable. No representa toda la gramática, expresión facial ni contexto de una lengua de señas.

### ¿Qué quedó fuera del alcance?

La integración con Arduino Uno Q por limitaciones de tiempo y hardware.

## 3. Preguntas de visión por computador

### ¿Qué hace MediaPipe?

Detecta la mano y entrega 21 landmarks por frame. Cada punto contiene `x`, `y` y `z` normalizados. El archivo `data/models/hand_landmarker.task` es el asset que usa la API MediaPipe Tasks.

### ¿Por qué 21 puntos?

Es la representación estándar de la topología de una mano: muñeca, articulaciones y puntas de los cinco dedos. Permite describir geometría sin alimentar al clasificador con todos los píxeles.

### ¿Por qué usar landmarks en vez de imágenes?

Los landmarks reducen dimensionalidad, costo y dependencia del fondo. También hacen posible visualizar qué información está usando el sistema. La desventaja es que si MediaPipe falla por oclusión o mala iluminación, el clasificador recibe información incompleta.

### ¿Qué hace OpenCV?

Captura frames con `VideoCapture`, los redimensiona a 640×480, los espeja horizontalmente, aplica blur, mejora contraste con CLAHE, normaliza iluminación y dibuja overlays.

### ¿Por qué se usa una mano?

Las pantallas activas crean `HandDetector(max_hands=1)`. Es una simplificación del prototipo para concentrarse en la forma y movimiento de una mano. Dos manos y postura corporal son una mejora futura.

### ¿Cómo se manejan frames perdidos?

La pantalla conserva el último conjunto válido durante pocos frames y marca `RECUPERANDO`. Si se supera el límite, marca `PERDIDO`, limpia buffers dinámicos y espera una detección nueva.

### ¿Cómo se reduce el parpadeo?

Se combina suavizado de landmarks, un buffer de predicciones, consenso mayoritario y umbrales de confianza. Para estáticas se requieren al menos 6 frames quietos; el consenso utiliza el historial disponible.

## 4. Preguntas de preprocesamiento

### ¿Qué normalización se aplica?

La normalización estática centra cada mano en la muñeca y escala por el tamaño de referencia asociado al punto 9. Así el modelo depende menos de la posición y distancia de la mano a la cámara.

### ¿Qué diferencia hay con la normalización dinámica?

La función `normalize_dynamic_sequence` pretende centrar la secuencia completa en la muñeca del primer frame y usar una sola escala basada en la mediana del tamaño de mano. De esa manera conserva el desplazamiento global.

### ¿Hay algún punto que debas reconocer si te preguntan con mucho detalle?

Sí. En la ruta actual de captura de `TeachSignScreen`, cada frame se normaliza con `normalize_single_hand` antes de guardarse. Eso puede eliminar parte de la trayectoria global que la normalización dinámica pretende conservar. La respuesta correcta es: *la arquitectura está preparada para trayectoria, pero esa ruta debe unificarse y validarse con datos reales antes de afirmar una demostración dinámica completa*.

### ¿Qué significa `(21, 3)`?

21 puntos, cada uno con 3 coordenadas. Para una muestra estática son 63 valores estructurados, no una imagen de 63 píxeles.

## 5. Preguntas de Machine Learning

### ¿Qué modelo estático usa?

El flujo activo en `services/training_service.py` usa entrada `(21, 3)`, capas Dense point-wise, BatchNormalization, Flatten, capas densas globales con Dropout y una salida Softmax con el número de clases.

### ¿Qué modelo dinámico usa?

Entrada `(20, 21, 3)`, procesamiento TimeDistributed, dos capas BiLSTM, Dropout, una capa Dense y Softmax. Las 20 posiciones representan la ventana temporal normalizada.

### ¿Por qué usar BiLSTM?

Una LSTM modela dependencia temporal. La variante bidireccional puede aprovechar el patrón completo de la ventana desde ambas direcciones durante la clasificación. La contrapartida es más memoria y latencia que una pose estática.

### ¿Qué optimizador y pérdida se usan?

Adam con learning rate 0,001 y `sparse_categorical_crossentropy`, adecuada cuando `y` contiene índices enteros de clase.

### ¿Cuántas épocas?

50 para estático y 80 para dinámico en el `TrainingService`. El batch se limita a 8 para facilitar datasets pequeños.

### ¿Cómo se valida?

Se usa `validation_split=0,2` cuando hay al menos 10 muestras. Eso es una validación interna del entrenamiento, no una evaluación independiente. Para reportar desempeño serio se necesita un conjunto de test separado por usuario/sesión.

### ¿Qué pasa si hay una sola clase?

El servicio no entrena y explica que hacen falta al menos dos señas distintas. Con una clase el modelo no puede aprender una frontera de clasificación útil.

### ¿Cómo se manejan clases sin ejemplos?

Pueden aparecer en el registro y en `labels`, pero no tienen muestras en `y`. El entrenamiento usa las clases presentes; la aplicación debe capturar datos antes de esperar que una clase sin ejemplos sea reconocida correctamente.

### ¿Qué es `NO_SENA`?

En el entrenamiento dinámico activo se generan secuencias negativas de quietud, transición y deriva. Se etiqueta `NO_SENA` para que el modelo aprenda a no convertir cualquier movimiento en una seña. Durante inferencia esa etiqueta se traduce a `unknown`.

### ¿Cuál es el umbral de confianza?

El clasificador devuelve `unknown` debajo de 0,5. La pantalla, además, exige que las predicciones del consenso superen 0,7. Son filtros distintos: uno pertenece al clasificador y el otro a la lógica temporal de la UI.

### ¿Por qué no afirmar 99% de exactitud?

Porque la exactitud final depende del conjunto de prueba y el snapshot no contiene una evaluación independiente reproducible como métrica oficial. La aplicación sí muestra accuracy de entrenamiento y validación al entrenar, pero eso no reemplaza un test separado.

### ¿Hay data augmentation?

El entrenamiento dinámico genera negativos sintéticos. El dataset estático no muestra una política completa de aumentos geométricos para todas las clases; la mejora natural es incorporar variaciones controladas y evaluar que no alteren el significado de la seña.

### ¿Qué diferencia hay entre `ml/train.py` y `services/training_service.py`?

`TrainingService` es la ruta usada por la aplicación cuando el usuario acepta capturas. `ml/train.py` y `ml/train_dynamic.py` son scripts de entrenamiento independientes. Si hay alguna diferencia entre ellos, para explicar el comportamiento de la app se debe priorizar `services/training_service.py`.

### ¿Qué diferencia técnica importante existe en el script dinámico?

El script independiente `ml/train_dynamic.py` contiene una normalización por frame distinta a la función dinámica del servicio y no representa necesariamente el camino activo de la UI. Por eso no se deben mezclar sus resultados con la ejecución de la aplicación.

## 6. Preguntas sobre el flujo de entrenamiento

### ¿Cómo se agrega una nueva seña?

En `Gestionar señas` se escribe su nombre y se elige si tendrá captura estática, dinámica o ambas. El `SignService` crea un ID normalizado, evita duplicados y exporta labels.

### ¿Cómo se captura una muestra estática?

Con la cámara activa, se presiona `T`. Se guarda el landmark normalizado de la mano actual. Se pueden capturar muchas poses en una sesión.

### ¿Cómo se captura una dinámica?

`T` inicia la grabación, cada frame válido se acumula, y `T` la detiene. Si hay menos de 5 frames se descarta. La secuencia se remuestrea a 20 frames antes de quedar pendiente.

### ¿Por qué hay una etapa pendiente?

Para que el usuario revise un promedio visual o una animación y pueda aceptar/rechazar antes de alterar el dataset oficial. Es un control humano de calidad.

### ¿Qué ocurre al aceptar?

`CaptureService` agrega `X`, `y` y labels al JSON correspondiente, actualiza conteos del registro, elimina la sesión pendiente y llama al entrenamiento del tipo seleccionado.

### ¿Qué ocurre al eliminar una seña?

Se elimina del registro, datasets, capturas pendientes y labels; se reindexan las clases restantes, se invalidan modelos antiguos y se reentrenan los modelos restantes desde la pantalla de gestión.

### ¿Qué hace `Resetear todo`?

Borra los datos generados por la app —registro, datasets, labels, modelos, pendientes y previews—, pero conserva `hand_landmarker.task`. Tiene confirmación visual y exige escribir `RESET`.

## 7. Preguntas sobre transcripción

### ¿El modelo reconoce palabras directamente?

La clasificación produce una etiqueta de seña. La transcripción posterior puede interpretar letras compactas, tokens de control o palabras registradas. No es la misma tarea que clasificar directamente una frase.

### ¿Cómo convierte `HOLASOYSANTI` en una frase?

Primero busca memoria exacta; luego aplica correcciones y mapas; después segmenta la cadena con un vocabulario ponderado por frecuencia; intenta reparar errores de una edición y finalmente formatea la frase.

### ¿Qué algoritmo usa para segmentar?

Programación dinámica sobre candidatos del vocabulario. Cada segmento obtiene un puntaje usando frecuencia de palabra, penalización por inserción y costo para caracteres desconocidos.

### ¿Qué aporta `wordfreq`?

Un listado de palabras frecuentes por idioma y una frecuencia Zipf que ayuda a preferir palabras más probables. El código carga hasta 50.000 palabras y filtra por longitud y frecuencia mínima.

### ¿Cómo se agregan tildes?

Los `word_map` y las palabras mostradas por el vocabulario conservan la forma escrita, mientras que las claves internas se normalizan sin tildes para comparar.

### ¿Qué idiomas hay?

Español, inglés, portugués, francés, italiano y alemán. El idioma se guarda en `data/transcription/rules.json`; el snapshot actual está configurado en español.

### ¿La memoria reentrena TensorFlow?

No. La memoria solo aprende la interpretación de texto y se guarda en `memory.json`. Es más rápida y segura para corregir frases sin cambiar el clasificador visual.

## 8. Preguntas sobre extensiones

### ¿Cómo funciona la extensión de voz?

`ExtensionService` descubre `extensions/voz/extension.py`, importa la clase, ejecuta `setup(context)` y notifica a la extensión cuando cambia la transcripción. `voz` inicia un temporizador de aproximadamente 2 segundos y usa `pyttsx3` en un hilo para hablar sin bloquear la UI.

### ¿Qué pasa si la extensión falla?

El error se registra en `context.extensions.errors` y la aplicación principal continúa. La pantalla de extensiones muestra el estado.

### ¿Por qué usar plugins?

Para desacoplar funcionalidades opcionales, permitir activarlas/desactivarlas y evitar modificar el núcleo cada vez que se incorpora una nueva salida.

### ¿La voz necesita internet?

No. `pyttsx3` utiliza motores instalados localmente. Si no existe el motor o la dependencia, la función se informa como no disponible.

## 9. Preguntas sobre estado actual y demo

### ¿Cuántas señas hay hoy?

Hay 22 registros/labels en el snapshot revisado. Son: `A, B, C, D, E, F, H, I, K, L, M, N, O, P, Q, R, T, U, V, W, Y, Z`.

### ¿Cuántas tienen muestras?

10 clases tienen muestras estáticas: `A, B, C, D, E, F, H, I, L, O`. El total es 632. Las otras 12 están registradas pero tienen conteo cero.

### ¿El alfabeto completo está listo?

La interfaz tiene una función para importar A-Z completo, pero el snapshot actual tiene 22 labels. Faltan `G, J, S, X` y, además, no todas las labels existentes tienen muestras.

### ¿El modelo dinámico está listo?

No en el snapshot revisado. El código y la pantalla soportan capturas/entrenamiento dinámico, pero `dataset_dynamic.json` indica 0 muestras y no existe `model_dynamic.h5`. Para la demo inmediata se debe usar el flujo estático o preparar antes datos dinámicos reales.

### ¿La app puede iniciar sin modelo?

Sí, la pantalla muestra el estado de modelos disponibles. Sin modelo, el clasificador devuelve `unknown` y no se genera una predicción útil.

### ¿Qué pasa si la cámara falla?

Se muestra un mensaje de cámara no disponible y se libera el recurso. La demo debe tener buena iluminación, cámara funcional y una alternativa grabada.

## 10. Preguntas de arquitectura y calidad

### ¿Por qué `AppContext`?

Centraliza dependencias y evita que cada pantalla cree sus propios servicios o rutas. También facilita pruebas al poder inyectar un contexto.

### ¿Qué patrón de navegación se usa?

Una ventana principal con `QStackedWidget`, señales Qt y métodos para cambiar la pantalla activa.

### ¿Dónde está la lógica de negocio?

En `services/`: registro, capturas, entrenamiento, transcripción, rutas, documentos, librerías y extensiones. La UI coordina eventos y presenta resultados.

### ¿Cómo se conserva la consistencia entre labels y datasets?

`SignService` exporta labels y `CaptureService` guarda el índice de la clase. Al eliminar una seña se reindexan `y` y se reescriben labels.

### ¿Por qué JSON y no una base de datos?

Para el prototipo ofrece transparencia y cero configuración. El costo es que archivos grandes son menos eficientes y no hay transacciones sofisticadas; SQLite sería una evolución razonable.

### ¿Cómo se registra un error?

Los clasificadores usan el logger centralizado; la UI comunica errores críticos con `QMessageBox`; el servicio de extensiones conserva errores por carpeta.

### ¿Hay pruebas?

Existe `_test_deteccion.py`, un script de integración que ejercita el pipeline estático, la lógica temporal y un escenario sintético de trayectoria. Antes de presentarlo como evidencia, hay que ejecutarlo y registrar sus salidas; por sí solo no equivale a una suite de métricas de producción.

### ¿Qué prueba faltaría?

Una evaluación con train/validation/test separado, usuarios distintos, clases balanceadas, métricas por clase, pruebas de cámara y un conjunto dinámico real capturado desde la UI.

## 11. Preguntas difíciles y respuestas honestas

### ¿Entonces el sistema traduce lenguaje de señas o solo deletrea?

En el estado actual, el flujo demostrado se basa principalmente en reconocer señas etiquetadas —incluidas letras— y luego transcribir tokens. Es correcto describirlo como un prototipo de reconocimiento y transcripción de señas, no como un traductor lingüístico completo.

### ¿Por qué la documentación dice que reconoce dinámicas si no hay modelo?

Porque la capacidad está implementada en código: captura de secuencias, normalización, modelo temporal, negativos y lógica de selección. La diferencia es que el artefacto de datos actual no contiene secuencias, por lo que la funcionalidad aún no está lista para demostrarse con el snapshot sin una etapa adicional de captura y entrenamiento.

### ¿Qué riesgo tiene entrenar con pocas muestras?

Sobreajuste y mala generalización a otras personas, ángulos o iluminaciones. La accuracy de entrenamiento puede ser alta sin que el modelo funcione fuera de las muestras.

### ¿Qué harías primero para mejorar el proyecto?

Unificaría la captura y normalización dinámica, generaría un dataset real balanceado, separaría test por usuario y reportaría métricas. Después optimizaría persistencia y ejecución del entrenamiento en segundo plano.

### ¿Por qué no usar un modelo grande preentrenado?

El objetivo del prototipo es que el usuario pueda enseñar su vocabulario localmente. Un modelo grande exigiría más datos, recursos y posiblemente un servicio externo; la representación de landmarks permite comenzar con un modelo pequeño y auditable.

### ¿Qué seguridad o privacidad tiene?

La cámara, landmarks, modelos, reglas y voz se procesan localmente. Eso reduce exposición de imágenes, aunque el usuario debe proteger los archivos locales porque contienen datos y modelos del proyecto.

## 12. Frases útiles durante la exposición

- “Diferencio el alcance diseñado de los artefactos presentes en este snapshot.”
- “La confidence de una predicción no es una métrica global de exactitud.”
- “La UI usa un consenso temporal para convertir una predicción ruidosa en un token estable.”
- “La memoria de frases y el entrenamiento visual son dos mecanismos distintos.”
- “La arquitectura dinámica está implementada; la evidencia reproducible requiere un modelo y un dataset dinámicos.”
- “La principal mejora metodológica es evaluar con datos separados por usuario y sesión.”

## 13. Cosas que no conviene decir

- “Reconoce cualquier lengua de señas sin entrenamiento adicional.”
- “Tiene 100% de precisión.”
- “El modelo dinámico está funcionando en el estado actual” si no se generó `model_dynamic.h5`.
- “Las 26 letras tienen datos” cuando hay 22 labels y 10 clases con muestras.
- “La transcripción entiende toda la gramática” cuando usa reglas, vocabulario y segmentación de tokens.
- “La confianza 0,9 significa 90% de exactitud real.”

## 14. Cierre de defensa

> T.L.S demuestra un flujo completo y extensible: captura, representación geométrica, clasificación, transcripción, aprendizaje de correcciones y salida de voz. Su fortaleza principal es que el usuario controla el vocabulario y todo funciona localmente. La limitación que reconocemos es que la calidad final depende de los datos; por eso la siguiente etapa debe consolidar el pipeline dinámico y evaluar el sistema con un conjunto de prueba independiente.
