# MIMOAPP

*Software capaz de aprender **cualquier** lenguaje de señas mediante el reconocimiento
de la mano en 21 puntos a través de una cámara, y traducirlo a texto y voz en tiempo real.*

**Python** · **PySide6** · **MediaPipe** · **TensorFlow** · 100% local, sin APIs de pago

---

## Desarrollador Principal

| Nombre y Apellido | Usuario GitHub | Correo | LinkedIn |
| ----------------- | -------------- | ------ | ------- |
| Santiago Silva | @Santouc | ssilvap@usm.cl | [https://www.linkedin.com/in/santiago-silva-06b44a416](https://www.linkedin.com/in/santiago-silva-06b44a416/)/) |

> *"Nuestra motivación es facilitar la comunicación entre las personas sordomudas y el resto de la sociedad."*

---

## ¿Qué hace la app?

| Función | Descripción |
| ------- | ----------- |
| **Reconocimiento de señas** | Detecta la mano en 21 puntos con la cámara y clasifica señas **estáticas** y **dinámicas** |
| **Entrenamiento propio** | Cualquier usuario puede enseñarle señas nuevas: captura muestras, acepta/rechaza y el modelo se reentrena automáticamente |
| **Transcripción inteligente** | Convierte las letras deletreadas en frases con sentido usando un diccionario de ~50.000 palabras (segmentación probabilística, tildes automáticas) en 6 idiomas: español, inglés, portugués, francés, italiano y alemán |
| **Memoria que aprende** | Con la tecla `C` puedes corregir una interpretación y la app la recordará para siempre |
| **Voz automática** | Al detectar una pausa, la app dice la frase en voz alta (extensión `voz`, texto a voz local) |
| **Sistema de extensiones** | Funcionalidades opcionales que se activan/desactivan desde el menú, sin tocar el código principal |

---

## Objetivos

**Objetivo general**

> Crear un software capaz de aprender cualquier lenguaje de señas en base a un
> entrenamiento mediante fotogramas y la detección de la forma de la mano.

**Objetivos específicos**

- Detectar sin mayores problemas la forma y posición de la mano.
- Traducir señas a texto natural y voz para lograr una comunicación fluida.
- Permitir que el propio usuario amplíe el vocabulario de señas y las interpretaciones.

---

## Alcance del proyecto

**Dentro del alcance**: modelo capaz de aprender señas estáticas y dinámicas,
transcripción a frases naturales, voz local y sistema de extensiones.

**Fuera del alcance**: integración con Arduino Uno Q, por limitaciones de tiempo y hardware.

---

## Tecnologías utilizadas

| Tecnología | Uso |
| ---------- | --- |
| **Python 3.9 – 3.12** | Lenguaje principal |
| **PySide6 (Qt)** | Interfaz gráfica de escritorio |
| **MediaPipe** | Detección de la mano en 21 puntos |
| **TensorFlow / Keras** | Modelos de clasificación de señas |
| **OpenCV** | Captura y procesamiento de video |
| **wordfreq** | Diccionarios de frecuencias multilingües para la transcripción |
| **pyttsx3** | Texto a voz local (extensión `voz`) |
| **JSON** | Persistencia de datasets, reglas y memoria |

---

## Estructura del repositorio

```
T.L.S/
├── app/                                    # Aplicación de escritorio (PySide6)
│   ├── screens/                            # Pantallas de la interfaz
│   │   ├── __init__.py
│   │   ├── document_screen.py              # Visor de documentos (Créditos, Manual)
│   │   ├── extensions_screen.py            # Gestor de extensiones
│   │   ├── home_screen.py                  # Menú principal
│   │   ├── manage_signs_screen.py          # Gestión de señas
│   │   ├── teach_sign_screen.py            # Entrenamiento de señas
│   │   └── translate_screen.py             # Traducción en vivo
│   ├── widgets/                            # Componentes reutilizables
│   │   ├── __init__.py
│   │   ├── landmark_preview.py             # Vista previa de landmarks
│   │   └── transcription_correction_dialog.py  # Diálogo para enseñar frases
│   ├── __init__.py
│   ├── app_context.py                      # Contenedor de servicios compartidos
│   └── main_window.py                      # Ventana principal y navegación
├── archive/                                # Código legado (solo referencia)
│   ├── api/                                # API experimental archivada
│   │   ├── client_examples.py
│   │   ├── docker-compose.yml
│   │   ├── Dockerfile
│   │   ├── main.py
│   │   ├── README.md
│   │   └── requirements.txt
│   ├── legacy_console/                     # Versión antigua de consola
│   │   ├── main.py
│   │   └── teaching.py
│   └── ml_extras/                          # Utilidades ML no conectadas
│       ├── data_capture.py
│       ├── data_processor.py
│       ├── deployment_manager.py
│       ├── model_evaluator.py
│       └── realtime_optimizer.py
├── core/                                   # Visión por computadora
│   ├── dataset_utils.py                    # Utilidades para datasets
│   ├── hand_detector.py                    # Detección de manos (MediaPipe)
│   ├── image_processor.py                  # Procesamiento de imágenes
│   └── preprocessing.py                    # Preprocesamiento de landmarks
├── data/                                   # Datos generados por la app
│   ├── datasets/                           # Muestras de entrenamiento
│   ├── models/                             # Modelos entrenados y etiquetas
│   │   ├── hand_landmarker.task            # Modelo de landmarks (MediaPipe)
│   │   ├── labels.json                     # Etiquetas estáticas
│   │   └── labels_dynamic.json             # Etiquetas dinámicas
│   ├── pending_captures/                   # Capturas pendientes de revisión
│   ├── previews/                           # Vistas previas de señas
│   ├── signs/
│   │   └── signs.json                      # Registro de señas
│   ├── transcription/
│   │   ├── memory.json                     # Frases aprendidas por el usuario
│   │   └── rules.json                      # Reglas y correcciones de transcripción
│   └── extensions.json                     # Estado activado/desactivado de extensiones
├── docs/                                   # Documentación
│   ├── anexo_tecnico.md                    # Anexo técnico
│   ├── conceptos.md                        # Conceptos teóricos
│   └── manual_uso.md                       # Manual de uso
├── extensions/                             # Extensiones opcionales
│   ├── voz/
│   │   └── extension.py                    # Texto a voz automático
│   └── README.md                           # Guía para crear extensiones
├── ml/                                     # Machine Learning
│   ├── __init__.py
│   ├── clasificador.py                     # Clasificador de señas estáticas
│   ├── dynamic_classifier.py               # Clasificador de señas dinámicas
│   ├── train.py                            # Entrenamiento del modelo estático
│   └── train_dynamic.py                    # Entrenamiento del modelo dinámico
├── services/                               # Lógica de negocio
│   ├── __init__.py
│   ├── capture_service.py                  # Capturas de muestras
│   ├── document_service.py                 # Lectura de documentos Markdown
│   ├── extension_service.py                # Carga y gestión de extensiones
│   ├── library_service.py                  # Biblioteca de señas (alfabeto)
│   ├── path_service.py                     # Rutas del proyecto
│   ├── sign_service.py                     # Registro de señas
│   ├── training_service.py                 # Entrenamiento automático
│   └── transcription_service.py            # Letras → frases naturales
├── utils/                                  # Utilidades generales
│   ├── __init__.py
│   ├── config.py                           # Configuración del proyecto
│   └── logger.py                           # Sistema de logging
├── visualization/                          # Visualización de landmarks
│   ├── __init__.py
│   └── landmark_average.py                 # Promedio animado de capturas
├── .gitignore                              # Archivos ignorados por Git
├── desktop_app.py                          # Punto de entrada de la aplicación
├── README.md                               # Este archivo
└── requirements.txt                        # Dependencias del proyecto
```

---

## Instalación y uso

**1.** Clona el repositorio:

```powershell
git clone <URL-del-repositorio>
cd "T.L.S Beta"
```

**2.** Instala las dependencias (Python 3.9 – 3.12):

```powershell
py -m pip install -r requirements.txt
```

**3.** Ejecuta la aplicación:

```powershell
py desktop_app.py
```

### Flujo recomendado

1. **Gestionar señas** → agrega señas o el alfabeto occidental completo.
2. **Entrenamiento** → captura muestras; el modelo se reentrena solo.
3. **Traducir en vivo** → haz señas frente a la cámara y observa la frase.
4. Haz una pausa de ~2 segundos y la app **dirá la frase en voz alta**. 
5. ¿Interpretó mal? Presiona `C`, corrígela, y la recordará. 

> Más detalles en el **Manual de uso** dentro de la app o en `docs/manual_uso.md`.

---

## Extensiones

La app se puede ampliar sin modificar su código: basta con crear una carpeta en
`extensions/` con un archivo `extension.py`. Desde el menú **Extensiones** se
activan y desactivan con un clic.

| Extensión incluida | Descripción |
| ------------------ | ----------- |
| `voz` | Dice automáticamente la frase transcrita al detectar una pausa |

> Guía completa para desarrolladores en `extensions/README.md`.

---

## Bibliografía

Todas las librerías y herramientas utilizadas en el desarrollo del proyecto:

### Librerías principales

| Librería | Uso en el proyecto | Documentación |
| -------- | ------------------ | ------------- |
| **PySide6** (≥ 6.6.0) | Interfaz gráfica de escritorio (Qt) | [doc.qt.io/qtforpython-6](https://doc.qt.io/qtforpython-6/) |
| **MediaPipe** (≥ 0.10.0) | Detección de la mano en 21 puntos (hand landmarker) | [ai.google.dev/edge/mediapipe](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker) |
| **TensorFlow / Keras** (≥ 2.13.0) | Entrenamiento y ejecución de los clasificadores de señas | [tensorflow.org](https://www.tensorflow.org/guide/keras) |
| **OpenCV** (opencv-python ≥ 4.8.0) | Captura de cámara y procesamiento de video/imágenes | [docs.opencv.org](https://docs.opencv.org/) |
| **NumPy** (≥ 1.24.0) | Operaciones numéricas sobre landmarks y datasets | [numpy.org](https://numpy.org/doc/) |
| **wordfreq** (≥ 3.0.0) | Diccionarios de frecuencias multilingües (español, inglés y más) para la transcripción de frases | [pypi.org/project/wordfreq](https://pypi.org/project/wordfreq/) |
| **pyttsx3** (≥ 2.90) | Texto a voz local (extensión `voz`) | [pypi.org/project/pyttsx3](https://pypi.org/project/pyttsx3/) |

### Librerías de apoyo

| Librería | Uso en el proyecto | Documentación |
| -------- | ------------------ | ------------- |
| **SciPy** (≥ 1.10.0) | Procesamiento numérico y de señales | [scipy.org](https://docs.scipy.org/doc/scipy/) |
| **Matplotlib** (≥ 3.7.0) | Gráficos y visualización de datos | [matplotlib.org](https://matplotlib.org/stable/) |
| **Pillow** (≥ 10.0.0) | Manipulación de imágenes | [pillow.readthedocs.io](https://pillow.readthedocs.io/) |

### Herramientas de desarrollo

| Herramienta | Uso en el proyecto | Documentación |
| ----------- | ------------------ | ------------- |
| **pytest** (≥ 7.4.0) | Pruebas automatizadas | [docs.pytest.org](https://docs.pytest.org/) |
| **black** (≥ 23.0.0) | Formateo automático de código | [black.readthedocs.io](https://black.readthedocs.io/) |
| **Git / GitHub** | Control de versiones y trabajo colaborativo | [docs.github.com](https://docs.github.com/) |

### Librería estándar de Python

Módulos integrados de Python utilizados (no requieren instalación): `json`, `re`,
`threading`, `queue`, `pathlib`, `importlib`, `dataclasses`, `collections`, `time`,
`sys`, `os`, `functools` y `logging`.

---

## Notas adicionales

> El código legado de consola se conserva en `archive/legacy_console/` solo como
> referencia histórica. La aplicación oficial es `desktop_app.py`.
