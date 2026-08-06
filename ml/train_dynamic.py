#!/usr/bin/env python3
"""
Entrenamiento de modelo temporal para señas dinámicas.

Este script entrena el modelo que reconoce señas CON MOVIMIENTO (por ejemplo,
letras como J o Z, o palabras que requieren desplazamiento de la mano). El
flujo completo es:

1. Cargar el dataset dinámico (dataset_dynamic.json) con secuencias de
   landmarks capturadas desde la app de escritorio: forma (N, T, 21, 3),
   donde N es el número de muestras y T la cantidad de frames por secuencia.
2. Validar el dataset (formas, consistencia de muestras, mínimo de 2 clases)
   y compactar los índices de clase a un rango contiguo 0..K-1.
3. Normalizar cada frame de cada secuencia con el preprocesado centralizado.
4. Construir una red neuronal temporal Keras (capas TimeDistributed + LSTM
   bidireccionales) y entrenarla.
5. Guardar los artefactos resultantes: model_dynamic.h5 y labels_dynamic.json,
   que luego consume ml/dynamic_classifier.py durante la inferencia.

Se ejecuta directamente desde consola: py ml/train_dynamic.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

# Raíz del proyecto (carpeta padre de ml/), necesaria para resolver rutas
# de datos y para poder importar los paquetes core/ y utils/
BASE_DIR = Path(__file__).resolve().parent.parent

# Agregar la raíz del proyecto al path de Python si aún no está,
# de modo que el script funcione al ejecutarse directamente desde consola
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from core.preprocessing import normalize_landmarks
from utils.logger import get_logger

logger = get_logger(__name__)


# Rutas de entrada (dataset grabado desde la app) y de salida (artefactos
# del entrenamiento que consumirá el clasificador dinámico)
DATASET_PATH = BASE_DIR / "data" / "datasets" / "dataset_dynamic.json"
MODEL_PATH = BASE_DIR / "data" / "models" / "model_dynamic.h5"
LABELS_PATH = BASE_DIR / "data" / "models" / "labels_dynamic.json"


def load_dynamic_dataset(filepath: Path):
    """
    Carga y valida el dataset de señas dinámicas desde un archivo JSON.

    Además de leer los datos, realiza validaciones críticas antes de entrenar:
    - Forma esperada de X: (N, T, 21, 3) — N secuencias de T frames.
    - Misma cantidad de muestras en X e y.
    - Presencia de al menos 2 clases distintas (imprescindible para clasificar).

    También "compacta" las etiquetas: si el dataset tiene clases sin muestras,
    se remapean los índices a un rango contiguo 0..K-1 para que coincidan con
    las neuronas de salida del modelo.

    Args:
        filepath: Ruta al archivo dataset_dynamic.json.

    Returns:
        Tupla (X, y, compact_labels) con las secuencias, los índices de clase
        remapeados y la lista de nombres de las clases realmente usadas.

    Raises:
        FileNotFoundError: Si el dataset no existe.
        ValueError: Si la forma es inválida, hay inconsistencias o menos de 2 clases.
    """
    # Verificar que el dataset exista antes de intentar leerlo
    if not filepath.exists():
        raise FileNotFoundError(
            "No se encontró dataset_dynamic.json. "
            "Graba señas dinámicas desde la app de escritorio antes de entrenar."
        )

    # Leer el JSON del dataset con encoding UTF-8
    with open(filepath, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    # Convertir a arrays NumPy: X = secuencias, y = índices de clase
    X = np.array(dataset.get("X", []), dtype=np.float32)
    y = np.array(dataset.get("y", []), dtype=np.int32)
    labels = dataset.get("labels", [])

    # Validar que cada muestra sea una secuencia de frames de 21 landmarks 3D
    if X.ndim != 4 or X.shape[2:] != (21, 3):
        raise ValueError(f"Shape inválido de dataset dinámico: {X.shape}, esperado (N, T, 21, 3)")

    # Cada secuencia debe tener su etiqueta correspondiente
    if len(X) != len(y):
        raise ValueError(f"Cantidad inconsistente de muestras: X={len(X)}, y={len(y)}")

    # Exigir al menos 2 clases distintas: con una sola clase no hay clasificación posible
    unique_classes = np.unique(y)
    if len(unique_classes) < 2:
        class_name = labels[int(unique_classes[0])] if len(unique_classes) == 1 and int(unique_classes[0]) < len(labels) else "desconocida"
        raise ValueError(
            f"El dataset dinámico solo contiene una clase ({class_name}). "
            "Captura al menos 2 señas dinámicas distintas antes de entrenar."
        )
    
    # Compactar etiquetas: conservar solo las clases con muestras reales y
    # remapear sus índices a un rango contiguo 0..K-1 (requisito de softmax)
    compact_labels = [labels[int(class_idx)] for class_idx in unique_classes]
    class_to_compact = {int(class_idx): compact_idx for compact_idx, class_idx in enumerate(unique_classes)}
    y = np.array([class_to_compact[int(class_idx)] for class_idx in y], dtype=np.int32)

    logger.info(f"Dataset dinámico cargado: {X.shape[0]} secuencias, {X.shape[1]} frames, {len(compact_labels)} clases usadas")
    logger.info(f"Clases dinámicas usadas: {compact_labels}")
    return X, y, compact_labels


def normalize_dynamic_sequences(X: np.ndarray) -> np.ndarray:
    """
    Normaliza todas las secuencias del dataset frame por frame.

    Estrategia: se "aplanan" las secuencias juntando todos los frames de todas
    las muestras en un solo lote (N*T, 21, 3), se normaliza cada frame con el
    mismo módulo centralizado que usa la inferencia (normalize_landmarks) y
    luego se restaura la forma original (N, T, 21, 3). Así se garantiza que
    entrenamiento e inferencia apliquen exactamente el mismo preprocesado.

    Args:
        X: Array de secuencias con forma (N, T, 21, 3).

    Returns:
        Array normalizado con la misma forma (N, T, 21, 3).
    """
    # Aplanar la dimensión temporal para normalizar todos los frames en lote
    samples, timesteps, points, coords = X.shape
    X_flat = X.reshape(samples * timesteps, points, coords)
    X_norm = normalize_landmarks(X_flat)
    # Restaurar la estructura de secuencias original
    return X_norm.reshape(samples, timesteps, points, coords)


def create_dynamic_model(sequence_length: int, num_classes: int) -> tf.keras.Model:
    """
    Construye y compila la red neuronal temporal para señas dinámicas.

    Arquitectura (de entrada a salida):
    - Input (T, 21, 3): secuencia de T frames con 21 landmarks 3D cada uno.
    - TimeDistributed(Dense 64 + Flatten): extrae características espaciales
      de cada frame por separado, produciendo un vector por instante de tiempo.
    - Bidirectional LSTM 64 (return_sequences) + Dropout 0.3: primera capa
      recurrente que modela la evolución temporal en ambos sentidos.
    - Bidirectional LSTM 32: segunda capa recurrente que condensa toda la
      secuencia en un único vector resumen.
    - Dense 64 (ReLU) + Dropout 0.3: capa densa final con regularización.
    - Dense softmax: distribución de probabilidad sobre las clases.

    Se compila con el optimizador Adam (lr=0.001) y la pérdida
    sparse_categorical_crossentropy (etiquetas como índices enteros).

    Args:
        sequence_length: Cantidad de frames T de cada secuencia de entrada.
        num_classes: Número de señas dinámicas a clasificar.

    Returns:
        Modelo Keras compilado y listo para entrenar.
    """
    # Entrada: secuencia completa de landmarks (T frames de 21 puntos 3D)
    inputs = tf.keras.Input(shape=(sequence_length, 21, 3))

    # Extracción de características espaciales frame a frame (TimeDistributed
    # aplica la misma capa Dense a cada frame de forma independiente)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(64, activation="relu"))(inputs)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten())(x)
    # Modelado temporal: LSTM bidireccionales capturan el movimiento
    # hacia adelante y hacia atrás en el tiempo; Dropout evita sobreajuste
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True))(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32))(x)
    # Cabezal de clasificación: capa densa + softmax sobre las clases
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    # Compilar el modelo con Adam y entropía cruzada para etiquetas enteras
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def train_dynamic_model(X: np.ndarray, y: np.ndarray, labels: list):
    """
    Entrena el modelo dinámico y guarda los artefactos en disco.

    Pasos: normaliza las secuencias, construye el modelo según la longitud de
    secuencia y el número de clases, entrena (80 épocas, batch adaptativo) y
    guarda el modelo (.h5) junto con las etiquetas (.json).

    Nota: la validación (validation_split=0.2) solo se activa cuando hay al
    menos 10 muestras; con datasets muy pequeños se entrena con todo.

    Args:
        X: Secuencias de entrenamiento con forma (N, T, 21, 3).
        y: Índices de clase de cada secuencia, forma (N,).
        labels: Lista de nombres de las clases.

    Returns:
        Tupla (model, history) con el modelo entrenado y el historial de Keras.
    """
    # Normalizar las secuencias y crear el modelo acorde al dataset
    X_processed = normalize_dynamic_sequences(X)
    model = create_dynamic_model(sequence_length=X_processed.shape[1], num_classes=len(labels))

    # Reservar un 20% para validación solo si hay suficientes muestras
    validation_split = 0.2 if len(X_processed) >= 10 else 0.0

    # Entrenamiento: muchas épocas y batch pequeño, adecuados para
    # datasets reducidos capturados manualmente desde la app
    history = model.fit(
        X_processed,
        y,
        epochs=80,
        batch_size=min(8, len(X_processed)),
        validation_split=validation_split,
        verbose=1
    )

    # Guardar el modelo entrenado (crear la carpeta de destino si no existe)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_PATH)

    # Guardar las etiquetas en el mismo orden que las salidas del modelo
    with open(LABELS_PATH, "w", encoding="utf-8") as f:
        json.dump(labels, f, indent=2)

    logger.info(f"Modelo dinámico guardado en {MODEL_PATH}")
    logger.info(f"Etiquetas dinámicas guardadas en {LABELS_PATH}")
    return model, history


def main():
    """
    Punto de entrada del script: carga el dataset, entrena y reporta el resultado.

    Cualquier error (dataset inexistente, datos inválidos, fallo de
    entrenamiento) se captura, se registra en el log y se informa por consola
    sin propagar la excepción.
    """
    try:
        # Flujo completo: cargar/validar dataset y entrenar el modelo dinámico
        X, y, labels = load_dynamic_dataset(DATASET_PATH)
        train_dynamic_model(X, y, labels)
        print("✅ Modelo dinámico entrenado correctamente")
        print(f"   Model: {MODEL_PATH}")
        print(f"   Labels: {LABELS_PATH}")
    except Exception as e:
        # Registrar y mostrar el error de forma amigable
        logger.error(f"Error en entrenamiento dinámico: {e}")
        print(f"Dynamic training failed: {e}")


if __name__ == "__main__":
    main()
