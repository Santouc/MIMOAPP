#!/usr/bin/env python3
"""
Entrenamiento de modelo temporal para señas dinámicas.
"""

import json
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

BASE_DIR = Path(__file__).resolve().parent.parent

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from core.preprocessing import normalize_landmarks
from utils.logger import get_logger

logger = get_logger(__name__)


DATASET_PATH = BASE_DIR / "data" / "datasets" / "dataset_dynamic.json"
MODEL_PATH = BASE_DIR / "data" / "models" / "model_dynamic.h5"
LABELS_PATH = BASE_DIR / "data" / "models" / "labels_dynamic.json"


def load_dynamic_dataset(filepath: Path):
    if not filepath.exists():
        raise FileNotFoundError(
            "No se encontró dataset_dynamic.json. "
            "Graba señas dinámicas desde la app de escritorio antes de entrenar."
        )

    with open(filepath, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    X = np.array(dataset.get("X", []), dtype=np.float32)
    y = np.array(dataset.get("y", []), dtype=np.int32)
    labels = dataset.get("labels", [])

    if X.ndim != 4 or X.shape[2:] != (21, 3):
        raise ValueError(f"Shape inválido de dataset dinámico: {X.shape}, esperado (N, T, 21, 3)")

    if len(X) != len(y):
        raise ValueError(f"Cantidad inconsistente de muestras: X={len(X)}, y={len(y)}")

    unique_classes = np.unique(y)
    if len(unique_classes) < 2:
        class_name = labels[int(unique_classes[0])] if len(unique_classes) == 1 and int(unique_classes[0]) < len(labels) else "desconocida"
        raise ValueError(
            f"El dataset dinámico solo contiene una clase ({class_name}). "
            "Captura al menos 2 señas dinámicas distintas antes de entrenar."
        )
    
    compact_labels = [labels[int(class_idx)] for class_idx in unique_classes]
    class_to_compact = {int(class_idx): compact_idx for compact_idx, class_idx in enumerate(unique_classes)}
    y = np.array([class_to_compact[int(class_idx)] for class_idx in y], dtype=np.int32)

    logger.info(f"Dataset dinámico cargado: {X.shape[0]} secuencias, {X.shape[1]} frames, {len(compact_labels)} clases usadas")
    logger.info(f"Clases dinámicas usadas: {compact_labels}")
    return X, y, compact_labels


def normalize_dynamic_sequences(X: np.ndarray) -> np.ndarray:
    samples, timesteps, points, coords = X.shape
    X_flat = X.reshape(samples * timesteps, points, coords)
    X_norm = normalize_landmarks(X_flat)
    return X_norm.reshape(samples, timesteps, points, coords)


def create_dynamic_model(sequence_length: int, num_classes: int) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(sequence_length, 21, 3))

    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(64, activation="relu"))(inputs)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten())(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True))(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32))(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def train_dynamic_model(X: np.ndarray, y: np.ndarray, labels: list):
    X_processed = normalize_dynamic_sequences(X)
    model = create_dynamic_model(sequence_length=X_processed.shape[1], num_classes=len(labels))

    validation_split = 0.2 if len(X_processed) >= 10 else 0.0

    history = model.fit(
        X_processed,
        y,
        epochs=80,
        batch_size=min(8, len(X_processed)),
        validation_split=validation_split,
        verbose=1
    )

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_PATH)

    with open(LABELS_PATH, "w", encoding="utf-8") as f:
        json.dump(labels, f, indent=2)

    logger.info(f"Modelo dinámico guardado en {MODEL_PATH}")
    logger.info(f"Etiquetas dinámicas guardadas en {LABELS_PATH}")
    return model, history


def main():
    try:
        X, y, labels = load_dynamic_dataset(DATASET_PATH)
        train_dynamic_model(X, y, labels)
        print("✅ Modelo dinámico entrenado correctamente")
        print(f"   Model: {MODEL_PATH}")
        print(f"   Labels: {LABELS_PATH}")
    except Exception as e:
        logger.error(f"Error en entrenamiento dinámico: {e}")
        print(f"Dynamic training failed: {e}")


if __name__ == "__main__":
    main()
