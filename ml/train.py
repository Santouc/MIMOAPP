#!/usr/bin/env python3
"""
Script de entrenamiento para modelo de reconocimiento de señas
Genera model.h5 y labels.json para el sistema de inferencia
"""

import numpy as np
import tensorflow as tf
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parent.parent

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from utils.logger import get_logger
from core.preprocessing import normalize_landmarks
from core.dataset_utils import load_dataset as load_dataset_centralized

logger = get_logger(__name__)

def create_model(num_classes: int) -> tf.keras.Model:
    """
    Crea modelo para entrenamiento
    
    Args:
        num_classes: Número de clases de salida
        
    Returns:
        Modelo TensorFlow
    """
    inputs = tf.keras.Input(shape=(21, 3))
    
    # Capas densas
    x = tf.keras.layers.Dense(64, activation='relu')(inputs)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    
    # Capa de salida
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    
    model = tf.keras.Model(inputs, outputs)
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_model(X: np.ndarray, y: np.ndarray, labels: list):
    """
    Entrena el modelo y guarda artefactos
    
    Args:
        X: Datos de entrenamiento
        y: Etiquetas de entrenamiento
        labels: Lista de nombres de clases
    """
    logger.info("Iniciando entrenamiento del modelo...")
    
    # Crear modelo
    num_classes = len(labels)
    model = create_model(num_classes)
    
    # =========================
    # #notas: ajustar para datasets pequeños
    # #notas: más epochs con regularización
    # #notas: batch_size pequeño para datasets pequeños
    # =========================
    # Entrenar
    history = model.fit(
        X, y,
        epochs=50,  # Más epochs para datasets pequeños
        batch_size=min(8, len(X)),  # Batch size adaptativo
        validation_split=0.2,
        verbose=1
    )
    
    # Guardar modelo
    model_path = "data/models/model.h5"
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)
    logger.info(f"Modelo guardado en {model_path}")
    
    # Guardar etiquetas
    labels_path = "data/models/labels.json"
    with open(labels_path, 'w') as f:
        import json
        json.dump(labels, f, indent=2)
    logger.info(f"Etiquetas guardadas en {labels_path}")
    
    # Mostrar resultados
    final_accuracy = history.history['accuracy'][-1]
    final_val_accuracy = history.history['val_accuracy'][-1]
    
    logger.info(f"Entrenamiento completado:")
    logger.info(f"  Accuracy final: {final_accuracy:.4f}")
    logger.info(f"  Val accuracy final: {final_val_accuracy:.4f}")
    
    return model, history

def main():
    """Función principal de entrenamiento"""
    try:
        # Cargar dataset
        dataset_path = BASE_DIR / "data" / "datasets" / "dataset_static.json"
        if not dataset_path.exists():
            dataset_path = BASE_DIR / "data" / "datasets" / "dataset_final.json"
        if not dataset_path.exists():
            raise FileNotFoundError(
                "No se encontró ningún dataset para entrenar. "
                "Captura datos desde la app de escritorio antes de entrenar."
            )
        X, y, labels = load_dataset_centralized(str(dataset_path))
        
        unique_classes = np.unique(y)
        if len(unique_classes) < 2:
            class_name = labels[int(unique_classes[0])] if len(unique_classes) == 1 and int(unique_classes[0]) < len(labels) else "desconocida"
            raise ValueError(
                f"El dataset solo contiene una clase ({class_name}). "
                "Captura al menos 2 letras distintas antes de entrenar."
            )
        
        # Preprocesar datos usando módulo centralizado
        X_processed = normalize_landmarks(X)
        print(f"Shape después de preprocesar: {X_processed.shape}")
        
        # Entrenar modelo
        model, history = train_model(X_processed, y, labels)
        
        print("✅ Model trained and saved successfully!")
        print(f"   Model: data/models/model.h5")
        print(f"   Labels: data/models/labels.json")
        print(f"   Classes: {labels}")
        
    except Exception as e:
        logger.error(f"Error en entrenamiento: {e}")
        print(f"Training failed: {e}")

if __name__ == "__main__":
    main()
