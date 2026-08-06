#!/usr/bin/env python3
"""
Script de entrenamiento para modelo de reconocimiento de señas
Genera model.h5 y labels.json para el sistema de inferencia

Este script entrena el modelo de señas ESTÁTICAS (posturas fijas de la mano,
como la mayoría de las letras del alfabeto dactilológico). A diferencia del
modelo dinámico (ml/train_dynamic.py), aquí cada muestra es un único frame
de 21 landmarks 3D de MediaPipe, sin componente temporal.

Flujo general:
1. Cargar el dataset estático (dataset_static.json o dataset_final.json)
   mediante el cargador centralizado de core.dataset_utils.
2. Validar que existan al menos 2 clases distintas.
3. Normalizar los landmarks con core.preprocessing (mismo preprocesado que
   usa la inferencia en ml/clasificador.py).
4. Construir y entrenar una red neuronal densa (MLP) en Keras.
5. Guardar los artefactos: data/models/model.h5 y data/models/labels.json.

Se ejecuta directamente desde consola: py ml/train.py
"""

import numpy as np
import tensorflow as tf
from pathlib import Path
import sys

# Raíz del proyecto (carpeta padre de ml/), necesaria para resolver rutas
# de datos y para poder importar los paquetes core/ y utils/
BASE_DIR = Path(__file__).resolve().parent.parent

# Agregar la raíz del proyecto al path de Python si aún no está,
# de modo que el script funcione al ejecutarse directamente desde consola
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

    Arquitectura (de entrada a salida):
    - Input (21, 3): un frame con 21 landmarks 3D de la mano (MediaPipe).
    - Dense(64, ReLU) aplicada punto a punto: extrae características de cada landmark.
    - Flatten: aplana la representación para las capas globales.
    - Dense(128, ReLU): combina la información de todos los landmarks.
    - Dense(num_classes, softmax): distribución de probabilidad sobre las señas.

    Se compila con Adam y sparse_categorical_crossentropy (las etiquetas
    llegan como índices enteros, no en one-hot).
    """
    # Entrada: un único frame de landmarks (sin componente temporal)
    inputs = tf.keras.Input(shape=(21, 3))
    
    # Capas densas
    x = tf.keras.layers.Dense(64, activation='relu')(inputs)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    
    # Capa de salida
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    
    model = tf.keras.Model(inputs, outputs)
    
    # Compilar el modelo: optimizador Adam y entropía cruzada para
    # etiquetas enteras, midiendo la exactitud durante el entrenamiento
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

    Returns:
        Tupla (model, history) con el modelo entrenado y el historial de Keras.

    Artefactos generados en disco:
        - data/models/model.h5: pesos y arquitectura del modelo.
        - data/models/labels.json: nombres de clases en el orden de la softmax.
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
    
    # Guardar modelo (crear la carpeta de destino si no existe)
    model_path = "data/models/model.h5"
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)
    logger.info(f"Modelo guardado en {model_path}")
    
    # Guardar etiquetas en el mismo orden que las salidas del modelo
    labels_path = "data/models/labels.json"
    with open(labels_path, 'w') as f:
        import json
        json.dump(labels, f, indent=2)
    logger.info(f"Etiquetas guardadas en {labels_path}")
    
    # Mostrar resultados finales del entrenamiento (última época)
    final_accuracy = history.history['accuracy'][-1]
    final_val_accuracy = history.history['val_accuracy'][-1]
    
    logger.info(f"Entrenamiento completado:")
    logger.info(f"  Accuracy final: {final_accuracy:.4f}")
    logger.info(f"  Val accuracy final: {final_val_accuracy:.4f}")
    
    return model, history

def main():
    """
    Función principal de entrenamiento

    Orquesta el flujo completo: localizar el dataset (con ruta de respaldo),
    validar que haya al menos 2 clases, normalizar los landmarks y entrenar.
    Los errores se capturan, se registran en el log y se informan por consola
    sin propagar la excepción.
    """
    try:
        # Cargar dataset: se prueba primero dataset_static.json y, si no
        # existe, se usa dataset_final.json como respaldo
        dataset_path = BASE_DIR / "data" / "datasets" / "dataset_static.json"
        if not dataset_path.exists():
            dataset_path = BASE_DIR / "data" / "datasets" / "dataset_final.json"
        if not dataset_path.exists():
            raise FileNotFoundError(
                "No se encontró ningún dataset para entrenar. "
                "Captura datos desde la app de escritorio antes de entrenar."
            )
        X, y, labels = load_dataset_centralized(str(dataset_path))
        
        # Validar que el dataset tenga al menos 2 clases distintas:
        # con una sola clase no es posible entrenar un clasificador
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
        # Registrar y mostrar el error de forma amigable
        logger.error(f"Error en entrenamiento: {e}")
        print(f"Training failed: {e}")

if __name__ == "__main__":
    main()
