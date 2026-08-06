#!/usr/bin/env python3
"""
Módulo centralizado de preprocesamiento de landmarks
Normalización de traducción y escala consistente en todo el sistema

Todas las etapas del proyecto (captura de dataset, entrenamiento e
inferencia en tiempo real) deben usar estas funciones para garantizar
que los landmarks lleguen al modelo con exactamente el mismo formato:

- normalize_landmarks: normalización estándar de manos estáticas
  (centrado en la muñeca + escala relativa al tamaño de la mano).
- normalize_dynamic_sequence(s): normalización de secuencias dinámicas
  que preserva la trayectoria global del gesto.
- preprocess_for_inference / preprocess_for_training: envoltorios de
  conveniencia con los shapes que espera TensorFlow.

Mantener este preprocesamiento centralizado evita inconsistencias entre
los datos de entrenamiento y los datos de inferencia.
"""

import numpy as np
from typing import Union, List, Tuple


def normalize_landmarks(landmarks: Union[np.ndarray, List[List[Tuple[float, float, float]]]]) -> np.ndarray:
    """
    Normalización de landmarks de manos (traducción + escala)
    
    Este método implementa el preprocesamiento estándar usado en todo el sistema:
    - Normalización de traducción (centrar en muñeca - landmark 0)
    - Normalización de escala (relativa a longitud del hueso índice)
    
    Args:
        landmarks: Array de landmarks shape (N, 21, 3) o lista de manos [[(x,y,z), ...], ...]
        
    Returns:
        Array normalizado shape (N, 21, 3) o (1, 21, 3) si es una sola mano
        
    Examples:
        >>> landmarks = np.random.rand(100, 21, 3)
        >>> normalized = normalize_landmarks(landmarks)
        >>> print(normalized.shape)
        (100, 21, 3)
        
        >>> single_hand = [[(0.5, 0.5, 0.0) for _ in range(21)]]
        >>> normalized = normalize_landmarks(single_hand)
        >>> print(normalized.shape)
        (1, 21, 3)
    """
    # Convertir a array numpy si es lista
    if isinstance(landmarks, list):
        if len(landmarks) == 0:
            return np.zeros((1, 21, 3), dtype=np.float32)
        
        # Si es lista de manos, tomar solo la primera
        hand = landmarks[0]
        if isinstance(hand, list) and len(hand) == 21:
            landmarks = np.array([hand], dtype=np.float32)
        else:
            landmarks = np.array(landmarks, dtype=np.float32)
    
    # Validar shape
    if len(landmarks.shape) == 2:
        landmarks = np.expand_dims(landmarks, axis=0)
    
    if landmarks.shape[1] != 21 or landmarks.shape[2] != 3:
        raise ValueError(f"Shape inválido: {landmarks.shape}, esperado (N, 21, 3)")
    
    X = landmarks.copy().astype(np.float32)
    
    # ====================
    # NORMALIZACIÓN DE TRADUCCIÓN
    # ====================
    # Centrar en la muñeca (landmark 0)
    X = X - X[:, 0:1, :]
    
    # ====================
    # NORMALIZACIÓN DE ESCALA
    # ====================
    # Usar longitud del hueso medio del dedo índice como referencia
    # landmarks[5] = MCP del índice, landmarks[6] = PIP del índice
    # Nota: como los datos ya están centrados en la muñeca, la norma del
    # landmark 9 (MCP del dedo medio) equivale a la distancia muñeca->nudillo,
    # una medida estable del tamaño de la mano en la imagen.
    scale = np.linalg.norm(X[:, 9:10, :], axis=2, keepdims=True)
    
    # Evitar división por cero
    scale[scale < 1e-6] = 1.0
    
    # Normalizar por escala
    X = X / scale
    
    return X


def normalize_dynamic_sequence(sequence: Union[np.ndarray, List[List[Tuple[float, float, float]]]]) -> np.ndarray:
    """
    Normalización de una secuencia dinámica que PRESERVA la trayectoria global.

    A diferencia de normalize_landmarks (que centra cada frame en su muñeca y
    elimina el movimiento de la mano por la pantalla), esta función:
    - Centra la secuencia COMPLETA en la muñeca del primer frame
    - Escala una sola vez con la mediana del tamaño de mano de la secuencia

    Así el modelo recibe tanto la forma de la mano como la ruta que recorre,
    de manera invariante a la posición inicial y a la distancia a la cámara.

    Args:
        sequence: Array o lista shape (T, 21, 3)

    Returns:
        Array normalizado shape (T, 21, 3)
    """
    # Convertir a array float32 y copiar para no modificar el original
    seq = np.asarray(sequence, dtype=np.float32).copy()

    # Validar que la secuencia tenga el formato (T frames, 21 landmarks, 3 coords)
    if seq.ndim != 3 or seq.shape[1] != 21 or seq.shape[2] != 3:
        raise ValueError(f"Shape inválido: {seq.shape}, esperado (T, 21, 3)")

    # Tamaño de mano por frame (muñeca -> MCP dedo medio), invariante a traslación
    hand_sizes = np.linalg.norm(seq[:, 9, :] - seq[:, 0, :], axis=1)
    scale = float(np.median(hand_sizes))
    if scale < 1e-6:
        scale = 1.0

    # Centrar toda la secuencia en la muñeca del primer frame (preserva trayectoria)
    seq = seq - seq[0:1, 0:1, :]

    return seq / scale


def normalize_dynamic_sequences(sequences: Union[np.ndarray, List]) -> np.ndarray:
    """
    Normalización de un batch de secuencias dinámicas shape (N, T, 21, 3),
    preservando la trayectoria de cada secuencia.
    """
    # Validar formato del batch antes de procesar
    batch = np.asarray(sequences, dtype=np.float32)
    if batch.ndim != 4 or batch.shape[2] != 21 or batch.shape[3] != 3:
        raise ValueError(f"Shape inválido: {batch.shape}, esperado (N, T, 21, 3)")
    # Normalizar cada secuencia de forma independiente y apilar el resultado
    return np.stack([normalize_dynamic_sequence(seq) for seq in batch])


def normalize_single_hand(landmarks: List[Tuple[float, float, float]]) -> np.ndarray:
    """
    Normalización para una sola mano (21 landmarks)
    
    Args:
        landmarks: Lista de 21 tuplas (x, y, z)
        
    Returns:
        Array normalizado shape (1, 21, 3)
    """
    return normalize_landmarks([landmarks])


def normalize_batch(landmarks: np.ndarray) -> np.ndarray:
    """
    Normalización para batch de manos (N manos)
    
    Args:
        landmarks: Array shape (N, 21, 3)
        
    Returns:
        Array normalizado shape (N, 21, 3)
    """
    return normalize_landmarks(landmarks)


def preprocess_for_inference(landmarks: List[List[Tuple[float, float, float]]]) -> np.ndarray:
    """
    Preprocesamiento específico para inferencia en tiempo real
    Retorna shape (1, 21, 3) para compatibilidad con TensorFlow
    
    Args:
        landmarks: Lista de manos detectadas [[(x,y,z), ...], ...]
        
    Returns:
        Array preprocesado shape (1, 21, 3)
    """
    # Si no hay manos detectadas, devolver un array de ceros del shape esperado
    if not landmarks or len(landmarks) == 0:
        return np.zeros((1, 21, 3), dtype=np.float32)
    
    # Tomar solo la primera mano
    hand = landmarks[0]
    
    # Descartar detecciones incompletas (deben ser exactamente 21 landmarks)
    if len(hand) != 21:
        return np.zeros((1, 21, 3), dtype=np.float32)
    
    # Normalizar
    normalized = normalize_landmarks([hand])
    
    return normalized


def preprocess_for_training(landmarks: np.ndarray) -> np.ndarray:
    """
    Preprocesamiento específico para entrenamiento
    Retorna shape (N, 21, 3) para batch de entrenamiento
    
    Args:
        landmarks: Array shape (N, 21, 3)
        
    Returns:
        Array preprocesado shape (N, 21, 3)
    """
    return normalize_landmarks(landmarks)


# Bloque de autoprueba: se ejecuta solo al correr este archivo directamente
# (py core/preprocessing.py) y verifica shapes y rangos de la normalización.
if __name__ == "__main__":
    # Prueba básica
    print("=== Prueba de Preprocesamiento ===")
    
    # Generar datos de prueba
    test_landmarks = np.random.rand(10, 21, 3)
    print(f"Input shape: {test_landmarks.shape}")
    
    # Normalizar
    normalized = normalize_landmarks(test_landmarks)
    print(f"Output shape: {normalized.shape}")
    print(f"Output dtype: {normalized.dtype}")
    print(f"Output range: [{normalized.min():.3f}, {normalized.max():.3f}]")
    
    # Prueba con lista
    single_hand = [[(0.5 + i*0.01, 0.5 + i*0.01, 0.0) for i in range(21)]]
    normalized_single = normalize_landmarks(single_hand)
    print(f"\nSingle hand input shape: (1, 21, 3)")
    print(f"Single hand output shape: {normalized_single.shape}")
    
    print("\n✅ Prueba completada exitosamente")
