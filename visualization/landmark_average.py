"""Utilidades de visualización para promediar landmarks de la mano.

Este módulo permite calcular la "forma promedio" de una seña a partir de
varias muestras capturadas, tanto para señas estáticas (una sola pose)
como dinámicas (secuencias de poses). El resultado se usa para dibujar
una vista previa representativa de la seña en la interfaz.
"""

from typing import Any

import numpy as np


# Pares de índices de landmarks que forman el esqueleto de la mano según
# el modelo de MediaPipe (21 puntos: muñeca, pulgar, índice, medio,
# anular y meñique). Cada tupla indica dos puntos que se unen con una línea
# al dibujar la mano.
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


def average_static_landmarks(samples: list[Any] | np.ndarray) -> np.ndarray:
    """Calcula la pose promedio de varias muestras de una seña estática.

    Args:
        samples: Colección de muestras con forma (N, 21, 3) o (N, 1, 21, 3),
            donde N es la cantidad de muestras, 21 los landmarks de la mano
            y 3 las coordenadas (x, y, z) de cada punto.

    Returns:
        Un arreglo (21, 3) con la posición promedio de cada landmark.

    Raises:
        ValueError: Si las muestras no tienen la forma esperada.
    """
    array = np.asarray(samples, dtype=np.float32)
    # Algunas capturas llegan con una dimensión extra (N, 1, 21, 3);
    # se elimina ese eje intermedio para trabajar con (N, 21, 3).
    if array.ndim == 4 and array.shape[1] == 1:
        array = array[:, 0, :, :]
    # Validar que la estructura final sea la esperada antes de promediar.
    if array.ndim != 3 or array.shape[1:] != (21, 3):
        raise ValueError(f"Muestras estáticas inválidas: {array.shape}, esperado (N, 21, 3)")
    # El promedio por eje 0 colapsa las N muestras en una sola pose media.
    return np.mean(array, axis=0)


def average_dynamic_sequences(sequences: list[Any] | np.ndarray) -> np.ndarray:
    """Calcula la secuencia promedio de varias muestras de una seña dinámica.

    Como cada grabación puede tener distinta cantidad de cuadros (T),
    primero se re-muestrean todas las secuencias a una longitud común
    (la mediana de las longitudes) y luego se promedian cuadro a cuadro.

    Args:
        sequences: Colección de secuencias con forma (N, T, 21, 3).

    Returns:
        Un arreglo (T, 21, 3) con la secuencia promedio.

    Raises:
        ValueError: Si las secuencias no tienen la forma esperada.
    """
    array = np.asarray(sequences, dtype=np.float32)
    # Validar la estructura general del lote de secuencias.
    if array.ndim != 4 or array.shape[2:] != (21, 3):
        raise ValueError(f"Secuencias dinámicas inválidas: {array.shape}, esperado (N, T, 21, 3)")
    # Elegir la longitud objetivo: la mediana de las longitudes reales,
    # o 20 cuadros como valor por defecto si el arreglo es heterogéneo.
    target_length = int(np.median([sequence.shape[0] for sequence in array])) if array.dtype != object else 20
    # Re-muestrear cada secuencia a la longitud común para poder promediarlas.
    normalized = np.asarray([resample_sequence(sequence, target_length) for sequence in array], dtype=np.float32)
    # Promediar todas las secuencias ya alineadas temporalmente.
    return np.mean(normalized, axis=0)


def resample_sequence(sequence: np.ndarray, target_length: int) -> np.ndarray:
    """Ajusta una secuencia de landmarks a una longitud de cuadros dada.

    Usa interpolación lineal independiente por cada coordenada de cada
    landmark, de modo que el movimiento se estira o comprime en el tiempo
    sin perder su forma general.

    Args:
        sequence: Secuencia original con forma (T, 21, 3).
        target_length: Cantidad de cuadros deseada en la salida.

    Returns:
        Un arreglo (target_length, 21, 3) con la secuencia re-muestreada.

    Raises:
        ValueError: Si la secuencia no tiene la forma esperada.
    """
    sequence = np.asarray(sequence, dtype=np.float32)
    # Validar la forma de la secuencia individual.
    if sequence.ndim != 3 or sequence.shape[1:] != (21, 3):
        raise ValueError(f"Secuencia inválida: {sequence.shape}, esperado (T, 21, 3)")
    # Si ya tiene la longitud pedida no hay nada que hacer.
    if sequence.shape[0] == target_length:
        return sequence
    # Con un solo cuadro no se puede interpolar: se repite ese cuadro.
    if sequence.shape[0] == 1:
        return np.repeat(sequence, target_length, axis=0)
    # Posiciones temporales de origen y destino para la interpolación.
    source_positions = np.linspace(0, sequence.shape[0] - 1, sequence.shape[0])
    target_positions = np.linspace(0, sequence.shape[0] - 1, target_length)
    resampled = np.empty((target_length, 21, 3), dtype=np.float32)
    # Interpolar por separado cada coordenada (x, y, z) de cada landmark.
    for landmark_idx in range(21):
        for coord_idx in range(3):
            resampled[:, landmark_idx, coord_idx] = np.interp(
                target_positions,
                source_positions,
                sequence[:, landmark_idx, coord_idx],
            )
    return resampled
