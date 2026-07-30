from typing import Any

import numpy as np


HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


def average_static_landmarks(samples: list[Any] | np.ndarray) -> np.ndarray:
    array = np.asarray(samples, dtype=np.float32)
    if array.ndim == 4 and array.shape[1] == 1:
        array = array[:, 0, :, :]
    if array.ndim != 3 or array.shape[1:] != (21, 3):
        raise ValueError(f"Muestras estáticas inválidas: {array.shape}, esperado (N, 21, 3)")
    return np.mean(array, axis=0)


def average_dynamic_sequences(sequences: list[Any] | np.ndarray) -> np.ndarray:
    array = np.asarray(sequences, dtype=np.float32)
    if array.ndim != 4 or array.shape[2:] != (21, 3):
        raise ValueError(f"Secuencias dinámicas inválidas: {array.shape}, esperado (N, T, 21, 3)")
    target_length = int(np.median([sequence.shape[0] for sequence in array])) if array.dtype != object else 20
    normalized = np.asarray([resample_sequence(sequence, target_length) for sequence in array], dtype=np.float32)
    return np.mean(normalized, axis=0)


def resample_sequence(sequence: np.ndarray, target_length: int) -> np.ndarray:
    sequence = np.asarray(sequence, dtype=np.float32)
    if sequence.ndim != 3 or sequence.shape[1:] != (21, 3):
        raise ValueError(f"Secuencia inválida: {sequence.shape}, esperado (T, 21, 3)")
    if sequence.shape[0] == target_length:
        return sequence
    if sequence.shape[0] == 1:
        return np.repeat(sequence, target_length, axis=0)
    source_positions = np.linspace(0, sequence.shape[0] - 1, sequence.shape[0])
    target_positions = np.linspace(0, sequence.shape[0] - 1, target_length)
    resampled = np.empty((target_length, 21, 3), dtype=np.float32)
    for landmark_idx in range(21):
        for coord_idx in range(3):
            resampled[:, landmark_idx, coord_idx] = np.interp(
                target_positions,
                source_positions,
                sequence[:, landmark_idx, coord_idx],
            )
    return resampled
