#!/usr/bin/env python3
"""
Clasificador temporal para señas dinámicas usando secuencias de landmarks.
"""

import json
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import tensorflow as tf

from core.preprocessing import normalize_dynamic_sequence
from utils.logger import get_logger

logger = get_logger(__name__)

NO_SIGN_LABEL = "NO_SENA"


class DynamicSignClassifier:
    def __init__(self, model_path: Optional[str] = None, labels_path: Optional[str] = None, sequence_length: int = 20):
        self.model_path = Path(model_path) if model_path else Path("data/models/model_dynamic.h5")
        self.labels_path = Path(labels_path) if labels_path else Path("data/models/labels_dynamic.json")
        self.sequence_length = sequence_length
        self.model = None
        self.labels = []
        self.num_classes = 0

        self._load_model()
        self._load_labels()

    def _load_model(self):
        if not self.model_path.exists():
            logger.warning(f"No se encontró modelo dinámico en {self.model_path}")
            self.model = None
            return

        try:
            self.model = tf.keras.models.load_model(str(self.model_path))
            logger.info(f"Modelo dinámico cargado desde {self.model_path}")
        except Exception as e:
            logger.error(f"Error cargando modelo dinámico: {e}")
            self.model = None

    def _load_labels(self):
        if not self.labels_path.exists():
            logger.warning(f"No se encontraron etiquetas dinámicas en {self.labels_path}")
            self.labels = []
            return

        try:
            with open(self.labels_path, "r", encoding="utf-8") as f:
                self.labels = json.load(f)
            self.num_classes = len(self.labels)
            logger.info(f"Etiquetas dinámicas cargadas: {len(self.labels)} clases")
        except Exception as e:
            logger.error(f"Error cargando etiquetas dinámicas: {e}")
            self.labels = []

    def preprocess_sequence(self, sequence: List[List[Tuple[float, float, float]]]) -> np.ndarray:
        sequence_array = np.array(sequence, dtype=np.float32)

        if sequence_array.ndim != 3 or sequence_array.shape[1:] != (21, 3):
            raise ValueError(f"Secuencia dinámica inválida: {sequence_array.shape}, esperado (T, 21, 3)")

        if len(sequence_array) > self.sequence_length:
            sequence_array = sequence_array[-self.sequence_length:]
        elif len(sequence_array) < self.sequence_length:
            padding = np.repeat(sequence_array[-1][np.newaxis, ...], self.sequence_length - len(sequence_array), axis=0)
            sequence_array = np.concatenate([sequence_array, padding], axis=0)

        normalized = normalize_dynamic_sequence(sequence_array)
        return normalized[np.newaxis, ...]

    def classify_sequence(self, sequence: List[List[Tuple[float, float, float]]]) -> Tuple[str, float]:
        if self.model is None:
            return "unknown", 0.0

        if len(sequence) == 0:
            return "unknown", 0.0

        try:
            x = self.preprocess_sequence(sequence)
            preds = self.model.predict(x, verbose=0)
            idx = int(np.argmax(preds[0]))
            confidence = float(np.max(preds[0]))

            if confidence < 0.5:
                return "unknown", confidence

            if idx < len(self.labels):
                label = self.labels[idx]
                if label == NO_SIGN_LABEL:
                    return "unknown", confidence
                logger.info(f"Predicción dinámica: {label} (confianza: {confidence:.3f})")
                return label, confidence

            return "unknown", confidence
        except Exception as e:
            logger.error(f"Error en predicción dinámica: {e}")
            return "unknown", 0.0
