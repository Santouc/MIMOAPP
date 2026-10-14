#!/usr/bin/env python3
"""
Clasificador temporal para señas dinámicas usando secuencias de landmarks.

A diferencia del clasificador estático (que analiza una sola "foto" de la mano,
es decir, un único frame de 21 landmarks), este módulo interpreta MOVIMIENTOS:
recibe una secuencia de frames consecutivos (T, 21, 3) capturados con MediaPipe
y la clasifica con un modelo temporal Keras (entrenado en ml/train_dynamic.py,
basado en capas LSTM bidireccionales).

Responsabilidades principales:
- Cargar el modelo dinámico (model_dynamic.h5) y sus etiquetas (labels_dynamic.json).
- Normalizar y ajustar la longitud de las secuencias de entrada (recorte/padding).
- Clasificar la secuencia y devolver siempre una tupla (etiqueta, confianza).

La clase especial NO_SENA representa "ausencia de seña" (muestras negativas):
si el modelo predice esa clase, el resultado se traduce a "unknown" para que
la interfaz no muestre una seña inexistente.
"""

import json
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import tensorflow as tf

from core.preprocessing import normalize_dynamic_sequence
from utils.logger import get_logger

logger = get_logger(__name__)

# Etiqueta reservada para la clase negativa "sin seña" (manos en reposo o
# movimiento aleatorio). Sirve para que el modelo aprenda a rechazar entradas
# que no corresponden a ninguna seña real.
NO_SIGN_LABEL = "NO_SENA"


class DynamicSignClassifier:
    """
    Clasificador de señas dinámicas (con movimiento) basado en secuencias.

    Encapsula la carga del modelo temporal Keras y de sus etiquetas, el
    preprocesamiento de secuencias de landmarks (normalización y ajuste de
    longitud) y la inferencia. Su contrato público principal es
    classify_sequence(), que siempre devuelve (etiqueta, confianza).

    Atributos:
        model_path: Ruta al modelo dinámico entrenado (.h5).
        labels_path: Ruta al JSON con la lista de nombres de clases.
        sequence_length: Cantidad fija de frames (T) que espera el modelo.
        model: Modelo Keras cargado, o None si no está disponible.
        labels: Lista de etiquetas; el índice coincide con la salida softmax.
        num_classes: Número de clases cargadas.
    """

    def __init__(self, model_path: Optional[str] = None, labels_path: Optional[str] = None, sequence_length: int = 20):
        """
        Inicializa el clasificador dinámico y carga sus artefactos.

        Args:
            model_path: Ruta al modelo .h5 (por defecto data/models/model_dynamic.h5).
            labels_path: Ruta al JSON de etiquetas (por defecto data/models/labels_dynamic.json).
            sequence_length: Longitud fija de secuencia (frames) que espera el modelo.
        """
        # Configurar rutas con valores por defecto del proyecto
        self.model_path = Path(model_path) if model_path else Path("data/models/model_dynamic.h5")
        self.labels_path = Path(labels_path) if labels_path else Path("data/models/labels_dynamic.json")
        self.sequence_length = sequence_length
        self.model = None
        self.labels = []
        self.num_classes = 0

        # Intentar cargar modelo y etiquetas al construir la instancia;
        # si fallan, el clasificador queda operativo pero responde "unknown".
        self._load_model()
        self._load_labels()

    def _load_model(self):
        """
        Carga el modelo dinámico Keras desde disco (método interno).

        Si el archivo no existe o la carga falla, deja self.model en None
        y registra el problema en el log en lugar de lanzar una excepción,
        para que la aplicación pueda seguir funcionando sin modelo dinámico.
        """
        # Verificar existencia del archivo antes de intentar cargarlo
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
        """
        Carga la lista de etiquetas desde el archivo JSON (método interno).

        El orden de la lista debe coincidir con el orden de las neuronas de
        salida del modelo. Si el archivo no existe o está corrupto, deja
        self.labels vacía y lo registra en el log.
        """
        # Verificar existencia del archivo de etiquetas
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
        """
        Prepara una secuencia de landmarks para la inferencia del modelo.

        Pasos que realiza:
        1. Convierte la secuencia a un array NumPy float32 y valida su forma.
        2. Ajusta la longitud temporal a sequence_length: si sobran frames se
           conservan los últimos (los más recientes); si faltan, se rellena
           repitiendo el último frame (padding por repetición).
        3. Normaliza la secuencia con el módulo centralizado de preprocesado.

        Args:
            sequence: Lista de frames; cada frame contiene 21 landmarks (x, y, z).

        Returns:
            Array con forma (1, sequence_length, 21, 3) listo para model.predict.

        Raises:
            ValueError: Si la secuencia no tiene la forma esperada (T, 21, 3).
        """
        # Convertir a array NumPy para operar de forma vectorizada
        sequence_array = np.array(sequence, dtype=np.float32)

        # Validar que cada frame tenga exactamente 21 landmarks con 3 coordenadas
        if sequence_array.ndim != 3 or sequence_array.shape[1:] != (21, 3):
            raise ValueError(f"Secuencia dinámica inválida: {sequence_array.shape}, esperado (T, 21, 3)")

        # Ajustar la longitud temporal a la que espera el modelo:
        # - Si hay frames de más, quedarse con los últimos (más recientes).
        # - Si faltan frames, rellenar repitiendo el último frame capturado.
        if len(sequence_array) > self.sequence_length:
            sequence_array = sequence_array[-self.sequence_length:]
        elif len(sequence_array) < self.sequence_length:
            padding = np.repeat(sequence_array[-1][np.newaxis, ...], self.sequence_length - len(sequence_array), axis=0)
            sequence_array = np.concatenate([sequence_array, padding], axis=0)

        # Normalizar con el módulo centralizado (traslación/escala consistentes
        # con las usadas durante el entrenamiento) y agregar dimensión de batch
        normalized = normalize_dynamic_sequence(sequence_array)
        return normalized[np.newaxis, ...]

    def classify_sequence(self, sequence: List[List[Tuple[float, float, float]]]) -> Tuple[str, float]:
        """
        Clasifica una secuencia de landmarks como una seña dinámica.

        Contrato: siempre devuelve una tupla (etiqueta, confianza), donde la
        etiqueta pertenece a self.labels o es "unknown", y la confianza está
        en el rango [0, 1]. Se devuelve "unknown" cuando:
        - No hay modelo cargado o la secuencia está vacía.
        - La confianza de la predicción es menor que 0.5.
        - La clase predicha es NO_SENA (clase negativa "sin seña").
        - Ocurre cualquier error durante la inferencia.

        Args:
            sequence: Lista de frames; cada frame contiene 21 landmarks (x, y, z).

        Returns:
            Tupla (etiqueta, confianza) con la seña detectada.
        """
        # Sin modelo cargado no es posible clasificar
        if self.model is None:
            return "unknown", 0.0

        # Una secuencia vacía no aporta información
        if len(sequence) == 0:
            return "unknown", 0.0

        try:
            # Preprocesar la secuencia y ejecutar la inferencia del modelo
            x = self.preprocess_sequence(sequence)
            preds = self.model.predict(x, verbose=0)
            # Tomar la clase con mayor probabilidad y su confianza asociada
            idx = int(np.argmax(preds[0]))
            confidence = float(np.max(preds[0]))

            # Umbral de confianza: por debajo de 0.5 se descarta la predicción
            if confidence < 0.5:
                return "unknown", confidence

            # Traducir el índice de clase a su etiqueta legible
            if idx < len(self.labels):
                label = self.labels[idx]
                # La clase negativa NO_SENA se traduce a "unknown" para la interfaz
                if label == NO_SIGN_LABEL:
                    return "unknown", confidence
                logger.info(f"Predicción dinámica: {label} (confianza: {confidence:.3f})")
                return label, confidence

            # Índice fuera del rango de etiquetas conocidas
            return "unknown", confidence
        except Exception as e:
            # Cualquier error de inferencia se registra y se degrada a "unknown"
            logger.error(f"Error en predicción dinámica: {e}")
            return "unknown", 0.0
