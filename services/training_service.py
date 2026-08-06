"""Servicio de entrenamiento de modelos de la aplicación MIMO.

Este módulo define `TrainingService`, encargado de entrenar los dos modelos
de clasificación de señas con TensorFlow/Keras:

- Modelo ESTÁTICO: clasifica una postura de mano a partir de 21 landmarks 3D
  de MediaPipe (entrada de forma (21, 3)). Es una red densa (MLP).
- Modelo DINÁMICO: clasifica secuencias de posturas (entrada de forma
  (T, 21, 3), con T pasos de tiempo). Usa capas LSTM bidireccionales.

Para el modelo dinámico, además, se generan automáticamente muestras
"negativas" sintéticas (movimientos que NO son señas: manos quietas,
transiciones entre posturas y desplazamientos aleatorios) que se etiquetan
con la clase especial `NO_SIGN_LABEL`. Esto ayuda al clasificador a rechazar
movimientos que no corresponden a ninguna seña real.

El resultado de cada entrenamiento se devuelve como un `TrainingResult` con
métricas y rutas de los archivos generados (modelo .h5 y labels .json).
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import tensorflow as tf

from core.dataset_utils import load_dataset
from core.preprocessing import normalize_landmarks, normalize_dynamic_sequences
from ml.dynamic_classifier import NO_SIGN_LABEL
from .path_service import PathService


@dataclass
class TrainingResult:
    """Resultado de un intento de entrenamiento.

    Atributos:
        trained: True si el modelo se entrenó correctamente.
        model_type: "static" o "dynamic".
        message: mensaje descriptivo para mostrar al usuario.
        model_path: ruta del modelo .h5 guardado (si se entrenó).
        labels_path: ruta del archivo de etiquetas guardado (si se entrenó).
        samples: cantidad de muestras usadas en el entrenamiento.
        classes: cantidad de clases del modelo.
        accuracy: exactitud final sobre el conjunto de entrenamiento.
        validation_accuracy: exactitud final de validación (si hubo split).
    """

    trained: bool
    model_type: Literal["static", "dynamic"]
    message: str
    model_path: str | None = None
    labels_path: str | None = None
    samples: int = 0
    classes: int = 0
    accuracy: float | None = None
    validation_accuracy: float | None = None


class TrainingService:
    """Entrena los modelos Keras estático y dinámico a partir de los datasets."""

    def __init__(self, paths: PathService | None = None):
        """Inicializa el servicio y garantiza la estructura de carpetas.

        Args:
            paths: instancia de `PathService` (se crea una por defecto si falta).
        """
        self.paths = paths or PathService()
        self.paths.ensure_app_dirs()

    def train(self, capture_type: Literal["static", "dynamic"]) -> TrainingResult:
        """Entrena el modelo indicado por tipo.

        Args:
            capture_type: "static" o "dynamic".

        Returns:
            El `TrainingResult` del entrenamiento correspondiente.

        Raises:
            ValueError: si el tipo no es válido.
        """
        # Delegar en el método específico según el tipo solicitado.
        if capture_type == "static":
            return self.train_static()
        if capture_type == "dynamic":
            return self.train_dynamic()
        raise ValueError("El tipo de entrenamiento debe ser 'static' o 'dynamic'")

    def train_static(self) -> TrainingResult:
        """Entrena el modelo estático con el dataset de posturas.

        Pasos: carga el dataset, valida que haya al menos 2 clases,
        normaliza los landmarks, mezcla las muestras, entrena la red densa
        durante 50 épocas y guarda el modelo y sus etiquetas.

        Returns:
            `TrainingResult` con el resultado (éxito o motivo del fallo).
        """
        # Sin dataset no hay nada que entrenar.
        if not self.paths.static_dataset_path.exists():
            return TrainingResult(False, "static", "No existe dataset estático para entrenar.")

        # Cargar el dataset y validar que tenga suficientes clases.
        X, y, labels = load_dataset(str(self.paths.static_dataset_path))
        validation = self._validate_dataset(y, labels, "static")
        if validation is not None:
            return validation

        # Normalizar los landmarks y mezclar las muestras de forma
        # reproducible antes de entrenar.
        X_processed = normalize_landmarks(X)
        X_processed, y = self._shuffle_samples(X_processed, y)
        model = self._create_static_model(len(labels))
        # Reservar un 20% para validación solo si hay muestras suficientes.
        validation_split = 0.2 if len(X_processed) >= 10 else 0.0
        history = model.fit(
            X_processed,
            y,
            epochs=50,
            batch_size=min(8, len(X_processed)),
            validation_split=validation_split,
            verbose=0,
        )

        # Guardar el modelo entrenado y las etiquetas asociadas.
        self.paths.static_model_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(self.paths.static_model_path)
        self._save_labels(self.paths.static_labels_path, labels)
        return self._build_result("static", history, self.paths.static_model_path, self.paths.static_labels_path, len(X_processed), len(labels))

    def train_dynamic(self) -> TrainingResult:
        """Entrena el modelo dinámico con el dataset de secuencias.

        Además de las señas reales, genera muestras negativas sintéticas
        (clase `NO_SIGN_LABEL`) a partir del dataset estático para que el
        modelo aprenda a rechazar movimientos que no son señas. Luego
        normaliza, mezcla, entrena la red LSTM durante 80 épocas y guarda
        el modelo y las etiquetas de entrenamiento.

        Returns:
            `TrainingResult` con el resultado (éxito o motivo del fallo).
        """
        # Sin dataset dinámico no hay nada que entrenar.
        if not self.paths.dynamic_dataset_path.exists():
            return TrainingResult(False, "dynamic", "No existe dataset dinámico para entrenar.")

        # Cargar y validar que el dataset no esté vacío.
        X, y, labels = self._load_dynamic_dataset(self.paths.dynamic_dataset_path)
        if len(y) == 0:
            return TrainingResult(False, "dynamic", "El dataset dinámico está vacío.")

        # Generar negativos sintéticos; si se obtienen, se agrega la clase
        # extra NO_SIGN_LABEL al final de las etiquetas de entrenamiento.
        negatives = self._generate_dynamic_negatives(X.shape[1], int(np.bincount(y).max()))
        if len(negatives) > 0:
            training_labels = list(labels) + [NO_SIGN_LABEL]
            X = np.concatenate([X, negatives])
            y = np.concatenate([y, np.full(len(negatives), len(labels), dtype=np.int32)])
        else:
            # Sin negativos, se exige que el dataset tenga al menos 2 clases.
            training_labels = list(labels)
            validation = self._validate_dataset(y, labels, "dynamic")
            if validation is not None:
                return validation

        # Normalizar las secuencias, mezclar y entrenar la red recurrente.
        X_processed = normalize_dynamic_sequences(X)
        X_processed, y = self._shuffle_samples(X_processed, y)
        model = self._create_dynamic_model(X_processed.shape[1], len(training_labels))
        # Reservar un 20% para validación solo si hay muestras suficientes.
        validation_split = 0.2 if len(X_processed) >= 10 else 0.0
        history = model.fit(
            X_processed,
            y,
            epochs=80,
            batch_size=min(8, len(X_processed)),
            validation_split=validation_split,
            verbose=0,
        )

        # Guardar el modelo entrenado y las etiquetas (incluye NO_SIGN_LABEL).
        self.paths.dynamic_model_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(self.paths.dynamic_model_path)
        self._save_labels(self.paths.dynamic_labels_path, training_labels)
        return self._build_result("dynamic", history, self.paths.dynamic_model_path, self.paths.dynamic_labels_path, len(X_processed), len(training_labels))

    def _shuffle_samples(self, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Mezcla X e y con la misma permutación, usando semilla fija (42).

        La semilla fija hace que el entrenamiento sea reproducible.

        Args:
            X: matriz de muestras.
            y: vector de etiquetas.

        Returns:
            Tupla (X mezclado, y mezclado) manteniendo la correspondencia.
        """
        order = np.random.default_rng(42).permutation(len(X))
        return X[order], np.asarray(y)[order]

    def _validate_dataset(self, y: np.ndarray, labels: list[str], model_type: Literal["static", "dynamic"]) -> TrainingResult | None:
        """Valida que el dataset sea entrenable.

        Comprueba que no esté vacío y que existan al menos 2 clases
        distintas (una red softmax con una sola clase no aprende nada útil).

        Args:
            y: vector de etiquetas del dataset.
            labels: nombres de las clases.
            model_type: tipo de modelo, para el mensaje de error.

        Returns:
            Un `TrainingResult` de fallo si el dataset no es válido, o
            `None` si se puede entrenar.
        """
        if len(y) == 0:
            return TrainingResult(False, model_type, "El dataset está vacío.")
        # Contar las clases distintas presentes en las etiquetas.
        unique_classes = np.unique(y)
        if len(unique_classes) < 2:
            # Con una sola clase, informar cuál es para orientar al usuario.
            class_name = labels[int(unique_classes[0])] if len(unique_classes) == 1 and int(unique_classes[0]) < len(labels) else "desconocida"
            return TrainingResult(
                False,
                model_type,
                f"Aún no se entrenó el modelo porque solo hay una clase ({class_name}). Agrega al menos 2 señas distintas.",
                samples=len(y),
                classes=len(unique_classes),
            )
        return None

    def _load_dynamic_dataset(self, path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """Carga y valida el dataset dinámico desde disco.

        Args:
            path: ruta al archivo JSON del dataset dinámico.

        Returns:
            Tupla (X, y, labels), donde X tiene forma (N, T, 21, 3).

        Raises:
            ValueError: si la forma de X es inválida o si la cantidad de
                muestras no coincide con la de etiquetas.
        """
        # Leer el JSON y convertir a arrays de NumPy con los tipos correctos.
        with path.open("r", encoding="utf-8") as file:
            dataset = json.load(file)
        X = np.array(dataset.get("X", []), dtype=np.float32)
        y = np.array(dataset.get("y", []), dtype=np.int32)
        labels = dataset.get("labels", [])
        # Dataset vacío: devolver arrays con la forma esperada pero sin filas.
        if len(X) == 0:
            return np.zeros((0, 20, 21, 3), dtype=np.float32), np.zeros((0,), dtype=np.int32), labels
        # Validar la forma (N, T, 21, 3) y la consistencia entre X e y.
        if X.ndim != 4 or X.shape[2:] != (21, 3):
            raise ValueError(f"Shape inválido de dataset dinámico: {X.shape}, esperado (N, T, 21, 3)")
        if len(X) != len(y):
            raise ValueError(f"Cantidad inconsistente de muestras: X={len(X)}, y={len(y)}")
        return X, y, labels

    def _generate_dynamic_negatives(self, timesteps: int, max_class_count: int) -> np.ndarray:
        """Genera secuencias negativas sintéticas ("no seña").

        A partir de posturas del dataset estático, fabrica secuencias que se
        parecen a movimientos cotidianos pero NO a señas dinámicas reales.
        Se alternan tres tipos de negativos:
        - "hold": la mano se mantiene casi quieta con leve deriva.
        - "transition": interpolación suave entre dos posturas distintas.
        - "drift": la mano se desplaza siguiendo un camino aleatorio.

        Args:
            timesteps: cantidad de cuadros por secuencia (T).
            max_class_count: cantidad de muestras de la clase más numerosa;
                se usa para dimensionar cuántos negativos generar.

        Returns:
            Array de forma (M, T, 21, 3) con las secuencias negativas, o un
            array vacío si no hay dataset estático utilizable.
        """
        empty = np.zeros((0, timesteps, 21, 3), dtype=np.float32)
        # Sin dataset estático no se pueden fabricar negativos.
        if not self.paths.static_dataset_path.exists():
            return empty
        # Cargar el dataset estático de forma tolerante: ante cualquier
        # error se devuelve el array vacío en lugar de interrumpir.
        try:
            X, y, _ = load_dataset(str(self.paths.static_dataset_path))
        except Exception:
            return empty
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)
        if len(X) == 0:
            return empty

        # Calcular cuántos negativos generar: ~120% de la clase mayoritaria,
        # acotado entre 6 y 60 para no desbalancear el dataset.
        rng = np.random.default_rng(7)
        total = int(np.clip(round(max_class_count * 1.2), 6, 60))
        negatives = []
        # Alternar cíclicamente los tres tipos de negativos hasta el total.
        while len(negatives) < total:
            kind = len(negatives) % 3
            base_idx = int(rng.integers(len(X)))
            base = X[base_idx]
            if kind == 0:
                # Tipo 0: mano casi quieta (hold).
                frames = self._negative_hold(base, timesteps, rng)
            elif kind == 1:
                # Tipo 1: transición entre dos posturas de clases distintas.
                other_indices = np.flatnonzero(y != y[base_idx])
                other_idx = int(rng.choice(other_indices)) if len(other_indices) else int(rng.integers(len(X)))
                frames = self._negative_transition(base, X[other_idx], timesteps, rng)
            else:
                # Tipo 2: desplazamiento aleatorio (drift).
                frames = self._negative_drift(base, timesteps, rng)
            negatives.append(frames.astype(np.float32))
        return np.asarray(negatives, dtype=np.float32)

    def _negative_hold(self, base: np.ndarray, timesteps: int, rng: np.random.Generator) -> np.ndarray:
        """Genera un negativo tipo "hold": mano casi quieta con leve deriva.

        Args:
            base: postura base de forma (21, 3).
            timesteps: cantidad de cuadros de la secuencia.
            rng: generador aleatorio (para reproducibilidad).

        Returns:
            Secuencia de forma (T, 21, 3).
        """
        # Deriva lineal muy pequeña (menor en el eje Z) a lo largo del tiempo.
        drift = rng.normal(0.0, 0.008, size=3) * np.array([1.0, 1.0, 0.2])
        progress = np.linspace(0.0, 1.0, timesteps)[:, None]
        path = progress * drift[None, :]
        # Ruido fino por landmark para simular el temblor natural de la mano.
        jitter = rng.normal(0.0, 0.002, size=(timesteps, 21, 3))
        return base[None, :, :] + path[:, None, :] + jitter

    def _negative_transition(self, start: np.ndarray, end: np.ndarray, timesteps: int, rng: np.random.Generator) -> np.ndarray:
        """Genera un negativo tipo "transition": cambio suave entre posturas.

        Interpola con suavizado (smoothstep) entre una postura inicial y una
        final, agregando un desplazamiento global y ruido leve.

        Args:
            start: postura inicial de forma (21, 3).
            end: postura final de forma (21, 3).
            timesteps: cantidad de cuadros de la secuencia.
            rng: generador aleatorio.

        Returns:
            Secuencia de forma (T, 21, 3).
        """
        # Curva de interpolación suavizada (smoothstep: 3t^2 - 2t^3).
        t = np.linspace(0.0, 1.0, timesteps)
        ease = t * t * (3.0 - 2.0 * t)
        # Mezcla progresiva entre la postura inicial y la final.
        frames = start[None, :, :] * (1.0 - ease)[:, None, None] + end[None, :, :] * ease[:, None, None]
        # Desplazamiento global acotado que acompaña la transición.
        offset = np.clip(rng.normal(0.0, 0.05, size=3), -0.15, 0.15) * np.array([1.0, 1.0, 0.2])
        path = ease[:, None] * offset[None, :]
        # Ruido fino por landmark.
        jitter = rng.normal(0.0, 0.002, size=(timesteps, 21, 3))
        return frames + path[:, None, :] + jitter

    def _negative_drift(self, base: np.ndarray, timesteps: int, rng: np.random.Generator) -> np.ndarray:
        """Genera un negativo tipo "drift": la mano se desplaza al azar.

        Construye un camino aleatorio (random walk) acumulado, lo reescala a
        una amplitud controlada y lo aplica a la postura base.

        Args:
            base: postura base de forma (21, 3).
            timesteps: cantidad de cuadros de la secuencia.
            rng: generador aleatorio.

        Returns:
            Secuencia de forma (T, 21, 3).
        """
        # Camino aleatorio acumulado, con menor amplitud en el eje Z, y
        # anclado para que comience en el origen.
        steps = rng.normal(0.0, 1.0, size=(timesteps, 3)) * np.array([1.0, 1.0, 0.2])
        path = np.cumsum(steps, axis=0)
        path = path - path[0]
        # Reescalar el camino para que su alcance en el plano XY quede
        # dentro de un rango razonable (entre 0.08 y 0.3 aprox.).
        span = float(np.max(np.linalg.norm(path[:, :2], axis=1)))
        if span < 1e-6:
            span = 1.0
        path = path * (rng.uniform(0.08, 0.3) / span)
        # Ruido fino por landmark.
        jitter = rng.normal(0.0, 0.002, size=(timesteps, 21, 3))
        return base[None, :, :] + path[:, None, :] + jitter

    def _create_static_model(self, num_classes: int) -> tf.keras.Model:
        """Construye y compila la red densa del clasificador estático.

        Arquitectura: dos capas densas por landmark (con BatchNormalization),
        aplanado y luego capas densas con Dropout para regularizar, terminando
        en una salida softmax con una neurona por clase.

        Args:
            num_classes: cantidad de clases de salida.

        Returns:
            Modelo Keras compilado (Adam + sparse_categorical_crossentropy).
        """
        # Entrada: 21 landmarks con 3 coordenadas cada uno.
        inputs = tf.keras.Input(shape=(21, 3))
        # Extracción de características por landmark.
        x = tf.keras.layers.Dense(64, activation="relu")(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dense(64, activation="relu")(x)
        # Aplanar y clasificar con capas densas regularizadas con Dropout.
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(256, activation="relu")(x)
        x = tf.keras.layers.Dropout(0.4)(x)
        x = tf.keras.layers.Dense(128, activation="relu")(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        # Salida softmax: probabilidad por cada clase (seña).
        outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
        model = tf.keras.Model(inputs, outputs)
        # Compilar con Adam y entropía cruzada para etiquetas enteras.
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model

    def _create_dynamic_model(self, sequence_length: int, num_classes: int) -> tf.keras.Model:
        """Construye y compila la red recurrente del clasificador dinámico.

        Arquitectura: proyección densa aplicada a cada cuadro (TimeDistributed),
        dos capas LSTM bidireccionales para modelar la dinámica temporal en
        ambos sentidos, y capas densas finales con Dropout hasta la salida
        softmax.

        Args:
            sequence_length: cantidad de cuadros por secuencia (T).
            num_classes: cantidad de clases de salida (incluye "no seña").

        Returns:
            Modelo Keras compilado (Adam + sparse_categorical_crossentropy).
        """
        # Entrada: secuencia de T cuadros, cada uno con 21 landmarks 3D.
        inputs = tf.keras.Input(shape=(sequence_length, 21, 3))
        # Procesar cada cuadro por separado y aplanarlo a un vector.
        x = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(64, activation="relu"))(inputs)
        x = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten())(x)
        # Dos LSTM bidireccionales capturan la evolución temporal del gesto.
        x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True))(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32))(x)
        # Cabezal de clasificación con regularización.
        x = tf.keras.layers.Dense(64, activation="relu")(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        # Salida softmax: probabilidad por cada clase.
        outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        # Compilar con Adam y entropía cruzada para etiquetas enteras.
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model

    def _save_labels(self, path: Path, labels: list[str]) -> None:
        """Guarda la lista de etiquetas en un archivo JSON legible.

        Args:
            path: ruta destino del archivo de labels.
            labels: nombres de las clases, en el orden usado por el modelo.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(labels, ensure_ascii=False, indent=2), encoding="utf-8")

    def _build_result(
        self,
        model_type: Literal["static", "dynamic"],
        history,
        model_path: Path,
        labels_path: Path,
        samples: int,
        classes: int,
    ) -> TrainingResult:
        """Arma el `TrainingResult` exitoso a partir del historial de Keras.

        Args:
            model_type: "static" o "dynamic".
            history: objeto History devuelto por `model.fit`.
            model_path: ruta donde se guardó el modelo.
            labels_path: ruta donde se guardaron las etiquetas.
            samples: cantidad de muestras entrenadas.
            classes: cantidad de clases del modelo.

        Returns:
            `TrainingResult` con las métricas finales del entrenamiento.
        """
        # Tomar la exactitud de la última época (entrenamiento y validación).
        accuracy = float(history.history.get("accuracy", [0.0])[-1])
        validation_values = history.history.get("val_accuracy", [])
        validation_accuracy = float(validation_values[-1]) if validation_values else None
        # Armar el mensaje en español según el tipo de modelo.
        type_name = "estático" if model_type == "static" else "dinámico"
        return TrainingResult(
            True,
            model_type,
            f"Modelo {type_name} entrenado correctamente.",
            model_path=str(model_path),
            labels_path=str(labels_path),
            samples=samples,
            classes=classes,
            accuracy=accuracy,
            validation_accuracy=validation_accuracy,
        )
