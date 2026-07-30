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
    def __init__(self, paths: PathService | None = None):
        self.paths = paths or PathService()
        self.paths.ensure_app_dirs()

    def train(self, capture_type: Literal["static", "dynamic"]) -> TrainingResult:
        if capture_type == "static":
            return self.train_static()
        if capture_type == "dynamic":
            return self.train_dynamic()
        raise ValueError("El tipo de entrenamiento debe ser 'static' o 'dynamic'")

    def train_static(self) -> TrainingResult:
        if not self.paths.static_dataset_path.exists():
            return TrainingResult(False, "static", "No existe dataset estático para entrenar.")

        X, y, labels = load_dataset(str(self.paths.static_dataset_path))
        validation = self._validate_dataset(y, labels, "static")
        if validation is not None:
            return validation

        X_processed = normalize_landmarks(X)
        X_processed, y = self._shuffle_samples(X_processed, y)
        model = self._create_static_model(len(labels))
        validation_split = 0.2 if len(X_processed) >= 10 else 0.0
        history = model.fit(
            X_processed,
            y,
            epochs=50,
            batch_size=min(8, len(X_processed)),
            validation_split=validation_split,
            verbose=0,
        )

        self.paths.static_model_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(self.paths.static_model_path)
        self._save_labels(self.paths.static_labels_path, labels)
        return self._build_result("static", history, self.paths.static_model_path, self.paths.static_labels_path, len(X_processed), len(labels))

    def train_dynamic(self) -> TrainingResult:
        if not self.paths.dynamic_dataset_path.exists():
            return TrainingResult(False, "dynamic", "No existe dataset dinámico para entrenar.")

        X, y, labels = self._load_dynamic_dataset(self.paths.dynamic_dataset_path)
        if len(y) == 0:
            return TrainingResult(False, "dynamic", "El dataset dinámico está vacío.")

        negatives = self._generate_dynamic_negatives(X.shape[1], int(np.bincount(y).max()))
        if len(negatives) > 0:
            training_labels = list(labels) + [NO_SIGN_LABEL]
            X = np.concatenate([X, negatives])
            y = np.concatenate([y, np.full(len(negatives), len(labels), dtype=np.int32)])
        else:
            training_labels = list(labels)
            validation = self._validate_dataset(y, labels, "dynamic")
            if validation is not None:
                return validation

        X_processed = normalize_dynamic_sequences(X)
        X_processed, y = self._shuffle_samples(X_processed, y)
        model = self._create_dynamic_model(X_processed.shape[1], len(training_labels))
        validation_split = 0.2 if len(X_processed) >= 10 else 0.0
        history = model.fit(
            X_processed,
            y,
            epochs=80,
            batch_size=min(8, len(X_processed)),
            validation_split=validation_split,
            verbose=0,
        )

        self.paths.dynamic_model_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(self.paths.dynamic_model_path)
        self._save_labels(self.paths.dynamic_labels_path, training_labels)
        return self._build_result("dynamic", history, self.paths.dynamic_model_path, self.paths.dynamic_labels_path, len(X_processed), len(training_labels))

    def _shuffle_samples(self, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        order = np.random.default_rng(42).permutation(len(X))
        return X[order], np.asarray(y)[order]

    def _validate_dataset(self, y: np.ndarray, labels: list[str], model_type: Literal["static", "dynamic"]) -> TrainingResult | None:
        if len(y) == 0:
            return TrainingResult(False, model_type, "El dataset está vacío.")
        unique_classes = np.unique(y)
        if len(unique_classes) < 2:
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
        with path.open("r", encoding="utf-8") as file:
            dataset = json.load(file)
        X = np.array(dataset.get("X", []), dtype=np.float32)
        y = np.array(dataset.get("y", []), dtype=np.int32)
        labels = dataset.get("labels", [])
        if len(X) == 0:
            return np.zeros((0, 20, 21, 3), dtype=np.float32), np.zeros((0,), dtype=np.int32), labels
        if X.ndim != 4 or X.shape[2:] != (21, 3):
            raise ValueError(f"Shape inválido de dataset dinámico: {X.shape}, esperado (N, T, 21, 3)")
        if len(X) != len(y):
            raise ValueError(f"Cantidad inconsistente de muestras: X={len(X)}, y={len(y)}")
        return X, y, labels

    def _generate_dynamic_negatives(self, timesteps: int, max_class_count: int) -> np.ndarray:
        empty = np.zeros((0, timesteps, 21, 3), dtype=np.float32)
        if not self.paths.static_dataset_path.exists():
            return empty
        try:
            X, y, _ = load_dataset(str(self.paths.static_dataset_path))
        except Exception:
            return empty
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)
        if len(X) == 0:
            return empty

        rng = np.random.default_rng(7)
        total = int(np.clip(round(max_class_count * 1.2), 6, 60))
        negatives = []
        while len(negatives) < total:
            kind = len(negatives) % 3
            base_idx = int(rng.integers(len(X)))
            base = X[base_idx]
            if kind == 0:
                frames = self._negative_hold(base, timesteps, rng)
            elif kind == 1:
                other_indices = np.flatnonzero(y != y[base_idx])
                other_idx = int(rng.choice(other_indices)) if len(other_indices) else int(rng.integers(len(X)))
                frames = self._negative_transition(base, X[other_idx], timesteps, rng)
            else:
                frames = self._negative_drift(base, timesteps, rng)
            negatives.append(frames.astype(np.float32))
        return np.asarray(negatives, dtype=np.float32)

    def _negative_hold(self, base: np.ndarray, timesteps: int, rng: np.random.Generator) -> np.ndarray:
        drift = rng.normal(0.0, 0.008, size=3) * np.array([1.0, 1.0, 0.2])
        progress = np.linspace(0.0, 1.0, timesteps)[:, None]
        path = progress * drift[None, :]
        jitter = rng.normal(0.0, 0.002, size=(timesteps, 21, 3))
        return base[None, :, :] + path[:, None, :] + jitter

    def _negative_transition(self, start: np.ndarray, end: np.ndarray, timesteps: int, rng: np.random.Generator) -> np.ndarray:
        t = np.linspace(0.0, 1.0, timesteps)
        ease = t * t * (3.0 - 2.0 * t)
        frames = start[None, :, :] * (1.0 - ease)[:, None, None] + end[None, :, :] * ease[:, None, None]
        offset = np.clip(rng.normal(0.0, 0.05, size=3), -0.15, 0.15) * np.array([1.0, 1.0, 0.2])
        path = ease[:, None] * offset[None, :]
        jitter = rng.normal(0.0, 0.002, size=(timesteps, 21, 3))
        return frames + path[:, None, :] + jitter

    def _negative_drift(self, base: np.ndarray, timesteps: int, rng: np.random.Generator) -> np.ndarray:
        steps = rng.normal(0.0, 1.0, size=(timesteps, 3)) * np.array([1.0, 1.0, 0.2])
        path = np.cumsum(steps, axis=0)
        path = path - path[0]
        span = float(np.max(np.linalg.norm(path[:, :2], axis=1)))
        if span < 1e-6:
            span = 1.0
        path = path * (rng.uniform(0.08, 0.3) / span)
        jitter = rng.normal(0.0, 0.002, size=(timesteps, 21, 3))
        return base[None, :, :] + path[:, None, :] + jitter

    def _create_static_model(self, num_classes: int) -> tf.keras.Model:
        inputs = tf.keras.Input(shape=(21, 3))
        x = tf.keras.layers.Dense(64, activation="relu")(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dense(64, activation="relu")(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(256, activation="relu")(x)
        x = tf.keras.layers.Dropout(0.4)(x)
        x = tf.keras.layers.Dense(128, activation="relu")(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
        model = tf.keras.Model(inputs, outputs)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model

    def _create_dynamic_model(self, sequence_length: int, num_classes: int) -> tf.keras.Model:
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
            metrics=["accuracy"],
        )
        return model

    def _save_labels(self, path: Path, labels: list[str]) -> None:
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
        accuracy = float(history.history.get("accuracy", [0.0])[-1])
        validation_values = history.history.get("val_accuracy", [])
        validation_accuracy = float(validation_values[-1]) if validation_values else None
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
