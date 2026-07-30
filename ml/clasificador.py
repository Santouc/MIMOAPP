#!/usr/bin/env python3
"""
Módulo de clasificación de señas usando TensorFlow
Responsable de interpretar los landmarks y clasificar las señas
Versión híbrida: combina modelo ML con reglas fallback y logging profesional
"""

import numpy as np
import tensorflow as tf
import json
from pathlib import Path
from typing import List, Tuple, Optional, Dict

from utils.logger import get_logger
from core.preprocessing import normalize_landmarks

logger = get_logger(__name__)

class SignClassifier:
    """
    Clasificador híbrido de señas de lenguaje de señas
    - Resiliencia con reglas fallback
    - Logging profesional
    - Validación robusta
    """
    
    def __init__(self, model_path: Optional[str] = None, labels_path: Optional[str] = None):
        """
        Inicializa el clasificador híbrido de señas
        
        Args:
            model_path: Ruta al modelo entrenado (.h5 o .pb)
            labels_path: Ruta al archivo de etiquetas JSON
        """
            
        # Configurar rutas con valores por defecto
        self.model_path = Path(model_path) if model_path else Path("model.h5")
        self.labels_path = Path(labels_path) if labels_path else Path("labels.json")
        self.input_shape = (21, 3)
        self.num_classes = 0
        
        # Sistema puro basado en modelo - sin reglas fallback
        # El modelo aprende directamente desde landmarks (21×3)
        
        self.model = None
        self.labels = []
        
        self._load_model()
        self._load_labels()
        logger.info("SignClassifier híbrido inicializado")
    
    def _load_model(self):
        """
        Carga modelo Keras con validación (solo formato .h5)
        """
        if self.model_path.exists():
            try:
                # Forzar uso de modelos Keras .h5 para consistencia
                if self.model_path.suffix != '.h5':
                    logger.error("Solo se soportan modelos .h5 para consistencia del sistema")
                    self.model = None
                    return
                
                # Cargar modelo Keras
                self.model = tf.keras.models.load_model(str(self.model_path))
                logger.info(f"Modelo Keras cargado desde {self.model_path}")
                
            except Exception as e:
                logger.error(f"Error cargando modelo Keras: {e}")
                self.model = None
        else:
            logger.warning(f"No se encontró modelo en {self.model_path}")
            self.model = None
    
    def load_model(self, model_path: str) -> bool:
        """
        Carga un modelo pre-entrenado (método público para compatibilidad)
        
        Args:
            model_path: Ruta al archivo del modelo
            
        Returns:
            True si se cargó exitosamente, False en caso contrario
        """
        try:
            self.model_path = Path(model_path)
            self._load_model()
            return self.model is not None
        except Exception as e:
            logger.error(f"Error en load_model: {e}")
            return False
    
    def _load_labels(self):
        """
        Carga etiquetas con encoding UTF-8 (interno)
        """
        if self.labels_path.exists():
            try:
                with open(self.labels_path, "r", encoding='utf-8') as f:
                    self.labels = json.load(f)
                self.num_classes = len(self.labels)
                logger.info(f"Etiquetas cargadas: {len(self.labels)} clases")
            except Exception as e:
                logger.error(f"Error cargando etiquetas: {e}")
                self.labels = []
        else:
            logger.warning(f"No se encontraron etiquetas en {self.labels_path}")
            self.labels = []
    
    def load_labels(self, labels_path: str) -> bool:
        """
        Carga las etiquetas desde un archivo JSON (método público para compatibilidad)
        
        Args:
            labels_path: Ruta al archivo JSON de etiquetas
            
        Returns:
            True si se cargaron exitosamente, False en caso contrario
        """
        try:
            self.labels_path = Path(labels_path)
            self._load_labels()
            return len(self.labels) > 0
        except Exception as e:
            logger.error(f"Error en load_labels: {e}")
            return False
    
    def preprocess(self, landmarks: List[List[Tuple[float, float, float]]]) -> np.ndarray:
        """
        Preprocesamiento usando módulo centralizado
        
        Usa core.preprocessing.normalize_landmarks para consistencia en todo el sistema.
        """
        if landmarks is None or len(landmarks) == 0:
            logger.warning("No se detectaron landmarks")
            return np.zeros((1, 21, 3), dtype=np.float32)
        
        try:
            # Usar módulo centralizado
            normalized = normalize_landmarks(landmarks)
            return normalized
        except Exception as e:
            logger.error(f"Error en preprocess: {e}")
            return np.zeros((1, 21, 3), dtype=np.float32)
    
    
    def classify_with_model(self, landmarks: List[List[Tuple[float, float, float]]]) -> Tuple[str, float]:
        """
        Clasificación usando modelo sin fallback
        
        Retorna siempre (label, confidence) con contrato consistente
        """
        if self.model is None:
            logger.warning("No hay modelo disponible")
            return "unknown", 0.0
        
        try:
            x = self.preprocess(landmarks)
            
            # Validar consistencia de input shape
            if x.shape != (1, 21, 3):
                raise ValueError(f"Input shape inválido: {x.shape}, esperado (1, 21, 3)")
            
            preds = self.model.predict(x, verbose=0)
            
            idx = int(np.argmax(preds[0]))
            confidence = float(np.max(preds[0]))
            
            # Si confianza baja, retornar "unknown" en lugar de None
            if confidence < 0.5:
                logger.info(f"Baja confianza ({confidence:.3f})")
                return "unknown", confidence
            
            if idx < len(self.labels):
                label = self.labels[idx]
                logger.info(f"Predicción modelo: {label} (confianza: {confidence:.3f})")
            else:
                label = "unknown"
                logger.warning(f"Índice {idx} fuera de rango de etiquetas ({len(self.labels)} clases)")
            
            return label, confidence
            
        except Exception as e:
            logger.error(f"Error en predicción modelo: {e}")
            return "unknown", 0.0
    
    
    def classify(self, landmarks: List[List[Tuple[float, float, float]]]) -> Tuple[str, float]:
        """
        Clasificación pura basada en modelo
        
        Retorna siempre (label, confidence) donde:
        - label ∈ labels ∪ {"unknown"}
        - confidence ∈ [0, 1]
        """
        return self.classify_with_model(landmarks)
    
    
    
    def create_advanced_model(self, num_classes: Optional[int] = None, use_geometric: bool = False) -> tf.keras.Model:
        """
        Crea modelo avanzado de alta capacidad para clasificación de señas
        
        Arquitecturas disponibles:
        - Baseline MLP: landmarks → flatten → dense → softmax
        - Structured: landmarks → point-wise → global → softmax
        - Geometric: landmarks + derived features → dense → softmax
        
        Args:
            num_classes: Número de clases de salida
            use_geometric: Si usar características geométricas derivadas
        
        Returns:
            Modelo TensorFlow compilado
        """
        # Usar número de clases de etiquetas si no se especifica
        if num_classes is None:
            num_classes = len(self.labels) if self.labels else 9  # Default para A-I
        
        if use_geometric:
            # ====================
            # MODELO CON CARACTERÍSTICAS GEOMÉTRICAS
            # ====================
            # Input: características concatenadas (~100+ dimensiones)
            input_shape = (None,)  # Se determinará dinámicamente
            
            model = tf.keras.Sequential([
                tf.keras.layers.Input(shape=input_shape),
                # Primera capa densa para procesar características de alta dimensionalidad
                tf.keras.layers.Dense(256, activation='relu'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dropout(0.4),
                
                # Segunda capa para aprender interacciones complejas
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dropout(0.3),
                
                # Capa de compactación
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dropout(0.2),
                
                # Capa de salida
                tf.keras.layers.Dense(num_classes, activation='softmax')
            ])
            
        else:
            # ====================
            # MODELO ESTRUCTURADO (PREFERIDO)
            # ====================
            # Input: (21, 3) landmarks
            
            # Capa de entrada
            inputs = tf.keras.layers.Input(shape=(21, 3))
            
            # Capas point-wise: procesa cada landmark individualmente
            x = tf.keras.layers.Dense(64, activation='relu')(inputs)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Dropout(0.1)(x)
            
            # Segunda capa point-wise
            x = tf.keras.layers.Dense(64, activation='relu')(x)
            x = tf.keras.layers.BatchNormalization()(x)
            
            # Aplanar para procesamiento global
            x = tf.keras.layers.Flatten()(x)
            
            # Capas densas globales
            x = tf.keras.layers.Dense(256, activation='relu')(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Dropout(0.4)(x)
            
            x = tf.keras.layers.Dense(128, activation='relu')(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Dropout(0.3)(x)
            
            # Capa de salida
            outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
            
            model = tf.keras.Model(inputs=inputs, outputs=outputs)
        
        # ====================
        # COMPILACIÓN AVANZADA
        # ====================
        
        # Optimizador con learning rate adaptativo
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        
        # Compilar con métricas adicionales
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=[
                'accuracy',
                tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='top3_accuracy')
            ]
        )
        
        model_type = "geométrico" if use_geometric else "estructurado"
        logger.info(f"Modelo avanzado {model_type} creado con {num_classes} clases")
        
        return model
    
    def create_training_model(self, num_classes: int) -> tf.keras.Model:
        """
        Crea modelo optimizado para entrenamiento según especificación
        
        Arquitectura: (21,3) → Dense(64) → BatchNorm → Dense(64) → Flatten → 
                    Dense(256) → Dropout → Dense(128) → Dropout → Dense(num_classes)
        """
        inputs = tf.keras.Input(shape=(21, 3))
        
        # Capas point-wise
        x = tf.keras.layers.Dense(64, activation='relu')(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dense(64, activation='relu')(x)
        
        # Aplanar para capas globales
        x = tf.keras.layers.Flatten()(x)
        
        # Capas densas profundas
        x = tf.keras.layers.Dense(256, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.4)(x)
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        
        # Salida
        outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
        
        model = tf.keras.Model(inputs, outputs)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-3),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        logger.info(f"Modelo de entrenamiento creado con {num_classes} clases")
        return model
    
    def preprocess_batch(self, X: np.ndarray) -> np.ndarray:
        """
        Preprocesamiento batch usando módulo centralizado
        
        Args:
            X: Array de landmarks shape (N, 21, 3)
            
        Returns:
            Array normalizado shape (N, 21, 3)
        """
        try:
            # Usar módulo centralizado
            normalized = normalize_landmarks(X)
            return normalized
        except Exception as e:
            logger.error(f"Error en preprocess_batch: {e}")
            return X.astype(np.float32)
    
    def train_model(self, X: np.ndarray, y: np.ndarray, 
                   validation_split: float = 0.2, 
                   epochs: int = 50, 
                   batch_size: int = 32,
                   save_path: str = "model.h5") -> Dict:
        """
        Entrena el modelo con datos proporcionados
        
        Args:
            X: Array de landmarks shape (N, 21, 3)
            y: Array de etiquetas shape (N,)
            validation_split: Fracción para validación
            epochs: Número de épocas
            batch_size: Tamaño de batch
            save_path: Ruta para guardar modelo
            
        Returns:
            Diccionario con historial de entrenamiento
        """
        from sklearn.model_selection import train_test_split
        
        # Validar datos de entrada
        if X.shape[1:] != (21, 3):
            raise ValueError(f"Shape de X inválido: {X.shape}, esperado (N, 21, 3)")
        
        # División train/val
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, stratify=y, random_state=42
        )
        
        # Preprocesamiento
        X_train_processed = self.preprocess_batch(X_train)
        X_val_processed = self.preprocess_batch(X_val)
        
        # Crear modelo
        num_classes = len(np.unique(y))
        model = self.create_training_model(num_classes)
        
        # Callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                patience=5, restore_best_weights=True, monitor='val_loss'
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                patience=3, factor=0.5, monitor='val_loss'
            )
        ]
        
        # Entrenamiento
        logger.info(f"Iniciando entrenamiento: {X_train_processed.shape[0]} muestras, {num_classes} clases")
        
        history = model.fit(
            X_train_processed, y_train,
            validation_data=(X_val_processed, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        # Guardar artefactos
        model.save(save_path)
        logger.info(f"Modelo guardado en {save_path}")
        
        # Actualizar modelo actual
        self.model = model
        self.model_path = Path(save_path)
        
        return {
            'history': history.history,
            'train_samples': X_train_processed.shape[0],
            'val_samples': X_val_processed.shape[0],
            'num_classes': num_classes,
            'model_path': save_path
        }
    
    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """
        Evalúa el modelo con métricas completas
        
        Args:
            X_test: Datos de prueba shape (N, 21, 3)
            y_test: Etiquetas de prueba shape (N,)
            
        Returns:
            Diccionario con métricas de evaluación
        """
        if self.model is None:
            raise ValueError("No hay modelo cargado para evaluación")
        
        # Preprocesar datos
        X_test_processed = self.preprocess_batch(X_test)
        
        # Predicciones
        y_pred_proba = self.model.predict(X_test_processed, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # Métricas
        from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
        
        accuracy = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=self.labels, output_dict=True)
        
        # Detectar confusiones problemáticas
        problematic_pairs = []
        for i in range(len(cm)):
            for j in range(len(cm)):
                if i != j and cm[i, j] > 5:  # Más de 5 confusiones
                    problematic_pairs.append((self.labels[i], self.labels[j], cm[i, j]))
        
        logger.info(f"Evaluación completada: accuracy={accuracy:.3f}")
        if problematic_pairs:
            logger.warning(f"Pares problemáticos detectados: {problematic_pairs}")
        
        return {
            'accuracy': accuracy,
            'confusion_matrix': cm.tolist(),
            'classification_report': report,
            'problematic_pairs': problematic_pairs,
            'test_samples': X_test.shape[0]
        }
    
    def save_training_data(self, X: np.ndarray, y: np.ndarray, 
                          labels: List[str], data_path: str = "training_data.json") -> bool:
        """
        Serializa datos de entrenamiento en formato recomendado
        
        Args:
            X: Array de landmarks shape (N, 21, 3)
            y: Array de etiquetas shape (N,)
            labels: Lista de nombres de clases
            data_path: Ruta para guardar datos
            
        Returns:
            True si se guardó exitosamente
        """
        try:
            dataset = {
                "X": X.tolist(),
                "y": y.tolist(),
                "labels": labels,
                "metadata": {
                    "num_samples": X.shape[0],
                    "num_classes": len(labels),
                    "landmark_format": "mediapipe_21_3d",
                    "coordinate_system": "normalized"
                }
            }
            
            with open(data_path, 'w') as f:
                json.dump(dataset, f, indent=2)
            
            logger.info(f"Datos de entrenamiento guardados en {data_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error guardando datos: {e}")
            return False
    
    def load_training_data(self, data_path: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Carga datos de entrenamiento serializados
        
        Args:
            data_path: Ruta al archivo de datos
            
        Returns:
            Tuple (X, y, labels)
        """
        try:
            with open(data_path, 'r') as f:
                dataset = json.load(f)
            
            X = np.array(dataset['X'], dtype=np.float32)
            y = np.array(dataset['y'], dtype=np.int32)
            labels = dataset['labels']
            
            logger.info(f"Datos cargados: {X.shape[0]} muestras, {len(labels)} clases")
            return X, y, labels
            
        except Exception as e:
            logger.error(f"Error cargando datos: {e}")
            return np.array([]), np.array([]), []
    
    def save_model(self, model_path: str) -> bool:
        """
        Guarda el modelo entrenado
        
        Args:
            model_path: Ruta donde guardar el modelo
            
        Returns:
            True si se guardó exitosamente
        """
        try:
            # Verificar que haya un modelo cargado para guardar
            if self.model is None:
                logger.error("No hay modelo para guardar")
                return False
            
            # Guardar el modelo en la ruta especificada
            self.model.save(model_path)
            logger.info(f"Modelo guardado en: {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error guardando modelo: {e}")
            return False
    
    def save_labels(self, labels_path: str) -> bool:
        """
        Guarda las etiquetas en formato JSON
        
        Args:
            labels_path: Ruta donde guardar las etiquetas
            
        Returns:
            True si se guardaron exitosamente
        """
        try:
            # Abrir archivo en modo escritura con encoding UTF-8
            with open(labels_path, 'w', encoding='utf-8') as f:
                # Guardar etiquetas como JSON con indentación y sin escapar caracteres ASCII
                json.dump(self.labels, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Etiquetas guardadas en: {labels_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error guardando etiquetas: {e}")
            return False
    
    def get_class_info(self) -> Dict:
        """
        Información del clasificador (sistema puro basado en modelo)
        """
        return {
            "num_classes": len(self.labels),  # Número total de clases disponibles
            "classes": self.labels,  # Lista de nombres de las clases
            "input_shape": self.input_shape,  # Formato esperado de entrada (21, 3)
            "model_loaded": self.model is not None,  # Si hay modelo ML cargado
            "has_fallback_rules": False,  # Sistema puro - sin reglas fallback
            "confidence_threshold": 0.5  # Umbral de confianza para aceptar predicciones
        }


# -----------------------------
# INTERFAZ EXTERNA CON PERSISTENCIA
# -----------------------------

# Instancia global persistente para evitar recreación
_global_classifier = None
_global_config = None

def predict_sign(landmarks, model_path: str = "model.h5", labels_path: str = "labels.json") -> Dict:
    """
    Función externa con respuesta estructurada y persistencia de instancia
    """
    global _global_classifier, _global_config
    
    try:
        # Verificar si necesitamos crear/recrear el clasificador
        current_config = (model_path, labels_path)
        
        if _global_classifier is None or _global_config != current_config:
            logger.info("Creando nueva instancia de SignClassifier")
            # Crear instancia del clasificador con rutas especificadas
            _global_classifier = SignClassifier(model_path=model_path, labels_path=labels_path)
            _global_config = current_config
        else:
            logger.debug("Reusando instancia existente de SignClassifier")
        
        # Validación robusta de entrada
        if landmarks is None:
            raise ValueError("landmarks no puede ser None")
        
        if not isinstance(landmarks, list) or len(landmarks) == 0:
            raise ValueError("landmarks debe ser una lista no vacía")
        
        if not all(isinstance(hand, list) and len(hand) == 21 for hand in landmarks):
            raise ValueError("Cada mano debe tener exactamente 21 landmarks")
        
        # Realizar clasificación híbrida
        sign, confidence = _global_classifier.classify(landmarks)
        
        # Construir respuesta estructurada
        return {
            "sign": sign,  # Letra o seña detectada
            "confidence": confidence,  # Nivel de confianza (0.0 a 1.0)
            "success": sign != "unknown",  # Si se detectó algo válido
            "method": "model",  # Siempre usa modelo
            "classifier_info": _global_classifier.get_class_info()  # Información del clasificador
        }
    except Exception as e:
        # Manejo de errores con respuesta estructurada
        logger.error(f"Error en predict_sign: {e}")
        return {
            "sign": "unknown",
            "confidence": 0.0,
            "success": False,
            "error": str(e),
            "method": "error"
        }


if __name__ == "__main__":
    # ====================
    # PRUEBA BÁSICA DEL CLASIFICADOR
    # ====================
    
    # Crear datos de prueba simulados
    test_landmarks = [[
        (0.1, 0.2, 0.0),  # muñeca (landmark 0)
        (0.15, 0.18, 0.02),  # pulgar base (landmark 1)
        (0.2, 0.16, 0.01),   # pulgar medio (landmark 2)
    ]]
    
    # Completar con datos aleatorios hasta tener 21 landmarks
    while len(test_landmarks[0]) < 21:
        # Generar coordenadas aleatorias dentro de rangos realistas
        test_landmarks[0].append((
            np.random.random(),  # X entre 0 y 1
            np.random.random(),  # Y entre 0 y 1
            np.random.random() * 0.1  # Z entre 0 y 0.1 (menor profundidad)
        ))
    
    # Probar la función externa de predicción
    result = predict_sign(test_landmarks)
    
    # Mostrar resultado de la prueba
    print("Resultado prueba híbrida:", result)
