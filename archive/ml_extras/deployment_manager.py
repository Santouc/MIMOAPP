#!/usr/bin/env python3
"""
Sistema de deployment para múltiples plataformas
Desktop, Edge (Raspberry Pi/Jetson), Mobile (Android)
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

# Importaciones de TensorFlow
try:
    import tensorflow as tf
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False

# Importaciones de TFLite Runtime
try:
    import tflite_runtime.interpreter as tflite
    HAS_TFLITE_RUNTIME = True
except ImportError:
    HAS_TFLITE_RUNTIME = False

from utils.logger import get_logger
from core.preprocessing import normalize_landmarks

logger = get_logger(__name__)

class ModelExporter:
    """Sistema de exportación de modelos para diferentes plataformas"""
    
    def __init__(self, model, labels: List[str]):
        """
        Inicializa exportador de modelos
        
        Args:
            model: Modelo TensorFlow/Keras entrenado
            labels: Lista de nombres de clases
        """
        self.model = model
        self.labels = labels
        
        if not HAS_TENSORFLOW:
            raise ImportError("TensorFlow es requerido para exportación")
        
        logger.info(f"ModelExporter inicializado con {len(labels)} clases")
    
    def export_savedmodel(self, output_dir: str) -> bool:
        """
        Exporta modelo a formato SavedModel
        
        Args:
            output_dir: Directorio de salida
            
        Returns:
            True si exportación exitosa
        """
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            # Guardar como SavedModel
            self.model.save(output_dir)
            
            # Guardar etiquetas
            labels_path = os.path.join(output_dir, "labels.json")
            with open(labels_path, 'w') as f:
                json.dump(self.labels, f, indent=2)
            
            logger.info(f"Modelo exportado como SavedModel en {output_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Error exportando SavedModel: {e}")
            return False
    
    def export_tflite(self, output_path: str, quantize: bool = True) -> bool:
        """
        Exporta modelo a formato TensorFlow Lite
        
        Args:
            output_path: Ruta del archivo .tflite
            quantize: Si aplicar cuantización
            
        Returns:
            True si exportación exitosa
        """
        try:
            # Crear conversor
            converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
            
            # Optimizaciones
            if quantize:
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                # Para cuantización int8 necesitas dataset representativo
                # converter.representative_dataset = self.representative_dataset
            
            # Convertir modelo
            tflite_model = converter.convert()
            
            # Guardar modelo
            with open(output_path, 'wb') as f:
                f.write(tflite_model)
            
            # Guardar etiquetas
            labels_path = output_path.replace('.tflite', '_labels.json')
            with open(labels_path, 'w') as f:
                json.dump(self.labels, f, indent=2)
            
            size_mb = len(tflite_model) / (1024 * 1024)
            logger.info(f"Modelo TFLite guardado en {output_path} ({size_mb:.2f} MB)")
            return True
            
        except Exception as e:
            logger.error(f"Error exportando TFLite: {e}")
            return False
    
    def export_h5(self, output_path: str) -> bool:
        """
        Exporta modelo a formato H5
        
        Args:
            output_path: Ruta del archivo .h5
            
        Returns:
            True si exportación exitosa
        """
        try:
            self.model.save(output_path)
            
            # Guardar etiquetas
            labels_path = output_path.replace('.h5', '_labels.json')
            with open(labels_path, 'w') as f:
                json.dump(self.labels, f, indent=2)
            
            logger.info(f"Modelo H5 guardado en {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error exportando H5: {e}")
            return False
    
    def export_all_formats(self, base_dir: str = "deployment") -> Dict[str, bool]:
        """
        Exporta modelo en todos los formatos soportados
        
        Args:
            base_dir: Directorio base para exportación
            
        Returns:
            Diccionario con éxito de cada formato
        """
        os.makedirs(base_dir, exist_ok=True)
        
        results = {}
        
        # H5
        h5_path = os.path.join(base_dir, "model.h5")
        results['h5'] = self.export_h5(h5_path)
        
        # SavedModel
        savedmodel_dir = os.path.join(base_dir, "saved_model")
        results['savedmodel'] = self.export_savedmodel(savedmodel_dir)
        
        # TFLite
        tflite_path = os.path.join(base_dir, "model.tflite")
        results['tflite'] = self.export_tflite(tflite_path, quantize=True)
        
        logger.info(f"Exportación completada: {results}")
        return results


class DesktopRuntime:
    """Runtime de deployment para Desktop (Python)"""
    
    def __init__(self, model_path: str, labels_path: str):
        """
        Inicializa runtime desktop
        
        Args:
            model_path: Ruta del modelo (.h5 o SavedModel)
            labels_path: Ruta de etiquetas JSON
        """
        if not HAS_TENSORFLOW:
            raise ImportError("TensorFlow es requerido para Desktop Runtime")
        
        # Cargar modelo
        if model_path.endswith('.h5'):
            self.model = tf.keras.models.load_model(model_path)
        else:
            self.model = tf.saved_model.load(model_path)
        
        # Cargar etiquetas
        with open(labels_path, 'r') as f:
            self.labels = json.load(f)
        
        logger.info(f"DesktopRuntime inicializado con {len(self.labels)} clases")
    
    def preprocess(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Preprocesamiento usando módulo centralizado
        
        Args:
            landmarks: Array (21, 3) de landmarks
            
        Returns:
            Array (1, 21, 3) preprocesado
        """
        try:
            # Usar módulo centralizado
            normalized = normalize_landmarks(landmarks)
            return normalized
        except Exception as e:
            logger.error(f"Error en preprocess: {e}")
            return np.zeros((1, 21, 3), dtype=np.float32)
    
    def predict(self, landmarks: np.ndarray) -> Tuple[str, float]:
        """
        Predicción minimal y rápida
        
        Args:
            landmarks: Array (21, 3) de landmarks
            
        Returns:
            Tuple (label, confidence)
        """
        x = self.preprocess(landmarks)
        probs = self.model(x, training=False).numpy()[0]
        
        idx = np.argmax(probs)
        confidence = probs[idx]
        
        if confidence < 0.6:
            return "unknown", confidence
        else:
            return self.labels[idx], confidence


class EdgeRuntime:
    """Runtime de deployment para Edge devices (TFLite)"""
    
    def __init__(self, model_path: str, labels_path: str):
        """
        Inicializa runtime edge
        
        Args:
            model_path: Ruta del modelo TFLite
            labels_path: Ruta de etiquetas JSON
        """
        if not HAS_TFLITE_RUNTIME:
            raise ImportError("TFLite Runtime es requerido para Edge Runtime")
        
        # Cargar intérprete TFLite
        self.interpreter = tflite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        
        # Obtener detalles de entrada/salida
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # Cargar etiquetas
        with open(labels_path, 'r') as f:
            self.labels = json.load(f)
        
        logger.info(f"EdgeRuntime inicializado con {len(self.labels)} clases")
        logger.info(f"Input shape: {self.input_details[0]['shape']}")
    
    def preprocess(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Preprocesamiento usando módulo centralizado
        
        Args:
            landmarks: Array (21, 3) de landmarks
            
        Returns:
            Array preprocesado con shape correcto
        """
        try:
            # Usar módulo centralizado
            normalized = normalize_landmarks(landmarks)
            
            # Ajustar shape según input del modelo
            input_shape = self.input_details[0]['shape']
            if len(input_shape) == 3:
                return normalized.reshape(1, 21, 3).astype(np.float32)
            else:
                return normalized.flatten().astype(np.float32)
        except Exception as e:
            logger.error(f"Error en preprocess: {e}")
            return np.zeros((1, 21, 3), dtype=np.float32)
    
    def predict(self, landmarks: np.ndarray) -> Tuple[str, float]:
        """
        Predicción con TFLite
        
        Args:
            landmarks: Array (21, 3) de landmarks
            
        Returns:
            Tuple (label, confidence)
        """
        x = self.preprocess(landmarks)
        
        # Setear tensor de entrada
        self.interpreter.set_tensor(self.input_details[0]['index'], x)
        
        # Invocar inferencia
        self.interpreter.invoke()
        
        # Obtener salida
        probs = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        
        idx = np.argmax(probs)
        confidence = probs[idx]
        
        if confidence < 0.6:
            return "unknown", confidence
        else:
            return self.labels[idx], confidence




class UnifiedDeploymentPipeline:
    """Pipeline unificado de deployment para todas las plataformas"""
    
    def __init__(self, model_path: str, labels_path: str, 
                 platform: str = "desktop"):
        """
        Inicializa pipeline unificado
        
        Args:
            model_path: Ruta del modelo
            labels_path: Ruta de etiquetas
            platform: Plataforma ("desktop", "edge")
        """
        self.platform = platform
        
        # Inicializar runtime según plataforma
        if platform == "desktop":
            self.runtime = DesktopRuntime(model_path, labels_path)
        elif platform == "edge":
            self.runtime = EdgeRuntime(model_path, labels_path)
        else:
            raise ValueError(f"Plataforma no soportada: {platform}")
        
        logger.info(f"UnifiedDeploymentPipeline inicializado para {platform}")
    
    def predict(self, landmarks: np.ndarray) -> Tuple[str, float]:
        """
        Predicción usando el runtime configurado
        
        Args:
            landmarks: Array (21, 3) de landmarks
            
        Returns:
            Tuple (label, confidence)
        """
        return self.runtime.predict(landmarks)
    
    def cleanup(self):
        """Limpieza de recursos"""
        logger.info("Pipeline limpiado")




def main():
    """Función principal para demostración"""
    print("=== Sistema de Deployment Multiplataforma ===")
    print("Este sistema permite deployment en Desktop, Edge y Mobile.")
    print()
    
    print("Componentes implementados:")
    print("✅ Exportación de modelos (SavedModel, TFLite, H5)")
    print("✅ Runtime Desktop (TensorFlow)")
    print("✅ Runtime Edge (TFLite)")
    print("✅ Pipeline unificado")
    print()
    
    print("Para deployment:")
    print("1. Exportar modelo: ModelExporter.export_all_formats()")
    print("2. Elegir runtime: DesktopRuntime() o EdgeRuntime()")
    print("3. O usar pipeline unificado: UnifiedDeploymentPipeline()")
    print()
    
    print("Ejemplo Desktop:")
    print("runtime = DesktopRuntime('model.h5', 'labels.json')")
    print("label, conf = runtime.predict(landmarks)")
    print()
    
    print("Ejemplo Edge:")
    print("runtime = EdgeRuntime('model.tflite', 'labels.json')")
    print("label, conf = runtime.predict(landmarks)")


if __name__ == "__main__":
    main()
