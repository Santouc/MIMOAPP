#!/usr/bin/env python3
"""
Sistema de captura automatizada de dataset para entrenamiento de señas
Genera datos etiquetados X ∈ ℝ^(N,21,3), y ∈ ℕ^(N) usando MediaPipe
"""

import cv2
import mediapipe as mp
import numpy as np
import json
import time
from typing import List, Tuple, Dict
from pathlib import Path
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DatasetCapture:
    """Sistema de captura de dataset de señas con MediaPipe"""
    
    def __init__(self, labels: List[str] = None, min_samples_per_class: int = 200):
        """
        Inicializa el sistema de captura
        
        Args:
            labels: Lista de etiquetas de clases
            min_samples_per_class: Mínimo de muestras por clase requerido
        """
        self.labels = labels or ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
                                 "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
        self.min_samples_per_class = min_samples_per_class
        self.current_label_idx = 0
        
        # Inicializar MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Almacenamiento de datos
        self.X = []  # landmarks
        self.y = []  # etiquetas
        
        # Control temporal para evitar frames idénticos
        self.last_capture_time = 0
        self.min_capture_interval = 0.3  # segundos
        
        # Estadísticas
        self.samples_per_class = {label: 0 for label in self.labels}
        
        logger.info(f"DatasetCapture inicializado con {len(self.labels)} clases")
        logger.info(f"Objetivo: {min_samples_per_class} muestras por clase")
    
    def extract_landmarks(self, hand_landmarks) -> np.ndarray:
        """
        Extrae landmarks de MediaPipe en formato (21, 3)
        
        Args:
            hand_landmarks: Objeto landmarks de MediaPipe
            
        Returns:
            Array numpy shape (21, 3) con coordenadas [x, y, z]
        """
        sample = []
        for lm in hand_landmarks.landmark:
            sample.append([lm.x, lm.y, lm.z])
        return np.array(sample, dtype=np.float32)
    
    def validate_sample(self, sample: np.ndarray) -> bool:
        """
        Valida calidad de la muestra capturada
        
        Args:
            sample: Array (21, 3) de landmarks
            
        Returns:
            True si la muestra es válida
        """
        # Verificar dimensiones
        if sample.shape != (21, 3):
            return False
        
        # Verificar que no haya valores NaN o infinitos
        if not np.all(np.isfinite(sample)):
            return False
        
        # Verificar rango de coordenadas (MediaPipe usa [0, 1])
        if not np.all((sample >= 0) & (sample <= 1)):
            return False
        
        return True
    
    def capture_sample(self, sample: np.ndarray) -> bool:
        """
        Captura una muestra con control temporal
        
        Args:
            sample: Array (21, 3) de landmarks
            
        Returns:
            True si se capturó exitosamente
        """
        current_time = time.time()
        
        # Control de espaciado temporal
        if current_time - self.last_capture_time < self.min_capture_interval:
            return False
        
        # Validar muestra
        if not self.validate_sample(sample):
            logger.warning("Muestra inválida descartada")
            return False
        
        # Guardar muestra
        self.X.append(sample.tolist())
        self.y.append(self.current_label_idx)
        
        # Actualizar estadísticas
        current_label = self.labels[self.current_label_idx]
        self.samples_per_class[current_label] += 1
        self.last_capture_time = current_time
        
        logger.info(f"Muestra capturada: {current_label} ({self.samples_per_class[current_label]}/{self.min_samples_per_class})")
        return True
    
    def next_label(self):
        """Avanza a la siguiente etiqueta"""
        self.current_label_idx = (self.current_label_idx + 1) % len(self.labels)
        current_label = self.labels[self.current_label_idx]
        logger.info(f"Etiqueta cambiada a: {current_label}")
    
    def is_dataset_complete(self) -> bool:
        """
        Verifica si se ha alcanzado el objetivo de muestras por clase
        
        Returns:
            True si el dataset está completo
        """
        return all(count >= self.min_samples_per_class for count in self.samples_per_class.values())
    
    def get_progress(self) -> Dict[str, int]:
        """
        Obtiene progreso actual de captura
        
        Returns:
            Diccionario con progreso por clase
        """
        return self.samples_per_class.copy()
    
    def draw_interface(self, frame: np.ndarray, hand_detected: bool = False) -> np.ndarray:
        """
        Dibuja interfaz de usuario en el frame
        
        Args:
            frame: Frame de video
            hand_detected: Si se detecta mano
            
        Returns:
            Frame con interfaz dibujada
        """
        height, width = frame.shape[:2]
        
        # Etiqueta actual
        current_label = self.labels[self.current_label_idx]
        color = (0, 255, 0) if hand_detected else (0, 0, 255)
        
        cv2.putText(frame, f"Label: {current_label}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        # Progreso
        progress_text = f"Progress: {self.samples_per_class[current_label]}/{self.min_samples_per_class}"
        cv2.putText(frame, progress_text, (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Controles
        controls = [
            "Controls:",
            "S - Save sample",
            "N - Next label", 
            "ESC - Exit"
        ]
        
        for i, text in enumerate(controls):
            cv2.putText(frame, text, (10, height - 100 + i*25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Estado de detección
        status = "Hand detected" if hand_detected else "No hand detected"
        status_color = (0, 255, 0) if hand_detected else (0, 0, 255)
        cv2.putText(frame, status, (width - 250, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        return frame
    
    def save_dataset(self, filepath: str = "dataset.json") -> bool:
        """
        Guarda el dataset capturado
        
        Args:
            filepath: Ruta del archivo JSON
            
        Returns:
            True si se guardó exitosamente
        """
        try:
            dataset = {
                "X": self.X,
                "y": self.y,
                "labels": self.labels,
                "metadata": {
                    "num_samples": len(self.X),
                    "num_classes": len(self.labels),
                    "samples_per_class": self.samples_per_class,
                    "min_samples_per_class": self.min_samples_per_class,
                    "landmark_format": "mediapipe_21_3d",
                    "coordinate_system": "normalized_0_1",
                    "capture_timestamp": time.time()
                }
            }
            
            with open(filepath, 'w') as f:
                json.dump(dataset, f, indent=2)
            
            logger.info(f"Dataset guardado en {filepath}")
            logger.info(f"Total muestras: {len(self.X)}")
            logger.info(f"Muestras por clase: {self.samples_per_class}")
            return True
            
        except Exception as e:
            logger.error(f"Error guardando dataset: {e}")
            return False
    
    def run_capture_session(self, camera_index: int = 0, save_path: str = "dataset.json") -> bool:
        """
        Ejecuta sesión completa de captura
        
        Args:
            camera_index: Índice de cámara
            save_path: Ruta para guardar dataset
            
        Returns:
            True si la sesión se completó exitosamente
        """
        # Inicializar cámara
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            logger.error(f"No se pudo abrir cámara {camera_index}")
            return False
        
        logger.info("Iniciando sesión de captura. Presiona ESC para salir.")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    logger.warning("No se pudo leer frame de cámara")
                    break
                
                # Convertir a RGB para MediaPipe
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.hands.process(frame_rgb)
                
                hand_detected = False
                if results.multi_hand_landmarks:
                    hand_landmarks = results.multi_hand_landmarks[0]
                    sample = self.extract_landmarks(hand_landmarks)
                    hand_detected = True
                    
                    # Dibujar landmarks en el frame
                    mp.solutions.drawing_utils.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                    )
                
                # Dibujar interfaz
                frame = self.draw_interface(frame, hand_detected)
                
                # Mostrar frame
                cv2.imshow("Dataset Capture", frame)
                
                # Manejar input de teclado
                key = cv2.waitKey(1) & 0xFF
                
                if key == 27:  # ESC
                    logger.info("Sesión terminada por usuario")
                    break
                elif key == ord('s') and hand_detected:  # Save sample
                    if self.capture_sample(sample):
                        # Verificar si dataset está completo
                        if self.is_dataset_complete():
                            logger.info("¡Dataset completo!")
                            break
                elif key == ord('n'):  # Next label
                    self.next_label()
            
            # Guardar dataset si hay datos
            if len(self.X) > 0:
                return self.save_dataset(save_path)
            else:
                logger.warning("No se capturaron muestras")
                return False
                
        except KeyboardInterrupt:
            logger.info("Captura interrumpida")
            return len(self.X) > 0 and self.save_dataset(save_path)
            
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.hands.close()
    
    def augment_dataset(self, X: np.ndarray, y: np.ndarray, 
                       noise_factor: float = 0.01, 
                       rotation_range: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Aplica augmentación offline al dataset
        
        Args:
            X: Array (N, 21, 3) de landmarks
            y: Array (N,) de etiquetas
            noise_factor: Factor de ruido gaussiano
            rotation_range: Rango de rotación en radianes
            
        Returns:
            Tuple (X_augmented, y_augmented)
        """
        X_aug = X.copy()
        y_aug = y.copy()
        
        # Ruido gaussiano
        noise = np.random.normal(0, noise_factor, X.shape)
        X_aug = X_aug + noise
        
        # Pequeñas rotaciones (simplificado - rotación en Z)
        for i in range(len(X_aug)):
            angle = np.random.uniform(-rotation_range, rotation_range)
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            
            # Rotar coordenadas X, Y
            x_rot = X_aug[i, :, 0] * cos_a - X_aug[i, :, 1] * sin_a
            y_rot = X_aug[i, :, 0] * sin_a + X_aug[i, :, 1] * cos_a
            
            X_aug[i, :, 0] = x_rot
            X_aug[i, :, 1] = y_rot
        
        # Asegurar que los valores permanezcan en rango válido [0, 1]
        X_aug = np.clip(X_aug, 0, 1)
        
        # Combinar original y augmentado
        X_total = np.vstack([X, X_aug])
        y_total = np.hstack([y, y_aug])
        
        logger.info(f"Dataset augmentado: {len(X)} → {len(X_total)} muestras")
        return X_total, y_total


def main():
    """Función principal para ejecutar captura de dataset"""
    print("=== Sistema de Captura de Dataset de Señas ===")
    print("Este sistema capturará datos para entrenar el modelo de reconocimiento de señas.")
    print()
    
    # Configuración
    labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
              "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
    min_samples = 200  # Mínimo por clase
    
    # Crear sistema de captura
    capture = DatasetCapture(labels=labels, min_samples_per_class=min_samples)
    
    # Ejecutar captura
    success = capture.run_capture_session(camera_index=0, save_path="dataset.json")
    
    if success:
        print("\n¡Captura completada exitosamente!")
        print(f"Dataset guardado en: dataset.json")
        print(f"Total muestras: {len(capture.X)}")
        print("Progreso por clase:")
        for label, count in capture.samples_per_class.items():
            print(f"  {label}: {count}/{min_samples}")
        
        # Validar antes de entrenamiento
        X = np.array(capture.X)
        y = np.array(capture.y)
        print(f"\nValidación - Shape X: {X.shape}")
        print(f"Validación - Clases únicas: {len(set(y))}")
        
        if capture.is_dataset_complete():
            print("✅ Dataset completo y listo para entrenamiento")
        else:
            print("⚠️ Dataset incompleto - se recomienda capturar más muestras")
    else:
        print("❌ Error en la captura de dataset")


if __name__ == "__main__":
    main()
