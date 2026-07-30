#!/usr/bin/env python3
"""
Traductor de Lenguaje de Señas Chileno
Archivo principal que orquesta el sistema completo
"""

import cv2
import numpy as np
import time
import threading
import sys
from collections import deque
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Importar módulos personalizados
from core.image_processor import ImageProcessor
from core.hand_detector import HandDetector
from utils.config import Config
from utils.logger import get_logger

# Importar el clasificador híbrido real
from ml.clasificador import SignClassifier
from ml.dynamic_classifier import DynamicSignClassifier

class SignLanguageTranslator:
    """Clase principal que coordina todos los componentes del sistema"""
    
    def __init__(self):
        # Configuración inicial
        self.config = Config()
        self.logger = get_logger(__name__)
        
        # Inicializar componentes
        self.hand_detector = HandDetector()
        self.image_processor = ImageProcessor()
        self.sign_classifier = SignClassifier(
            model_path=str(PROJECT_ROOT / "data" / "models" / "model.h5"),
            labels_path=str(PROJECT_ROOT / "data" / "models" / "labels.json")
        )
        self.dynamic_classifier = DynamicSignClassifier(
            model_path=str(PROJECT_ROOT / "data" / "models" / "model_dynamic.h5"),
            labels_path=str(PROJECT_ROOT / "data" / "models" / "labels_dynamic.json"),
            sequence_length=20
        )
        
        # Variables de estado
        self.running = False
        self.current_sign = None
        self.sign_buffer = deque(maxlen=10)  # Buffer para estabilizar detección
        self.dynamic_sequence = deque(maxlen=20)
        self.dynamic_buffer = deque(maxlen=5)
        self.confidence_threshold = 0.7
        self.last_landmarks = None
        self.missed_frames = 0
        self.max_missed_frames = 4
        self.smoothing_alpha = 0.65
        self.tracking_status = "PERDIDO"
        
        self.logger.info("Sistema de traducción inicializado")
    
    def process_frame(self, frame):
        """Procesa un frame individual y devuelve el resultado"""
        try:
            # 1. Procesamiento de imagen con OpenCV
            processed_frame = self.image_processor.preprocess(frame)
            
            # 2. Detección de landmarks con MediaPipe
            raw_landmarks = self.hand_detector.detect(processed_frame)
            landmarks = self._stabilize_landmarks(raw_landmarks)
            
            if landmarks is not None:
                # 3. Dibujar landmarks en el frame
                processed_frame = self.hand_detector.draw_landmarks(processed_frame, landmarks)
                
                # 4. Clasificación con TensorFlow
                sign, confidence = self.sign_classifier.classify(landmarks)
                
                dynamic_sign = self._process_dynamic_prediction(landmarks)
                
                # 5. Aplicar buffer para estabilidad
                self.sign_buffer.append((sign, confidence))
                
                # 6. Determinar sign final basado en consenso
                if len(self.sign_buffer) >= 5:
                    final_sign = self._get_consensus_sign()
                    return processed_frame, final_sign, dynamic_sign
            
            return processed_frame, None, None
            
        except Exception as e:
            self.logger.error(f"Error procesando frame: {e}")
            return frame, None, None
    
    def _stabilize_landmarks(self, landmarks):
        if landmarks is None or len(landmarks) == 0:
            if self.last_landmarks is not None and self.missed_frames < self.max_missed_frames:
                self.missed_frames += 1
                self.tracking_status = "RECUPERANDO"
                return self.last_landmarks
            
            self.last_landmarks = None
            self.missed_frames = 0
            self.tracking_status = "PERDIDO"
            self.dynamic_sequence.clear()
            return None
        
        if self.last_landmarks is None:
            self.last_landmarks = landmarks
            self.missed_frames = 0
            self.tracking_status = "OK"
            return landmarks
        
        try:
            stabilized = []
            used_prev = set()
            
            for curr_hand in landmarks:
                if not isinstance(curr_hand, list) or len(curr_hand) != 21:
                    stabilized.append(curr_hand)
                    continue
                
                curr = np.array(curr_hand, dtype=np.float32)
                
                best_prev_idx = None
                best_dist = 1e9
                for j, prev_hand in enumerate(self.last_landmarks if isinstance(self.last_landmarks, list) else []):
                    if j in used_prev or not isinstance(prev_hand, list) or len(prev_hand) != 21:
                        continue
                    prev_wrist = np.array(prev_hand[0], dtype=np.float32)
                    curr_wrist = np.array(curr_hand[0], dtype=np.float32)
                    dist = float(np.linalg.norm(prev_wrist[:2] - curr_wrist[:2]))
                    if dist < best_dist:
                        best_dist = dist
                        best_prev_idx = j
                
                if best_prev_idx is not None:
                    previous = np.array(self.last_landmarks[best_prev_idx], dtype=np.float32)
                    used_prev.add(best_prev_idx)
                else:
                    previous = curr
                
                if curr.shape != previous.shape or curr.shape != (21, 3):
                    stabilized.append(curr_hand)
                    continue
                
                smoothed = self.smoothing_alpha * curr + (1.0 - self.smoothing_alpha) * previous
                stabilized.append(smoothed.tolist())
            
            self.last_landmarks = stabilized
            self.missed_frames = 0
            self.tracking_status = "OK"
            return stabilized
        except Exception:
            self.last_landmarks = landmarks
            self.missed_frames = 0
            self.tracking_status = "OK"
            return landmarks
    
    def _process_dynamic_prediction(self, landmarks):
        if landmarks is None or len(landmarks) == 0:
            self.dynamic_sequence.clear()
            return None
        
        hand_landmarks = landmarks[0]
        if len(hand_landmarks) != 21:
            return None
        
        self.dynamic_sequence.append(hand_landmarks)
        
        if len(self.dynamic_sequence) < self.dynamic_sequence.maxlen:
            return None
        
        sign, confidence = self.dynamic_classifier.classify_sequence(list(self.dynamic_sequence))
        self.dynamic_buffer.append((sign, confidence))
        
        if len(self.dynamic_buffer) < self.dynamic_buffer.maxlen:
            return None
        
        return self._get_consensus_sign(self.dynamic_buffer)
    
    def _get_consensus_sign(self, buffer=None):
        """Determina el signo con mayor consenso en el buffer"""
        target_buffer = buffer if buffer is not None else self.sign_buffer
        signs = [item[0] for item in target_buffer if item[1] > self.confidence_threshold and item[0] != "unknown"]
        if not signs:
            return None
        
        # Contar frecuencia y devolver el más común
        from collections import Counter
        most_common = Counter(signs).most_common(1)[0][0]
        return most_common
    
    def run(self):
        """Bucle principal del sistema"""
        self.logger.info("Iniciando sistema de traducción...")
        self.running = True
        
        # Inicializar cámara
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        if not cap.isOpened():
            self.logger.error("No se pudo abrir la cámara")
            return
        
        try:
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    continue
                
                # Procesar frame
                processed_frame, detected_sign, dynamic_sign = self.process_frame(frame)
                
                # Mostrar resultados
                self._display_results(processed_frame, detected_sign, dynamic_sign)
                
                # Salir con 'q'
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        except KeyboardInterrupt:
            self.logger.info("Interrupción del usuario")
        
        finally:
            self.cleanup(cap)
    
    def _display_results(self, frame, sign, dynamic_sign=None):
        """Muestra los resultados en la ventana"""
        # Espejar imagen para mejor experiencia
        display_frame = cv2.flip(frame, 1)
        
        # Mostrar signo detectado
        if sign:
            cv2.putText(display_frame, f"Estatico: {sign}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(display_frame, "Estatico: None", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        if dynamic_sign:
            cv2.putText(display_frame, f"Dinamico: {dynamic_sign}", (10, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        else:
            cv2.putText(display_frame, "Dinamico: None", (10, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        tracking_color = (0, 255, 0) if self.tracking_status == "OK" else (0, 255, 255) if self.tracking_status == "RECUPERANDO" else (0, 0, 255)
        cv2.putText(display_frame, f"Tracking: {self.tracking_status}", (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, tracking_color, 2)
        
        # Mostrar FPS
        fps_text = f"FPS: {self._calculate_fps():.1f}"
        cv2.putText(display_frame, fps_text, (10, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Mostrar controles
        cv2.putText(display_frame, "Presiona 'q' para salir", (10, display_frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        cv2.imshow('Traductor de Lenguaje de Señas', display_frame)
    
    def _calculate_fps(self):
        """Calcula los FPS actuales"""
        if not hasattr(self, 'last_time'):
            self.last_time = time.time()
            self.frame_count = 0
            self.fps = 0
        
        self.frame_count += 1
        current_time = time.time()
        
        if current_time - self.last_time >= 1.0:
            self.fps = self.frame_count / (current_time - self.last_time)
            self.last_time = current_time
            self.frame_count = 0
        
        return self.fps
    
    def cleanup(self, cap):
        """Limpia recursos al finalizar"""
        self.running = False
        cap.release()
        cv2.destroyAllWindows()
        self.logger.info("Sistema finalizado")

def main():
    """Función principal de entrada"""
    translator = SignLanguageTranslator()
    translator.run()

if __name__ == "__main__":
    main()
