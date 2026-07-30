#!/usr/bin/env python3
"""
Traductor de Lenguaje de Señas Chileno con Captura Interactiva
Sistema completo con FSM para captura estática y dinámica
"""

import cv2
import numpy as np
import time
import json
from collections import deque
from pathlib import Path

# Importar módulos personalizados
import sys
from pathlib import Path

# =========================
# #notas: script archivado; PROJECT_ROOT apunta a la raíz real del proyecto
# =========================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASE_DIR = PROJECT_ROOT

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Importar clases directamente
from core.image_processor import ImageProcessor
from core.hand_detector import HandDetector
from utils.config import Config
from utils.logger import get_logger
from ml.clasificador import SignClassifier

# Estados del FSM
STATE_CAMERA = 0      # default: live inference / idle
STATE_MENU = 1        # letter selection UI
STATE_CAPTURE = 2     # static capture
STATE_RECORDING = 3   # dynamic sequence recording

# =========================
# #notas: códigos normalizados de teclas
# #notas: soporte Windows + Linux + fallback WASD
# =========================
KEY_UP = [ord('w'), 2490368, 65362]
KEY_DOWN = [ord('s'), 2621440, 65364]

class InteractiveSignCapture:
    """Sistema de captura interactiva con FSM
    
    # =========================
    # #notas: FSM controla flujo del sistema
    # #notas: evita estados inválidos
    # =========================
    """
    
    def __init__(self):
        # Configuración inicial
        self.config = Config()
        self.logger = get_logger(__name__)
        
        # Paths relativos al directorio del script
        # =========================
        # #notas: reutilizar BASE_DIR global (raíz del proyecto)
        # =========================
        BASE_DIR = PROJECT_ROOT
        
        # Inicializar componentes
        self.hand_detector = HandDetector()
        self.image_processor = ImageProcessor()
        self.sign_classifier = SignClassifier(
            model_path=str(BASE_DIR / "data" / "models" / "model.h5"),
            labels_path=str(BASE_DIR / "data" / "models" / "labels.json")
        )
        
        # Cargar etiquetas para el menú
        try:
            with open(BASE_DIR / "data" / "models" / "labels.json", 'r') as f:
                self.labels = json.load(f)
        except:
            self.labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", 
                          "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
        
        # Precomputar lookup de etiquetas para optimización
        self.label_to_idx = {label: i for i, label in enumerate(self.labels)}
        
        # Variables de estado del FSM
        self.state = STATE_CAMERA
        self.current_label_idx = 0
        self.current_label = self.labels[0] if self.labels else "A"
        self.mode = "static"  # "static" or "dynamic"
        
        # Buffers de captura
        self.frame_buffer = []
        self.max_seq_len = 20
        
        # Variables de estabilización
        self.capture_counter = 0
        self.capture_delay = 5  # frames para estabilización
        
        # Datasets
        self.dataset_static = []
        self.dataset_dynamic = []
        self.load_existing_datasets()
        
        # Variables de inferencia
        self.running = False
        self.sign_buffer = deque(maxlen=10)
        self.confidence_threshold = 0.7
        self.last_landmarks = None
        self.missed_frames = 0
        self.max_missed_frames = 4
        self.smoothing_alpha = 0.65
        self.tracking_status = "PERDIDO"
        
        # Variables de UI
        self.menu_selection = 0
        self.recording_frames = 0
        
        self.logger.info("Sistema de captura interactiva inicializado")
    
    def stabilize_landmarks(self, landmarks):
        if landmarks is None or len(landmarks) == 0:
            if self.last_landmarks is not None and self.missed_frames < self.max_missed_frames:
                self.missed_frames += 1
                self.tracking_status = "RECUPERANDO"
                return self.last_landmarks
            
            self.last_landmarks = None
            self.missed_frames = 0
            self.tracking_status = "PERDIDO"
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
    
    def load_existing_datasets(self):
        static_path = BASE_DIR / "data" / "datasets" / "dataset_static.json"
        dynamic_path = BASE_DIR / "data" / "datasets" / "dataset_dynamic.json"
        
        if static_path.exists():
            try:
                with open(static_path, 'r', encoding='utf-8') as f:
                    dataset = json.load(f)
                X = dataset.get("X", [])
                y = dataset.get("y", [])
                self.dataset_static = [{"X": x, "y": label} for x, label in zip(X, y)]
                self.logger.info(f"Dataset estático existente cargado: {len(self.dataset_static)} muestras")
            except Exception as e:
                self.logger.warning(f"No se pudo cargar dataset estático existente: {e}")
        
        if dynamic_path.exists():
            try:
                with open(dynamic_path, 'r', encoding='utf-8') as f:
                    dataset = json.load(f)
                X = dataset.get("X", [])
                y = dataset.get("y", [])
                self.dataset_dynamic = [{"X": x, "y": label} for x, label in zip(X, y)]
                self.logger.info(f"Dataset dinámico existente cargado: {len(self.dataset_dynamic)} muestras")
            except Exception as e:
                self.logger.warning(f"No se pudo cargar dataset dinámico existente: {e}")
    
    def extract_landmarks(self, hand_landmarks):
        """Extrae landmarks de MediaPipe
        
        # =========================
        # #notas: hand_landmarks puede ser lista o objeto MediaPipe
        # #notas: manejar ambos casos para compatibilidad
        # =========================
        """
        if hasattr(hand_landmarks, 'landmark'):
            # Caso MediaPipe: objeto con atributo landmark
            return np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
        else:
            # Caso lista: ya es lista de coordenadas (x, y, z)
            return np.array(hand_landmarks)
    
    def preprocess_frame(self, X):
        """Preprocesamiento consistente con entrenamiento
        
        # =========================
        # #notas: normalización crítica
        # #notas: debe ser idéntica a entrenamiento
        # =========================
        """
        X = X - X[0]  # Normalización de traducción
        scale = np.linalg.norm(X[9])
        if scale > 0:
            X = X / scale  # Normalización de escala
        return X
    
    def pad_or_truncate(self, seq, T):
        """Normaliza longitud de secuencia"""
        if len(seq) > T:
            return seq[:T]
        elif len(seq) < T:
            pad = np.repeat(seq[-1][np.newaxis,...], T - len(seq), axis=0)
            return np.concatenate([seq, pad], axis=0)
        return seq
    
    def handle_key_input(self, key, key_ext):
        """Manejo de input de teclado
        
        # =========================
        # #notas: se usa waitKeyEx para capturar teclas extendidas (flechas)
        # #notas: key → ASCII (letras)
        # #notas: key_ext → teclas especiales dependientes del sistema
        # =========================
        """
        if self.state == STATE_CAMERA:
            if key == ord('m'):
                self.state = STATE_MENU
                self.menu_selection = 0
                self.logger.info("Entrando al menú de selección")
            
            elif key == ord('t') and self.current_label:
                if self.mode == "static":
                    self.state = STATE_CAPTURE
                    self.capture_counter = 0
                    self.logger.info(f"Iniciando captura estática para: {self.current_label}")
                else:
                    self.state = STATE_RECORDING
                    self.frame_buffer = []
                    self.recording_frames = 0
                    self.logger.info(f"Iniciando grabación dinámica: {self.current_label}")
        
        elif self.state == STATE_RECORDING:
            if key == ord('t'):
                self.stop_recording()
        
        elif self.state == STATE_MENU:
            # =========================
            # #notas: navegación del menú
            # #notas: soporta WASD + flechas
            # #notas: compatibilidad multiplataforma
            # =========================
            
            # ARRIBA
            if key in KEY_UP or key_ext in KEY_UP:
                self.menu_selection = (self.menu_selection - 1) % len(self.labels)
            
            # ABAJO
            elif key in KEY_DOWN or key_ext in KEY_DOWN:
                self.menu_selection = (self.menu_selection + 1) % len(self.labels)
            
            # ENTER
            elif key == 13:
                # =========================
                # #notas: validación de etiquetas
                # #notas: evita acceso a lista vacía
                # =========================
                if not self.labels:
                    self.logger.error("Lista de etiquetas vacía")
                    self.state = STATE_CAMERA
                    return
                    
                self.current_label_idx = self.menu_selection
                self.current_label = self.labels[self.menu_selection]
                self.state = STATE_CAMERA
            
            # SALIR
            elif key == ord('m'):
                self.state = STATE_CAMERA
    
    def process_static_capture(self, landmarks):
        """Procesa captura estática con estabilización
        
        # =========================
        # #notas: captura estática requiere estabilización
        # #notas: evita ruido en dataset
        # =========================
        """
        self.capture_counter += 1
        
        if self.capture_counter >= self.capture_delay:
            X = self.extract_landmarks(landmarks)
            X = self.preprocess_frame(X)
            
            self.dataset_static.append({
                "X": X.tolist(),
                "y": self.label_to_idx[self.current_label]
            })
            
            self.logger.info(f"Captura estática guardada: {self.current_label}")
            self.state = STATE_CAMERA
            self.capture_counter = 0
    
    def process_dynamic_recording(self, landmarks):
        """Procesa grabación dinámica
        
        # =========================
        # #notas: buffer almacena secuencia temporal
        # #notas: limitado a max_seq_len para consistencia
        # =========================
        """
        X = self.extract_landmarks(landmarks)
        X = self.preprocess_frame(X)
        
        self.frame_buffer.append(X)
        self.recording_frames += 1
        
        if len(self.frame_buffer) > self.max_seq_len:
            self.frame_buffer.pop(0)
    
    def stop_recording(self):
        """Detiene grabación y guarda secuencia"""
        if len(self.frame_buffer) > 0:
            sequence = np.array(self.frame_buffer)  # shape (T,21,3)
            sequence = self.pad_or_truncate(sequence, self.max_seq_len)
            
            self.dataset_dynamic.append({
                "X": sequence.tolist(),
                "y": self.label_to_idx[self.current_label]
            })
            
            self.logger.info(f"Secuencia dinámica guardada: {self.current_label} ({len(self.frame_buffer)} frames)")
        
        self.frame_buffer = []
        self.recording_frames = 0
        self.state = STATE_CAMERA
    
    def draw_ui_overlay(self, frame):
        """Dibuja UI con OpenCV"""
        height, width = frame.shape[:2]
        
        # Fondo semitransparente para texto
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (width, 60), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # =========================
        # #notas: interfaz completamente en español
        # #notas: evitar mezcla de idiomas en UI
        # =========================
        
        # Estado actual
        state_text = {
            STATE_CAMERA: "CÁMARA - 'm': menú, 't': capturar",
            STATE_MENU: "MENÚ - flechas/WASD: navegar, Enter: seleccionar",
            STATE_CAPTURE: "CAPTURANDO ESTÁTICO...",
            STATE_RECORDING: f"GRABANDO DINÁMICO... {self.recording_frames} frames"
        }
        
        cv2.putText(frame, state_text[self.state], (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Etiqueta actual
        label_text = f"Etiqueta: {self.current_label} (idx: {self.current_label_idx})"
        cv2.putText(frame, label_text, (10, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Modo actual
        mode_text = f"Modo: {self.mode.upper()}"
        cv2.putText(frame, mode_text, (width - 150, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Contadores de datasets
        static_count = len(self.dataset_static)
        dynamic_count = len(self.dataset_dynamic)
        count_text = f"Estático: {static_count} | Dinámico: {dynamic_count}"
        cv2.putText(frame, count_text, (width - 250, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)
        
        tracking_color = (0, 255, 0) if self.tracking_status == "OK" else (0, 255, 255) if self.tracking_status == "RECUPERANDO" else (0, 0, 255)
        cv2.putText(frame, f"Tracking: {self.tracking_status}", (10, 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, tracking_color, 2)
        
        # Menú de selección
        if self.state == STATE_MENU:
            self.draw_menu_overlay(frame)
        
        # Indicador de grabación
        if self.state == STATE_RECORDING:
            # Círculo rojo parpadeante
            color = (0, 0, 255) if int(time.time() * 2) % 2 == 0 else (0, 100, 255)
            cv2.circle(frame, (width - 50, 50), 15, color, -1)
            cv2.putText(frame, "GRABANDO", (width - 110, 55), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return frame
    
    def draw_menu_overlay(self, frame):
        """Dibuja menú de selección de etiquetas"""
        height, width = frame.shape[:2]
        
        # Fondo del menú
        menu_height = min(len(self.labels) * 30 + 40, height - 100)
        menu_y = (height - menu_height) // 2
        menu_width = 200
        menu_x = (width - menu_width) // 2
        
        cv2.rectangle(frame, (menu_x, menu_y), 
                     (menu_x + menu_width, menu_y + menu_height), (50, 50, 50), -1)
        cv2.rectangle(frame, (menu_x, menu_y), 
                     (menu_x + menu_width, menu_y + menu_height), (200, 200, 200), 2)
        
        # Título
        cv2.putText(frame, "SELECCIONAR ETIQUETA", (menu_x + 20, menu_y + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Opciones del menú
        start_idx = max(0, self.menu_selection - 5)
        end_idx = min(len(self.labels), start_idx + 10)
        
        for i in range(start_idx, end_idx):
            y_pos = menu_y + 50 + (i - start_idx) * 30
            
            if i == self.menu_selection:
                # Opción seleccionada
                cv2.rectangle(frame, (menu_x + 10, y_pos - 20), 
                             (menu_x + menu_width - 10, y_pos), (100, 200, 100), -1)
                color = (255, 255, 255)
            else:
                color = (200, 200, 200)
            
            label_text = f"{self.labels[i]}"
            cv2.putText(frame, label_text, (menu_x + 20, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    def save_datasets(self):
        """Guarda datasets en formato JSON
        
        # =========================
        # #notas: formato JSON compatible con pipeline de entrenamiento
        # =========================
        """
        if self.dataset_static:
            static_data = {
                "X": [item["X"] for item in self.dataset_static],
                "y": [item["y"] for item in self.dataset_static],
                "labels": self.labels,
                "metadata": {
                    "type": "static",
                    "num_samples": len(self.dataset_static),
                    "landmark_format": "mediapipe_21_3d",
                    "coordinate_system": "normalized"
                }
            }
            
            dataset_path = BASE_DIR / "data" / "datasets" / "dataset_static.json"
            dataset_path.parent.mkdir(parents=True, exist_ok=True)
            with open(dataset_path, 'w') as f:
                json.dump(static_data, f, indent=2)
            self.logger.info(f"Dataset estático guardado: {len(self.dataset_static)} muestras")
        
        if self.dataset_dynamic:
            dynamic_data = {
                "X": [item["X"] for item in self.dataset_dynamic],
                "y": [item["y"] for item in self.dataset_dynamic],
                "labels": self.labels,
                "metadata": {
                    "type": "dynamic",
                    "num_samples": len(self.dataset_dynamic),
                    "sequence_length": self.max_seq_len,
                    "landmark_format": "mediapipe_21_3d",
                    "coordinate_system": "normalized"
                }
            }
            
            dataset_path = BASE_DIR / "data" / "datasets" / "dataset_dynamic.json"
            dataset_path.parent.mkdir(parents=True, exist_ok=True)
            with open(dataset_path, 'w') as f:
                json.dump(dynamic_data, f, indent=2)
            self.logger.info(f"Dataset dinámico guardado: {len(self.dataset_dynamic)} muestras")
    
    def process_inference(self, frame, landmarks):
        """Procesamiento de inferencia normal"""
        if landmarks is not None:
            frame = self.hand_detector.draw_landmarks(frame, landmarks)
            
            if isinstance(landmarks, list) and len(landmarks) > 0:
                primary = landmarks[0]
            else:
                primary = landmarks
            
            if hasattr(primary, 'landmark'):
                num_landmarks = len(primary.landmark)
            else:
                num_landmarks = len(primary) if isinstance(primary, list) else 0
            
            if num_landmarks != 21:
                return frame, None
            
            lm_array = [self.extract_landmarks(primary)]
            
            sign, confidence = self.sign_classifier.classify(lm_array)
            
            self.sign_buffer.append((sign, confidence))
            
            if len(self.sign_buffer) >= 5:
                final_sign = self._get_consensus_sign()
                return frame, final_sign
        
        return frame, None
    
    def _get_consensus_sign(self):
        """Determina el signo con mayor consenso"""
        signs = [item[0] for item in self.sign_buffer if item[1] > self.confidence_threshold]
        if not signs:
            return None
        
        from collections import Counter
        most_common = Counter(signs).most_common(1)[0][0]
        return most_common
    
    def run(self):
        """Bucle principal del sistema"""
        self.logger.info("Iniciando sistema de captura interactiva...")
        self.running = True
        
        # Inicializar cámara
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        if not cap.isOpened():
            self.logger.error("No se pudo abrir la cámara")
            return
        
        print("\n=== SISTEMA DE CAPTURA INTERACTIVA ===")
        print("Controles:")
        print("  'm' - Abrir menú de selección")
        print("  Flechas/WASD - Navegar menú")
        print("  Enter - Confirmar selección")
        print("  't' - Capturar estática / Iniciar-Detener grabación dinámica")
        print("  'q' - Salir y guardar datasets")
        print("  's' - Cambiar modo (estático/dinámico)")
        print(f"  Modo actual: {self.mode.upper()}")
        print(f"  Etiqueta actual: {self.current_label}")
        print("=====================================\n")
        
        try:
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    continue
                
                # =========================
                # #notas: loop principal
                # #notas: flujo por iteración:
                # #notas: 1) captura frame
                # #notas: 2) detección landmarks
                # #notas: 3) FSM (estado)
                # #notas: 4) render UI
                # #notas: 5) input teclado
                # =========================
                
                # Procesamiento de imagen
                # Flip ANTES de procesar para consistencia
                frame = cv2.flip(frame, 1)
                processed_frame = self.image_processor.preprocess(frame)
                
                # =========================
                # #notas: detección MediaPipe
                # #notas: puede retornar None si no hay mano
                # =========================
                raw_landmarks = self.hand_detector.detect(processed_frame)
                landmarks = self.stabilize_landmarks(raw_landmarks)
                
                # =========================
                # #notas: evaluación de estado FSM
                # #notas: define comportamiento del sistema
                # =========================
                if self.state == STATE_CAMERA:
                    # Inferencia normal
                    processed_frame, detected_sign = self.process_inference(processed_frame, landmarks)
                    
                    # Mostrar resultado de inferencia
                    if detected_sign:
                        # =========================
                        # #notas: salida de inferencia localizada
                        # =========================
                        cv2.putText(processed_frame, f"Detectado: {detected_sign}", (10, 100), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                elif self.state == STATE_CAPTURE and landmarks is not None:
                    # Captura estática con validación
                    # =========================
                    # #notas: landmarks es List[List[Tuple]] de HandDetector
                    # #notas: landmarks[0] es la primera mano (21 puntos)
                    # #notas: validar longitud de landmarks por mano
                    # =========================
                    
                    # =========================
                    # #notas: extraer primera mano
                    # #notas: HandDetector retorna lista de manos
                    # =========================
                    if isinstance(landmarks, list) and len(landmarks) > 0:
                        hand_landmarks = landmarks[0]  # Primera mano
                        num_landmarks = len(hand_landmarks)
                    else:
                        hand_landmarks = landmarks
                        num_landmarks = 0
                    
                    if num_landmarks == 21:
                        self.process_static_capture(hand_landmarks)
                    else:
                        # =========================
                        # #notas: no volver a CAMERA inmediatamente
                        # #notas: permitir que MediaPipe estabilice
                        # #notas: solo loggear si es persistente
                        # =========================
                        if not hasattr(self, '_incomplete_landmark_count'):
                            self._incomplete_landmark_count = 0
                        
                        self._incomplete_landmark_count += 1
                        
                        if self._incomplete_landmark_count % 10 == 0:  # Log cada 10 frames
                            self.logger.warning(f"Landmarks incompletos: {num_landmarks}/21")
                        
                        # Mantener en STATE_CAPTURE para permitir estabilización
                        # Solo volver a CAMERA después de muchos frames incompletos
                        if self._incomplete_landmark_count > 30:
                            self.logger.warning("Demasiados intentos con landmarks incompletos, cancelando captura")
                            self.state = STATE_CAMERA
                            self._incomplete_landmark_count = 0
                
                elif self.state == STATE_RECORDING:
                    if landmarks is not None:
                        processed_frame = self.hand_detector.draw_landmarks(processed_frame, landmarks)
                        
                        if isinstance(landmarks, list) and len(landmarks) > 0:
                            hand_landmarks = landmarks[0]
                            num_landmarks = len(hand_landmarks)
                        elif hasattr(landmarks, 'landmark'):
                            hand_landmarks = landmarks
                            num_landmarks = len(landmarks.landmark)
                        else:
                            hand_landmarks = landmarks
                            num_landmarks = 0
                        
                        if num_landmarks == 21:
                            self.process_dynamic_recording(hand_landmarks)
                    else:
                        if len(self.frame_buffer) > 5:
                            self.stop_recording()
                        else:
                            self.state = STATE_CAMERA
                
                # Dibujar UI
                display_frame = self.draw_ui_overlay(processed_frame.copy())
                
                # =========================
                # #notas: validación de input
                # #notas: -1 indica ausencia de tecla
                # =========================
                key_ext = cv2.waitKeyEx(1)
                
                if key_ext == -1:
                    key = -1
                else:
                    key = key_ext & 0xFF
                
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    # Cambiar modo
                    self.mode = "dynamic" if self.mode == "static" else "static"
                    print(f"Modo cambiado a: {self.mode.upper()}")
                else:
                    self.handle_key_input(key, key_ext)
                
                # Mostrar frame (sin flip adicional)
                cv2.imshow('Interactive Sign Capture', display_frame)
        
        except KeyboardInterrupt:
            self.logger.info("Interrupción del usuario")
        
        finally:
            # Guardar datasets
            self.save_datasets()
            
            # Limpieza
            self.running = False
            cap.release()
            cv2.destroyAllWindows()
            self.logger.info("Sistema finalizado")

def main():
    """Función principal"""
    capture_system = InteractiveSignCapture()
    capture_system.run()

if __name__ == "__main__":
    main()
