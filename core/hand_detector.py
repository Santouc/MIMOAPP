#!/usr/bin/env python3
"""
Módulo de detección de manos usando MediaPipe
Responsable de detectar landmarks de manos en tiempo real
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional

# Importar MediaPipe
try:
    import mediapipe as mp
except ImportError:
    print("MediaPipe no está instalado")
    mp = None

class HandDetector:
    """Detector de manos basado en MediaPipe Hands"""
    
    def __init__(self, max_hands: int = 2, detection_confidence: float = 0.5, 
                 tracking_confidence: float = 0.5, model_path: str = None,
                 use_simulated_fallback: bool = False):
        """
        Inicializa el detector de manos
        
        Args:
            max_hands: Número máximo de manos a detectar
            detection_confidence: Confianza mínima para detección
            tracking_confidence: Confianza mínima para seguimiento
            model_path: Ruta al archivo del modelo (default: data/models/hand_landmarker.task)
            use_simulated_fallback: Activa landmarks simulados si MediaPipe falla
        """
        self.max_hands = max_hands
        self.detection_confidence = detection_confidence
        self.tracking_confidence = tracking_confidence
        self.use_simulated_fallback = use_simulated_fallback
        self.hand_landmarker = None
        
        if mp is None:
            print("Error: MediaPipe no está disponible")
            return
        
        # Determinar path del modelo
        if model_path is None:
            # Buscar en data/models/ primero, luego en raíz
            model_path = Path(__file__).parent.parent / "data" / "models" / "hand_landmarker.task"
            if not model_path.exists():
                model_path = Path(__file__).parent.parent / "hand_landmarker.task"
        
        self.model_path = str(model_path)
        
        # Usar la nueva API de MediaPipe Tasks con modelo descargado
        try:
            HandLandmarker = mp.tasks.vision.HandLandmarker
            VisionRunningMode = mp.tasks.vision.RunningMode
            
            # Crear detector con configuración básica
            self.hand_landmarker = HandLandmarker.create_from_options(
                mp.tasks.vision.HandLandmarkerOptions(
                    base_options=mp.tasks.BaseOptions(
                        model_asset_path=self.model_path
                    ),
                    running_mode=VisionRunningMode.IMAGE,
                    num_hands=max_hands,
                    min_hand_detection_confidence=detection_confidence,
                    min_hand_presence_confidence=detection_confidence,
                    min_tracking_confidence=tracking_confidence
                )
            )
            print(f"HandLandmarker creado exitosamente con modelo: {self.model_path}")
        except Exception as e:
            print(f"Error creando HandLandmarker: {e}")
            print(f"Intentando usar modelo en: {self.model_path}")
            # Fallback a versión simple simulada
            self.hand_landmarker = None
        
        print("HandDetector inicializado con MediaPipe Tasks")
    
    def detect(self, frame: np.ndarray) -> Optional[List[List[Tuple[float, float, float]]]]:
        """
        Detecta landmarks de manos en el frame
        
        Args:
            frame: Frame de la cámara en formato BGR
            
        Returns:
            Lista de landmarks para cada mano detectada, o None si no se detectan manos
            Cada mano tiene 21 landmarks con coordenadas (x, y, z)
        """
        # Si MediaPipe no se inicializó, usar detección simulada
        if self.hand_landmarker is None:
            if self.use_simulated_fallback:
                return self._detect_simulated(frame)
            return None
        
        try:
            # Convertir BGR a RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Crear imagen MediaPipe
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            # Procesar con MediaPipe Tasks
            results = self.hand_landmarker.detect(mp_image)
            
            if results.hand_landmarks:
                hand_landmarks_list = []
                
                for hand_landmarks in results.hand_landmarks:
                    # Extraer coordenadas normalizadas
                    landmarks = []
                    for landmark in hand_landmarks:
                        landmarks.append((landmark.x, landmark.y, landmark.z))
                    
                    hand_landmarks_list.append(landmarks)
                
                return hand_landmarks_list
            
            return None
            
        except Exception as e:
            print(f"Error en detección de manos: {e}")
            if self.use_simulated_fallback:
                return self._detect_simulated(frame)
            return None
    
    def _detect_simulated(self, frame: np.ndarray) -> List[List[Tuple[float, float, float]]]:
        """Detección simulada como fallback"""
        h, w = frame.shape[:2]
        
        # Crear puntos simulados en forma de mano
        landmarks = []
        
        # Puntos de la muñeca y palma
        landmarks.append((0.5, 0.9, 0.0))  # 0: Muñeca
        landmarks.append((0.45, 0.8, 0.0))  # 1: Pulgar base
        landmarks.append((0.4, 0.7, 0.0))   # 2: Pulgar articulación
        landmarks.append((0.35, 0.6, 0.0))  # 3: Pulgar medio
        landmarks.append((0.3, 0.5, 0.0))   # 4: Pulgar punta
        
        # Índice
        landmarks.append((0.45, 0.7, 0.0))  # 5: Índice base
        landmarks.append((0.42, 0.6, 0.0))  # 6: Índice articulación
        landmarks.append((0.4, 0.5, 0.0))   # 7: Índice medio
        landmarks.append((0.38, 0.4, 0.0))  # 8: Índice punta
        
        # Medio
        landmarks.append((0.5, 0.7, 0.0))   # 9: Medio base
        landmarks.append((0.5, 0.6, 0.0))   # 10: Medio articulación
        landmarks.append((0.5, 0.5, 0.0))   # 11: Medio medio
        landmarks.append((0.5, 0.4, 0.0))   # 12: Medio punta
        
        # Anular
        landmarks.append((0.55, 0.7, 0.0))  # 13: Anular base
        landmarks.append((0.58, 0.6, 0.0))  # 14: Anular articulación
        landmarks.append((0.6, 0.5, 0.0))   # 15: Anular medio
        landmarks.append((0.62, 0.4, 0.0))  # 16: Anular punta
        
        # Meñique
        landmarks.append((0.6, 0.75, 0.0))  # 17: Meñique base
        landmarks.append((0.65, 0.65, 0.0))  # 18: Meñique articulación
        landmarks.append((0.7, 0.55, 0.0))  # 19: Meñique medio
        landmarks.append((0.75, 0.45, 0.0)) # 20: Meñique punta
        
        return [landmarks]
    
    def draw_landmarks(self, frame: np.ndarray, landmarks_list: List[List[Tuple[float, float, float]]]) -> np.ndarray:
        """
        Dibuja los landmarks en el frame
        
        Args:
            frame: Frame original
            landmarks_list: Lista de landmarks a dibujar
            
        Returns:
            Frame con landmarks dibujados
        """
        if not landmarks_list:
            return frame
        
        # Crear objeto de landmarks para dibujar
        annotated_frame = frame.copy()
        
        for landmarks in landmarks_list:
            # Convertir landmarks normalizados a coordenadas de píxeles
            h, w = frame.shape[:2]
            pixel_landmarks = []
            
            for x, y, z in landmarks:
                pixel_x = int(x * w)
                pixel_y = int(y * h)
                pixel_landmarks.append((pixel_x, pixel_y))
            
            # Dibujar puntos
            for i, (px, py) in enumerate(pixel_landmarks):
                cv2.circle(annotated_frame, (px, py), 5, (0, 255, 0), -1)
                cv2.putText(annotated_frame, str(i), (px + 8, py - 8),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
            
            # Dibujar conexiones
            self._draw_connections(annotated_frame, pixel_landmarks)
        
        return annotated_frame
    
    def _draw_connections(self, frame: np.ndarray, landmarks: List[Tuple[int, int]]) -> None:
        """
        Dibuja las conexiones entre landmarks
        
        Args:
            frame: Frame donde dibujar
            landmarks: Lista de coordenadas de landmarks
        """
        # Conexiones de la mano según MediaPipe
        connections = [
            # Pulgar
            (0, 1), (1, 2), (2, 3), (3, 4),
            # Índice
            (0, 5), (5, 6), (6, 7), (7, 8),
            # Medio
            (5, 9), (9, 10), (10, 11), (11, 12),
            # Anular
            (9, 13), (13, 14), (14, 15), (15, 16),
            # Meñique
            (13, 17), (17, 18), (18, 19), (19, 20),
            # Conexión muñeca-meñique
            (0, 17)
        ]
        
        for start_idx, end_idx in connections:
            if start_idx < len(landmarks) and end_idx < len(landmarks):
                start_point = landmarks[start_idx]
                end_point = landmarks[end_idx]
                cv2.line(frame, start_point, end_point, (255, 0, 0), 2)
    
    def get_finger_states(self, landmarks: List[Tuple[float, float, float]]) -> List[bool]:
        """
        Determina qué dedos están extendidos
        
        Args:
            landmarks: Lista de 21 landmarks de una mano
            
        Returns:
            Lista de 5 booleanos: [pulgar, indice, medio, anular, menique]
        """
        if len(landmarks) != 21:
            return [False] * 5
        
        # Puntos clave para cada dedo
        thumb_tip = landmarks[4]
        thumb_ip = landmarks[3]
        index_tip = landmarks[8]
        index_pip = landmarks[6]
        middle_tip = landmarks[12]
        middle_pip = landmarks[10]
        ring_tip = landmarks[16]
        ring_pip = landmarks[14]
        pinky_tip = landmarks[20]
        pinky_pip = landmarks[18]
        
        # Determinar si cada dedo está extendido (comparando posiciones Y)
        finger_states = [
            thumb_tip[1] < thumb_ip[1],  # Pulgar
            index_tip[1] < index_pip[1],  # Índice
            middle_tip[1] < middle_pip[1],  # Medio
            ring_tip[1] < ring_pip[1],  # Anular
            pinky_tip[1] < pinky_pip[1]  # Meñique
        ]
        
        return finger_states
    
    def cleanup(self):
        """Libera recursos de MediaPipe"""
        if hasattr(self, 'hand_landmarker'):
            self.hand_landmarker.close()
