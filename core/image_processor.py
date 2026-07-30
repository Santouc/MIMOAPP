#!/usr/bin/env python3
"""
Módulo de procesamiento de imágenes con OpenCV
Responsable del preprocesamiento y postprocesamiento de imágenes
"""

import cv2
import numpy as np
from typing import Tuple, Optional

class ImageProcessor:
    """Procesador de imágenes usando OpenCV"""
    
    def __init__(self, target_size: Tuple[int, int] = (640, 480)):
        """
        Inicializa el procesador de imágenes
        
        Args:
            target_size: Tamaño objetivo para las imágenes (ancho, alto)
        """
        self.target_width, self.target_height = target_size
        self.skin_lower = np.array([0, 20, 70], dtype=np.uint8)
        self.skin_upper = np.array([20, 255, 255], dtype=np.uint8)
        
        print("ImageProcessor inicializado con OpenCV")
    
    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocesa un frame para mejorar la detección
        
        Args:
            frame: Frame original en formato BGR
            
        Returns:
            Frame preprocesado
        """
        try:
            # 1. Redimensionar si es necesario
            if frame.shape[1] != self.target_width or frame.shape[0] != self.target_height:
                frame = cv2.resize(frame, (self.target_width, self.target_height))
            
            # 2. Aplicar suavizado para reducir ruido
            frame_smooth = cv2.GaussianBlur(frame, (5, 5), 0)
            
            # 3. Mejorar contraste
            frame_contrast = self._enhance_contrast(frame_smooth)
            
            # 4. Normalizar iluminación
            frame_normalized = self._normalize_lighting(frame_contrast)
            
            return frame_normalized
            
        except Exception as e:
            print(f"Error en preprocesamiento: {e}")
            return frame
    
    def _enhance_contrast(self, frame: np.ndarray) -> np.ndarray:
        """
        Mejora el contraste de la imagen usando CLAHE
        
        Args:
            frame: Frame de entrada
            
        Returns:
            Frame con contraste mejorado
        """
        # Convertir a LAB color space
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Aplicar CLAHE al canal L
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_clahe = clahe.apply(l)
        
        # Combinar canales y volver a BGR
        lab_clahe = cv2.merge([l_clahe, a, b])
        return cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
    
    def _normalize_lighting(self, frame: np.ndarray) -> np.ndarray:
        """
        Normaliza la iluminación de la imagen
        
        Args:
            frame: Frame de entrada
            
        Returns:
            Frame con iluminación normalizada
        """
        # Convertir a escala de grises
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Crear máscara de iluminación
        blur = cv2.GaussianBlur(gray, (101, 101), 0)
        mask = cv2.divide(gray, blur, scale=255.0)
        
        # Aplicar máscara a cada canal
        result = frame.copy()
        for i in range(3):
            result[:, :, i] = cv2.multiply(frame[:, :, i], mask, scale=1/255.0)
        
        return result
    
    def detect_skin_region(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Detecta regiones de piel en la imagen
        
        Args:
            frame: Frame de entrada en formato BGR
            
        Returns:
            Máscara binaria de regiones de piel, o None si hay error
        """
        try:
            # Convertir a HSV
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # Crear máscara de piel
            skin_mask = cv2.inRange(hsv, self.skin_lower, self.skin_upper)
            
            # Aplicar operaciones morfológicas para limpiar la máscara
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
            skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
            
            return skin_mask
            
        except Exception as e:
            print(f"Error en detección de piel: {e}")
            return None
    
    def extract_hand_roi(self, frame: np.ndarray, landmarks: list, 
                        padding: int = 50) -> Optional[np.ndarray]:
        """
        Extrae la región de interés (ROI) de la mano
        
        Args:
            frame: Frame completo
            landmarks: Lista de landmarks de la mano
            padding: Padding adicional alrededor de la mano
            
        Returns:
            ROI de la mano, o None si hay error
        """
        try:
            if not landmarks or len(landmarks) != 21:
                return None
            
            h, w = frame.shape[:2]
            
            # Convertir landmarks a coordenadas de píxeles
            pixel_coords = []
            for x, y, z in landmarks:
                px = int(x * w)
                py = int(y * h)
                pixel_coords.append((px, py))
            
            # Encontrar bounding box
            x_coords = [coord[0] for coord in pixel_coords]
            y_coords = [coord[1] for coord in pixel_coords]
            
            x_min = max(0, min(x_coords) - padding)
            x_max = min(w, max(x_coords) + padding)
            y_min = max(0, min(y_coords) - padding)
            y_max = min(h, max(y_coords) + padding)
            
            # Extraer ROI
            roi = frame[y_min:y_max, x_min:x_max]
            
            return roi
            
        except Exception as e:
            print(f"Error extrayendo ROI: {e}")
            return None
    
    def apply_edge_detection(self, frame: np.ndarray) -> np.ndarray:
        """
        Aplica detección de bordes para resaltar contornos
        
        Args:
            frame: Frame de entrada
            
        Returns:
            Frame con bordes detectados
        """
        try:
            # Convertir a escala de grises
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Aplicar filtro Gaussiano
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Detección de bordes Canny
            edges = cv2.Canny(blurred, 50, 150)
            
            # Convertir a BGR para compatibilidad
            edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            
            return edges_bgr
            
        except Exception as e:
            print(f"Error en detección de bordes: {e}")
            return frame
    
    def create_overlay_info(self, frame: np.ndarray, info_text: str, 
                           position: Tuple[int, int] = (10, 30)) -> np.ndarray:
        """
        Crea una overlay con información en el frame
        
        Args:
            frame: Frame original
            info_text: Texto a mostrar
            position: Posición del texto (x, y)
            
        Returns:
            Frame con overlay de información
        """
        overlay = frame.copy()
        
        # Crear fondo semitransparente para el texto
        text_size = cv2.getTextSize(info_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        cv2.rectangle(overlay, 
                      (position[0] - 5, position[1] - text_size[1] - 5),
                      (position[0] + text_size[0] + 5, position[1] + 5),
                      (0, 0, 0), -1)
        
        # Añadir transparencia
        alpha = 0.7
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        # Añadir texto
        cv2.putText(frame, info_text, position, cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, (0, 255, 0), 2)
        
        return frame
    
    def resize_with_aspect_ratio(self, frame: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """
        Redimensiona manteniendo el aspect ratio
        
        Args:
            frame: Frame original
            target_size: Tamaño objetivo (ancho, alto)
            
        Returns:
            Frame redimensionado con padding si es necesario
        """
        h, w = frame.shape[:2]
        target_w, target_h = target_size
        
        # Calcular escala
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Redimensionar
        resized = cv2.resize(frame, (new_w, new_h))
        
        # Crear canvas con padding
        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        
        # Calcular posición para centrar la imagen
        x_offset = (target_w - new_w) // 2
        y_offset = (target_h - new_h) // 2
        
        # Colocar imagen en el centro
        canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
        
        return canvas
