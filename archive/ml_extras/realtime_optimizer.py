#!/usr/bin/env python3
"""
Sistema de optimización para inferencia en tiempo real
Minimiza latencia y maximiza estabilidad temporal para video → MediaPipe → modelo
"""

import numpy as np
import threading
import time
from queue import Queue, Empty
from collections import deque, Counter
from typing import Tuple, List, Optional, Dict, Any
from dataclasses import dataclass

from utils.logger import get_logger
from core.preprocessing import normalize_landmarks

logger = get_logger(__name__)

@dataclass
class PredictionResult:
    """Estructura para resultados de predicción"""
    label: str
    confidence: float
    probabilities: np.ndarray
    timestamp: float
    frame_id: int

class EMASmoother:
    """Exponential Moving Average para suavizado temporal de probabilidades"""
    
    def __init__(self, alpha: float = 0.6):
        """
        Inicializa suavizador EMA
        
        Args:
            alpha: Factor de suavizado (0.1-0.9 recomendado)
        """
        self.alpha = alpha
        self.state = None
        self.initialized = False
        
    def update(self, probs: np.ndarray) -> np.ndarray:
        """
        Actualiza estado EMA con nuevas probabilidades
        
        Args:
            probs: Array de probabilidades (num_classes,)
            
        Returns:
            Probabilidades suavizadas
        """
        if not self.initialized:
            self.state = probs.copy()
            self.initialized = True
        else:
            self.state = self.alpha * probs + (1 - self.alpha) * self.state
        
        return self.state.copy()
    
    def reset(self):
        """Reinicia estado del suavizador"""
        self.state = None
        self.initialized = False

class VotingSmoother:
    """Suavizado por votación de mayoría sobre ventana temporal"""
    
    def __init__(self, window_size: int = 5):
        """
        Inicializa suavizador por votación
        
        Args:
            window_size: Tamaño de ventana para votación
        """
        self.window_size = window_size
        self.buffer = deque(maxlen=window_size)
        
    def update(self, prediction: int) -> int:
        """
        Actualiza buffer y retorna predicción por mayoría
        
        Args:
            prediction: Índice de clase predicha
            
        Returns:
            Predicción suavizada por mayoría
        """
        self.buffer.append(prediction)
        
        if len(self.buffer) == self.window_size:
            # Votación de mayoría
            counter = Counter(self.buffer)
            return counter.most_common(1)[0][0]
        else:
            # Buffer no lleno, retornar predicción actual
            return prediction
    
    def reset(self):
        """Reinicia buffer de votación"""
        self.buffer.clear()

class ConfidenceGater:
    """Sistema de gating por confianza para predicciones estables"""
    
    def __init__(self, threshold: float = 0.6, unknown_label: str = "unknown"):
        """
        Inicializa gater de confianza
        
        Args:
            threshold: Umbral de confianza mínima
            unknown_label: Etiqueta para predicciones de baja confianza
        """
        self.threshold = threshold
        self.unknown_label = unknown_label
        
    def predict_label(self, probs: np.ndarray, labels: List[str]) -> Tuple[str, float]:
        """
        Predice etiqueta con gating por confianza
        
        Args:
            probs: Array de probabilidades
            labels: Lista de nombres de clases
            
        Returns:
            Tuple (label, confidence)
        """
        idx = np.argmax(probs)
        confidence = probs[idx]
        
        if confidence < self.threshold:
            return self.unknown_label, confidence
        else:
            return labels[idx], confidence

class RealtimeOptimizer:
    """Sistema completo de optimización para inferencia en tiempo real"""
    
    def __init__(self, model, labels: List[str], 
                 frame_skip: int = 2,
                 ema_alpha: float = 0.6,
                 confidence_threshold: float = 0.6,
                 voting_window: Optional[int] = None):
        """
        Inicializa optimizador de tiempo real
        
        Args:
            model: Modelo TensorFlow/Keras
            labels: Lista de nombres de clases
            frame_skip: Frames a saltar para reducir cómputo
            ema_alpha: Factor de suavizado EMA
            confidence_threshold: Umbral de confianza
            voting_window: Tamaño de ventana para votación (None = solo EMA)
        """
        self.model = model
        self.labels = labels
        self.num_classes = len(labels)
        self.frame_skip = frame_skip
        
        # Componentes de optimización
        self.ema_smoother = EMASmoother(ema_alpha)
        self.confidence_gater = ConfidenceGater(confidence_threshold)
        self.voting_smoother = VotingSmoother(voting_window) if voting_window else None
        
        # Estado de rendimiento
        self.frame_count = 0
        self.processed_frames = 0
        self.latency_samples = deque(maxlen=100)
        
        logger.info(f"RealtimeOptimizer inicializado:")
        logger.info(f"  Frame skip: {frame_skip}")
        logger.info(f"  EMA alpha: {ema_alpha}")
        logger.info(f"  Confidence threshold: {confidence_threshold}")
        logger.info(f"  Voting window: {voting_window}")
    
    def should_process_frame(self) -> bool:
        """
        Determina si el frame actual debe ser procesado
        
        Returns:
            True si se debe procesar el frame
        """
        return self.frame_count % self.frame_skip == 0
    
    def preprocess_landmarks(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Preprocesamiento usando módulo centralizado
        
        Args:
            landmarks: Array (21, 3) de landmarks
            
        Returns:
            Array preprocesado (1, 21, 3)
        """
        try:
            # Usar módulo centralizado
            normalized = normalize_landmarks(landmarks)
            return normalized
        except Exception as e:
            logger.error(f"Error en preprocess_landmarks: {e}")
            return np.zeros((1, 21, 3), dtype=np.float32)
    
    def predict_optimized(self, landmarks: np.ndarray) -> PredictionResult:
        """
        Predicción optimizada con suavizado temporal
        
        Args:
            landmarks: Array (21, 3) de landmarks
            
        Returns:
            PredictionResult con predicción suavizada
        """
        start_time = time.time()
        
        # Preprocesamiento
        x = self.preprocess_landmarks(landmarks)
        
        # Inferencia optimizada (evitar overhead de predict())
        probs = self.model(x, training=False).numpy()[0]
        
        # Suavizado EMA
        probs_smoothed = self.ema_smoother.update(probs)
        
        # Gating por confianza
        label, confidence = self.confidence_gater.predict_label(probs_smoothed, self.labels)
        
        # Suavizado por votación (si está habilitado)
        if self.voting_smoother:
            label_idx = self.labels.index(label) if label in self.labels else 0
            smoothed_idx = self.voting_smoother.update(label_idx)
            label = self.labels[smoothed_idx] if smoothed_idx < len(self.labels) else label
        
        # Medición de latencia
        latency = (time.time() - start_time) * 1000  # ms
        self.latency_samples.append(latency)
        
        self.processed_frames += 1
        
        return PredictionResult(
            label=label,
            confidence=confidence,
            probabilities=probs_smoothed,
            timestamp=time.time(),
            frame_id=self.frame_count
        )
    
    def update_frame_count(self):
        """Actualiza contador de frames"""
        self.frame_count += 1
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas de rendimiento
        
        Returns:
            Diccionario con métricas de rendimiento
        """
        if not self.latency_samples:
            return {
                "avg_latency_ms": 0,
                "max_latency_ms": 0,
                "min_latency_ms": 0,
                "fps": 0,
                "frame_skip_ratio": self.frame_skip,
                "processed_frames": 0,
                "total_frames": self.frame_count
            }
        
        avg_latency = np.mean(self.latency_samples)
        max_latency = np.max(self.latency_samples)
        min_latency = np.min(self.latency_samples)
        
        # Estimación de FPS basada en latencia
        estimated_fps = 1000 / avg_latency if avg_latency > 0 else 0
        
        return {
            "avg_latency_ms": float(avg_latency),
            "max_latency_ms": float(max_latency),
            "min_latency_ms": float(min_latency),
            "fps": float(estimated_fps),
            "frame_skip_ratio": self.frame_skip,
            "processed_frames": self.processed_frames,
            "total_frames": self.frame_count,
            "efficiency": self.processed_frames / max(self.frame_count, 1)
        }
    
    def reset_smoothers(self):
        """Reinicia todos los suavizadores"""
        self.ema_smoother.reset()
        if self.voting_smoother:
            self.voting_smoother.reset()

class AsyncRealtimePipeline:
    """Pipeline asíncrono para inferencia en tiempo real"""
    
    def __init__(self, model, labels: List[str], 
                 frame_queue_size: int = 2,
                 result_queue_size: int = 2,
                 **optimizer_kwargs):
        """
        Inicializa pipeline asíncrono
        
        Args:
            model: Modelo TensorFlow/Keras
            labels: Lista de nombres de clases
            frame_queue_size: Tamaño de cola de frames
            result_queue_size: Tamaño de cola de resultados
            **optimizer_kwargs: Argumentos para RealtimeOptimizer
        """
        self.model = model
        self.labels = labels
        
        # Colas para comunicación entre threads
        self.frame_queue = Queue(maxsize=frame_queue_size)
        self.result_queue = Queue(maxsize=result_queue_size)
        
        # Optimizador de tiempo real
        self.optimizer = RealtimeOptimizer(model, labels, **optimizer_kwargs)
        
        # Control de threads
        self.running = False
        self.threads = []
        
        logger.info("AsyncRealtimePipeline inicializado")
    
    def capture_loop(self, camera_source, stop_event):
        """
        Loop de captura de cámara en thread separado
        
        Args:
            camera_source: Fuente de cámara (cv2.VideoCapture)
            stop_event: Evento para detener ejecución
        """
        logger.info("Iniciando loop de captura")
        
        while not stop_event.is_set():
            ret, frame = camera_source.read()
            if not ret:
                continue
            
            # Agregar frame a cola si no está llena
            if not self.frame_queue.full():
                self.frame_queue.put((time.time(), frame))
            else:
                # Descartar frame si la cola está llena
                logger.debug("Frame descartado - cola llena")
    
    def inference_loop(self, mediapipe_hands, stop_event):
        """
        Loop de inferencia en thread separado
        
        Args:
            mediapipe_hands: Detector de manos MediaPipe
            stop_event: Evento para detener ejecución
        """
        logger.info("Iniciando loop de inferencia")
        
        while not stop_event.is_set():
            try:
                # Obtener frame con timeout
                timestamp, frame = self.frame_queue.get(timeout=0.1)
                
                # Extraer landmarks con MediaPipe
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = mediapipe_hands.process(frame_rgb)
                
                if results.multi_hand_landmarks:
                    # Extraer landmarks
                    hand_landmarks = results.multi_hand_landmarks[0]
                    landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
                    
                    # Actualizar contador de frames
                    self.optimizer.update_frame_count()
                    
                    # Procesar si corresponde (frame skipping)
                    if self.optimizer.should_process_frame():
                        # Predicción optimizada
                        result = self.optimizer.predict_optimized(landmarks)
                        
                        # Agregar resultado a cola
                        if not self.result_queue.full():
                            self.result_queue.put(result)
                    
            except Empty:
                # Timeout normal, continuar
                continue
            except Exception as e:
                logger.error(f"Error en loop de inferencia: {e}")
                continue
    
    def start_pipeline(self, camera_source, mediapipe_hands):
        """
        Inicia pipeline asíncrono
        
        Args:
            camera_source: Fuente de cámara
            mediapipe_hands: Detector de manos MediaPipe
        """
        if self.running:
            logger.warning("Pipeline ya está en ejecución")
            return
        
        self.running = True
        self.stop_event = threading.Event()
        
        # Crear y arrancar threads
        capture_thread = threading.Thread(
            target=self.capture_loop,
            args=(camera_source, self.stop_event),
            name="CaptureThread"
        )
        
        inference_thread = threading.Thread(
            target=self.inference_loop,
            args=(mediapipe_hands, self.stop_event),
            name="InferenceThread"
        )
        
        self.threads = [capture_thread, inference_thread]
        
        for thread in self.threads:
            thread.start()
            logger.info(f"Thread {thread.name} iniciado")
        
        logger.info("Pipeline asíncrono iniciado")
    
    def stop_pipeline(self):
        """Detiene pipeline asíncrono"""
        if not self.running:
            return
        
        logger.info("Deteniendo pipeline...")
        
        # Señalizar detención
        self.stop_event.set()
        
        # Esperar threads
        for thread in self.threads:
            thread.join(timeout=2.0)
            if thread.is_alive():
                logger.warning(f"Thread {thread.name} no terminó gracefully")
        
        self.running = False
        self.threads.clear()
        
        # Limpiar colas
        while not self.frame_queue.empty():
            self.frame_queue.get()
        while not self.result_queue.empty():
            self.result_queue.get()
        
        logger.info("Pipeline detenido")
    
    def get_latest_result(self, timeout: float = 0.1) -> Optional[PredictionResult]:
        """
        Obtiene resultado más reciente
        
        Args:
            timeout: Timeout para obtener resultado
            
        Returns:
            PredictionResult o None si no hay resultado
        """
        try:
            return self.result_queue.get(timeout=timeout)
        except Empty:
            return None
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas de rendimiento del optimizador"""
        return self.optimizer.get_performance_stats()


def main():
    """Función principal para demostración"""
    print("=== Sistema de Optimización para Inferencia en Tiempo Real ===")
    print("Este sistema optimiza latencia y estabilidad para reconocimiento de señas.")
    print()
    
    print("Componentes implementados:")
    print("✅ Suavizado temporal (EMA + Voting)")
    print("✅ Gating por confianza")
    print("✅ Frame skipping")
    print("✅ Pipeline asíncrono")
    print("✅ Optimización de inferencia")
    print()
    
    print("Para usar el sistema:")
    print("1. Crear RealtimeOptimizer para uso síncrono")
    print("2. O crear AsyncRealtimePipeline para uso asíncrono")
    print("3. Integrar con cámara y MediaPipe")
    print()
    
    print("Ejemplo síncrono:")
    print("optimizer = RealtimeOptimizer(model, labels)")
    print("result = optimizer.predict_optimized(landmarks)")
    print()
    print("Ejemplo asíncrono:")
    print("pipeline = AsyncRealtimePipeline(model, labels)")
    print("pipeline.start_pipeline(cap, hands)")
    print("result = pipeline.get_latest_result()")


if __name__ == "__main__":
    main()
