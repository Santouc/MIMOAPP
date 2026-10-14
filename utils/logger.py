#!/usr/bin/env python3
"""
Módulo de logging del sistema
Proporciona funcionalidad de registro centralizada

Este módulo define la clase Logger, un envoltorio (wrapper) sobre el módulo
estándar "logging" de Python, adaptado a las necesidades del traductor de
lenguaje de señas. Permite:

- Registrar mensajes en consola y/o archivo según la configuración.
- Métodos de conveniencia por nivel (debug, info, warning, error, critical).
- Registros especializados del dominio: rendimiento, detecciones de señas,
  información del sistema y de la cámara.
- Crear logs por sesión, cambiar el nivel en tiempo de ejecución, obtener
  estadísticas del archivo de log y limpiar logs antiguos.

También expone la función get_logger(), que devuelve una instancia global
compartida para usar el mismo logger en todo el sistema.
"""

import logging
import os
import sys
from datetime import datetime
from typing import Optional

class Logger:
    """
    Clase de logging centralizada para el sistema

    Encapsula un logger del módulo estándar "logging" y lo configura a
    partir de un diccionario de opciones (nivel, formato, salida a consola
    y/o archivo). Además de los métodos clásicos por nivel, incluye métodos
    específicos del dominio del traductor de señas (log_detection,
    log_performance, log_camera_info, etc.).
    """
    
    def __init__(self, name: str = "SignTranslator", config: Optional[dict] = None):
        """
        Inicializa el logger
        
        Args:
            name: Nombre del logger
            config: Configuración personalizada de logging
        """
        # Se guarda el nombre y se obtiene (o crea) el logger nativo de
        # Python asociado a ese nombre.
        self.name = name
        self.logger = logging.getLogger(name)
        
        # Configuración por defecto
        default_config = {
            'level': 'INFO',
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            'file_enabled': False,
            'console_enabled': True,
            'log_file': 'logs/sign_translator.log'
        }
        
        # Usar configuración personalizada si se proporciona
        self.config = config if config else default_config
        
        # Configurar el logger
        self._setup_logger()
    
    def _setup_logger(self):
        """Configura el logger con handlers y formatters"""
        
        # Limpiar handlers existentes
        self.logger.handlers.clear()
        
        # Establecer nivel de logging
        level = getattr(logging, self.config['level'].upper(), logging.INFO)
        self.logger.setLevel(level)
        
        # Crear formatter
        formatter = logging.Formatter(self.config['format'])
        
        # Handler para consola
        if self.config['console_enabled']:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(level)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
        
        # Handler para archivo
        if self.config['file_enabled']:
            # Asegurar que el directorio de logs exista
            log_dir = os.path.dirname(self.config['log_file'])
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir)
            
            file_handler = logging.FileHandler(self.config['log_file'])
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
        
        # Evitar duplicación de logs
        self.logger.propagate = False
    
    def debug(self, message: str, *args, **kwargs):
        """Registra un mensaje de nivel DEBUG"""
        self.logger.debug(message, *args, **kwargs)
    
    def info(self, message: str, *args, **kwargs):
        """Registra un mensaje de nivel INFO"""
        self.logger.info(message, *args, **kwargs)
    
    def warning(self, message: str, *args, **kwargs):
        """Registra un mensaje de nivel WARNING"""
        self.logger.warning(message, *args, **kwargs)
    
    def error(self, message: str, *args, **kwargs):
        """Registra un mensaje de nivel ERROR"""
        self.logger.error(message, *args, **kwargs)
    
    def critical(self, message: str, *args, **kwargs):
        """Registra un mensaje de nivel CRITICAL"""
        self.logger.critical(message, *args, **kwargs)
    
    def exception(self, message: str, *args, **kwargs):
        """Registra un mensaje de nivel ERROR con información de excepción"""
        self.logger.exception(message, *args, **kwargs)
    
    def log_performance(self, operation: str, duration_ms: float):
        """
        Registra métricas de rendimiento
        
        Args:
            operation: Nombre de la operación
            duration_ms: Duración en milisegundos
        """
        self.info(f"Performance: {operation} took {duration_ms:.2f}ms")
    
    def log_detection(self, sign: str, confidence: float, frame_count: int):
        """
        Registra información de detección de señas
        
        Args:
            sign: Seña detectada
            confidence: Confianza de la detección
            frame_count: Número de frame
        """
        self.info(f"Detection: Frame {frame_count} - Sign: {sign} (Confidence: {confidence:.3f})")
    
    def log_system_info(self):
        """Registra información del sistema"""
        # Importaciones locales para no cargar estas librerías si el método
        # nunca se utiliza.
        import platform
        import cv2
        
        # Datos básicos: sistema operativo, versión de Python y de OpenCV.
        self.info("=== System Information ===")
        self.info(f"Platform: {platform.system()} {platform.release()}")
        self.info(f"Python: {platform.python_version()}")
        self.info(f"OpenCV: {cv2.__version__}")
        
        # Se intenta registrar la versión de MediaPipe; si no está instalado
        # se deja una advertencia en lugar de fallar.
        try:
            import mediapipe as mp
            self.info(f"MediaPipe: {mp.__version__}")
        except ImportError:
            self.warning("MediaPipe not available")
        
        # Mismo tratamiento para TensorFlow: es opcional en tiempo de log.
        try:
            import tensorflow as tf
            self.info(f"TensorFlow: {tf.__version__}")
        except ImportError:
            self.warning("TensorFlow not available")
        
        self.info("==========================")
    
    def log_camera_info(self, camera_config: dict):
        """
        Registra información de la cámara
        
        Args:
            camera_config: Configuración de la cámara
        """
        self.info("=== Camera Configuration ===")
        self.info(f"Resolution: {camera_config.get('width', 'N/A')}x{camera_config.get('height', 'N/A')}")
        self.info(f"FPS: {camera_config.get('fps', 'N/A')}")
        self.info(f"Device ID: {camera_config.get('device_id', 'N/A')}")
        self.info("==============================")
    
    def create_session_log(self) -> str:
        """
        Crea un archivo de log específico para la sesión actual
        
        Returns:
            Ruta del archivo de log de la sesión
        """
        # El nombre del archivo incluye fecha y hora para que cada sesión
        # tenga su propio log y no se sobrescriban entre sí.
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_log_file = f"logs/session_{timestamp}.log"
        
        # Crear handler específico para esta sesión
        session_handler = logging.FileHandler(session_log_file)
        session_handler.setLevel(logging.DEBUG)
        
        # Formato detallado que incluye función y número de línea, útil
        # para depurar una sesión concreta.
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        session_handler.setFormatter(formatter)
        
        self.logger.addHandler(session_handler)
        
        self.info(f"Session log created: {session_log_file}")
        return session_log_file
    
    def set_level(self, level: str):
        """
        Cambia el nivel de logging
        
        Args:
            level: Nuevo nivel de logging (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        """
        # getattr lanza AttributeError si el nombre del nivel no existe en
        # el módulo logging; en ese caso se informa el error sin interrumpir.
        try:
            log_level = getattr(logging, level.upper())
            self.logger.setLevel(log_level)
            
            # Actualizar nivel de todos los handlers
            for handler in self.logger.handlers:
                handler.setLevel(log_level)
            
            self.info(f"Log level changed to: {level}")
        except AttributeError:
            self.error(f"Invalid log level: {level}")
    
    def get_log_stats(self) -> dict:
        """
        Obtiene estadísticas del archivo de log principal
        
        Returns:
            Diccionario con estadísticas del log
        """
        # Solo tiene sentido si el logging a archivo está habilitado y el
        # archivo realmente existe en disco.
        if not self.config['file_enabled']:
            return {"error": "File logging is disabled"}
        
        log_file = self.config['log_file']
        if not os.path.exists(log_file):
            return {"error": "Log file does not exist"}
        
        try:
            # Se lee el archivo completo en memoria línea por línea.
            with open(log_file, 'r') as f:
                lines = f.readlines()
            
            # Contar niveles de log
            level_counts = {
                'DEBUG': 0,
                'INFO': 0,
                'WARNING': 0,
                'ERROR': 0,
                'CRITICAL': 0
            }
            
            # Se recorre cada línea buscando el nivel de log embebido en el
            # formato estándar (" - NIVEL - ") para acumular los conteos.
            for line in lines:
                for level in level_counts:
                    if f' - {level} - ' in line:
                        level_counts[level] += 1
            
            # Se devuelven las estadísticas: total de líneas, conteo por
            # nivel, tamaño del archivo y fecha de última modificación.
            return {
                'total_lines': len(lines),
                'level_counts': level_counts,
                'file_size': os.path.getsize(log_file),
                'last_modified': os.path.getmtime(log_file)
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def cleanup_old_logs(self, days_to_keep: int = 7):
        """
        Limpia archivos de log antiguos
        
        Args:
            days_to_keep: Días de logs a mantener
        """
        import time
        
        # Si no hay directorio de logs configurado o no existe, no hay
        # nada que limpiar.
        log_dir = os.path.dirname(self.config['log_file'])
        if not log_dir or not os.path.exists(log_dir):
            return
        
        # Se calcula la fecha límite: todo archivo modificado antes de este
        # instante (en segundos desde época Unix) se considera antiguo.
        current_time = time.time()
        cutoff_time = current_time - (days_to_keep * 24 * 60 * 60)
        
        deleted_files = []
        
        # Se recorren los archivos .log del directorio y se eliminan los
        # que superan la antigüedad permitida.
        for filename in os.listdir(log_dir):
            if filename.endswith('.log'):
                file_path = os.path.join(log_dir, filename)
                if os.path.getmtime(file_path) < cutoff_time:
                    try:
                        os.remove(file_path)
                        deleted_files.append(filename)
                    except Exception as e:
                        self.error(f"Error deleting log file {filename}: {e}")
        
        # Se informa el resultado de la limpieza.
        if deleted_files:
            self.info(f"Cleaned up {len(deleted_files)} old log files")
        else:
            self.debug("No old log files to clean up")

# Logger global para uso fácil en todo el sistema
_global_logger = None

def get_logger(name: str = "SignTranslator", config: Optional[dict] = None) -> Logger:
    """
    Obtiene una instancia global del logger
    
    Args:
        name: Nombre del logger
        config: Configuración personalizada
        
    Returns:
        Instancia del logger
    """
    global _global_logger
    
    # Patrón singleton simple: se reutiliza la instancia existente salvo
    # que aún no exista o que se pida un logger con otro nombre.
    if _global_logger is None or _global_logger.name != name:
        _global_logger = Logger(name, config)
    
    return _global_logger
