#!/usr/bin/env python3
"""
Módulo de configuración del sistema
Centraliza todas las configuraciones y parámetros
"""

import os
from typing import Dict, Any

class Config:
    """Clase de configuración centralizada"""
    
    def __init__(self):
        """Inicializa la configuración con valores por defecto"""
        
        # Configuración de cámara
        self.camera_config = {
            'width': 640,
            'height': 480,
            'fps': 15,
            'device_id': 0
        }
        
        # Configuración de MediaPipe
        self.mediapipe_config = {
            'max_hands': 2,
            'detection_confidence': 0.5,
            'tracking_confidence': 0.5,
            'static_image_mode': False
        }
        
        # Configuración de TensorFlow
        self.tensorflow_config = {
            'model_path': 'data/models/model.h5',
            'labels_path': 'data/models/labels.json',
            'confidence_threshold': 0.7,
            'input_shape': (21, 3),
            'num_classes': 8
        }
        
        # Configuración de procesamiento
        self.processing_config = {
            'buffer_size': 10,
            'consensus_threshold': 0.8,
            'min_confidence': 0.5,
            'stabilization_frames': 5
        }
        
        # Configuración de interfaz
        self.interface_config = {
            'window_name': 'Traductor de Lenguaje de Señas',
            'show_fps': True,
            'show_landmarks': True,
            'show_confidence': True,
            'font_scale': 0.7,
            'font_thickness': 2
        }
        
        # Configuración de logging
        self.logging_config = {
            'level': 'INFO',
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            'file_enabled': False,
            'console_enabled': True,
            'log_file': 'logs/sign_translator.log'
        }
        
        # Rutas del sistema
        self.paths = {
            'base_dir': os.getcwd(),
            'models_dir': 'models',
            'logs_dir': 'logs',
            'data_dir': 'data',
            'temp_dir': 'temp'
        }
        
        # Configuración de rendimiento
        self.performance_config = {
            'target_fps': 15,
            'max_processing_time_ms': 66,  # ~15 FPS
            'enable_gpu': True,
            'optimize_for_speed': True
        }
        
        # Configuración de señas
        self.signs_config = {
            'supported_signs': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'],
            'transition_delay_ms': 500,
            'hold_time_ms': 1000,
            'max_signs_per_sequence': 20
        }
        
        # Cargar configuración desde variables de entorno si existen
        self._load_from_env()
        
        # Crear directorios necesarios
        self._create_directories()
    
    def _load_from_env(self):
        """Carga configuración desde variables de entorno"""
        env_mappings = {
            'CAMERA_WIDTH': ('camera_config', 'width', int),
            'CAMERA_HEIGHT': ('camera_config', 'height', int),
            'CAMERA_FPS': ('camera_config', 'fps', int),
            'MAX_HANDS': ('mediapipe_config', 'max_hands', int),
            'DETECTION_CONFIDENCE': ('mediapipe_config', 'detection_confidence', float),
            'CONFIDENCE_THRESHOLD': ('tensorflow_config', 'confidence_threshold', float),
            'LOG_LEVEL': ('logging_config', 'level', str),
            'ENABLE_GPU': ('performance_config', 'enable_gpu', bool)
        }
        
        for env_var, (config_section, config_key, type_func) in env_mappings.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                try:
                    if type_func == bool:
                        value = env_value.lower() in ('true', '1', 'yes', 'on')
                    else:
                        value = type_func(env_value)
                    
                    getattr(self, config_section)[config_key] = value
                    print(f"Configuración cargada desde {env_var}: {value}")
                except (ValueError, AttributeError) as e:
                    print(f"Error cargando {env_var}: {e}")
    
    def _create_directories(self):
        """Crea los directorios necesarios si no existen"""
        directories = [
            self.paths['models_dir'],
            self.paths['logs_dir'],
            self.paths['data_dir'],
            self.paths['temp_dir']
        ]
        
        for directory in directories:
            if not os.path.exists(directory):
                os.makedirs(directory)
                print(f"Directorio creado: {directory}")
    
    def get_camera_config(self) -> Dict[str, Any]:
        """Retorna la configuración de la cámara"""
        return self.camera_config.copy()
    
    def get_mediapipe_config(self) -> Dict[str, Any]:
        """Retorna la configuración de MediaPipe"""
        return self.mediapipe_config.copy()
    
    def get_tensorflow_config(self) -> Dict[str, Any]:
        """Retorna la configuración de TensorFlow"""
        return self.tensorflow_config.copy()
    
    def get_processing_config(self) -> Dict[str, Any]:
        """Retorna la configuración de procesamiento"""
        return self.processing_config.copy()
    
    def get_interface_config(self) -> Dict[str, Any]:
        """Retorna la configuración de interfaz"""
        return self.interface_config.copy()
    
    def get_logging_config(self) -> Dict[str, Any]:
        """Retorna la configuración de logging"""
        return self.logging_config.copy()
    
    def get_performance_config(self) -> Dict[str, Any]:
        """Retorna la configuración de rendimiento"""
        return self.performance_config.copy()
    
    def get_paths(self) -> Dict[str, str]:
        """Retorna las rutas del sistema"""
        return self.paths.copy()
    
    def update_config(self, section: str, key: str, value: Any):
        """
        Actualiza un valor de configuración
        
        Args:
            section: Sección de configuración
            key: Clave a actualizar
            value: Nuevo valor
        """
        if hasattr(self, f"{section}_config"):
            getattr(self, f"{section}_config")[key] = value
            print(f"Configuración actualizada: {section}.{key} = {value}")
        else:
            print(f"Sección de configuración no encontrada: {section}")
    
    def save_to_file(self, config_path: str = 'config.json'):
        """
        Guarda la configuración actual en un archivo JSON
        
        Args:
            config_path: Ruta donde guardar el archivo de configuración
        """
        import json
        
        config_dict = {}
        sections = ['camera', 'mediapipe', 'tensorflow', 'processing', 
                   'interface', 'logging', 'performance', 'signs']
        
        for section in sections:
            config_dict[section] = getattr(self, f"{section}_config")
        
        config_dict['paths'] = self.paths
        
        try:
            with open(config_path, 'w') as f:
                json.dump(config_dict, f, indent=2)
            print(f"Configuración guardada en: {config_path}")
        except Exception as e:
            print(f"Error guardando configuración: {e}")
    
    def load_from_file(self, config_path: str = 'config.json'):
        """
        Carga configuración desde un archivo JSON
        
        Args:
            config_path: Ruta del archivo de configuración
        """
        import json
        
        if not os.path.exists(config_path):
            print(f"Archivo de configuración no encontrado: {config_path}")
            return
        
        try:
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
            
            sections = ['camera', 'mediapipe', 'tensorflow', 'processing', 
                       'interface', 'logging', 'performance', 'signs']
            
            for section in sections:
                if section in config_dict:
                    setattr(self, f"{section}_config", config_dict[section])
            
            if 'paths' in config_dict:
                self.paths = config_dict['paths']
            
            print(f"Configuración cargada desde: {config_path}")
            
        except Exception as e:
            print(f"Error cargando configuración: {e}")
    
    def __str__(self) -> str:
        """Representación string de la configuración"""
        return f"Config(camera={self.camera_config}, mediapipe={self.mediapipe_config}, tensorflow={self.tensorflow_config})"
