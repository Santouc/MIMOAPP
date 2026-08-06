#!/usr/bin/env python3
"""
Módulo de configuración del sistema
Centraliza todas las configuraciones y parámetros

Define la clase Config, que agrupa en un solo lugar todos los parámetros
del traductor de lenguaje de señas: cámara, MediaPipe, TensorFlow,
procesamiento de frames, interfaz gráfica, logging, rutas del sistema,
rendimiento y señas soportadas.

Características principales:
- Valores por defecto razonables para todos los parámetros.
- Posibilidad de sobrescribir valores mediante variables de entorno
  (por ejemplo CAMERA_WIDTH, LOG_LEVEL, ENABLE_GPU).
- Creación automática de los directorios de trabajo necesarios.
- Guardado y carga de la configuración completa en formato JSON.
- Métodos "get_*" que devuelven copias de cada sección para evitar
  modificaciones accidentales del estado interno.
"""

import os
from typing import Dict, Any

class Config:
    """
    Clase de configuración centralizada

    Al instanciarse, inicializa cada sección de configuración como un
    diccionario de atributos (camera_config, mediapipe_config, etc.),
    aplica las variables de entorno definidas y crea los directorios
    de trabajo (models, logs, data, temp) si no existen.
    """
    
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
        """
        Carga configuración desde variables de entorno

        Recorre un mapeo de variables de entorno conocidas hacia la sección
        y clave de configuración correspondientes, convirtiendo el valor de
        texto al tipo adecuado (int, float, bool o str). Si la variable no
        está definida, se conserva el valor por defecto.
        """
        # Mapeo: nombre de variable de entorno -> (sección, clave, tipo).
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
        
        # Se procesa cada variable: si existe en el entorno, se convierte al
        # tipo esperado y se sobrescribe el valor por defecto de esa sección.
        for env_var, (config_section, config_key, type_func) in env_mappings.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                try:
                    # Los booleanos requieren un tratamiento especial porque
                    # bool("false") sería True; se aceptan varias formas
                    # comunes de expresar "verdadero".
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
        # Lista de directorios de trabajo que la aplicación necesita.
        directories = [
            self.paths['models_dir'],
            self.paths['logs_dir'],
            self.paths['data_dir'],
            self.paths['temp_dir']
        ]
        
        # Se crea cada directorio solo si aún no existe en disco.
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
        # Se busca el atributo "<section>_config"; si existe, se actualiza
        # la clave indicada; de lo contrario, se informa el error.
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
        
        # Se arma un diccionario con todas las secciones de configuración
        # para serializarlas juntas.
        config_dict = {}
        sections = ['camera', 'mediapipe', 'tensorflow', 'processing', 
                   'interface', 'logging', 'performance', 'signs']
        
        for section in sections:
            config_dict[section] = getattr(self, f"{section}_config")
        
        # Las rutas del sistema también forman parte del archivo guardado.
        config_dict['paths'] = self.paths
        
        # Se escribe el JSON con indentación para que sea legible; cualquier
        # error de escritura se informa sin interrumpir la aplicación.
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
        
        # Si el archivo no existe, se avisa y se mantienen los valores
        # actuales de configuración.
        if not os.path.exists(config_path):
            print(f"Archivo de configuración no encontrado: {config_path}")
            return
        
        try:
            # Se lee y parsea el archivo JSON completo.
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
            
            # Solo se reemplazan las secciones presentes en el archivo;
            # las ausentes conservan sus valores actuales.
            sections = ['camera', 'mediapipe', 'tensorflow', 'processing', 
                       'interface', 'logging', 'performance', 'signs']
            
            for section in sections:
                if section in config_dict:
                    setattr(self, f"{section}_config", config_dict[section])
            
            # Las rutas del sistema se restauran si estaban guardadas.
            if 'paths' in config_dict:
                self.paths = config_dict['paths']
            
            print(f"Configuración cargada desde: {config_path}")
            
        except Exception as e:
            print(f"Error cargando configuración: {e}")
    
    def __str__(self) -> str:
        """Representación string de la configuración"""
        return f"Config(camera={self.camera_config}, mediapipe={self.mediapipe_config}, tensorflow={self.tensorflow_config})"
