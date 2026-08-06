"""
Módulo de utilidades del sistema

Paquete que agrupa las utilidades transversales del traductor de lenguaje
de señas:

- Config: configuración centralizada de todo el sistema (cámara,
  MediaPipe, TensorFlow, interfaz, logging, rutas, etc.).
- Logger / get_logger: sistema de registro (logging) centralizado, con
  una función de fábrica para obtener una instancia global compartida.

Al importarlas aquí, estas clases quedan disponibles directamente desde el
paquete, por ejemplo: "from utils import Config, get_logger".
"""

# Se re-exportan las clases y funciones principales para simplificar los
# imports desde otras partes del proyecto.
from .config import Config
from .logger import Logger, get_logger

# Nombres públicos del paquete: lo que se importa con "from utils import *".
__all__ = ['Config', 'Logger', 'get_logger']
