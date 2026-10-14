"""Paquete de servicios de la aplicación MIMO.

Este paquete agrupa toda la lógica de negocio de la aplicación de escritorio,
separada de la interfaz gráfica (PySide6). Cada servicio se encarga de una
responsabilidad concreta:

- PathService: centraliza las rutas de archivos y carpetas de datos.
- SignService: administra el registro de señas (altas, bajas, consultas).
- CaptureService: gestiona las sesiones de captura pendientes y los datasets.
- DocumentService: lee documentos Markdown (créditos y manual de uso).
- LibraryService: importa bibliotecas de señas predefinidas (p. ej. el alfabeto).
- TrainingService: entrena los modelos Keras estático y dinámico.
- TranscriptionService: convierte las señas detectadas en texto legible.
- ExtensionService: carga y administra extensiones (plugins) de terceros.

Al importar el paquete, se reexportan las clases públicas para que el resto
de la aplicación pueda hacer `from services import ...` de forma directa.
"""

# Reexportación de las clases públicas de cada módulo del paquete.
from .path_service import PathService
from .sign_service import SignService
from .capture_service import CaptureService
from .document_service import DocumentService
from .library_service import LibraryService
from .training_service import TrainingService
from .transcription_service import TranscriptionService, TranscriptionState
from .extension_service import ExtensionService, TranslateAction

# Lista explícita de los nombres públicos del paquete (lo que expone
# `from services import *` y lo que se considera API estable).
__all__ = [
    "PathService",
    "SignService",
    "CaptureService",
    "DocumentService",
    "LibraryService",
    "TrainingService",
    "TranscriptionService",
    "TranscriptionState",
    "ExtensionService",
    "TranslateAction",
]
