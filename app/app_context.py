"""Contexto de aplicación de MIMO (T.L.S).

Este módulo define la clase :class:`AppContext`, que actúa como contenedor
central de dependencias (patrón "composition root" o inyección de dependencias
simple). Aquí se crean e interconectan todos los servicios que necesita la
aplicación: rutas de archivos, gestión de señas, capturas, documentos,
entrenamiento de modelos, librerías de señas, transcripción y extensiones.

Las pantallas de la interfaz reciben una instancia de este contexto y acceden
a los servicios a través de sus atributos, evitando así variables globales y
facilitando las pruebas.
"""

from services import CaptureService, DocumentService, ExtensionService, LibraryService, PathService, SignService, TrainingService, TranscriptionService


class AppContext:
    """Contenedor central de servicios compartidos por toda la aplicación.

    Atributos:
        paths: Servicio de rutas; resuelve y crea los directorios de datos.
        signs: Servicio de registro de señas (alta, listado, labels).
        captures: Servicio de capturas de muestras (pendientes y oficiales).
        documents: Servicio de lectura de documentos (créditos, manual).
        training: Servicio de entrenamiento de los modelos de clasificación.
        libraries: Servicio de librerías predefinidas (p. ej. alfabeto A-Z).
        transcription: Servicio de transcripción de señas a texto.
        extensions: Servicio de carga y gestión de extensiones opcionales.
    """

    def __init__(self):
        """Inicializa y conecta todos los servicios de la aplicación.

        El orden de creación importa: primero se resuelven las rutas y se
        garantiza que existan los directorios; luego se crean los servicios
        que dependen de esas rutas y, al final, se cargan las extensiones,
        que reciben el contexto completo ya construido.
        """
        # Servicio de rutas: base para localizar datasets, modelos y documentos.
        self.paths = PathService()
        # Asegura que existan las carpetas de datos de la aplicación.
        self.paths.ensure_app_dirs()
        # Registro de señas (nombres, tipos y labels exportadas).
        self.signs = SignService(self.paths)
        # Capturas de muestras: sesiones pendientes, aceptación y borrado.
        self.captures = CaptureService(self.paths, self.signs)
        # Lectura de documentos estáticos como créditos y manual de uso.
        self.documents = DocumentService(self.paths)
        # Entrenamiento de los modelos estático y dinámico.
        self.training = TrainingService(self.paths)
        # Importación de librerías de señas predefinidas.
        self.libraries = LibraryService(self.signs)
        # Conversión de señas reconocidas a texto (con correcciones aprendidas).
        self.transcription = TranscriptionService(self.paths)
        # Sistema de extensiones: se descubren y cargan al final,
        # cuando el resto del contexto ya está disponible.
        self.extensions = ExtensionService(self.paths)
        self.extensions.load_all(self)
