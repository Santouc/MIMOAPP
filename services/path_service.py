"""Servicio de rutas de la aplicación MIMO.

Este módulo define `PathService`, la clase que centraliza TODAS las rutas de
archivos y carpetas que utiliza la aplicación (datasets, modelos entrenados,
registro de señas, capturas pendientes, documentación, etc.).

Tener las rutas en un único lugar evita "rutas mágicas" repartidas por el
código: cualquier otro servicio recibe una instancia de `PathService` y
consulta aquí dónde leer o escribir sus datos.
"""

import shutil
from pathlib import Path


class PathService:
    """Centraliza y expone las rutas de datos y recursos de la aplicación.

    Atributos principales:
        base_dir: carpeta raíz del proyecto.
        data_dir: carpeta `data/` donde se guardan todos los datos generados.
        models_dir / datasets_dir / signs_dir: subcarpetas de datos.
        pending_captures_dir: sesiones de captura aún no aceptadas.
        previews_dir: imágenes o videos de vista previa.
        transcription_dir: reglas y memoria del servicio de transcripción.
        *_path: rutas a archivos concretos (JSON de señas, modelos .h5, etc.).
    """

    def __init__(self, base_dir: Path | None = None):
        """Inicializa todas las rutas a partir de un directorio base.

        Args:
            base_dir: carpeta raíz del proyecto. Si no se indica, se deduce
                automáticamente como la carpeta padre de `services/` (es decir,
                la raíz del repositorio donde vive este archivo).
        """
        # Determinar el directorio base: el recibido por parámetro o, en su
        # defecto, la carpeta que contiene al paquete `services`.
        if base_dir:
            self.base_dir = Path(base_dir).resolve()
            self.resource_dir = self.base_dir
        else:
            self.base_dir = Path(__file__).resolve().parent.parent
            self.resource_dir = self.base_dir

        # Carpetas principales de datos generados por la aplicación.
        self.data_dir = self.base_dir / "data"
        self.models_dir = self.data_dir / "models"
        self.datasets_dir = self.data_dir / "datasets"
        self.signs_dir = self.data_dir / "signs"
        self.pending_captures_dir = self.data_dir / "pending_captures"
        self.previews_dir = self.data_dir / "previews"
        self.transcription_dir = self.data_dir / "transcription"
        # Rutas de documentación (créditos y manual de uso en Markdown).
        self.docs_dir = self.resource_dir / "docs"
        self.readme_path = self.resource_dir / "README.md"
        self.manual_path = self.docs_dir / "manual_uso.md"
        # Archivos de datos concretos: registro de señas, datasets de
        # entrenamiento (estático y dinámico), etiquetas y modelos Keras.
        self.signs_path = self.signs_dir / "signs.json"
        self.static_dataset_path = self.datasets_dir / "dataset_static.json"
        self.dynamic_dataset_path = self.datasets_dir / "dataset_dynamic.json"
        self.static_labels_path = self.models_dir / "labels.json"
        self.dynamic_labels_path = self.models_dir / "labels_dynamic.json"
        self.static_model_path = self.models_dir / "model.h5"
        self.dynamic_model_path = self.models_dir / "model_dynamic.h5"
        # Archivos del servicio de transcripción: reglas configurables y
        # memoria de frases aprendidas por el usuario.
        self.transcription_rules_path = self.transcription_dir / "rules.json"
        self.transcription_memory_path = self.transcription_dir / "memory.json"

    def ensure_app_dirs(self) -> None:
        """Crea todas las carpetas de datos si aún no existen.

        Es idempotente: puede llamarse varias veces sin efectos secundarios,
        ya que usa `exist_ok=True`. Los demás servicios la invocan al
        inicializarse para garantizar que la estructura de carpetas exista.
        """
        # Recorrer cada carpeta requerida y crearla (incluyendo padres).
        for directory in (
            self.data_dir,
            self.models_dir,
            self.datasets_dir,
            self.signs_dir,
            self.pending_captures_dir,
            self.previews_dir,
            self.transcription_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)
