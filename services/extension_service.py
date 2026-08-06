"""Servicio de extensiones (plugins) de la aplicación MIMO.

Este módulo define `ExtensionService`, que permite ampliar la aplicación con
extensiones de terceros sin modificar el código base. Cada extensión vive en
una subcarpeta de `extensions/` y debe incluir un archivo `extension.py` con:

- Una clase `Extension` con métodos opcionales:
    * `setup(context)`: se llama al activarla, recibe el contexto de la app.
    * `translate_actions(screen)`: devuelve acciones para la pantalla de
      traducción (botones extra), como objetos `TranslateAction`.
    * `transcription_changed(state)`: se notifica cada cambio de transcripción.
    * `shutdown()`: se llama al desactivarla o al cerrar la aplicación.
- Constantes opcionales de módulo: `NAME`, `VERSION` y `DESCRIPTION`.

El servicio descubre las carpetas de extensiones, las carga dinámicamente con
`importlib`, permite habilitarlas/deshabilitarlas (persistiendo la lista de
deshabilitadas en `data/extensions.json`) y aísla los errores de cada
extensión para que un fallo no afecte al resto de la aplicación.
"""

import importlib.util
import json
from dataclasses import dataclass
from typing import Callable

from .path_service import PathService


@dataclass
class TranslateAction:
    """Acción que una extensión aporta a la pantalla de traducción.

    Atributos:
        label: texto visible del botón o acción.
        callback: función sin argumentos que se ejecuta al activar la acción.
        key: identificador opcional de la acción.
    """

    label: str
    callback: Callable[[], None]
    key: str | None = None


@dataclass
class ExtensionInfo:
    """Información descriptiva de una extensión para mostrar en la interfaz.

    Atributos:
        folder: nombre de la carpeta de la extensión.
        name: nombre visible.
        version: versión declarada.
        description: descripción breve.
        enabled: True si el usuario no la deshabilitó.
        active: True si está cargada y en funcionamiento.
        error: último mensaje de error asociado, si lo hubo.
    """

    folder: str
    name: str
    version: str
    description: str
    enabled: bool
    active: bool
    error: str | None = None


@dataclass
class LoadedExtension:
    """Extensión cargada en memoria y activa.

    Atributos:
        folder: nombre de la carpeta de la extensión.
        name: nombre visible.
        version: versión declarada.
        description: descripción breve.
        instance: instancia de la clase `Extension` del módulo cargado.
    """

    folder: str
    name: str
    version: str
    description: str
    instance: object


class ExtensionService:
    """Descubre, carga y administra las extensiones de la aplicación."""

    def __init__(self, paths: PathService):
        """Inicializa el servicio y carga la configuración persistida.

        Args:
            paths: instancia de `PathService` para ubicar la carpeta de
                extensiones y el archivo de configuración.
        """
        self.paths = paths
        # Carpeta donde viven las extensiones y archivo de configuración.
        self.extensions_dir = self.paths.base_dir / "extensions"
        self.config_path = self.paths.data_dir / "extensions.json"
        # Estado en memoria: extensiones activas, errores por carpeta y
        # contexto de la aplicación que se pasa a cada extensión.
        self.extensions: list[LoadedExtension] = []
        self.errors: dict[str, str] = {}
        self.context = None
        # Conjunto de carpetas deshabilitadas por el usuario (persistido).
        self._disabled: set[str] = self._load_config()

    def load_all(self, context) -> None:
        """Carga todas las extensiones habilitadas.

        Args:
            context: objeto de contexto de la aplicación que se entrega al
                método `setup` de cada extensión.
        """
        self.context = context
        self.extensions_dir.mkdir(parents=True, exist_ok=True)
        # Activar cada carpeta descubierta, salvo las deshabilitadas.
        for folder_name in self._discover_folders():
            if folder_name in self._disabled:
                continue
            self._activate(folder_name)

    def list_extensions(self) -> list[ExtensionInfo]:
        """Devuelve la información de todas las extensiones detectadas.

        Combina las extensiones activas (con sus metadatos en memoria) y las
        inactivas (leyendo sus metadatos del módulo sin activarlas).

        Returns:
            Lista de `ExtensionInfo`, una por carpeta descubierta.
        """
        infos: list[ExtensionInfo] = []
        # Índice rápido de extensiones activas por nombre de carpeta.
        active_by_folder = {loaded.folder: loaded for loaded in self.extensions}
        for folder_name in self._discover_folders():
            loaded = active_by_folder.get(folder_name)
            # Si está activa se usan sus metadatos; si no, se leen del módulo.
            if loaded is not None:
                name, version, description = loaded.name, loaded.version, loaded.description
            else:
                name, version, description = self._read_metadata(folder_name)
            infos.append(
                ExtensionInfo(
                    folder=folder_name,
                    name=name,
                    version=version,
                    description=description,
                    enabled=folder_name not in self._disabled,
                    active=loaded is not None,
                    error=self.errors.get(folder_name),
                )
            )
        return infos

    def set_enabled(self, folder_name: str, enabled: bool) -> bool:
        """Habilita o deshabilita una extensión y persiste la decisión.

        Args:
            folder_name: carpeta de la extensión.
            enabled: True para habilitarla (y activarla), False para
                deshabilitarla (y desactivarla).

        Returns:
            True si la operación fue exitosa; al habilitar, refleja si la
            activación funcionó.
        """
        if enabled:
            # Quitar de la lista de deshabilitadas, guardar y activar.
            self._disabled.discard(folder_name)
            self._save_config()
            return self._activate(folder_name)
        # Agregar a la lista de deshabilitadas, guardar y desactivar.
        self._disabled.add(folder_name)
        self._save_config()
        self._deactivate(folder_name)
        return True

    def translate_actions(self, screen) -> list[TranslateAction]:
        """Recolecta las acciones de traducción que aportan las extensiones.

        Consulta el método opcional `translate_actions(screen)` de cada
        extensión activa; los errores individuales se registran sin
        interrumpir a las demás.

        Args:
            screen: pantalla de traducción que se pasa a cada extensión.

        Returns:
            Lista combinada de `TranslateAction` de todas las extensiones.
        """
        actions: list[TranslateAction] = []
        for loaded in self.extensions:
            # Solo consultar extensiones que implementen el método.
            provider = getattr(loaded.instance, "translate_actions", None)
            if not callable(provider):
                continue
            # Aislar errores: se registran por extensión y no se propagan.
            try:
                actions.extend(provider(screen))
            except Exception as error:
                self.errors[loaded.folder] = str(error)
        return actions

    def notify_transcription(self, state) -> None:
        """Notifica a las extensiones un cambio en la transcripción.

        Llama al método opcional `transcription_changed(state)` de cada
        extensión activa, aislando los errores individuales.

        Args:
            state: estado actual de la transcripción (TranscriptionState).
        """
        for loaded in self.extensions:
            # Solo notificar a extensiones que implementen el método.
            handler = getattr(loaded.instance, "transcription_changed", None)
            if not callable(handler):
                continue
            # Aislar errores: se registran por extensión y no se propagan.
            try:
                handler(state)
            except Exception as error:
                self.errors[loaded.folder] = str(error)

    def shutdown(self) -> None:
        """Apaga todas las extensiones activas (al cerrar la aplicación)."""
        # Se itera sobre una copia porque _shutdown_instance modifica la lista.
        for loaded in list(self.extensions):
            self._shutdown_instance(loaded)

    def _activate(self, folder_name: str) -> bool:
        """Activa (carga e inicializa) una extensión por su carpeta.

        Importa el módulo `extension.py`, instancia su clase `Extension`,
        llama a `setup(context)` si existe y la registra como activa.

        Args:
            folder_name: carpeta de la extensión a activar.

        Returns:
            True si quedó activa (o ya lo estaba), False si falló.
        """
        # Evitar activar dos veces la misma extensión.
        if any(loaded.folder == folder_name for loaded in self.extensions):
            return True
        # Verificar que exista el punto de entrada extension.py.
        entry = self.extensions_dir / folder_name / "extension.py"
        if not entry.exists():
            self.errors[folder_name] = "No se encontró extension.py"
            return False
        try:
            # Importar el módulo, crear la instancia y ejecutar su setup.
            module = self._import_module(folder_name, entry)
            extension = module.Extension()
            setup = getattr(extension, "setup", None)
            if callable(setup):
                setup(self.context)
            # Registrar la extensión activa con sus metadatos de módulo.
            self.extensions.append(
                LoadedExtension(
                    folder=folder_name,
                    name=str(getattr(module, "NAME", folder_name)),
                    version=str(getattr(module, "VERSION", "0.1")),
                    description=str(getattr(module, "DESCRIPTION", "")),
                    instance=extension,
                )
            )
            # Limpiar cualquier error previo asociado a esta carpeta.
            self.errors.pop(folder_name, None)
            return True
        except Exception as error:
            # Registrar el error para mostrarlo en la lista de extensiones.
            self.errors[folder_name] = str(error)
            return False

    def _deactivate(self, folder_name: str) -> None:
        """Desactiva (apaga y quita) una extensión activa por su carpeta."""
        # Copia de la lista porque _shutdown_instance la modifica al remover.
        for loaded in list(self.extensions):
            if loaded.folder == folder_name:
                self._shutdown_instance(loaded)

    def _shutdown_instance(self, loaded: LoadedExtension) -> None:
        """Apaga una extensión: llama a su `shutdown` y la quita de la lista.

        Args:
            loaded: extensión activa a apagar.
        """
        # Llamar al método shutdown si la extensión lo define, ignorando
        # cualquier error para no interrumpir el cierre.
        handler = getattr(loaded.instance, "shutdown", None)
        if callable(handler):
            try:
                handler()
            except Exception:
                pass
        # Quitar la extensión de la lista de activas.
        if loaded in self.extensions:
            self.extensions.remove(loaded)

    def _discover_folders(self) -> list[str]:
        """Descubre las carpetas de extensiones válidas.

        Returns:
            Nombres (ordenados) de las subcarpetas de `extensions/` que
            contienen un archivo `extension.py`.
        """
        if not self.extensions_dir.exists():
            return []
        # Solo cuentan las carpetas que tengan el punto de entrada.
        return sorted(
            folder.name
            for folder in self.extensions_dir.iterdir()
            if folder.is_dir() and (folder / "extension.py").exists()
        )

    def _read_metadata(self, folder_name: str) -> tuple[str, str, str]:
        """Lee los metadatos (NAME, VERSION, DESCRIPTION) de una extensión.

        Importa el módulo sin activar la extensión, solo para consultar sus
        constantes. Si falla, registra el error y devuelve valores por defecto.

        Args:
            folder_name: carpeta de la extensión.

        Returns:
            Tupla (nombre, versión, descripción).
        """
        entry = self.extensions_dir / folder_name / "extension.py"
        try:
            # Importar el módulo y leer sus constantes con valores por defecto.
            module = self._import_module(folder_name, entry)
            return (
                str(getattr(module, "NAME", folder_name)),
                str(getattr(module, "VERSION", "0.1")),
                str(getattr(module, "DESCRIPTION", "")),
            )
        except Exception as error:
            # Registrar el error y devolver metadatos mínimos.
            self.errors[folder_name] = str(error)
            return folder_name, "?", ""

    def _load_config(self) -> set[str]:
        """Carga desde disco el conjunto de extensiones deshabilitadas.

        Returns:
            Conjunto de nombres de carpetas deshabilitadas; vacío si el
            archivo no existe o está corrupto.
        """
        # Lectura tolerante a fallos: cualquier problema devuelve un set vacío.
        try:
            if self.config_path.exists():
                data = json.loads(self.config_path.read_text(encoding="utf-8-sig"))
                disabled = data.get("disabled", [])
                if isinstance(disabled, list):
                    return {str(item) for item in disabled}
        except (OSError, json.JSONDecodeError):
            pass
        return set()

    def _save_config(self) -> None:
        """Persiste en disco el conjunto de extensiones deshabilitadas."""
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        self.config_path.write_text(
            json.dumps({"disabled": sorted(self._disabled)}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _import_module(self, folder_name: str, entry_path):
        """Importa dinámicamente el módulo `extension.py` de una carpeta.

        Usa `importlib` para cargar el archivo como un módulo con un nombre
        único (prefijo `tls_extension_`) y así evitar colisiones.

        Args:
            folder_name: carpeta de la extensión (para el nombre del módulo).
            entry_path: ruta al archivo extension.py.

        Returns:
            El módulo importado y ejecutado.

        Raises:
            ImportError: si no se pudo crear la especificación de importación.
        """
        # Crear la especificación de importación a partir del archivo.
        module_name = f"tls_extension_{folder_name}"
        spec = importlib.util.spec_from_file_location(module_name, entry_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"No se pudo cargar la extensión {folder_name}")
        # Crear el módulo y ejecutar su código de nivel superior.
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
