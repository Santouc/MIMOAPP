import importlib.util
import json
from dataclasses import dataclass
from typing import Callable

from .path_service import PathService


@dataclass
class TranslateAction:
    label: str
    callback: Callable[[], None]
    key: str | None = None


@dataclass
class ExtensionInfo:
    folder: str
    name: str
    version: str
    description: str
    enabled: bool
    active: bool
    error: str | None = None


@dataclass
class LoadedExtension:
    folder: str
    name: str
    version: str
    description: str
    instance: object


class ExtensionService:
    def __init__(self, paths: PathService):
        self.paths = paths
        self.extensions_dir = self.paths.base_dir / "extensions"
        self.config_path = self.paths.data_dir / "extensions.json"
        self.extensions: list[LoadedExtension] = []
        self.errors: dict[str, str] = {}
        self.context = None
        self._disabled: set[str] = self._load_config()

    def load_all(self, context) -> None:
        self.context = context
        self.extensions_dir.mkdir(parents=True, exist_ok=True)
        for folder_name in self._discover_folders():
            if folder_name in self._disabled:
                continue
            self._activate(folder_name)

    def list_extensions(self) -> list[ExtensionInfo]:
        infos: list[ExtensionInfo] = []
        active_by_folder = {loaded.folder: loaded for loaded in self.extensions}
        for folder_name in self._discover_folders():
            loaded = active_by_folder.get(folder_name)
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
        if enabled:
            self._disabled.discard(folder_name)
            self._save_config()
            return self._activate(folder_name)
        self._disabled.add(folder_name)
        self._save_config()
        self._deactivate(folder_name)
        return True

    def translate_actions(self, screen) -> list[TranslateAction]:
        actions: list[TranslateAction] = []
        for loaded in self.extensions:
            provider = getattr(loaded.instance, "translate_actions", None)
            if not callable(provider):
                continue
            try:
                actions.extend(provider(screen))
            except Exception as error:
                self.errors[loaded.folder] = str(error)
        return actions

    def notify_transcription(self, state) -> None:
        for loaded in self.extensions:
            handler = getattr(loaded.instance, "transcription_changed", None)
            if not callable(handler):
                continue
            try:
                handler(state)
            except Exception as error:
                self.errors[loaded.folder] = str(error)

    def shutdown(self) -> None:
        for loaded in list(self.extensions):
            self._shutdown_instance(loaded)

    def _activate(self, folder_name: str) -> bool:
        if any(loaded.folder == folder_name for loaded in self.extensions):
            return True
        entry = self.extensions_dir / folder_name / "extension.py"
        if not entry.exists():
            self.errors[folder_name] = "No se encontró extension.py"
            return False
        try:
            module = self._import_module(folder_name, entry)
            extension = module.Extension()
            setup = getattr(extension, "setup", None)
            if callable(setup):
                setup(self.context)
            self.extensions.append(
                LoadedExtension(
                    folder=folder_name,
                    name=str(getattr(module, "NAME", folder_name)),
                    version=str(getattr(module, "VERSION", "0.1")),
                    description=str(getattr(module, "DESCRIPTION", "")),
                    instance=extension,
                )
            )
            self.errors.pop(folder_name, None)
            return True
        except Exception as error:
            self.errors[folder_name] = str(error)
            return False

    def _deactivate(self, folder_name: str) -> None:
        for loaded in list(self.extensions):
            if loaded.folder == folder_name:
                self._shutdown_instance(loaded)

    def _shutdown_instance(self, loaded: LoadedExtension) -> None:
        handler = getattr(loaded.instance, "shutdown", None)
        if callable(handler):
            try:
                handler()
            except Exception:
                pass
        if loaded in self.extensions:
            self.extensions.remove(loaded)

    def _discover_folders(self) -> list[str]:
        if not self.extensions_dir.exists():
            return []
        return sorted(
            folder.name
            for folder in self.extensions_dir.iterdir()
            if folder.is_dir() and (folder / "extension.py").exists()
        )

    def _read_metadata(self, folder_name: str) -> tuple[str, str, str]:
        entry = self.extensions_dir / folder_name / "extension.py"
        try:
            module = self._import_module(folder_name, entry)
            return (
                str(getattr(module, "NAME", folder_name)),
                str(getattr(module, "VERSION", "0.1")),
                str(getattr(module, "DESCRIPTION", "")),
            )
        except Exception as error:
            self.errors[folder_name] = str(error)
            return folder_name, "?", ""

    def _load_config(self) -> set[str]:
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
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        self.config_path.write_text(
            json.dumps({"disabled": sorted(self._disabled)}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _import_module(self, folder_name: str, entry_path):
        module_name = f"tls_extension_{folder_name}"
        spec = importlib.util.spec_from_file_location(module_name, entry_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"No se pudo cargar la extensión {folder_name}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
