import shutil
from pathlib import Path


class PathService:
    def __init__(self, base_dir: Path | None = None):
        if base_dir:
            self.base_dir = Path(base_dir).resolve()
            self.resource_dir = self.base_dir
        else:
            self.base_dir = Path(__file__).resolve().parent.parent
            self.resource_dir = self.base_dir

        self.data_dir = self.base_dir / "data"
        self.models_dir = self.data_dir / "models"
        self.datasets_dir = self.data_dir / "datasets"
        self.signs_dir = self.data_dir / "signs"
        self.pending_captures_dir = self.data_dir / "pending_captures"
        self.previews_dir = self.data_dir / "previews"
        self.transcription_dir = self.data_dir / "transcription"
        self.docs_dir = self.resource_dir / "docs"
        self.readme_path = self.resource_dir / "README.md"
        self.manual_path = self.docs_dir / "manual_uso.md"
        self.signs_path = self.signs_dir / "signs.json"
        self.static_dataset_path = self.datasets_dir / "dataset_static.json"
        self.dynamic_dataset_path = self.datasets_dir / "dataset_dynamic.json"
        self.static_labels_path = self.models_dir / "labels.json"
        self.dynamic_labels_path = self.models_dir / "labels_dynamic.json"
        self.static_model_path = self.models_dir / "model.h5"
        self.dynamic_model_path = self.models_dir / "model_dynamic.h5"
        self.transcription_rules_path = self.transcription_dir / "rules.json"
        self.transcription_memory_path = self.transcription_dir / "memory.json"

    def ensure_app_dirs(self) -> None:
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
