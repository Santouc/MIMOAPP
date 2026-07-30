import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np

from .path_service import PathService
from .sign_service import SignService
from visualization.landmark_average import average_dynamic_sequences, average_static_landmarks


class CaptureService:
    def __init__(self, paths: PathService | None = None, sign_service: SignService | None = None):
        self.paths = paths or PathService()
        self.sign_service = sign_service or SignService(self.paths)
        self.paths.ensure_app_dirs()

    def create_pending_session(self, sign_id: str, capture_type: str, samples: list[Any]) -> dict[str, Any]:
        sign = self.sign_service.get_sign(sign_id)
        if sign is None:
            raise ValueError(f"No existe una seña con id '{sign_id}'")
        capture_type = self._validate_capture_type(capture_type)
        if not samples:
            raise ValueError("No se puede crear una sesión sin muestras")

        session_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}"
        session_dir = self.paths.pending_captures_dir / session_id
        session_dir.mkdir(parents=True, exist_ok=False)

        metadata = {
            "session_id": session_id,
            "sign_id": sign_id,
            "sign_name": sign["name"],
            "capture_type": capture_type,
            "sample_count": len(samples),
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "status": "pending",
        }

        samples_path = session_dir / "samples.json"
        metadata_path = session_dir / "metadata.json"
        summary_path = session_dir / "summary.json"

        samples_path.write_text(json.dumps({"samples": samples}, ensure_ascii=False), encoding="utf-8")
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        summary = self.build_session_summary(session_dir)
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        return summary

    def list_pending_sessions(self) -> list[dict[str, Any]]:
        sessions = []
        if not self.paths.pending_captures_dir.exists():
            return sessions
        for session_dir in self.paths.pending_captures_dir.iterdir():
            if not session_dir.is_dir():
                continue
            metadata_path = session_dir / "metadata.json"
            if metadata_path.exists():
                sessions.append(json.loads(metadata_path.read_text(encoding="utf-8")))
        return sorted(sessions, key=lambda item: item["created_at"])

    def build_session_summary(self, session_dir: str | Path) -> dict[str, Any]:
        session_dir = Path(session_dir)
        metadata = json.loads((session_dir / "metadata.json").read_text(encoding="utf-8"))
        samples = json.loads((session_dir / "samples.json").read_text(encoding="utf-8"))["samples"]
        if metadata["capture_type"] == "static":
            average = average_static_landmarks(samples).tolist()
        else:
            average = average_dynamic_sequences(samples).tolist()
        return {
            **metadata,
            "average_landmarks": average,
        }

    def accept_pending_session(self, session_id: str) -> dict[str, Any]:
        session_dir = self.paths.pending_captures_dir / session_id
        if not session_dir.exists():
            raise FileNotFoundError(f"Sesión pendiente no encontrada: {session_id}")
        metadata = json.loads((session_dir / "metadata.json").read_text(encoding="utf-8"))
        samples = json.loads((session_dir / "samples.json").read_text(encoding="utf-8"))["samples"]
        labels = self.sign_service.export_labels()
        label_index = labels.index(metadata["sign_name"])

        if metadata["capture_type"] == "static":
            self._append_static_samples(samples, label_index, labels)
        else:
            self._append_dynamic_samples(samples, label_index, labels)

        self._refresh_counts(metadata["sign_id"])
        shutil.rmtree(session_dir)
        return {**metadata, "status": "accepted"}

    def reject_pending_session(self, session_id: str) -> dict[str, Any]:
        session_dir = self.paths.pending_captures_dir / session_id
        if not session_dir.exists():
            raise FileNotFoundError(f"Sesión pendiente no encontrada: {session_id}")
        metadata = json.loads((session_dir / "metadata.json").read_text(encoding="utf-8"))
        shutil.rmtree(session_dir)
        return {**metadata, "status": "rejected"}

    def delete_sign_everywhere(self, sign_id: str) -> dict[str, Any]:
        sign = self.sign_service.get_sign(sign_id)
        if sign is None:
            raise ValueError(f"No existe una seña con id '{sign_id}'")
        sign_name = sign["name"]

        for session in self.list_pending_sessions():
            if session["sign_id"] == sign_id:
                shutil.rmtree(self.paths.pending_captures_dir / session["session_id"], ignore_errors=True)

        deleted_static = self._remove_from_dataset(self.paths.static_dataset_path, sign_name)
        deleted_dynamic = self._remove_from_dataset(self.paths.dynamic_dataset_path, sign_name)
        deleted_registry = self.sign_service.delete_sign_registry_entry(sign_id)
        labels = self.sign_service.export_labels()
        self._rewrite_dataset_labels(self.paths.static_dataset_path, labels)
        self._rewrite_dataset_labels(self.paths.dynamic_dataset_path, labels)
        invalidated_models = self._invalidate_models()
        return {
            "sign": deleted_registry,
            "deleted_static_samples": deleted_static,
            "deleted_dynamic_samples": deleted_dynamic,
            "invalidated_models": invalidated_models,
        }

    def reset_all_data(self) -> dict[str, Any]:
        sign_count = len(self.sign_service.list_signs())
        pending_count = len(self.list_pending_sessions())
        for directory in (self.paths.pending_captures_dir, self.paths.previews_dir):
            if directory.exists():
                for item in directory.iterdir():
                    if item.is_dir():
                        shutil.rmtree(item, ignore_errors=True)
                    else:
                        item.unlink(missing_ok=True)
        removed_files = []
        for path in (
            self.paths.static_dataset_path,
            self.paths.dynamic_dataset_path,
            self.paths.static_labels_path,
            self.paths.dynamic_labels_path,
            self.paths.static_model_path,
            self.paths.dynamic_model_path,
        ):
            if path.exists():
                path.unlink()
                removed_files.append(str(path))
        for pattern in ("model.deleted_sign_*.h5", "model_dynamic.deleted_sign_*.h5"):
            for path in self.paths.models_dir.glob(pattern):
                path.unlink(missing_ok=True)
                removed_files.append(str(path))
        self.sign_service.reset_registry()
        return {
            "deleted_signs": sign_count,
            "deleted_pending_sessions": pending_count,
            "removed_files": removed_files,
        }

    def _append_static_samples(self, samples: list[Any], label_index: int, labels: list[str]) -> None:
        dataset = self._load_dataset(self.paths.static_dataset_path, labels)
        dataset["X"].extend(samples)
        dataset["y"].extend([label_index] * len(samples))
        dataset["labels"] = labels
        dataset["metadata"] = self._metadata("static", len(dataset["X"]), labels)
        self._save_dataset(self.paths.static_dataset_path, dataset)

    def _append_dynamic_samples(self, samples: list[Any], label_index: int, labels: list[str]) -> None:
        dataset = self._load_dataset(self.paths.dynamic_dataset_path, labels)
        dataset["X"].extend(samples)
        dataset["y"].extend([label_index] * len(samples))
        dataset["labels"] = labels
        dataset["metadata"] = self._metadata("dynamic", len(dataset["X"]), labels)
        self._save_dataset(self.paths.dynamic_dataset_path, dataset)

    def _load_dataset(self, path: Path, labels: list[str]) -> dict[str, Any]:
        if not path.exists():
            return {"X": [], "y": [], "labels": labels, "metadata": {}}
        return json.loads(path.read_text(encoding="utf-8"))

    def _save_dataset(self, path: Path, dataset: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(dataset, ensure_ascii=False, indent=2), encoding="utf-8")

    def _remove_from_dataset(self, path: Path, sign_name: str) -> int:
        if not path.exists():
            return 0
        dataset = json.loads(path.read_text(encoding="utf-8"))
        labels = dataset.get("labels", [])
        if sign_name not in labels:
            return 0
        removed_index = labels.index(sign_name)
        new_labels = [label for label in labels if label != sign_name]
        index_map = {old_idx: new_labels.index(label) for old_idx, label in enumerate(labels) if label != sign_name}
        new_x = []
        new_y = []
        removed = 0
        for sample, label_idx in zip(dataset.get("X", []), dataset.get("y", [])):
            if int(label_idx) == removed_index:
                removed += 1
                continue
            new_x.append(sample)
            new_y.append(index_map[int(label_idx)])
        dataset["X"] = new_x
        dataset["y"] = new_y
        dataset["labels"] = new_labels
        dataset["metadata"] = self._metadata(dataset.get("metadata", {}).get("type", "unknown"), len(new_x), new_labels)
        self._save_dataset(path, dataset)
        return removed

    def _rewrite_dataset_labels(self, path: Path, labels: list[str]) -> None:
        if not path.exists():
            return
        dataset = json.loads(path.read_text(encoding="utf-8"))
        dataset["labels"] = labels
        dataset["metadata"] = self._metadata(dataset.get("metadata", {}).get("type", "unknown"), len(dataset.get("X", [])), labels)
        self._save_dataset(path, dataset)

    def _refresh_counts(self, sign_id: str) -> None:
        sign = self.sign_service.get_sign(sign_id)
        if sign is None:
            return
        self.sign_service.update_sample_counts(
            sign_id,
            static_samples=self._count_samples(self.paths.static_dataset_path, sign["name"]),
            dynamic_samples=self._count_samples(self.paths.dynamic_dataset_path, sign["name"]),
        )

    def _count_samples(self, path: Path, sign_name: str) -> int:
        if not path.exists():
            return 0
        dataset = json.loads(path.read_text(encoding="utf-8"))
        labels = dataset.get("labels", [])
        if sign_name not in labels:
            return 0
        label_index = labels.index(sign_name)
        return sum(1 for item in dataset.get("y", []) if int(item) == label_index)

    def _metadata(self, dataset_type: str, sample_count: int, labels: list[str]) -> dict[str, Any]:
        return {
            "type": dataset_type,
            "num_samples": sample_count,
            "num_classes": len(labels),
            "landmark_format": "mediapipe_21_3d",
            "coordinate_system": "normalized",
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        }

    def _validate_capture_type(self, capture_type: str) -> str:
        normalized = str(capture_type).strip().lower()
        if normalized not in {"static", "dynamic"}:
            raise ValueError("El tipo de captura debe ser 'static' o 'dynamic'")
        return normalized

    def _invalidate_models(self) -> list[str]:
        invalidated = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        for model_path in (self.paths.static_model_path, self.paths.dynamic_model_path):
            if model_path.exists():
                invalid_path = model_path.with_suffix(f".deleted_sign_{timestamp}{model_path.suffix}")
                model_path.rename(invalid_path)
                invalidated.append(str(invalid_path))
        return invalidated
