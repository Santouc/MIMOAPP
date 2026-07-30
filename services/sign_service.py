import json
import re
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from .path_service import PathService


class SignService:
    def __init__(self, paths: PathService | None = None):
        self.paths = paths or PathService()
        self.paths.ensure_app_dirs()

    def list_signs(self) -> list[dict[str, Any]]:
        data = self._load_registry()
        return sorted(data["signs"], key=lambda sign: sign["name"].lower())

    def get_sign(self, sign_id: str) -> dict[str, Any] | None:
        for sign in self.list_signs():
            if sign["id"] == sign_id:
                return sign
        return None

    def find_by_name(self, name: str) -> dict[str, Any] | None:
        normalized = self._normalize_name(name)
        for sign in self.list_signs():
            if self._normalize_name(sign["name"]) == normalized:
                return sign
        return None

    def add_sign(self, name: str, sign_types: list[str] | None = None) -> dict[str, Any]:
        clean_name = self._clean_name(name)
        if not clean_name:
            raise ValueError("El nombre de la seña no puede estar vacío")
        if self.find_by_name(clean_name):
            raise ValueError(f"La seña '{clean_name}' ya existe")

        now = self._now()
        sign = {
            "id": self._build_sign_id(clean_name),
            "name": clean_name,
            "types": self._normalize_types(sign_types or []),
            "created_at": now,
            "updated_at": now,
            "static_samples": 0,
            "dynamic_samples": 0,
        }

        data = self._load_registry()
        existing_ids = {item["id"] for item in data["signs"]}
        if sign["id"] in existing_ids:
            sign["id"] = f"{sign['id']}_{uuid4().hex[:8]}"
        data["signs"].append(sign)
        self._save_registry(data)
        return sign

    def update_sample_counts(self, sign_id: str, static_samples: int | None = None, dynamic_samples: int | None = None) -> dict[str, Any]:
        data = self._load_registry()
        for sign in data["signs"]:
            if sign["id"] == sign_id:
                if static_samples is not None:
                    sign["static_samples"] = int(static_samples)
                    if "static" not in sign["types"] and static_samples > 0:
                        sign["types"].append("static")
                if dynamic_samples is not None:
                    sign["dynamic_samples"] = int(dynamic_samples)
                    if "dynamic" not in sign["types"] and dynamic_samples > 0:
                        sign["types"].append("dynamic")
                sign["types"] = self._normalize_types(sign["types"])
                sign["updated_at"] = self._now()
                self._save_registry(data)
                return sign
        raise ValueError(f"No existe una seña con id '{sign_id}'")

    def delete_sign_registry_entry(self, sign_id: str) -> dict[str, Any]:
        data = self._load_registry()
        remaining = []
        deleted = None
        for sign in data["signs"]:
            if sign["id"] == sign_id:
                deleted = sign
            else:
                remaining.append(sign)
        if deleted is None:
            raise ValueError(f"No existe una seña con id '{sign_id}'")
        data["signs"] = remaining
        self._save_registry(data)
        return deleted

    def export_labels(self) -> list[str]:
        labels = [sign["name"] for sign in self.list_signs()]
        self.paths.static_labels_path.parent.mkdir(parents=True, exist_ok=True)
        self.paths.dynamic_labels_path.parent.mkdir(parents=True, exist_ok=True)
        self.paths.static_labels_path.write_text(json.dumps(labels, ensure_ascii=False, indent=2), encoding="utf-8")
        self.paths.dynamic_labels_path.write_text(json.dumps(labels, ensure_ascii=False, indent=2), encoding="utf-8")
        return labels

    def reset_registry(self) -> None:
        self._save_registry({"version": 1, "signs": []})

    def _load_registry(self) -> dict[str, Any]:
        if not self.paths.signs_path.exists():
            return {"version": 1, "signs": []}
        with self.paths.signs_path.open("r", encoding="utf-8-sig") as file:
            data = json.load(file)
        if "signs" not in data or not isinstance(data["signs"], list):
            raise ValueError(f"Registro de señas inválido: {self.paths.signs_path}")
        return data

    def _save_registry(self, data: dict[str, Any]) -> None:
        self.paths.signs_path.parent.mkdir(parents=True, exist_ok=True)
        with self.paths.signs_path.open("w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=2)

    def _build_sign_id(self, name: str) -> str:
        normalized = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
        normalized = re.sub(r"[^a-zA-Z0-9]+", "_", normalized).strip("_").lower()
        return normalized or f"sign_{uuid4().hex[:8]}"

    def _clean_name(self, name: str) -> str:
        return " ".join(str(name).strip().split())

    def _normalize_name(self, name: str) -> str:
        return self._clean_name(name).casefold()

    def _normalize_types(self, sign_types: list[str]) -> list[str]:
        valid = []
        for sign_type in sign_types:
            normalized = str(sign_type).strip().lower()
            if normalized in {"static", "dynamic"} and normalized not in valid:
                valid.append(normalized)
        return valid

    def _now(self) -> str:
        return datetime.now().isoformat(timespec="seconds")
