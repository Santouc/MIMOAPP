"""Servicio de capturas y datasets de la aplicación MIMO.

Este módulo define `CaptureService`, encargado del ciclo de vida de las
muestras capturadas con la cámara:

1. Cuando el usuario captura muestras de una seña, se crea una "sesión
   pendiente" en `data/pending_captures/<session_id>/` con tres archivos:
   `samples.json` (las muestras crudas), `metadata.json` (información de la
   sesión) y `summary.json` (metadatos + landmarks promedio para previsualizar).
2. El usuario puede aceptar la sesión (las muestras pasan al dataset de
   entrenamiento correspondiente) o rechazarla (se descarta todo).
3. También ofrece operaciones destructivas controladas: eliminar una seña de
   todos lados (registro, datasets, sesiones pendientes, invalidando los
   modelos entrenados) y reiniciar por completo los datos de la aplicación.

Los datasets son archivos JSON con claves "X" (muestras), "y" (índices de
clase), "labels" (nombres de señas) y "metadata" (información descriptiva).
"""

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
    """Gestiona sesiones de captura pendientes y los datasets de entrenamiento."""

    def __init__(self, paths: PathService | None = None, sign_service: SignService | None = None):
        """Inicializa el servicio con sus dependencias.

        Args:
            paths: instancia de `PathService` (se crea una por defecto si falta).
            sign_service: instancia de `SignService` (se crea una por defecto
                compartiendo el mismo `PathService`).
        """
        self.paths = paths or PathService()
        self.sign_service = sign_service or SignService(self.paths)
        self.paths.ensure_app_dirs()

    def create_pending_session(self, sign_id: str, capture_type: str, samples: list[Any]) -> dict[str, Any]:
        """Crea una sesión de captura pendiente de revisión.

        Guarda las muestras crudas y los metadatos en una carpeta nueva
        dentro de `pending_captures/`, y genera un resumen con los landmarks
        promedio para poder previsualizar la seña capturada.

        Args:
            sign_id: id de la seña a la que pertenecen las muestras.
            capture_type: "static" o "dynamic".
            samples: lista de muestras capturadas (landmarks de MediaPipe).

        Returns:
            El resumen de la sesión (metadatos + landmarks promedio).

        Raises:
            ValueError: si la seña no existe, el tipo es inválido o no hay muestras.
        """
        # Validar que la seña exista, que el tipo sea correcto y que haya muestras.
        sign = self.sign_service.get_sign(sign_id)
        if sign is None:
            raise ValueError(f"No existe una seña con id '{sign_id}'")
        capture_type = self._validate_capture_type(capture_type)
        if not samples:
            raise ValueError("No se puede crear una sesión sin muestras")

        # Generar un id de sesión único (marca de tiempo + sufijo aleatorio)
        # y crear su carpeta dedicada dentro de pending_captures.
        session_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}"
        session_dir = self.paths.pending_captures_dir / session_id
        session_dir.mkdir(parents=True, exist_ok=False)

        # Metadatos descriptivos de la sesión (estado inicial: pendiente).
        metadata = {
            "session_id": session_id,
            "sign_id": sign_id,
            "sign_name": sign["name"],
            "capture_type": capture_type,
            "sample_count": len(samples),
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "status": "pending",
        }

        # Rutas de los tres archivos que componen una sesión pendiente.
        samples_path = session_dir / "samples.json"
        metadata_path = session_dir / "metadata.json"
        summary_path = session_dir / "summary.json"

        # Persistir muestras y metadatos, y calcular el resumen con el
        # promedio de landmarks para la previsualización.
        samples_path.write_text(json.dumps({"samples": samples}, ensure_ascii=False), encoding="utf-8")
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        summary = self.build_session_summary(session_dir)
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        return summary

    def list_pending_sessions(self) -> list[dict[str, Any]]:
        """Lista los metadatos de todas las sesiones pendientes.

        Returns:
            Lista de metadatos de sesión ordenada por fecha de creación.
        """
        sessions = []
        # Si nunca se creó la carpeta, no hay sesiones pendientes.
        if not self.paths.pending_captures_dir.exists():
            return sessions
        # Recorrer cada subcarpeta de sesión y leer su metadata.json.
        for session_dir in self.paths.pending_captures_dir.iterdir():
            if not session_dir.is_dir():
                continue
            metadata_path = session_dir / "metadata.json"
            if metadata_path.exists():
                sessions.append(json.loads(metadata_path.read_text(encoding="utf-8")))
        return sorted(sessions, key=lambda item: item["created_at"])

    def build_session_summary(self, session_dir: str | Path) -> dict[str, Any]:
        """Construye el resumen de una sesión (metadatos + landmarks promedio).

        Args:
            session_dir: carpeta de la sesión pendiente.

        Returns:
            Diccionario con los metadatos de la sesión más la clave
            "average_landmarks" (promedio de las muestras capturadas).
        """
        session_dir = Path(session_dir)
        # Cargar metadatos y muestras crudas desde los archivos de la sesión.
        metadata = json.loads((session_dir / "metadata.json").read_text(encoding="utf-8"))
        samples = json.loads((session_dir / "samples.json").read_text(encoding="utf-8"))["samples"]
        # Promediar según el tipo: landmarks estáticos o secuencias dinámicas.
        if metadata["capture_type"] == "static":
            average = average_static_landmarks(samples).tolist()
        else:
            average = average_dynamic_sequences(samples).tolist()
        return {
            **metadata,
            "average_landmarks": average,
        }

    def accept_pending_session(self, session_id: str) -> dict[str, Any]:
        """Acepta una sesión pendiente: incorpora sus muestras al dataset.

        Agrega las muestras al dataset estático o dinámico (según el tipo de
        la sesión), actualiza los contadores de la seña y elimina la carpeta
        de la sesión.

        Args:
            session_id: id de la sesión a aceptar.

        Returns:
            Los metadatos de la sesión con el estado "accepted".

        Raises:
            FileNotFoundError: si la sesión no existe.
        """
        session_dir = self.paths.pending_captures_dir / session_id
        if not session_dir.exists():
            raise FileNotFoundError(f"Sesión pendiente no encontrada: {session_id}")
        # Cargar la información de la sesión y obtener el índice de clase que
        # le corresponde a la seña dentro de las etiquetas exportadas.
        metadata = json.loads((session_dir / "metadata.json").read_text(encoding="utf-8"))
        samples = json.loads((session_dir / "samples.json").read_text(encoding="utf-8"))["samples"]
        labels = self.sign_service.export_labels()
        label_index = labels.index(metadata["sign_name"])

        # Anexar las muestras al dataset correspondiente según el tipo.
        if metadata["capture_type"] == "static":
            self._append_static_samples(samples, label_index, labels)
        else:
            self._append_dynamic_samples(samples, label_index, labels)

        # Refrescar los contadores de la seña y eliminar la sesión aceptada.
        self._refresh_counts(metadata["sign_id"])
        shutil.rmtree(session_dir)
        return {**metadata, "status": "accepted"}

    def reject_pending_session(self, session_id: str) -> dict[str, Any]:
        """Rechaza una sesión pendiente: descarta sus muestras.

        Args:
            session_id: id de la sesión a rechazar.

        Returns:
            Los metadatos de la sesión con el estado "rejected".

        Raises:
            FileNotFoundError: si la sesión no existe.
        """
        session_dir = self.paths.pending_captures_dir / session_id
        if not session_dir.exists():
            raise FileNotFoundError(f"Sesión pendiente no encontrada: {session_id}")
        # Conservar los metadatos para informar al usuario y borrar la carpeta.
        metadata = json.loads((session_dir / "metadata.json").read_text(encoding="utf-8"))
        shutil.rmtree(session_dir)
        return {**metadata, "status": "rejected"}

    def delete_sign_everywhere(self, sign_id: str) -> dict[str, Any]:
        """Elimina una seña de TODOS los lugares donde aparece.

        Pasos que realiza:
        1. Borra las sesiones pendientes asociadas a la seña.
        2. Elimina sus muestras de los datasets estático y dinámico.
        3. Quita la seña del registro y reexporta las etiquetas.
        4. Reescribe las etiquetas de los datasets para mantener coherencia.
        5. Invalida (renombra) los modelos entrenados, ya que sus clases
           dejan de coincidir con las etiquetas actuales.

        Args:
            sign_id: id de la seña a eliminar.

        Returns:
            Resumen con la seña borrada, la cantidad de muestras eliminadas
            de cada dataset y los modelos invalidados.

        Raises:
            ValueError: si la seña no existe.
        """
        sign = self.sign_service.get_sign(sign_id)
        if sign is None:
            raise ValueError(f"No existe una seña con id '{sign_id}'")
        sign_name = sign["name"]

        # 1. Borrar las sesiones pendientes que pertenezcan a esta seña.
        for session in self.list_pending_sessions():
            if session["sign_id"] == sign_id:
                shutil.rmtree(self.paths.pending_captures_dir / session["session_id"], ignore_errors=True)

        # 2 y 3. Quitar las muestras de ambos datasets y la entrada del registro.
        deleted_static = self._remove_from_dataset(self.paths.static_dataset_path, sign_name)
        deleted_dynamic = self._remove_from_dataset(self.paths.dynamic_dataset_path, sign_name)
        deleted_registry = self.sign_service.delete_sign_registry_entry(sign_id)
        # 4. Reexportar etiquetas y sincronizarlas en los datasets.
        labels = self.sign_service.export_labels()
        self._rewrite_dataset_labels(self.paths.static_dataset_path, labels)
        self._rewrite_dataset_labels(self.paths.dynamic_dataset_path, labels)
        # 5. Invalidar los modelos entrenados (quedaron desactualizados).
        invalidated_models = self._invalidate_models()
        return {
            "sign": deleted_registry,
            "deleted_static_samples": deleted_static,
            "deleted_dynamic_samples": deleted_dynamic,
            "invalidated_models": invalidated_models,
        }

    def reset_all_data(self) -> dict[str, Any]:
        """Borra por completo los datos de la aplicación.

        Elimina sesiones pendientes, previsualizaciones, datasets, archivos
        de etiquetas, modelos entrenados (incluidos los invalidados) y
        reinicia el registro de señas.

        Returns:
            Resumen con la cantidad de señas y sesiones borradas y la lista
            de archivos eliminados.
        """
        # Registrar cuántas señas y sesiones había antes de limpiar.
        sign_count = len(self.sign_service.list_signs())
        pending_count = len(self.list_pending_sessions())
        # Vaciar las carpetas de capturas pendientes y previsualizaciones.
        for directory in (self.paths.pending_captures_dir, self.paths.previews_dir):
            if directory.exists():
                for item in directory.iterdir():
                    if item.is_dir():
                        shutil.rmtree(item, ignore_errors=True)
                    else:
                        item.unlink(missing_ok=True)
        # Eliminar datasets, etiquetas y modelos entrenados si existen.
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
        # Borrar también los modelos invalidados por eliminaciones previas.
        for pattern in ("model.deleted_sign_*.h5", "model_dynamic.deleted_sign_*.h5"):
            for path in self.paths.models_dir.glob(pattern):
                path.unlink(missing_ok=True)
                removed_files.append(str(path))
        # Dejar el registro de señas vacío.
        self.sign_service.reset_registry()
        return {
            "deleted_signs": sign_count,
            "deleted_pending_sessions": pending_count,
            "removed_files": removed_files,
        }

    def _append_static_samples(self, samples: list[Any], label_index: int, labels: list[str]) -> None:
        """Agrega muestras estáticas al dataset estático.

        Args:
            samples: muestras a incorporar (landmarks de una mano).
            label_index: índice de clase que les corresponde.
            labels: lista completa de etiquetas vigente.
        """
        # Cargar el dataset actual, anexar las muestras con su etiqueta y
        # actualizar las etiquetas y los metadatos antes de guardar.
        dataset = self._load_dataset(self.paths.static_dataset_path, labels)
        dataset["X"].extend(samples)
        dataset["y"].extend([label_index] * len(samples))
        dataset["labels"] = labels
        dataset["metadata"] = self._metadata("static", len(dataset["X"]), labels)
        self._save_dataset(self.paths.static_dataset_path, dataset)

    def _append_dynamic_samples(self, samples: list[Any], label_index: int, labels: list[str]) -> None:
        """Agrega muestras dinámicas (secuencias) al dataset dinámico.

        Args:
            samples: secuencias de landmarks a incorporar.
            label_index: índice de clase que les corresponde.
            labels: lista completa de etiquetas vigente.
        """
        # Misma lógica que el caso estático pero sobre el dataset dinámico.
        dataset = self._load_dataset(self.paths.dynamic_dataset_path, labels)
        dataset["X"].extend(samples)
        dataset["y"].extend([label_index] * len(samples))
        dataset["labels"] = labels
        dataset["metadata"] = self._metadata("dynamic", len(dataset["X"]), labels)
        self._save_dataset(self.paths.dynamic_dataset_path, dataset)

    def _load_dataset(self, path: Path, labels: list[str]) -> dict[str, Any]:
        """Carga un dataset desde disco o devuelve uno vacío si no existe.

        Args:
            path: ruta al archivo JSON del dataset.
            labels: etiquetas a usar si el dataset se crea desde cero.

        Returns:
            Diccionario con las claves "X", "y", "labels" y "metadata".
        """
        if not path.exists():
            return {"X": [], "y": [], "labels": labels, "metadata": {}}
        return json.loads(path.read_text(encoding="utf-8"))

    def _save_dataset(self, path: Path, dataset: dict[str, Any]) -> None:
        """Guarda un dataset en disco en formato JSON legible."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(dataset, ensure_ascii=False, indent=2), encoding="utf-8")

    def _remove_from_dataset(self, path: Path, sign_name: str) -> int:
        """Elimina de un dataset todas las muestras de una seña.

        Además de quitar las muestras, reindexa las etiquetas restantes para
        que los valores de "y" sigan apuntando a la clase correcta.

        Args:
            path: ruta al archivo JSON del dataset.
            sign_name: nombre de la seña cuyas muestras se eliminan.

        Returns:
            Cantidad de muestras eliminadas (0 si el dataset o la seña no existen).
        """
        # Si el dataset no existe o la seña no figura entre sus etiquetas,
        # no hay nada que eliminar.
        if not path.exists():
            return 0
        dataset = json.loads(path.read_text(encoding="utf-8"))
        labels = dataset.get("labels", [])
        if sign_name not in labels:
            return 0
        # Calcular el índice a eliminar y el mapeo de índices viejos a nuevos
        # para las etiquetas que permanecen.
        removed_index = labels.index(sign_name)
        new_labels = [label for label in labels if label != sign_name]
        index_map = {old_idx: new_labels.index(label) for old_idx, label in enumerate(labels) if label != sign_name}
        # Reconstruir X e y descartando las muestras de la seña eliminada y
        # reindexando el resto según el nuevo orden de etiquetas.
        new_x = []
        new_y = []
        removed = 0
        for sample, label_idx in zip(dataset.get("X", []), dataset.get("y", [])):
            if int(label_idx) == removed_index:
                removed += 1
                continue
            new_x.append(sample)
            new_y.append(index_map[int(label_idx)])
        # Guardar el dataset actualizado con metadatos frescos.
        dataset["X"] = new_x
        dataset["y"] = new_y
        dataset["labels"] = new_labels
        dataset["metadata"] = self._metadata(dataset.get("metadata", {}).get("type", "unknown"), len(new_x), new_labels)
        self._save_dataset(path, dataset)
        return removed

    def _rewrite_dataset_labels(self, path: Path, labels: list[str]) -> None:
        """Sobrescribe la lista de etiquetas de un dataset existente.

        Se usa tras eliminar una seña para que el dataset quede alineado con
        las etiquetas exportadas por `SignService`.

        Args:
            path: ruta al archivo JSON del dataset.
            labels: nueva lista de etiquetas.
        """
        if not path.exists():
            return
        # Reemplazar etiquetas y refrescar los metadatos del dataset.
        dataset = json.loads(path.read_text(encoding="utf-8"))
        dataset["labels"] = labels
        dataset["metadata"] = self._metadata(dataset.get("metadata", {}).get("type", "unknown"), len(dataset.get("X", [])), labels)
        self._save_dataset(path, dataset)

    def _refresh_counts(self, sign_id: str) -> None:
        """Recalcula y actualiza los contadores de muestras de una seña.

        Cuenta las muestras reales presentes en cada dataset y las guarda en
        el registro de señas mediante `SignService`.

        Args:
            sign_id: id de la seña a actualizar.
        """
        sign = self.sign_service.get_sign(sign_id)
        if sign is None:
            return
        # Contar muestras en ambos datasets y actualizar el registro.
        self.sign_service.update_sample_counts(
            sign_id,
            static_samples=self._count_samples(self.paths.static_dataset_path, sign["name"]),
            dynamic_samples=self._count_samples(self.paths.dynamic_dataset_path, sign["name"]),
        )

    def _count_samples(self, path: Path, sign_name: str) -> int:
        """Cuenta cuántas muestras de una seña hay en un dataset.

        Args:
            path: ruta al archivo JSON del dataset.
            sign_name: nombre de la seña a contar.

        Returns:
            Cantidad de muestras cuyo índice de clase corresponde a la seña.
        """
        # Sin dataset o sin la seña entre las etiquetas, el conteo es cero.
        if not path.exists():
            return 0
        dataset = json.loads(path.read_text(encoding="utf-8"))
        labels = dataset.get("labels", [])
        if sign_name not in labels:
            return 0
        # Contar las entradas de "y" que coinciden con el índice de la seña.
        label_index = labels.index(sign_name)
        return sum(1 for item in dataset.get("y", []) if int(item) == label_index)

    def _metadata(self, dataset_type: str, sample_count: int, labels: list[str]) -> dict[str, Any]:
        """Construye el bloque de metadatos descriptivos de un dataset.

        Args:
            dataset_type: "static", "dynamic" u otro identificador.
            sample_count: cantidad total de muestras del dataset.
            labels: etiquetas vigentes.

        Returns:
            Diccionario de metadatos (tipo, tamaños, formato y fecha).
        """
        return {
            "type": dataset_type,
            "num_samples": sample_count,
            "num_classes": len(labels),
            "landmark_format": "mediapipe_21_3d",
            "coordinate_system": "normalized",
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        }

    def _validate_capture_type(self, capture_type: str) -> str:
        """Valida y normaliza el tipo de captura.

        Args:
            capture_type: valor recibido ("static" o "dynamic", en cualquier
                combinación de mayúsculas/minúsculas).

        Returns:
            El tipo normalizado en minúsculas.

        Raises:
            ValueError: si el valor no es "static" ni "dynamic".
        """
        normalized = str(capture_type).strip().lower()
        if normalized not in {"static", "dynamic"}:
            raise ValueError("El tipo de captura debe ser 'static' o 'dynamic'")
        return normalized

    def _invalidate_models(self) -> list[str]:
        """Invalida los modelos entrenados renombrándolos con un sufijo.

        Tras eliminar una seña, los modelos guardados dejan de coincidir con
        las etiquetas actuales; en lugar de borrarlos, se renombran con la
        marca `deleted_sign_<timestamp>` para conservarlos como respaldo.

        Returns:
            Lista de rutas de los modelos invalidados (renombrados).
        """
        invalidated = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Renombrar cada modelo existente agregando el sufijo de invalidación.
        for model_path in (self.paths.static_model_path, self.paths.dynamic_model_path):
            if model_path.exists():
                invalid_path = model_path.with_suffix(f".deleted_sign_{timestamp}{model_path.suffix}")
                model_path.rename(invalid_path)
                invalidated.append(str(invalid_path))
        return invalidated
