"""Servicio de gestión del registro de señas.

Este módulo define `SignService`, responsable de administrar el catálogo de
señas conocidas por la aplicación. El registro se persiste en un archivo JSON
(`data/signs/signs.json`) con la forma:

    {"version": 1, "signs": [{...}, {...}]}

Cada seña es un diccionario con: id, name, types ("static" y/o "dynamic"),
fechas de creación/actualización y contadores de muestras capturadas.

El servicio también exporta las etiquetas (nombres de señas) a los archivos
de labels que consumen los modelos de clasificación.
"""

import json
import re
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from .path_service import PathService


class SignService:
    """Administra las operaciones CRUD sobre el registro de señas.

    Ofrece métodos para listar, buscar, crear y eliminar señas, actualizar
    sus contadores de muestras y exportar las etiquetas para entrenamiento.
    """

    def __init__(self, paths: PathService | None = None):
        """Inicializa el servicio y garantiza la estructura de carpetas.

        Args:
            paths: instancia de `PathService` a utilizar. Si no se indica,
                se crea una con las rutas por defecto del proyecto.
        """
        self.paths = paths or PathService()
        self.paths.ensure_app_dirs()

    def list_signs(self) -> list[dict[str, Any]]:
        """Devuelve todas las señas registradas, ordenadas alfabéticamente.

        Returns:
            Lista de diccionarios de señas, ordenada por nombre sin
            distinguir mayúsculas/minúsculas.
        """
        data = self._load_registry()
        return sorted(data["signs"], key=lambda sign: sign["name"].lower())

    def get_sign(self, sign_id: str) -> dict[str, Any] | None:
        """Busca una seña por su identificador único.

        Args:
            sign_id: id de la seña a buscar.

        Returns:
            El diccionario de la seña o `None` si no existe.
        """
        # Recorrer el listado hasta encontrar una coincidencia exacta de id.
        for sign in self.list_signs():
            if sign["id"] == sign_id:
                return sign
        return None

    def find_by_name(self, name: str) -> dict[str, Any] | None:
        """Busca una seña por nombre, ignorando mayúsculas y espacios extra.

        Args:
            name: nombre de la seña (se normaliza antes de comparar).

        Returns:
            El diccionario de la seña o `None` si no existe.
        """
        # Normalizar el nombre buscado y compararlo con cada seña registrada
        # usando la misma normalización (casefold + espacios colapsados).
        normalized = self._normalize_name(name)
        for sign in self.list_signs():
            if self._normalize_name(sign["name"]) == normalized:
                return sign
        return None

    def add_sign(self, name: str, sign_types: list[str] | None = None) -> dict[str, Any]:
        """Crea y registra una nueva seña.

        Args:
            name: nombre visible de la seña (se limpia de espacios extra).
            sign_types: lista opcional de tipos ("static" y/o "dynamic").

        Returns:
            El diccionario de la seña recién creada.

        Raises:
            ValueError: si el nombre queda vacío tras limpiarlo o si ya
                existe otra seña con el mismo nombre.
        """
        # Validar que el nombre no quede vacío y que no exista un duplicado.
        clean_name = self._clean_name(name)
        if not clean_name:
            raise ValueError("El nombre de la seña no puede estar vacío")
        if self.find_by_name(clean_name):
            raise ValueError(f"La seña '{clean_name}' ya existe")

        # Construir el diccionario de la nueva seña con contadores en cero.
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

        # Si el id generado a partir del nombre ya está ocupado (por ejemplo,
        # nombres que se normalizan igual), se le agrega un sufijo aleatorio.
        data = self._load_registry()
        existing_ids = {item["id"] for item in data["signs"]}
        if sign["id"] in existing_ids:
            sign["id"] = f"{sign['id']}_{uuid4().hex[:8]}"
        # Agregar la seña al registro y persistir el cambio en disco.
        data["signs"].append(sign)
        self._save_registry(data)
        return sign

    def update_sample_counts(self, sign_id: str, static_samples: int | None = None, dynamic_samples: int | None = None) -> dict[str, Any]:
        """Actualiza los contadores de muestras de una seña.

        Además de actualizar los números, agrega automáticamente el tipo
        ("static"/"dynamic") a la seña cuando su contador pasa a ser mayor
        que cero, y refresca la fecha de última actualización.

        Args:
            sign_id: id de la seña a modificar.
            static_samples: nuevo total de muestras estáticas (o None para
                no modificarlo).
            dynamic_samples: nuevo total de muestras dinámicas (o None para
                no modificarlo).

        Returns:
            El diccionario de la seña actualizada.

        Raises:
            ValueError: si no existe una seña con ese id.
        """
        data = self._load_registry()
        # Buscar la seña dentro del registro cargado para modificarla in situ.
        for sign in data["signs"]:
            if sign["id"] == sign_id:
                # Actualizar el contador estático y marcar el tipo si aplica.
                if static_samples is not None:
                    sign["static_samples"] = int(static_samples)
                    if "static" not in sign["types"] and static_samples > 0:
                        sign["types"].append("static")
                # Actualizar el contador dinámico y marcar el tipo si aplica.
                if dynamic_samples is not None:
                    sign["dynamic_samples"] = int(dynamic_samples)
                    if "dynamic" not in sign["types"] and dynamic_samples > 0:
                        sign["types"].append("dynamic")
                # Normalizar tipos, refrescar la fecha y guardar el registro.
                sign["types"] = self._normalize_types(sign["types"])
                sign["updated_at"] = self._now()
                self._save_registry(data)
                return sign
        raise ValueError(f"No existe una seña con id '{sign_id}'")

    def delete_sign_registry_entry(self, sign_id: str) -> dict[str, Any]:
        """Elimina una seña del registro (solo del JSON de señas).

        Nota: este método NO borra las muestras de los datasets; para una
        eliminación completa se usa `CaptureService.delete_sign_everywhere`.

        Args:
            sign_id: id de la seña a eliminar.

        Returns:
            El diccionario de la seña eliminada.

        Raises:
            ValueError: si no existe una seña con ese id.
        """
        data = self._load_registry()
        # Separar la seña a borrar del resto, conservando las demás.
        remaining = []
        deleted = None
        for sign in data["signs"]:
            if sign["id"] == sign_id:
                deleted = sign
            else:
                remaining.append(sign)
        if deleted is None:
            raise ValueError(f"No existe una seña con id '{sign_id}'")
        # Persistir el registro sin la seña eliminada.
        data["signs"] = remaining
        self._save_registry(data)
        return deleted

    def export_labels(self) -> list[str]:
        """Exporta los nombres de todas las señas a los archivos de labels.

        Escribe la misma lista (ordenada alfabéticamente) tanto en el archivo
        de etiquetas del modelo estático como en el del dinámico, para que
        los índices de clase coincidan con los usados en los datasets.

        Returns:
            La lista de nombres de señas exportada.
        """
        # Obtener los nombres en el mismo orden que usa list_signs().
        labels = [sign["name"] for sign in self.list_signs()]
        # Asegurar que las carpetas destino existan antes de escribir.
        self.paths.static_labels_path.parent.mkdir(parents=True, exist_ok=True)
        self.paths.dynamic_labels_path.parent.mkdir(parents=True, exist_ok=True)
        # Guardar las etiquetas como JSON legible (UTF-8, con tildes intactas).
        self.paths.static_labels_path.write_text(json.dumps(labels, ensure_ascii=False, indent=2), encoding="utf-8")
        self.paths.dynamic_labels_path.write_text(json.dumps(labels, ensure_ascii=False, indent=2), encoding="utf-8")
        return labels

    def reset_registry(self) -> None:
        """Reinicia el registro dejándolo vacío (sin señas)."""
        self._save_registry({"version": 1, "signs": []})

    def _load_registry(self) -> dict[str, Any]:
        """Carga el registro de señas desde disco.

        Returns:
            El diccionario del registro. Si el archivo no existe todavía,
            devuelve un registro vacío con la versión actual.

        Raises:
            ValueError: si el JSON existe pero no tiene la estructura esperada.
        """
        # Si nunca se guardó un registro, devolver uno vacío por defecto.
        if not self.paths.signs_path.exists():
            return {"version": 1, "signs": []}
        # Leer el JSON tolerando el BOM de UTF-8 (encoding utf-8-sig).
        with self.paths.signs_path.open("r", encoding="utf-8-sig") as file:
            data = json.load(file)
        # Validar la estructura mínima esperada antes de devolverla.
        if "signs" not in data or not isinstance(data["signs"], list):
            raise ValueError(f"Registro de señas inválido: {self.paths.signs_path}")
        return data

    def _save_registry(self, data: dict[str, Any]) -> None:
        """Guarda el registro de señas en disco en formato JSON legible.

        Args:
            data: diccionario completo del registro a persistir.
        """
        # Garantizar que la carpeta exista y escribir el JSON con indentación.
        self.paths.signs_path.parent.mkdir(parents=True, exist_ok=True)
        with self.paths.signs_path.open("w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=2)

    def _build_sign_id(self, name: str) -> str:
        """Genera un id "slug" a partir del nombre de la seña.

        Quita tildes y caracteres no ASCII, reemplaza todo lo que no sea
        alfanumérico por guiones bajos y pasa a minúsculas. Si el resultado
        queda vacío, genera un id aleatorio.

        Args:
            name: nombre visible de la seña.

        Returns:
            Identificador apto para usar como nombre de archivo o clave.
        """
        # Descomponer acentos (NFKD) y descartar los caracteres no ASCII.
        normalized = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
        # Sustituir secuencias no alfanuméricas por "_" y normalizar bordes.
        normalized = re.sub(r"[^a-zA-Z0-9]+", "_", normalized).strip("_").lower()
        return normalized or f"sign_{uuid4().hex[:8]}"

    def _clean_name(self, name: str) -> str:
        """Limpia un nombre: recorta bordes y colapsa espacios múltiples."""
        return " ".join(str(name).strip().split())

    def _normalize_name(self, name: str) -> str:
        """Normaliza un nombre para comparaciones sin distinción de mayúsculas."""
        return self._clean_name(name).casefold()

    def _normalize_types(self, sign_types: list[str]) -> list[str]:
        """Filtra y deduplica la lista de tipos de una seña.

        Solo se aceptan los valores "static" y "dynamic"; el resto se
        descarta. Se conserva el orden de aparición sin duplicados.

        Args:
            sign_types: lista de tipos posiblemente sucia.

        Returns:
            Lista limpia de tipos válidos.
        """
        valid = []
        # Normalizar cada tipo y aceptarlo solo si es válido y no repetido.
        for sign_type in sign_types:
            normalized = str(sign_type).strip().lower()
            if normalized in {"static", "dynamic"} and normalized not in valid:
                valid.append(normalized)
        return valid

    def _now(self) -> str:
        """Devuelve la fecha y hora actual en formato ISO (segundos)."""
        return datetime.now().isoformat(timespec="seconds")
