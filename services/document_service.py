"""Servicio de lectura de documentos de la aplicación MIMO.

Este módulo define `DocumentService`, una clase muy simple que se encarga de
leer los documentos Markdown que la interfaz muestra al usuario:

- Los créditos del proyecto (archivo README.md en la raíz).
- El manual de uso (docs/manual_uso.md).

Si un archivo no existe, devuelve un mensaje amigable en lugar de fallar.
"""

from .path_service import PathService


class DocumentService:
    """Lee los documentos Markdown (créditos y manual) para la interfaz."""

    def __init__(self, paths: PathService | None = None):
        """Inicializa el servicio.

        Args:
            paths: instancia de `PathService` con las rutas de los documentos.
                Si no se indica, se crea una con las rutas por defecto.
        """
        self.paths = paths or PathService()

    def read_credits(self) -> str:
        """Devuelve el contenido del README (créditos del proyecto)."""
        return self._read_markdown(self.paths.readme_path)

    def read_manual(self) -> str:
        """Devuelve el contenido del manual de uso."""
        return self._read_markdown(self.paths.manual_path)

    def _read_markdown(self, path) -> str:
        """Lee un archivo Markdown de forma segura.

        Args:
            path: ruta al archivo a leer.

        Returns:
            El texto del archivo, o un mensaje indicando que no se encontró.
        """
        # Si el archivo no existe, devolver un aviso en vez de lanzar error.
        if not path.exists():
            return f"No se encontró el archivo: {path}"
        # Leer tolerando el BOM de UTF-8 (encoding utf-8-sig).
        return path.read_text(encoding="utf-8-sig")
