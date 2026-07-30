from .path_service import PathService


class DocumentService:
    def __init__(self, paths: PathService | None = None):
        self.paths = paths or PathService()

    def read_credits(self) -> str:
        return self._read_markdown(self.paths.readme_path)

    def read_manual(self) -> str:
        return self._read_markdown(self.paths.manual_path)

    def _read_markdown(self, path) -> str:
        if not path.exists():
            return f"No se encontró el archivo: {path}"
        return path.read_text(encoding="utf-8-sig")
