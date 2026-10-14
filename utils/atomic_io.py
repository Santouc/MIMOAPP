"""Escritura atómica de archivos JSON.

Escribir directamente sobre el archivo destino deja una ventana en la que
una interrupción (cierre forzado, corte de energía, excepción) puede dejar
el archivo truncado o vacío. Para evitarlo, el contenido se escribe primero
en un archivo temporal oculto dentro de la misma carpeta y, una vez
completado y volcado a disco, se reemplaza el destino con `os.replace()`,
que es atómico a nivel de sistema operativo cuando origen y destino están
en el mismo volumen.
"""

import json
import os
import tempfile
from pathlib import Path
from typing import Any


def atomic_write_text(path: str | Path, text: str, encoding: str = "utf-8") -> None:
    """Escribe `text` en `path` de forma atómica.

    Args:
        path: ruta del archivo destino.
        text: contenido completo a escribir.
        encoding: codificación del archivo (UTF-8 por defecto).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # El temporal vive en la misma carpeta para garantizar que el reemplazo
    # ocurra dentro del mismo volumen (requisito de atomicidad de os.replace).
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding=encoding, newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except BaseException:
        # Si algo falla antes del reemplazo, el destino sigue intacto; solo
        # se limpia el temporal para no dejar basura en la carpeta.
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise


def atomic_write_json(path: str | Path, data: Any, indent: int | None = 2, ensure_ascii: bool = False) -> None:
    """Serializa `data` como JSON y lo guarda atómicamente en `path`.

    La serialización se completa en memoria antes de tocar el disco, por lo
    que un objeto no serializable falla sin modificar el archivo existente.

    Args:
        path: ruta del archivo destino.
        data: objeto serializable a JSON.
        indent: indentación del JSON (None para formato compacto).
        ensure_ascii: si es False, conserva tildes y caracteres no ASCII.
    """
    atomic_write_text(path, json.dumps(data, ensure_ascii=ensure_ascii, indent=indent))
