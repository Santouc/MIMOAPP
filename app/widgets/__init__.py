"""Paquete de widgets reutilizables de la interfaz (vista previa y diálogos)."""

# Reexporta los widgets para importarlos directamente desde `app.widgets`.
from .landmark_preview import LandmarkPreview
from .transcription_correction_dialog import TranscriptionCorrectionDialog

__all__ = [
    "LandmarkPreview",
    "TranscriptionCorrectionDialog",
]
