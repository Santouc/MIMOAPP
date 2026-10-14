"""Paquete de visualización: promedios de landmarks para vistas previas."""

# Reexporta las funciones de promedio para importarlas directamente
# desde `visualization` sin conocer el módulo interno.
from .landmark_average import average_dynamic_sequences, average_static_landmarks

__all__ = [
    "average_static_landmarks",
    "average_dynamic_sequences",
]
