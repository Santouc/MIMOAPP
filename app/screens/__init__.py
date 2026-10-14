"""Paquete de pantallas de la aplicación MIMO (T.L.S).

Este módulo reexporta todas las pantallas (widgets de PySide6) que componen
la interfaz de usuario, de modo que puedan importarse de forma sencilla como
``from app.screens import HomeScreen, TranslateScreen, ...``.

Pantallas disponibles:
    - DocumentScreen: visor de documentos (créditos y manual de uso).
    - ExtensionsScreen: gestión de extensiones opcionales.
    - HomeScreen: menú principal de la aplicación.
    - ManageSignsScreen: alta, listado y eliminación de señas.
    - TeachSignScreen: captura de muestras para entrenar señas.
    - TranslateScreen: traducción en vivo con la cámara.
"""

from .document_screen import DocumentScreen
from .extensions_screen import ExtensionsScreen
from .home_screen import HomeScreen
from .manage_signs_screen import ManageSignsScreen
from .teach_sign_screen import TeachSignScreen
from .translate_screen import TranslateScreen

# Lista pública del paquete: define qué nombres se exportan con
# "from app.screens import *".
__all__ = [
    "DocumentScreen",
    "ExtensionsScreen",
    "HomeScreen",
    "ManageSignsScreen",
    "TeachSignScreen",
    "TranslateScreen",
]
