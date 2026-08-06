"""Pantalla genérica de visualización de documentos.

Define :class:`DocumentScreen`, un widget reutilizable que muestra un título
y un contenido en formato Markdown dentro de un visor de solo lectura. La
aplicación la utiliza para mostrar los créditos y el manual de uso, cuyo
contenido proviene del servicio de documentos (:class:`DocumentService`).
"""

from PySide6.QtCore import Signal
from PySide6.QtGui import QColor, QFont, QPalette
from PySide6.QtWidgets import QLabel, QPushButton, QTextBrowser, QVBoxLayout, QWidget


class DocumentScreen(QWidget):
    """Visor de documentos de solo lectura con soporte de Markdown.

    Señales:
        back_requested: se emite cuando el usuario pulsa "Volver al inicio";
            la ventana principal la usa para regresar al menú.
    """

    # Señal de navegación para volver a la pantalla de inicio.
    back_requested = Signal()

    def __init__(self, title: str, content: str, parent=None):
        """Construye la pantalla con el título y contenido indicados.

        Args:
            title: título mostrado en la parte superior de la pantalla.
            content: texto del documento en formato Markdown.
            parent: widget padre opcional (convención de Qt).
        """
        super().__init__(parent)
        # Layout vertical: título arriba, visor en el centro y botón abajo.
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        # Etiqueta de título con estilo de la hoja de estilos global.
        title_label = QLabel(title)
        title_label.setObjectName("TitleLabel")

        # Visor de texto enriquecido: solo lectura y con apertura de enlaces
        # externos en el navegador del sistema.
        self.viewer = QTextBrowser()
        self.viewer.setReadOnly(True)
        self.viewer.setOpenExternalLinks(True)
        # Fuente y márgenes para mejorar la legibilidad del documento.
        font = QFont("Segoe UI", 12)
        self.viewer.setFont(font)
        self.viewer.document().setDocumentMargin(24)
        # Ajusta el color de los enlaces para que contrasten con el tema oscuro.
        palette = self.viewer.palette()
        palette.setColor(QPalette.Link, QColor("#60a5fa"))
        self.viewer.setPalette(palette)
        # Carga el contenido inicial del documento.
        self.set_content(content)

        # Botón para regresar al menú principal.
        back_button = QPushButton("Volver al inicio")
        back_button.clicked.connect(self.back_requested)

        # Composición final del layout; el visor ocupa el espacio sobrante.
        layout.addWidget(title_label)
        layout.addWidget(self.viewer, 1)
        layout.addWidget(back_button)

    def set_content(self, content: str) -> None:
        """Reemplaza el contenido mostrado en el visor.

        Args:
            content: nuevo texto del documento en formato Markdown; Qt lo
                convierte internamente a texto enriquecido para mostrarlo.
        """
        self.viewer.setMarkdown(content)
