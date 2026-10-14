"""Pantalla de inicio (menú principal) de la aplicación MIMO (T.L.S).

Define :class:`HomeScreen`, el widget que muestra el título de la aplicación
y los botones de navegación hacia las demás pantallas. Esta pantalla no
contiene lógica de negocio: se limita a emitir señales de Qt cuando el usuario
pulsa un botón, y es la ventana principal (:class:`MainWindow`) quien decide
qué pantalla mostrar en respuesta a cada señal.
"""

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QLabel, QPushButton, QVBoxLayout, QWidget


class HomeScreen(QWidget):
    """Menú principal con accesos a todas las funciones de la aplicación.

    Señales emitidas (una por cada botón del menú):
        teach_requested: el usuario quiere ir a la pantalla de entrenamiento.
        manage_requested: el usuario quiere gestionar las señas registradas.
        translate_requested: el usuario quiere traducir en vivo con la cámara.
        extensions_requested: el usuario quiere ver/gestionar extensiones.
        credits_requested: el usuario quiere ver los créditos.
        manual_requested: el usuario quiere ver el manual de uso.
        exit_requested: el usuario quiere cerrar la aplicación.
    """

    # Señales de navegación; la ventana principal las conecta a sus manejadores.
    teach_requested = Signal()
    manage_requested = Signal()
    translate_requested = Signal()
    extensions_requested = Signal()
    credits_requested = Signal()
    manual_requested = Signal()
    exit_requested = Signal()

    def __init__(self, parent=None):
        """Construye la interfaz del menú principal.

        Args:
            parent: widget padre opcional (convención de Qt).
        """
        super().__init__(parent)
        # Layout vertical que apila título, subtítulo y botones.
        layout = QVBoxLayout(self)
        layout.setSpacing(14)

        # Título y subtítulo de la aplicación; los objectName permiten
        # aplicarles estilos específicos desde la hoja de estilos global.
        title = QLabel("Traductor de Lengua de Señas")
        title.setObjectName("TitleLabel")
        subtitle = QLabel("Aplicación de escritorio para enseñar, gestionar y reconocer señas.")
        subtitle.setObjectName("SubtitleLabel")

        # Botones del menú: uno por cada función principal de la aplicación.
        teach_button = QPushButton("Entrenamiento")
        manage_button = QPushButton("Gestionar señas")
        translate_button = QPushButton("Traducir en vivo")
        extensions_button = QPushButton("Extensiones")
        credits_button = QPushButton("Créditos")
        manual_button = QPushButton("Manual de uso")
        exit_button = QPushButton("Salir")

        # Cada clic de botón se traduce directamente en la emisión de la
        # señal de navegación correspondiente.
        teach_button.clicked.connect(self.teach_requested)
        manage_button.clicked.connect(self.manage_requested)
        translate_button.clicked.connect(self.translate_requested)
        extensions_button.clicked.connect(self.extensions_requested)
        credits_button.clicked.connect(self.credits_requested)
        manual_button.clicked.connect(self.manual_requested)
        exit_button.clicked.connect(self.exit_requested)

        # Composición final del layout: espaciadores elásticos arriba y abajo
        # para centrar verticalmente el contenido del menú.
        layout.addStretch(1)
        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addSpacing(20)
        layout.addWidget(teach_button)
        layout.addWidget(manage_button)
        layout.addWidget(translate_button)
        layout.addWidget(extensions_button)
        layout.addWidget(credits_button)
        layout.addWidget(manual_button)
        layout.addWidget(exit_button)
        layout.addStretch(2)
