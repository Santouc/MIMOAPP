from PySide6.QtCore import Signal
from PySide6.QtWidgets import QLabel, QPushButton, QVBoxLayout, QWidget


class HomeScreen(QWidget):
    teach_requested = Signal()
    manage_requested = Signal()
    translate_requested = Signal()
    extensions_requested = Signal()
    credits_requested = Signal()
    manual_requested = Signal()
    exit_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setSpacing(14)

        title = QLabel("Traductor de Lengua de Señas")
        title.setObjectName("TitleLabel")
        subtitle = QLabel("Aplicación de escritorio para enseñar, gestionar y reconocer señas.")
        subtitle.setObjectName("SubtitleLabel")

        teach_button = QPushButton("Entrenamiento")
        manage_button = QPushButton("Gestionar señas")
        translate_button = QPushButton("Traducir en vivo")
        extensions_button = QPushButton("Extensiones")
        credits_button = QPushButton("Créditos")
        manual_button = QPushButton("Manual de uso")
        exit_button = QPushButton("Salir")

        teach_button.clicked.connect(self.teach_requested)
        manage_button.clicked.connect(self.manage_requested)
        translate_button.clicked.connect(self.translate_requested)
        extensions_button.clicked.connect(self.extensions_requested)
        credits_button.clicked.connect(self.credits_requested)
        manual_button.clicked.connect(self.manual_requested)
        exit_button.clicked.connect(self.exit_requested)

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
