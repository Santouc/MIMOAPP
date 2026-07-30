from PySide6.QtCore import Signal
from PySide6.QtGui import QColor, QFont, QPalette
from PySide6.QtWidgets import QLabel, QPushButton, QTextBrowser, QVBoxLayout, QWidget


class DocumentScreen(QWidget):
    back_requested = Signal()

    def __init__(self, title: str, content: str, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        title_label = QLabel(title)
        title_label.setObjectName("TitleLabel")

        self.viewer = QTextBrowser()
        self.viewer.setReadOnly(True)
        self.viewer.setOpenExternalLinks(True)
        font = QFont("Segoe UI", 12)
        self.viewer.setFont(font)
        self.viewer.document().setDocumentMargin(24)
        palette = self.viewer.palette()
        palette.setColor(QPalette.Link, QColor("#60a5fa"))
        self.viewer.setPalette(palette)
        self.set_content(content)

        back_button = QPushButton("Volver al inicio")
        back_button.clicked.connect(self.back_requested)

        layout.addWidget(title_label)
        layout.addWidget(self.viewer, 1)
        layout.addWidget(back_button)

    def set_content(self, content: str) -> None:
        self.viewer.setMarkdown(content)
