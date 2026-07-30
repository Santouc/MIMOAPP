from PySide6.QtWidgets import QDialog, QDialogButtonBox, QLabel, QLineEdit, QPlainTextEdit, QVBoxLayout


class TranscriptionCorrectionDialog(QDialog):
    def __init__(self, raw_text: str = "", interpreted_text: str = "", parent=None):
        super().__init__(parent)
        self.setWindowTitle("Corregir interpretación")
        self.setMinimumWidth(520)

        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        raw_label = QLabel("Letras detectadas:")
        self.raw_input = QLineEdit(raw_text)
        self.raw_input.setPlaceholderText("Ejemplo: HOLASOYSANTI")

        interpreted_label = QLabel("Interpretación correcta:")
        self.interpreted_input = QPlainTextEdit(interpreted_text)
        self.interpreted_input.setPlaceholderText("Ejemplo: Hola, soy Santi.")
        self.interpreted_input.setMinimumHeight(90)

        self.error_label = QLabel("")
        self.error_label.setObjectName("BodyLabel")

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._validate_and_accept)
        buttons.rejected.connect(self.reject)

        layout.addWidget(raw_label)
        layout.addWidget(self.raw_input)
        layout.addWidget(interpreted_label)
        layout.addWidget(self.interpreted_input)
        layout.addWidget(self.error_label)
        layout.addWidget(buttons)

    def values(self) -> tuple[str, str]:
        return self.raw_input.text().strip(), self.interpreted_input.toPlainText().strip()

    def _validate_and_accept(self) -> None:
        raw_text, interpreted_text = self.values()
        if not raw_text:
            self.error_label.setText("Debes ingresar las letras detectadas.")
            return
        if not interpreted_text:
            self.error_label.setText("Debes ingresar la interpretación correcta.")
            return
        self.accept()
