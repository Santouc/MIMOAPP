"""
Diálogo de corrección de transcripciones.

Este módulo define TranscriptionCorrectionDialog, un cuadro de diálogo de Qt
que permite al usuario corregir manualmente una transcripción producida por
el traductor de señas. El usuario puede editar dos campos:

- Las letras detectadas en crudo (por ejemplo "HOLASOYSANTI").
- La interpretación correcta en texto natural (por ejemplo
  "Hola, soy Santi.").

El diálogo valida que ambos campos estén completos antes de aceptarse y
expone el método values() para que el código llamador recupere los textos
ya normalizados (sin espacios sobrantes).
"""

# Widgets de Qt necesarios: diálogo base, caja estándar de botones
# Aceptar/Cancelar, etiquetas, campos de texto de una y varias líneas,
# y layout vertical.
from PySide6.QtWidgets import QDialog, QDialogButtonBox, QLabel, QLineEdit, QPlainTextEdit, QVBoxLayout


class TranscriptionCorrectionDialog(QDialog):
    """
    Diálogo modal para corregir la interpretación de una transcripción.

    Presenta un campo de una línea con las letras detectadas y un área de
    texto con la interpretación en lenguaje natural. Al pulsar "Ok" se
    valida que ninguno de los dos campos esté vacío; si alguno falta, se
    muestra un mensaje de error y el diálogo permanece abierto.
    """

    def __init__(self, raw_text: str = "", interpreted_text: str = "", parent=None):
        """
        Construye el diálogo y arma su interfaz.

        Args:
            raw_text: texto inicial con las letras detectadas (prellenado).
            interpreted_text: texto inicial con la interpretación (prellenado).
            parent: widget padre dentro de la jerarquía de Qt (opcional).
        """
        super().__init__(parent)
        # Configuración básica de la ventana del diálogo.
        self.setWindowTitle("Corregir interpretación")
        self.setMinimumWidth(520)

        # Layout vertical que apila todos los elementos del formulario.
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        # Campo de una línea para las letras detectadas en crudo.
        raw_label = QLabel("Letras detectadas:")
        self.raw_input = QLineEdit(raw_text)
        self.raw_input.setPlaceholderText("Ejemplo: HOLASOYSANTI")

        # Área de texto multilínea para la interpretación en lenguaje natural.
        interpreted_label = QLabel("Interpretación correcta:")
        self.interpreted_input = QPlainTextEdit(interpreted_text)
        self.interpreted_input.setPlaceholderText("Ejemplo: Hola, soy Santi.")
        self.interpreted_input.setMinimumHeight(90)

        # Etiqueta reservada para mostrar mensajes de validación; comienza
        # vacía y se rellena solo cuando falta algún dato.
        self.error_label = QLabel("")
        self.error_label.setObjectName("BodyLabel")

        # Botones estándar: "Ok" pasa por la validación propia antes de
        # aceptar; "Cancel" cierra el diálogo rechazándolo.
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._validate_and_accept)
        buttons.rejected.connect(self.reject)

        # Se agregan todos los elementos al layout en orden vertical.
        layout.addWidget(raw_label)
        layout.addWidget(self.raw_input)
        layout.addWidget(interpreted_label)
        layout.addWidget(self.interpreted_input)
        layout.addWidget(self.error_label)
        layout.addWidget(buttons)

    def values(self) -> tuple[str, str]:
        """
        Devuelve los textos ingresados por el usuario, ya normalizados.

        Returns:
            tuple[str, str]: una tupla (letras_detectadas, interpretacion)
            con los espacios en blanco de los extremos eliminados.
        """
        return self.raw_input.text().strip(), self.interpreted_input.toPlainText().strip()

    def _validate_and_accept(self) -> None:
        """
        Valida los campos y acepta el diálogo solo si ambos están completos.

        Si falta alguno de los dos textos, se muestra el mensaje de error
        correspondiente en la etiqueta y el diálogo permanece abierto para
        que el usuario complete la información.
        """
        raw_text, interpreted_text = self.values()
        # Validación del campo de letras detectadas.
        if not raw_text:
            self.error_label.setText("Debes ingresar las letras detectadas.")
            return
        # Validación del campo de interpretación.
        if not interpreted_text:
            self.error_label.setText("Debes ingresar la interpretación correcta.")
            return
        # Ambos campos son válidos: se cierra el diálogo con estado aceptado.
        self.accept()
