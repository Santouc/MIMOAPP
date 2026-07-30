from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


class ExtensionsScreen(QWidget):
    back_requested = Signal()

    def __init__(self, context, parent=None):
        super().__init__(parent)
        self.context = context
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        title = QLabel("Extensiones")
        title.setObjectName("TitleLabel")
        subtitle = QLabel(
            "Las extensiones agregan funcionalidades opcionales a la app. "
            "Se detectan automáticamente desde la carpeta extensions/."
        )
        subtitle.setObjectName("SubtitleLabel")
        subtitle.setWordWrap(True)

        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["Extensión", "Versión", "Descripción", "Estado", ""])
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeToContents)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionMode(QTableWidget.NoSelection)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)

        self.status_label = QLabel("")
        self.status_label.setObjectName("BodyLabel")

        buttons = QHBoxLayout()
        refresh_button = QPushButton("Actualizar lista")
        back_button = QPushButton("Volver al inicio")
        refresh_button.clicked.connect(self.refresh)
        back_button.clicked.connect(self.back_requested)
        buttons.addWidget(refresh_button)
        buttons.addStretch(1)
        buttons.addWidget(back_button)

        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addWidget(self.table, 1)
        layout.addWidget(self.status_label)
        layout.addLayout(buttons)

    def refresh(self) -> None:
        extensions = self.context.extensions.list_extensions()
        self.table.setRowCount(len(extensions))
        for row, info in enumerate(extensions):
            name_item = QTableWidgetItem(info.name)
            version_item = QTableWidgetItem(info.version)
            description_item = QTableWidgetItem(info.description)
            if info.error:
                state_text = f"Error: {info.error}"
            elif info.active:
                state_text = "Activa"
            elif info.enabled:
                state_text = "Habilitada (no cargada)"
            else:
                state_text = "Desactivada"
            state_item = QTableWidgetItem(state_text)
            for item in (name_item, version_item, description_item, state_item):
                item.setTextAlignment(Qt.AlignVCenter | Qt.AlignLeft)
            self.table.setItem(row, 0, name_item)
            self.table.setItem(row, 1, version_item)
            self.table.setItem(row, 2, description_item)
            self.table.setItem(row, 3, state_item)

            toggle_button = QPushButton("Desactivar" if info.enabled else "Activar")
            toggle_button.clicked.connect(
                lambda checked=False, folder=info.folder, enable=not info.enabled: self._toggle(folder, enable)
            )
            self.table.setCellWidget(row, 4, toggle_button)

        if not extensions:
            self.status_label.setText("No se detectaron extensiones en la carpeta extensions/.")
        else:
            active_count = sum(1 for info in extensions if info.active)
            self.status_label.setText(f"{len(extensions)} extensión(es) detectada(s), {active_count} activa(s).")

    def _toggle(self, folder: str, enable: bool) -> None:
        success = self.context.extensions.set_enabled(folder, enable)
        if enable and not success:
            error = self.context.extensions.errors.get(folder, "error desconocido")
            QMessageBox.warning(self, "No se pudo activar", f"La extensión '{folder}' falló al activarse:\n{error}")
        self.refresh()
