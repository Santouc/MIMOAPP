"""Pantalla de gestión de extensiones de la aplicación MIMO (T.L.S).

Define :class:`ExtensionsScreen`, que muestra en una tabla todas las
extensiones detectadas en la carpeta ``extensions/`` junto con su versión,
descripción y estado (activa, habilitada, desactivada o con error), y permite
activarlas o desactivarlas individualmente mediante un botón por fila.

Toda la lógica de descubrimiento y carga de extensiones vive en el servicio
``ExtensionService`` del contexto; esta pantalla solo presenta esa información
y delega las acciones del usuario.
"""

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
    """Tabla interactiva para listar y activar/desactivar extensiones.

    Señales:
        back_requested: se emite cuando el usuario pulsa "Volver al inicio".
    """

    # Señal de navegación para regresar al menú principal.
    back_requested = Signal()

    def __init__(self, context, parent=None):
        """Inicializa la pantalla y construye su interfaz.

        Args:
            context: contexto de aplicación (AppContext) con el servicio
                de extensiones.
            parent: widget padre opcional (convención de Qt).
        """
        super().__init__(parent)
        self.context = context
        self._build_ui()

    def _build_ui(self) -> None:
        """Construye la interfaz: título, tabla de extensiones y botones."""
        # Layout principal vertical de la pantalla.
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        # Encabezado con título y explicación breve del sistema de extensiones.
        title = QLabel("Extensiones")
        title.setObjectName("TitleLabel")
        subtitle = QLabel(
            "Las extensiones agregan funcionalidades opcionales a la app. "
            "Se detectan automáticamente desde la carpeta extensions/."
        )
        subtitle.setObjectName("SubtitleLabel")
        subtitle.setWordWrap(True)

        # Tabla de 5 columnas: nombre, versión, descripción, estado y un
        # botón de activar/desactivar por fila. No es editable ni seleccionable.
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

        # Etiqueta de estado con el resumen de extensiones detectadas/activas.
        self.status_label = QLabel("")
        self.status_label.setObjectName("BodyLabel")

        # Barra inferior de botones: refrescar la lista y volver al inicio.
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
        """Reconsulta las extensiones al servicio y reconstruye la tabla.

        Para cada extensión detectada crea una fila con sus datos y un botón
        que permite alternar su estado (activar/desactivar). Al final,
        actualiza la etiqueta de resumen con los totales.
        """
        # Obtiene la lista actualizada de extensiones desde el servicio.
        extensions = self.context.extensions.list_extensions()
        self.table.setRowCount(len(extensions))
        for row, info in enumerate(extensions):
            # Celdas informativas de la extensión.
            name_item = QTableWidgetItem(info.name)
            version_item = QTableWidgetItem(info.version)
            description_item = QTableWidgetItem(info.description)
            # Determina el texto de estado según error/activa/habilitada.
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

            # Botón por fila para alternar el estado de la extensión.
            # Los valores por defecto de la lambda capturan la carpeta y el
            # estado deseado en el momento de crear el botón.
            toggle_button = QPushButton("Desactivar" if info.enabled else "Activar")
            toggle_button.clicked.connect(
                lambda checked=False, folder=info.folder, enable=not info.enabled: self._toggle(folder, enable)
            )
            self.table.setCellWidget(row, 4, toggle_button)

        # Resumen final: cantidad de extensiones detectadas y activas.
        if not extensions:
            self.status_label.setText("No se detectaron extensiones en la carpeta extensions/.")
        else:
            active_count = sum(1 for info in extensions if info.active)
            self.status_label.setText(f"{len(extensions)} extensión(es) detectada(s), {active_count} activa(s).")

    def _toggle(self, folder: str, enable: bool) -> None:
        """Activa o desactiva una extensión y refresca la tabla.

        Args:
            folder: nombre de la carpeta de la extensión dentro de extensions/.
            enable: True para activarla, False para desactivarla.

        Si la activación falla, muestra el error reportado por el servicio.
        """
        # Delegación al servicio: intenta cambiar el estado de la extensión.
        success = self.context.extensions.set_enabled(folder, enable)
        # Si se intentó activar y falló, informa el motivo al usuario.
        if enable and not success:
            error = self.context.extensions.errors.get(folder, "error desconocido")
            QMessageBox.warning(self, "No se pudo activar", f"La extensión '{folder}' falló al activarse:\n{error}")
        self.refresh()
