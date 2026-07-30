from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app.app_context import AppContext


class ManageSignsScreen(QWidget):
    back_requested = Signal()

    def __init__(self, context: AppContext, parent=None):
        super().__init__(parent)
        self.context = context
        self._signs = []
        self._build_ui()
        self.refresh()

    def refresh(self) -> None:
        self._signs = self.context.signs.list_signs()
        self.table.setRowCount(len(self._signs))
        for row, sign in enumerate(self._signs):
            self.table.setItem(row, 0, QTableWidgetItem(sign["name"]))
            self.table.setItem(row, 1, QTableWidgetItem(self._format_types(sign.get("types", []))))
            self.table.setItem(row, 2, QTableWidgetItem(str(sign.get("static_samples", 0))))
            self.table.setItem(row, 3, QTableWidgetItem(str(sign.get("dynamic_samples", 0))))
            self.table.setItem(row, 4, QTableWidgetItem(sign.get("updated_at", "-")))
        self.status_label.setText(f"Señas registradas: {len(self._signs)}")

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        title = QLabel("Gestionar señas")
        title.setObjectName("TitleLabel")
        description = QLabel("Agrega frases o señas, revisa sus muestras y elimina de raíz cualquier seña que ya no quieras conservar.")
        description.setObjectName("BodyLabel")
        description.setWordWrap(True)

        form_layout = QHBoxLayout()
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("Nombre de la seña o frase")
        self.static_checkbox = QCheckBox("Estática")
        self.dynamic_checkbox = QCheckBox("Dinámica")
        add_button = QPushButton("Agregar seña")
        add_button.clicked.connect(self._add_sign)
        form_layout.addWidget(self.name_input, 1)
        form_layout.addWidget(self.static_checkbox)
        form_layout.addWidget(self.dynamic_checkbox)
        form_layout.addWidget(add_button)

        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["Seña", "Tipo", "Muestras estáticas", "Muestras dinámicas", "Actualizada"])
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.MultiSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeToContents)

        actions_layout = QHBoxLayout()
        refresh_button = QPushButton("Actualizar lista")
        alphabet_button = QPushButton("Agregar alfabeto occidental")
        delete_button = QPushButton("Eliminar señas seleccionadas")
        reset_button = QPushButton("Resetear todo")
        back_button = QPushButton("Volver al inicio")
        refresh_button.clicked.connect(self.refresh)
        alphabet_button.clicked.connect(self._import_western_alphabet)
        delete_button.clicked.connect(self._delete_selected_signs)
        reset_button.clicked.connect(self._reset_all_signs)
        back_button.clicked.connect(self.back_requested)
        actions_layout.addWidget(refresh_button)
        actions_layout.addWidget(alphabet_button)
        actions_layout.addWidget(delete_button)
        actions_layout.addWidget(reset_button)
        actions_layout.addStretch(1)
        actions_layout.addWidget(back_button)

        self.status_label = QLabel()
        self.status_label.setObjectName("BodyLabel")

        layout.addWidget(title)
        layout.addWidget(description)
        layout.addLayout(form_layout)
        layout.addWidget(self.table, 1)
        layout.addLayout(actions_layout)
        layout.addWidget(self.status_label)

    def _add_sign(self) -> None:
        name = self.name_input.text().strip()
        sign_types = []
        if self.static_checkbox.isChecked():
            sign_types.append("static")
        if self.dynamic_checkbox.isChecked():
            sign_types.append("dynamic")
        try:
            sign = self.context.signs.add_sign(name, sign_types)
            self.context.signs.export_labels()
        except ValueError as error:
            QMessageBox.warning(self, "No se pudo agregar", str(error))
            return
        self.name_input.clear()
        self.static_checkbox.setChecked(False)
        self.dynamic_checkbox.setChecked(False)
        self.refresh()
        QMessageBox.information(self, "Seña agregada", f"La seña '{sign['name']}' fue registrada correctamente.")

    def _import_western_alphabet(self) -> None:
        confirmation = QMessageBox.question(
            self,
            "Agregar alfabeto occidental",
            "Se agregarán las letras A-Z al registro de señas.\n\n"
            "No se duplicarán letras que ya existan y se actualizarán las labels exportadas.\n"
            "Luego podrás enseñar capturas para cada letra cuando lo necesites.\n\n"
            "¿Quieres continuar?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if confirmation != QMessageBox.Yes:
            return
        try:
            result = self.context.libraries.import_western_alphabet()
        except Exception as error:
            QMessageBox.critical(self, "No se pudo importar", str(error))
            return
        self.refresh()
        QMessageBox.information(
            self,
            "Librería importada",
            f"{result.library_name} agregado correctamente.\n\n"
            f"Nuevas señas creadas: {result.created_count}\n"
            f"Señas que ya existían: {result.existing_count}",
        )

    def _delete_selected_signs(self) -> None:
        selected_rows = self.table.selectionModel().selectedRows()
        if not selected_rows:
            QMessageBox.information(self, "Selecciona señas", "Primero selecciona una o más señas de la tabla.")
            return
        signs = [self._signs[index.row()] for index in selected_rows]
        if len(signs) == 1:
            if not self._confirm_delete(signs[0]):
                return
        elif not self._confirm_bulk_delete(signs):
            return
        deleted_static = 0
        deleted_dynamic = 0
        deleted_names = []
        errors = []
        for sign in signs:
            try:
                result = self.context.captures.delete_sign_everywhere(sign["id"])
            except Exception as error:
                errors.append(f"{sign['name']}: {error}")
                continue
            deleted_names.append(sign["name"])
            deleted_static += result["deleted_static_samples"]
            deleted_dynamic += result["deleted_dynamic_samples"]
        self.refresh()
        retrain_summary = self._retrain_models_after_delete() if deleted_names else ""
        if errors:
            QMessageBox.warning(
                self,
                "Eliminación parcial",
                "Algunas señas no pudieron eliminarse:\n\n"
                + "\n".join(errors)
                + f"\n\nSeñas eliminadas correctamente: {len(deleted_names)}"
                + (f"\n\n{retrain_summary}" if retrain_summary else ""),
            )
            return
        QMessageBox.information(
            self,
            "Señas eliminadas",
            f"Señas eliminadas: {len(deleted_names)}\n\n"
            f"Muestras estáticas eliminadas: {deleted_static}\n"
            f"Muestras dinámicas eliminadas: {deleted_dynamic}\n\n"
            "Las labels fueron actualizadas y los modelos se reentrenaron con las señas restantes:\n"
            f"{retrain_summary}",
        )

    def _retrain_models_after_delete(self) -> str:
        progress = QProgressDialog("Reentrenando modelos con las señas restantes...\nEsto puede tardar unos segundos.", "", 0, 0, self)
        progress.setWindowTitle("Entrenando")
        progress.setWindowModality(Qt.WindowModal)
        progress.setCancelButton(None)
        progress.setMinimumDuration(0)
        progress.show()
        QApplication.processEvents()
        summaries = []
        try:
            for capture_type, type_name in (("static", "estático"), ("dynamic", "dinámico")):
                try:
                    result = self.context.training.train(capture_type)
                    summaries.append(f"Modelo {type_name}: {'reentrenado correctamente.' if result.trained else result.message}")
                except Exception as error:
                    summaries.append(f"Modelo {type_name}: error al reentrenar ({error})")
                QApplication.processEvents()
        finally:
            progress.close()
        return "\n".join(summaries)

    def _reset_all_signs(self) -> None:
        has_resettable_data = any(
            path.exists()
            for path in (
                self.context.paths.static_dataset_path,
                self.context.paths.dynamic_dataset_path,
                self.context.paths.static_labels_path,
                self.context.paths.dynamic_labels_path,
                self.context.paths.static_model_path,
                self.context.paths.dynamic_model_path,
            )
        ) or bool(self.context.signs.list_signs()) or bool(self.context.captures.list_pending_sessions())
        if not has_resettable_data:
            QMessageBox.information(self, "Sin datos", "No hay señas, labels, datasets ni modelos para resetear.")
            return
        confirmation = QMessageBox.warning(
            self,
            "Resetear todo",
            "Vas a eliminar TODAS las señas de raíz.\n\n"
            "Esto borrará registros, datasets, capturas pendientes, labels exportadas y modelos entrenados.\n\n"
            "Esta acción no se puede deshacer.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if confirmation != QMessageBox.Yes:
            return
        typed_text, accepted = QInputDialog.getText(
            self,
            "Confirmación final",
            "Para confirmar el reseteo total, escribe exactamente:\nRESET",
        )
        if not accepted or typed_text.strip() != "RESET":
            return
        try:
            result = self.context.captures.reset_all_data()
        except Exception as error:
            QMessageBox.critical(self, "No se pudo resetear", str(error))
            return
        self.refresh()
        QMessageBox.information(
            self,
            "Reset completado",
            "Se eliminaron todos los datos gestionados por la app.\n\n"
            f"Señas eliminadas: {result['deleted_signs']}\n"
            f"Capturas pendientes eliminadas: {result['deleted_pending_sessions']}\n"
            f"Archivos eliminados: {len(result['removed_files'])}\n\n"
            "El programa quedó sin labels, datasets ni modelos entrenados.",
        )

    def _confirm_bulk_delete(self, signs: list[dict]) -> bool:
        names = "\n".join(f"- {sign['name']}" for sign in signs[:12])
        if len(signs) > 12:
            names += f"\n... y {len(signs) - 12} más"
        confirmation = QMessageBox.warning(
            self,
            "Eliminar señas seleccionadas",
            f"Vas a eliminar {len(signs)} señas de raíz:\n\n"
            f"{names}\n\n"
            "Esto borrará sus registros, muestras oficiales, capturas pendientes asociadas y labels exportadas.\n"
            "También se invalidarán modelos entrenados antiguos si existen.\n\n"
            "Esta acción no se puede deshacer.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        return confirmation == QMessageBox.Yes

    def _confirm_delete(self, sign: dict) -> bool:
        first_confirmation = QMessageBox.warning(
            self,
            "Eliminar seña de raíz",
            f"Vas a eliminar '{sign['name']}' de raíz.\n\n"
            "Esto borrará su registro, muestras oficiales, capturas pendientes asociadas y labels exportadas.\n"
            "También se invalidarán modelos entrenados antiguos si existen.\n\n"
            "Esta acción no se puede deshacer.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if first_confirmation != QMessageBox.Yes:
            return False
        typed_name, accepted = QInputDialog.getText(
            self,
            "Confirmación final",
            f"Para confirmar, escribe exactamente el nombre de la seña:\n{sign['name']}",
        )
        return accepted and typed_name.strip() == sign["name"]

    def _format_types(self, sign_types: list[str]) -> str:
        if not sign_types:
            return "Sin muestras"
        labels = {"static": "Estática", "dynamic": "Dinámica"}
        return ", ".join(labels.get(sign_type, sign_type) for sign_type in sign_types)
