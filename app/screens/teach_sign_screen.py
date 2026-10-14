import cv2
import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from app.app_context import AppContext
from app.camera_worker import CameraWorker
from app.widgets import LandmarkPreview, TranscriptionCorrectionDialog
from core.preprocessing import normalize_single_hand


class TeachSignScreen(QWidget):
    back_requested = Signal()

    def __init__(self, context: AppContext, parent=None):
        super().__init__(parent)
        self.context = context
        self.camera_worker = None
        self.current_landmarks = None
        self.samples = []
        self.frame_buffer = []
        self.is_dynamic_recording = False
        self.last_dynamic_discard_reason = ""
        self.last_valid_landmarks = None
        self.missed_landmark_frames = 0
        self.max_missed_landmark_frames = 12
        self.pending_summary = None
        self.pending_session_id = None
        self.preview_mode = False
        self._build_ui()
        self.refresh_signs()

    def refresh_signs(self) -> None:
        self.sign_combo.clear()
        for sign in self.context.signs.list_signs():
            self.sign_combo.addItem(sign["name"], sign["id"])
        self.start_button.setEnabled(self.sign_combo.count() > 0)
        self.preview_button.setEnabled(True)
        if self.sign_combo.count() == 0:
            self.status_label.setText("Puedes usar la vista previa general. Para capturar muestras, primero agrega una seña en Gestionar señas.")
        else:
            self.status_label.setText("Selecciona una seña para capturar muestras o abre la vista previa general para probar correcciones.")

    def stop_camera(self) -> None:
        if self.camera_worker is not None:
            worker = self.camera_worker
            self.camera_worker = None
            try:
                worker.frame_ready.disconnect(self._on_frame_ready)
                worker.camera_failed.disconnect(self._on_camera_failed)
                worker.stop()
            except RuntimeError:
                # El hilo ya terminó y Qt liberó el objeto (finished -> deleteLater).
                pass

    def _on_worker_finished(self) -> None:
        worker = self.sender()
        if worker is not None and worker is self.camera_worker:
            self.camera_worker = None

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        self.pages = QStackedWidget()
        root.addWidget(self.pages)

        self.warning_page = self._build_warning_page()
        self.setup_page = self._build_setup_page()
        self.camera_page = self._build_camera_page()
        self.summary_page = self._build_summary_page()

        self.pages.addWidget(self.warning_page)
        self.pages.addWidget(self.setup_page)
        self.pages.addWidget(self.camera_page)
        self.pages.addWidget(self.summary_page)

    def _build_warning_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setSpacing(14)
        title = QLabel("Antes de enseñar una seña")
        title.setObjectName("TitleLabel")
        body = QLabel(
            "Para obtener un modelo más confiable:\n\n"
            "1. Usa buena iluminación.\n"
            "2. Mantén la misma mano durante toda la sesión.\n"
            "3. Evita fondos muy cargados.\n"
            "4. Intenta capturar entre 100 y 200 muestras por sesión si es estática.\n"
            "5. En dinámicas, repite el movimiento varias veces con velocidad natural.\n\n"
            "Las capturas primero quedarán pendientes. Luego podrás aceptarlas o rechazarlas."
        )
        body.setObjectName("BodyLabel")
        body.setWordWrap(True)
        continue_button = QPushButton("Entendido, continuar")
        back_button = QPushButton("Volver al inicio")
        continue_button.clicked.connect(lambda: self.pages.setCurrentWidget(self.setup_page))
        back_button.clicked.connect(self.back_requested)
        layout.addStretch(1)
        layout.addWidget(title)
        layout.addWidget(body)
        layout.addSpacing(16)
        layout.addWidget(continue_button)
        layout.addWidget(back_button)
        layout.addStretch(2)
        return page

    def _build_setup_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setSpacing(12)
        title = QLabel("Configurar enseñanza")
        title.setObjectName("TitleLabel")
        self.sign_combo = QComboBox()
        self.type_combo = QComboBox()
        self.type_combo.addItem("Estática", "static")
        self.type_combo.addItem("Dinámica", "dynamic")
        self.status_label = QLabel()
        self.status_label.setObjectName("BodyLabel")
        self.start_button = QPushButton("Abrir cámara para capturar")
        self.preview_button = QPushButton("Vista previa general")
        refresh_button = QPushButton("Actualizar señas")
        back_button = QPushButton("Volver al inicio")
        self.start_button.clicked.connect(self._start_camera)
        self.preview_button.clicked.connect(self._start_preview_camera)
        refresh_button.clicked.connect(self.refresh_signs)
        back_button.clicked.connect(self.back_requested)
        layout.addStretch(1)
        layout.addWidget(title)
        layout.addWidget(QLabel("Seña:"))
        layout.addWidget(self.sign_combo)
        layout.addWidget(QLabel("Tipo de captura:"))
        layout.addWidget(self.type_combo)
        layout.addWidget(self.status_label)
        layout.addSpacing(12)
        layout.addWidget(self.start_button)
        layout.addWidget(self.preview_button)
        layout.addWidget(refresh_button)
        layout.addWidget(back_button)
        layout.addStretch(2)
        return page

    def _build_camera_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setSpacing(10)
        self.camera_title = QLabel("Cámara")
        self.camera_title.setObjectName("TitleLabel")
        self.camera_label = QLabel("Cámara no iniciada")
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setMinimumHeight(420)
        self.camera_label.setObjectName("CameraLabel")
        self.capture_status_label = QLabel("Controles: T captura/graba. C corrige interpretación. Q termina la sesión.")
        self.capture_status_label.setObjectName("BodyLabel")
        buttons = QHBoxLayout()
        capture_button = QPushButton("Capturar / Grabar dinámica (T)")
        correction_button = QPushButton("Corregir interpretación (C)")
        finish_button = QPushButton("Terminar sesión (Q)")
        cancel_button = QPushButton("Cancelar")
        capture_button.clicked.connect(self._capture_current)
        correction_button.clicked.connect(self._teach_transcription_interpretation)
        finish_button.clicked.connect(self._finish_session)
        cancel_button.clicked.connect(self._cancel_camera)
        buttons.addWidget(capture_button)
        buttons.addWidget(correction_button)
        buttons.addWidget(finish_button)
        buttons.addWidget(cancel_button)
        layout.addWidget(self.camera_title)
        layout.addWidget(self.camera_label, 1)
        layout.addWidget(self.capture_status_label)
        layout.addLayout(buttons)
        return page

    def _build_summary_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setSpacing(12)
        title = QLabel("Resumen de capturas")
        title.setObjectName("TitleLabel")
        self.summary_label = QLabel()
        self.summary_label.setObjectName("BodyLabel")
        self.summary_label.setWordWrap(True)
        self.preview = LandmarkPreview()
        buttons = QHBoxLayout()
        accept_button = QPushButton("Aceptar y guardar en dataset")
        reject_button = QPushButton("Rechazar y descartar")
        accept_button.clicked.connect(self._accept_pending)
        reject_button.clicked.connect(self._reject_pending)
        buttons.addWidget(accept_button)
        buttons.addWidget(reject_button)
        layout.addWidget(title)
        layout.addWidget(self.summary_label)
        layout.addWidget(self.preview, 1)
        layout.addLayout(buttons)
        return page

    def _start_camera(self) -> None:
        sign_id = self.sign_combo.currentData()
        if not sign_id:
            QMessageBox.warning(self, "Sin seña", "Primero selecciona una seña.")
            return
        self._reset_capture_state(preview_mode=False)
        self._launch_camera_worker()

    def _start_preview_camera(self) -> None:
        self._reset_capture_state(preview_mode=True)
        self.camera_title.setText("Vista previa general")
        self._launch_camera_worker()

    def _reset_capture_state(self, preview_mode: bool) -> None:
        self.preview_mode = preview_mode
        self.samples = []
        self.frame_buffer = []
        self.is_dynamic_recording = False
        self.last_dynamic_discard_reason = ""
        self.last_valid_landmarks = None
        self.missed_landmark_frames = 0
        self.current_landmarks = None
        self.pending_summary = None
        self.pending_session_id = None

    def _launch_camera_worker(self) -> None:
        self.stop_camera()
        self.camera_worker = CameraWorker(max_hands=1, parent=self)
        self.camera_worker.frame_ready.connect(self._on_frame_ready)
        self.camera_worker.camera_failed.connect(self._on_camera_failed)
        self.camera_worker.finished.connect(self._on_worker_finished)
        self.camera_label.setText("Abriendo cámara...")
        self.pages.setCurrentWidget(self.camera_page)
        self.camera_worker.start()
        self._update_capture_status()

    def _on_camera_failed(self, message: str) -> None:
        self.stop_camera()
        QMessageBox.critical(self, "Cámara no disponible", message)
        self.pages.setCurrentWidget(self.setup_page)

    def _on_frame_ready(self, frame, landmarks: list) -> None:
        # Este slot corre en el hilo de la UI: solo guarda landmarks en memoria
        # y pinta el frame; la captura y la detección ocurren en CameraWorker.
        worker = self.camera_worker
        if worker is None:
            return
        try:
            self.current_landmarks = self._get_predictive_landmarks(landmarks)
            if self.type_combo.currentData() == "dynamic" and self.is_dynamic_recording:
                self._record_dynamic_frame()
            self._show_frame(frame)
        finally:
            worker.frame_consumed()

    def _show_frame(self, frame) -> None:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channels = rgb.shape
        image = QImage(rgb.data, width, height, channels * width, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(image).scaled(
            self.camera_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.camera_label.setPixmap(pixmap)

    def _capture_current(self) -> None:
        if self.preview_mode:
            self.capture_status_label.setText("Vista previa general. Usa C para corregir interpretaciones o Q para volver.")
            return
        capture_type = self.type_combo.currentData()
        if self.current_landmarks is None:
            self.capture_status_label.setText("No hay mano detectada. Ajusta iluminación/posición e intenta de nuevo.")
            return
        if capture_type == "static":
            normalized = normalize_single_hand(self.current_landmarks)[0].tolist()
            self.samples.append(normalized)
        else:
            if self.is_dynamic_recording:
                self._stop_dynamic_recording()
            else:
                self._start_dynamic_recording()
        self._update_capture_status()

    def _finish_session(self) -> None:
        if self.preview_mode:
            self._cancel_camera()
            return
        capture_type = self.type_combo.currentData()
        if capture_type == "dynamic" and self.is_dynamic_recording:
            self._stop_dynamic_recording()
        if not self.samples:
            detail = f"\n\n{self.last_dynamic_discard_reason}" if self.last_dynamic_discard_reason else ""
            QMessageBox.information(self, "Sin muestras", f"No capturaste muestras para revisar.{detail}")
            return
        self.stop_camera()
        sign_id = self.sign_combo.currentData()
        try:
            self.pending_summary = self.context.captures.create_pending_session(sign_id, capture_type, self.samples)
        except Exception as error:
            QMessageBox.critical(self, "Error guardando capturas", str(error))
            self.pages.setCurrentWidget(self.setup_page)
            return
        self.pending_session_id = self.pending_summary["session_id"]
        self._show_summary()

    def _show_summary(self) -> None:
        average = self.pending_summary["average_landmarks"]
        capture_type = self.pending_summary["capture_type"]
        if capture_type == "dynamic":
            self.preview.set_sequence(average)
            type_text = "Dinámica"
        else:
            self.preview.set_landmarks(average)
            type_text = "Estática"
        self.summary_label.setText(
            f"Seña: {self.pending_summary['sign_name']}\n"
            f"Tipo: {type_text}\n"
            f"Muestras capturadas: {self.pending_summary['sample_count']}\n\n"
            "Revisa el promedio visual. Si aceptas, se integrará al dataset oficial. "
            "Si rechazas, se eliminará la captura pendiente."
        )
        self.pages.setCurrentWidget(self.summary_page)

    def _accept_pending(self) -> None:
        if not self.pending_session_id:
            return
        capture_type = self.pending_summary["capture_type"]
        try:
            self.context.captures.accept_pending_session(self.pending_session_id)
        except Exception as error:
            QMessageBox.critical(self, "Error al aceptar", str(error))
            return
        training_result = self._train_with_progress(capture_type)
        QMessageBox.information(self, "Capturas aceptadas", self._format_training_message(training_result))
        self._reset_to_setup()

    def _reject_pending(self) -> None:
        if not self.pending_session_id:
            return
        try:
            self.context.captures.reject_pending_session(self.pending_session_id)
        except Exception as error:
            QMessageBox.critical(self, "Error al rechazar", str(error))
            return
        QMessageBox.information(self, "Capturas rechazadas", "La sesión pendiente fue descartada.")
        self._reset_to_setup()

    def _cancel_camera(self) -> None:
        self.stop_camera()
        self.samples = []
        self.frame_buffer = []
        self.is_dynamic_recording = False
        self.last_dynamic_discard_reason = ""
        self.preview_mode = False
        self.camera_title.setText("Cámara")
        self.pages.setCurrentWidget(self.setup_page)

    def _reset_to_setup(self) -> None:
        self.samples = []
        self.frame_buffer = []
        self.is_dynamic_recording = False
        self.last_dynamic_discard_reason = ""
        self.last_valid_landmarks = None
        self.missed_landmark_frames = 0
        self.pending_summary = None
        self.pending_session_id = None
        self.preview_mode = False
        self.camera_title.setText("Cámara")
        self.preview.stop_animation()
        self.refresh_signs()
        self.pages.setCurrentWidget(self.setup_page)

    def _update_capture_status(self) -> None:
        if self.preview_mode:
            self.capture_status_label.setText("Vista previa general. No se guardan muestras. Usa C para corregir interpretación y Q para volver.")
            return
        capture_type = self.type_combo.currentData()
        if capture_type == "static":
            self.capture_status_label.setText(
                f"Modo estático. Presiona T para capturar una muestra. Muestras: {len(self.samples)}"
            )
        else:
            recording_text = "GRABANDO" if self.is_dynamic_recording else "detenida"
            self.capture_status_label.setText(
                f"Modo dinámico ({recording_text}). T inicia/detiene una secuencia completa. "
                f"Secuencias: {len(self.samples)} | Frames actuales: {len(self.frame_buffer)}"
            )

    def _start_dynamic_recording(self) -> None:
        self.frame_buffer = []
        self.last_dynamic_discard_reason = ""
        self.is_dynamic_recording = True
        self._record_dynamic_frame()

    def _stop_dynamic_recording(self) -> None:
        self.is_dynamic_recording = False
        if len(self.frame_buffer) < 5:
            self.frame_buffer = []
            self.last_dynamic_discard_reason = "La última secuencia dinámica fue descartada porque duró menos de 5 frames."
            self.capture_status_label.setText(self.last_dynamic_discard_reason)
            return
        self.samples.append(self._resample_sequence(self.frame_buffer, 20))
        self.frame_buffer = []

    def _record_dynamic_frame(self) -> None:
        if self.current_landmarks is None:
            return
        # Se guardan las coordenadas crudas (relativas a la imagen) sin centrar:
        # la normalización dinámica se aplica después sobre la secuencia completa
        # para conservar la trayectoria de la mano.
        raw = np.asarray(self.current_landmarks, dtype=np.float32).tolist()
        self.frame_buffer.append(raw)
        self._update_capture_status()

    def _get_predictive_landmarks(self, landmarks):
        current = self._extract_first_hand(landmarks)
        if current is not None:
            self.last_valid_landmarks = current
            self.missed_landmark_frames = 0
            return current
        if self.last_valid_landmarks is not None and self.missed_landmark_frames < self.max_missed_landmark_frames:
            self.missed_landmark_frames += 1
            return self.last_valid_landmarks
        self.last_valid_landmarks = None
        return None

    def _extract_first_hand(self, landmarks):
        if not landmarks or not isinstance(landmarks, list):
            return None
        first = landmarks[0]
        if isinstance(first, (list, tuple)) and len(first) == 21:
            return list(first)
        return None

    def _pad_or_truncate(self, sequence: list, target_length: int) -> list:
        if len(sequence) > target_length:
            return sequence[:target_length]
        if len(sequence) < target_length:
            if not sequence:
                return []
            sequence = sequence + [sequence[-1]] * (target_length - len(sequence))
        return sequence

    def _resample_sequence(self, sequence: list, target_length: int) -> list:
        array = np.asarray(sequence, dtype=np.float32)
        if len(array) == target_length:
            return array.tolist()
        if len(array) == 1:
            return np.repeat(array, target_length, axis=0).tolist()
        source_positions = np.linspace(0, len(array) - 1, num=len(array))
        target_positions = np.linspace(0, len(array) - 1, num=target_length)
        resampled = np.empty((target_length, 21, 3), dtype=np.float32)
        for point_index in range(21):
            for coord_index in range(3):
                resampled[:, point_index, coord_index] = np.interp(
                    target_positions,
                    source_positions,
                    array[:, point_index, coord_index],
                )
        return resampled.tolist()

    def _format_training_message(self, training_result) -> str:
        message = "Las muestras fueron integradas al dataset oficial.\n\n"
        message += training_result.message
        if training_result.trained:
            message += (
                f"\n\nMuestras usadas: {training_result.samples}"
                f"\nClases usadas: {training_result.classes}"
                f"\nAccuracy final: {training_result.accuracy:.3f}"
            )
            if training_result.validation_accuracy is not None:
                message += f"\nAccuracy de validación: {training_result.validation_accuracy:.3f}"
        return message

    def _teach_transcription_interpretation(self) -> None:
        dialog = TranscriptionCorrectionDialog(parent=self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        raw_text, interpreted_text = dialog.values()
        self.context.transcription.learn_phrase(raw_text, interpreted_text)
        QMessageBox.information(self, "Interpretación aprendida", "El sistema recordará esta corrección durante la traducción en vivo.")

    def _train_with_progress(self, capture_type: str):
        progress = QProgressDialog(
            "Entrenando modelo...\nEsto puede tardar unos segundos. Por favor espera.",
            None,
            0,
            0,
            self,
        )
        progress.setWindowTitle("Entrenamiento automático")
        progress.setWindowModality(Qt.ApplicationModal)
        progress.setCancelButton(None)
        progress.setMinimumDuration(0)
        progress.show()
        QApplication.processEvents()
        try:
            return self.context.training.train(capture_type)
        finally:
            progress.close()
            QApplication.processEvents()

    def keyPressEvent(self, event) -> None:
        if self.pages.currentWidget() == self.camera_page:
            if event.key() == Qt.Key_T:
                self._capture_current()
                return
            if event.key() == Qt.Key_C:
                self._teach_transcription_interpretation()
                return
            if event.key() == Qt.Key_Q:
                self._finish_session()
                return
        super().keyPressEvent(event)
