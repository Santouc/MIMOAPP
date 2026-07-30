import time
from collections import Counter, deque

import cv2
import numpy as np
from PySide6.QtCore import QTimer, Qt, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QComboBox, QDialog, QHBoxLayout, QLabel, QMessageBox, QPushButton, QVBoxLayout, QWidget

from app.app_context import AppContext
from core.hand_detector import HandDetector
from core.image_processor import ImageProcessor
from ml.clasificador import SignClassifier
from ml.dynamic_classifier import DynamicSignClassifier
from app.widgets import TranscriptionCorrectionDialog

DYNAMIC_MOTION_THRESHOLD = 0.012
DYNAMIC_STILL_FRAMES_RESET = 6
STATIC_MIN_STILL_FRAMES = 6


class TranslateScreen(QWidget):
    back_requested = Signal()

    def __init__(self, context: AppContext, parent=None):
        super().__init__(parent)
        self.context = context
        self.capture = None
        self.hand_detector = None
        self.image_processor = ImageProcessor()
        self.static_classifier = None
        self.dynamic_classifier = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._process_frame)
        self.sign_buffer = deque(maxlen=10)
        self.dynamic_sequence = deque(maxlen=20)
        self.dynamic_buffer = deque(maxlen=5)
        self.still_frames = 0
        self._last_motion_reference = None
        self._last_seen_accepted_at = 0.0
        self.transcription_state = None
        self._last_notified_transcription = ("", "")
        self.confidence_threshold = 0.7
        self.last_landmarks = None
        self.missed_frames = 0
        self.max_missed_frames = 4
        self.smoothing_alpha = 0.65
        self.tracking_status = "PERDIDO"
        self.last_time = time.time()
        self.frame_count = 0
        self.fps = 0.0
        self._build_ui()

    def stop_camera(self) -> None:
        self.timer.stop()
        if self.capture is not None:
            self.capture.release()
            self.capture = None
        if self.hand_detector is not None:
            self.hand_detector.cleanup()
            self.hand_detector = None

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        title = QLabel("Traducir en vivo")
        title.setObjectName("TitleLabel")
        self.video_label = QLabel("Presiona 'Iniciar traducción' para abrir la cámara")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumHeight(430)
        self.video_label.setObjectName("CameraLabel")

        self.translation_label = QLabel("-")
        self.translation_label.setAlignment(Qt.AlignCenter)
        self.translation_label.setObjectName("TranslationLabel")
        self.raw_text_label = QLabel("Letras: -")
        self.raw_text_label.setAlignment(Qt.AlignCenter)
        self.raw_text_label.setObjectName("BodyLabel")
        self.transcription_status_label = QLabel("Transcripción: esperando")
        self.transcription_status_label.setObjectName("BodyLabel")
        self.status_label = QLabel("Estado: cámara detenida")
        self.status_label.setObjectName("BodyLabel")

        buttons = QHBoxLayout()
        start_button = QPushButton("Iniciar traducción")
        stop_button = QPushButton("Detener cámara")
        backspace_button = QPushButton("Borrar letra")
        clear_button = QPushButton("Limpiar texto")
        teach_button = QPushButton("Enseñar interpretación")
        back_button = QPushButton("Volver al inicio")
        start_button.clicked.connect(self._start_camera)
        stop_button.clicked.connect(self._stop_and_reset)
        backspace_button.clicked.connect(self._backspace_transcription)
        clear_button.clicked.connect(self._clear_transcription)
        teach_button.clicked.connect(self._teach_interpretation)
        back_button.clicked.connect(self._go_back)
        buttons.addWidget(start_button)
        buttons.addWidget(stop_button)
        buttons.addWidget(backspace_button)
        buttons.addWidget(clear_button)
        buttons.addWidget(teach_button)
        self.extension_buttons_layout = QHBoxLayout()
        self.extension_actions = []
        buttons.addLayout(self.extension_buttons_layout)
        self.refresh_extension_actions()
        buttons.addStretch(1)
        buttons.addWidget(QLabel("Idioma:"))
        self.language_combo = QComboBox()
        for code, name in self.context.transcription.available_languages():
            self.language_combo.addItem(name, code)
        current_index = self.language_combo.findData(self.context.transcription.language)
        if current_index >= 0:
            self.language_combo.setCurrentIndex(current_index)
        self.language_combo.currentIndexChanged.connect(self._change_language)
        buttons.addWidget(self.language_combo)
        buttons.addWidget(back_button)

        layout.addWidget(title)
        layout.addWidget(self.video_label, 1)
        layout.addWidget(QLabel("Transcripción final:"))
        layout.addWidget(self.translation_label)
        layout.addWidget(self.raw_text_label)
        layout.addWidget(self.transcription_status_label)
        layout.addWidget(self.status_label)
        layout.addLayout(buttons)

    def _start_camera(self) -> None:
        self._load_classifiers()
        self.context.transcription.reset()
        self._reset_buffers()
        self.capture = cv2.VideoCapture(0)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.capture.set(cv2.CAP_PROP_FPS, 30)
        if not self.capture.isOpened():
            self.capture = None
            QMessageBox.critical(self, "Cámara no disponible", "No se pudo abrir la cámara.")
            return
        self.hand_detector = HandDetector(max_hands=1)
        self.timer.start(30)
        self.status_label.setText(self._model_status_text())

    def _load_classifiers(self) -> None:
        self.static_classifier = SignClassifier(
            model_path=str(self.context.paths.static_model_path),
            labels_path=str(self.context.paths.static_labels_path),
        )
        self.dynamic_classifier = DynamicSignClassifier(
            model_path=str(self.context.paths.dynamic_model_path),
            labels_path=str(self.context.paths.dynamic_labels_path),
            sequence_length=20,
        )

    def _process_frame(self) -> None:
        if self.capture is None:
            return
        ok, frame = self.capture.read()
        if not ok:
            return
        frame = cv2.flip(frame, 1)
        processed = self.image_processor.preprocess(frame)
        raw_landmarks = self.hand_detector.detect(processed) if self.hand_detector else None
        landmarks = self._stabilize_landmarks(raw_landmarks)

        static_sign = None
        dynamic_sign = None
        if landmarks is not None:
            processed = self.hand_detector.draw_landmarks(processed, landmarks)
            self._update_motion_state(landmarks)
            static_sign = self._process_static_prediction(landmarks)
            dynamic_sign = self._process_dynamic_prediction(landmarks)

        final_sign = self._select_final_sign(static_sign, dynamic_sign)
        self.transcription_state = self.context.transcription.process_sign(final_sign)
        accepted_at = self.context.transcription.last_accepted_at
        if accepted_at != self._last_seen_accepted_at:
            self._last_seen_accepted_at = accepted_at
            self._reset_dynamic_state()
        self._update_results(final_sign)
        self._draw_overlay(processed, final_sign)
        self._show_frame(processed)

    def _update_motion_state(self, landmarks) -> None:
        hand_landmarks = landmarks[0] if landmarks else None
        if hand_landmarks is None or len(hand_landmarks) != 21:
            return
        motion = self._measure_motion(hand_landmarks)
        if motion is not None and motion < DYNAMIC_MOTION_THRESHOLD:
            self.still_frames += 1
        else:
            self.still_frames = 0

    def _process_static_prediction(self, landmarks):
        if self.static_classifier is None or self.static_classifier.model is None:
            return None
        sign, confidence = self.static_classifier.classify(landmarks)
        self.sign_buffer.append((sign, confidence))
        if self.still_frames < STATIC_MIN_STILL_FRAMES:
            return None
        if len(self.sign_buffer) < 5:
            return None
        return self._get_consensus_sign(self.sign_buffer)

    def _process_dynamic_prediction(self, landmarks):
        if self.dynamic_classifier is None or self.dynamic_classifier.model is None:
            return None
        if not landmarks or len(landmarks) == 0:
            self._reset_dynamic_state(clear_reference=True)
            return None
        hand_landmarks = landmarks[0]
        if len(hand_landmarks) != 21:
            return None
        if self.still_frames >= DYNAMIC_STILL_FRAMES_RESET:
            if self.dynamic_sequence or self.dynamic_buffer:
                self.dynamic_sequence.clear()
                self.dynamic_buffer.clear()
            return None
        self.dynamic_sequence.append(hand_landmarks)
        if len(self.dynamic_sequence) < self.dynamic_sequence.maxlen:
            return None
        sign, confidence = self.dynamic_classifier.classify_sequence(list(self.dynamic_sequence))
        self.dynamic_buffer.append((sign, confidence))
        if len(self.dynamic_buffer) < self.dynamic_buffer.maxlen:
            return None
        return self._get_consensus_sign(self.dynamic_buffer)

    def _measure_motion(self, hand_landmarks) -> float | None:
        try:
            current = np.asarray(hand_landmarks, dtype=np.float32)[:, :2]
        except (TypeError, ValueError):
            return None
        previous = self._last_motion_reference
        self._last_motion_reference = current
        if previous is None or previous.shape != current.shape:
            return None
        return float(np.mean(np.linalg.norm(current - previous, axis=1)))

    def _reset_dynamic_state(self, clear_reference: bool = False) -> None:
        self.dynamic_sequence.clear()
        self.dynamic_buffer.clear()
        self.still_frames = 0
        if clear_reference:
            self._last_motion_reference = None

    def _get_consensus_sign(self, buffer) -> str | None:
        signs = [item[0] for item in buffer if item[1] > self.confidence_threshold and item[0] != "unknown"]
        if not signs:
            return None
        return Counter(signs).most_common(1)[0][0]

    def _stabilize_landmarks(self, landmarks):
        if landmarks is None or len(landmarks) == 0:
            if self.last_landmarks is not None and self.missed_frames < self.max_missed_frames:
                self.missed_frames += 1
                self.tracking_status = "RECUPERANDO"
                return self.last_landmarks
            self.last_landmarks = None
            self.missed_frames = 0
            self.tracking_status = "PERDIDO"
            self._reset_dynamic_state(clear_reference=True)
            return None

        if self.last_landmarks is None:
            self.last_landmarks = landmarks
            self.missed_frames = 0
            self.tracking_status = "OK"
            return landmarks

        try:
            stabilized = []
            used_previous = set()
            for current_hand in landmarks:
                if not isinstance(current_hand, list) or len(current_hand) != 21:
                    stabilized.append(current_hand)
                    continue
                current = np.array(current_hand, dtype=np.float32)
                best_previous_index = None
                best_distance = 1e9
                for index, previous_hand in enumerate(self.last_landmarks if isinstance(self.last_landmarks, list) else []):
                    if index in used_previous or not isinstance(previous_hand, list) or len(previous_hand) != 21:
                        continue
                    previous_wrist = np.array(previous_hand[0], dtype=np.float32)
                    current_wrist = np.array(current_hand[0], dtype=np.float32)
                    distance = float(np.linalg.norm(previous_wrist[:2] - current_wrist[:2]))
                    if distance < best_distance:
                        best_distance = distance
                        best_previous_index = index
                if best_previous_index is not None:
                    previous = np.array(self.last_landmarks[best_previous_index], dtype=np.float32)
                    used_previous.add(best_previous_index)
                else:
                    previous = current
                if current.shape != previous.shape or current.shape != (21, 3):
                    stabilized.append(current_hand)
                    continue
                smoothed = self.smoothing_alpha * current + (1.0 - self.smoothing_alpha) * previous
                stabilized.append(smoothed.tolist())
            self.last_landmarks = stabilized
            self.missed_frames = 0
            self.tracking_status = "OK"
            return stabilized
        except Exception:
            self.last_landmarks = landmarks
            self.missed_frames = 0
            self.tracking_status = "OK"
            return landmarks

    def _select_final_sign(self, static_sign, dynamic_sign) -> str | None:
        return dynamic_sign or static_sign

    def _notify_extensions(self) -> None:
        state = self.transcription_state
        raw_text = state.raw_text if state else ""
        output_text = state.output_text if state else ""
        snapshot = (raw_text, output_text)
        if snapshot == self._last_notified_transcription:
            return
        self._last_notified_transcription = snapshot
        self.context.extensions.notify_transcription(state)

    def _update_results(self, final_sign) -> None:
        state = self.transcription_state
        self._notify_extensions()
        output_text = state.output_text if state and state.output_text else "-"
        raw_text = state.raw_text if state and state.raw_text else "-"
        transcription_status = state.status if state else "Esperando seña estable"
        self.translation_label.setText(output_text)
        self.raw_text_label.setText(f"Letras: {raw_text}")
        self.transcription_status_label.setText(f"Transcripción: {transcription_status}")
        self.status_label.setText(f"{self._model_status_text()} | Tracking: {self.tracking_status} | FPS: {self._calculate_fps():.1f}")

    def _draw_overlay(self, frame, final_sign) -> None:
        output_text = self.transcription_state.output_text if self.transcription_state and self.transcription_state.output_text else "-"
        raw_text = self.transcription_state.raw_text if self.transcription_state and self.transcription_state.raw_text else "-"
        translation_text = f"Texto: {output_text}"
        letter_text = f"Letras: {raw_text} | Actual: {final_sign or '-'}"
        tracking_color = (0, 255, 0) if self.tracking_status == "OK" else (0, 255, 255) if self.tracking_status == "RECUPERANDO" else (0, 0, 255)
        cv2.putText(frame, translation_text, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.putText(frame, letter_text, (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
        cv2.putText(frame, f"Tracking: {self.tracking_status}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.65, tracking_color, 2)

    def _show_frame(self, frame) -> None:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channels = rgb.shape
        image = QImage(rgb.data, width, height, channels * width, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(image).scaled(
            self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.video_label.setPixmap(pixmap)

    def _calculate_fps(self) -> float:
        self.frame_count += 1
        current_time = time.time()
        elapsed = current_time - self.last_time
        if elapsed >= 1.0:
            self.fps = self.frame_count / elapsed
            self.last_time = current_time
            self.frame_count = 0
        return self.fps

    def _model_status_text(self) -> str:
        static_ready = self.static_classifier is not None and self.static_classifier.model is not None and len(self.static_classifier.labels) > 0
        dynamic_ready = self.dynamic_classifier is not None and self.dynamic_classifier.model is not None and len(self.dynamic_classifier.labels) > 0
        if static_ready and dynamic_ready:
            return "Estado: modelos estático y dinámico cargados"
        if static_ready:
            return "Estado: modelo estático cargado; dinámico no disponible"
        if dynamic_ready:
            return "Estado: modelo dinámico cargado; estático no disponible"
        return "Estado: no hay modelos entrenados disponibles"

    def _reset_buffers(self) -> None:
        self.sign_buffer.clear()
        self.dynamic_sequence.clear()
        self.dynamic_buffer.clear()
        self.last_landmarks = None
        self.missed_frames = 0
        self.tracking_status = "PERDIDO"
        self.last_time = time.time()
        self.frame_count = 0
        self.fps = 0.0
        self.transcription_state = None
        self.translation_label.setText("-")
        self.raw_text_label.setText("Letras: -")
        self.transcription_status_label.setText("Transcripción: esperando")

    def _backspace_transcription(self) -> None:
        self.transcription_state = self.context.transcription.backspace()
        self._update_transcription_labels()

    def _clear_transcription(self) -> None:
        self.transcription_state = self.context.transcription.clear()
        self._update_transcription_labels()

    def _teach_interpretation(self) -> None:
        raw_text = self.context.transcription.get_raw_text()
        if not raw_text:
            QMessageBox.information(self, "Sin letras", "Primero transcribe algunas letras para enseñar una interpretación.")
            return
        current_text = self.context.transcription.get_output_text()
        dialog = TranscriptionCorrectionDialog(raw_text, current_text, self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        corrected_raw, corrected_text = dialog.values()
        self.transcription_state = self.context.transcription.learn_phrase(corrected_raw, corrected_text)
        self._update_transcription_labels()
        QMessageBox.information(self, "Interpretación aprendida", "El sistema recordará esta corrección para futuras transcripciones similares.")

    def _change_language(self) -> None:
        code = self.language_combo.currentData()
        if not code or not self.context.transcription.set_language(code):
            return
        self.transcription_state = self.context.transcription.process_sign(None)
        self._update_transcription_labels()
        self.transcription_status_label.setText(f"Transcripción: idioma cambiado a {self.language_combo.currentText()}")

    def refresh_extension_actions(self) -> None:
        while self.extension_buttons_layout.count():
            item = self.extension_buttons_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self.extension_actions = self.context.extensions.translate_actions(self)
        for action in self.extension_actions:
            extension_button = QPushButton(action.label)
            extension_button.clicked.connect(action.callback)
            self.extension_buttons_layout.addWidget(extension_button)

    def _update_transcription_labels(self) -> None:
        state = self.transcription_state
        self._notify_extensions()
        self.translation_label.setText(state.output_text if state and state.output_text else "-")
        self.raw_text_label.setText(f"Letras: {state.raw_text if state and state.raw_text else '-'}")
        self.transcription_status_label.setText(f"Transcripción: {state.status if state else 'esperando'}")

    def _stop_and_reset(self) -> None:
        self.stop_camera()
        self.context.transcription.reset()
        self._reset_buffers()
        self.video_label.setText("Cámara detenida")
        self.status_label.setText("Estado: cámara detenida")

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key_C:
            self._teach_interpretation()
            return
        for action in self.extension_actions:
            if action.key and event.key() == getattr(Qt, f"Key_{action.key.upper()}", None):
                action.callback()
                return
        super().keyPressEvent(event)

    def _go_back(self) -> None:
        self._stop_and_reset()
        self.back_requested.emit()
