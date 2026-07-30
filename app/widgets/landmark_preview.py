import numpy as np
from PySide6.QtCore import QPointF, QTimer, Qt
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QWidget

from visualization.landmark_average import HAND_CONNECTIONS


class LandmarkPreview(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.landmarks = None
        self.sequence = None
        self.frame_index = 0
        self.animation_timer = QTimer(self)
        self.animation_timer.timeout.connect(self._advance_frame)
        self.setMinimumHeight(260)

    def set_landmarks(self, landmarks) -> None:
        self.stop_animation()
        self.sequence = None
        self.frame_index = 0
        if landmarks is None:
            self.landmarks = None
        else:
            array = np.asarray(landmarks, dtype=np.float32)
            self.landmarks = array if array.shape == (21, 3) else None
        self.update()

    def set_sequence(self, sequence, interval_ms: int = 120) -> None:
        self.stop_animation()
        self.landmarks = None
        self.sequence = None
        self.frame_index = 0
        if sequence is not None:
            array = np.asarray(sequence, dtype=np.float32)
            if array.ndim == 3 and array.shape[1:] == (21, 3) and len(array) > 0:
                self.sequence = array
                self.landmarks = self.sequence[0]
                if len(array) > 1:
                    self.animation_timer.start(interval_ms)
        self.update()

    def stop_animation(self) -> None:
        if self.animation_timer.isActive():
            self.animation_timer.stop()

    def _advance_frame(self) -> None:
        if self.sequence is None or len(self.sequence) == 0:
            self.stop_animation()
            return
        self.frame_index = (self.frame_index + 1) % len(self.sequence)
        self.landmarks = self.sequence[self.frame_index]
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor("#111827"))
        if self.landmarks is None:
            painter.setPen(QColor("#94a3b8"))
            painter.drawText(self.rect(), Qt.AlignCenter, "Sin vista previa disponible")
            return

        points = self._project_points()
        connection_pen = QPen(QColor("#38bdf8"), 3)
        point_pen = QPen(QColor("#f8fafc"), 8)
        wrist_pen = QPen(QColor("#22c55e"), 12)

        painter.setPen(connection_pen)
        for start, end in HAND_CONNECTIONS:
            painter.drawLine(points[start], points[end])

        painter.setPen(point_pen)
        for index, point in enumerate(points):
            if index == 0:
                continue
            painter.drawPoint(point)

        painter.setPen(wrist_pen)
        painter.drawPoint(points[0])
        if self.sequence is not None and len(self.sequence) > 1:
            painter.setPen(QColor("#cbd5e1"))
            painter.drawText(
                16,
                28,
                f"Animación dinámica {self.frame_index + 1}/{len(self.sequence)}",
            )

    def _project_points(self) -> list[QPointF]:
        coords = self.landmarks[:, :2].copy()
        min_xy = coords.min(axis=0)
        max_xy = coords.max(axis=0)
        center = (min_xy + max_xy) / 2
        centered = coords - center
        span = np.maximum(max_xy - min_xy, 1e-6)
        scale = min(
            max(self.width() - 48, 1) / span[0],
            max(self.height() - 48, 1) / span[1],
        )
        widget_center = np.array([self.width() / 2, self.height() / 2], dtype=np.float32)
        return [
            QPointF(*(widget_center + point * scale))
            for point in centered
        ]
