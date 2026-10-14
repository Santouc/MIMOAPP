"""
Widget de vista previa de landmarks de la mano.

Este módulo define LandmarkPreview, un widget de Qt que dibuja sobre un
lienzo oscuro el esqueleto de una mano a partir de sus 21 landmarks
(puntos de referencia) detectados por MediaPipe. Soporta dos modos:

- Seña estática: se muestra un único conjunto de 21 puntos (matriz 21x3).
- Seña dinámica: se recibe una secuencia de frames (matriz Nx21x3) y el
  widget la reproduce en bucle como una animación mediante un QTimer.

El dibujo escala y centra automáticamente los puntos para aprovechar el
tamaño disponible del widget, une los puntos según las conexiones
anatómicas de la mano (HAND_CONNECTIONS) y resalta la muñeca en un color
distinto.
"""

# NumPy se usa para manipular los arreglos de landmarks (validación de
# formas, normalización y escalado de coordenadas).
import numpy as np
# QPointF: punto 2D con coordenadas flotantes; QTimer: temporizador para la
# animación; Qt: constantes generales (alineación de texto, etc.).
from PySide6.QtCore import QPointF, QTimer, Qt
# Clases de dibujo: colores, pintor 2D y lápices (grosor/estilo de trazo).
from PySide6.QtGui import QColor, QPainter, QPen
# Clase base de todo widget visual de Qt.
from PySide6.QtWidgets import QWidget

# HAND_CONNECTIONS define los pares de índices de landmarks que deben
# unirse con líneas para dibujar el esqueleto de la mano.
from visualization.landmark_average import HAND_CONNECTIONS


class LandmarkPreview(QWidget):
    """
    Widget que renderiza los landmarks de una mano como un esqueleto 2D.

    Atributos:
        landmarks: matriz (21, 3) con el frame que se está dibujando
            actualmente, o None si no hay datos que mostrar.
        sequence: matriz (N, 21, 3) con la secuencia completa de una seña
            dinámica, o None si se muestra una seña estática.
        frame_index: índice del frame actual dentro de la secuencia.
        animation_timer: temporizador que avanza la animación frame a frame.
    """

    def __init__(self, parent=None):
        """
        Inicializa el widget sin datos y prepara el temporizador de animación.

        Args:
            parent: widget padre dentro de la jerarquía de Qt (opcional).
        """
        super().__init__(parent)
        # Estado inicial: sin landmarks ni secuencia cargados.
        self.landmarks = None
        self.sequence = None
        self.frame_index = 0
        # El temporizador dispara _advance_frame periódicamente para animar
        # las señas dinámicas.
        self.animation_timer = QTimer(self)
        self.animation_timer.timeout.connect(self._advance_frame)
        # Altura mínima para que la vista previa siempre sea visible.
        self.setMinimumHeight(260)

    def set_landmarks(self, landmarks) -> None:
        """
        Muestra una seña estática a partir de un único frame de landmarks.

        Detiene cualquier animación en curso y valida que los datos tengan
        la forma esperada (21 puntos con 3 coordenadas cada uno); si la
        forma no coincide, se descartan y no se dibuja nada.

        Args:
            landmarks: estructura convertible a un arreglo NumPy de forma
                (21, 3), o None para limpiar la vista previa.
        """
        # Al fijar un frame estático se cancela la animación y se descarta
        # cualquier secuencia previa.
        self.stop_animation()
        self.sequence = None
        self.frame_index = 0
        if landmarks is None:
            self.landmarks = None
        else:
            # Se convierte a NumPy y solo se acepta si tiene exactamente la
            # forma (21, 3); en caso contrario se ignora por seguridad.
            array = np.asarray(landmarks, dtype=np.float32)
            self.landmarks = array if array.shape == (21, 3) else None
        # Se solicita a Qt un repintado del widget con el nuevo estado.
        self.update()

    def set_sequence(self, sequence, interval_ms: int = 120) -> None:
        """
        Muestra una seña dinámica reproduciendo una secuencia de frames.

        Valida que la secuencia tenga forma (N, 21, 3); si es válida,
        muestra el primer frame y, cuando hay más de uno, arranca el
        temporizador para animar el resto en bucle.

        Args:
            sequence: estructura convertible a un arreglo NumPy de forma
                (N, 21, 3), o None para limpiar la vista previa.
            interval_ms: milisegundos entre frames de la animación.
        """
        # Se reinicia por completo el estado antes de cargar la secuencia.
        self.stop_animation()
        self.landmarks = None
        self.sequence = None
        self.frame_index = 0
        if sequence is not None:
            array = np.asarray(sequence, dtype=np.float32)
            # Solo se acepta una secuencia tridimensional no vacía cuyos
            # frames tengan la forma (21, 3).
            if array.ndim == 3 and array.shape[1:] == (21, 3) and len(array) > 0:
                self.sequence = array
                # Se muestra de inmediato el primer frame.
                self.landmarks = self.sequence[0]
                # Con más de un frame, se inicia la animación periódica.
                if len(array) > 1:
                    self.animation_timer.start(interval_ms)
        # Se solicita el repintado con el nuevo estado.
        self.update()

    def stop_animation(self) -> None:
        """Detiene el temporizador de animación si está en marcha."""
        if self.animation_timer.isActive():
            self.animation_timer.stop()

    def _advance_frame(self) -> None:
        """
        Avanza al siguiente frame de la secuencia animada.

        Se ejecuta en cada disparo del temporizador. Usa aritmética modular
        para volver al primer frame al llegar al final (reproducción en
        bucle). Si ya no hay secuencia válida, detiene la animación.
        """
        # Sin secuencia válida no hay nada que animar: se corta el timer.
        if self.sequence is None or len(self.sequence) == 0:
            self.stop_animation()
            return
        # Avance circular: tras el último frame se vuelve al primero.
        self.frame_index = (self.frame_index + 1) % len(self.sequence)
        self.landmarks = self.sequence[self.frame_index]
        # Se repinta el widget con el frame actualizado.
        self.update()

    def paintEvent(self, event) -> None:
        """
        Dibuja el contenido del widget (invocado automáticamente por Qt).

        Pinta el fondo oscuro y, si hay landmarks cargados, dibuja las
        conexiones del esqueleto de la mano, los puntos individuales y la
        muñeca resaltada. Si se está reproduciendo una animación, muestra
        además un indicador con el número de frame actual.

        Args:
            event: evento de repintado de Qt (no se usa directamente).
        """
        # Se prepara el pintor con suavizado de bordes (antialiasing) y se
        # rellena todo el fondo con un color oscuro.
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor("#111827"))
        # Sin datos, se muestra un mensaje informativo centrado y se termina.
        if self.landmarks is None:
            painter.setPen(QColor("#94a3b8"))
            painter.drawText(self.rect(), Qt.AlignCenter, "Sin vista previa disponible")
            return

        # Se proyectan los landmarks normalizados a coordenadas de píxel del
        # widget y se definen los lápices: celeste para las conexiones,
        # blanco para los puntos y verde (más grueso) para la muñeca.
        points = self._project_points()
        connection_pen = QPen(QColor("#38bdf8"), 3)
        point_pen = QPen(QColor("#f8fafc"), 8)
        wrist_pen = QPen(QColor("#22c55e"), 12)

        # Se dibujan primero las líneas que unen los landmarks según la
        # estructura anatómica de la mano.
        painter.setPen(connection_pen)
        for start, end in HAND_CONNECTIONS:
            painter.drawLine(points[start], points[end])

        # Luego se dibujan los puntos individuales, salvo la muñeca
        # (índice 0), que se pinta aparte con otro estilo.
        painter.setPen(point_pen)
        for index, point in enumerate(points):
            if index == 0:
                continue
            painter.drawPoint(point)

        # La muñeca se resalta con un punto verde de mayor tamaño.
        painter.setPen(wrist_pen)
        painter.drawPoint(points[0])
        # Si se está reproduciendo una secuencia, se indica el progreso de
        # la animación en la esquina superior izquierda.
        if self.sequence is not None and len(self.sequence) > 1:
            painter.setPen(QColor("#cbd5e1"))
            painter.drawText(
                16,
                28,
                f"Animación dinámica {self.frame_index + 1}/{len(self.sequence)}",
            )

    def _project_points(self) -> list[QPointF]:
        """
        Convierte los landmarks a coordenadas de píxel dentro del widget.

        Toma solo las coordenadas X e Y de los 21 landmarks, centra la nube
        de puntos respecto a su caja delimitadora y la escala de manera
        uniforme (manteniendo la proporción) para que ocupe el área del
        widget dejando un margen de 24 píxeles por lado.

        Returns:
            list[QPointF]: los 21 puntos proyectados, en el mismo orden que
            los landmarks originales, listos para dibujar con QPainter.
        """
        # Se descartan las coordenadas Z: la vista previa es 2D.
        coords = self.landmarks[:, :2].copy()
        # Caja delimitadora de la mano y su centro geométrico.
        min_xy = coords.min(axis=0)
        max_xy = coords.max(axis=0)
        center = (min_xy + max_xy) / 2
        # Se trasladan los puntos para que el centro quede en el origen.
        centered = coords - center
        # Se evita la división por cero cuando la mano ocupa un área nula.
        span = np.maximum(max_xy - min_xy, 1e-6)
        # Escala uniforme: se toma el factor menor entre ancho y alto para
        # que la mano quepa completa, con 48 px de margen total (24 por lado).
        scale = min(
            max(self.width() - 48, 1) / span[0],
            max(self.height() - 48, 1) / span[1],
        )
        # Se traslada la nube escalada al centro del widget y se convierte
        # cada punto a QPointF para el dibujado.
        widget_center = np.array([self.width() / 2, self.height() / 2], dtype=np.float32)
        return [
            QPointF(*(widget_center + point * scale))
            for point in centered
        ]
