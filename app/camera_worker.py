"""Hilo de captura de cámara y detección de landmarks.

`CameraWorker` encapsula en un `QThread` propio todo el trabajo pesado del
flujo de video: abrir la cámara con OpenCV, leer cada fotograma,
preprocesarlo, ejecutar MediaPipe para obtener los landmarks y dibujarlos.
El hilo de la interfaz solo recibe el resultado a través de señales de Qt,
de modo que su única responsabilidad es pintar el pixmap y almacenar los
landmarks en memoria; así el renderizado de PySide6 no se bloquea aunque la
detección tarde más que el intervalo de refresco.
"""

import time

import cv2
from PySide6.QtCore import QThread, Signal

from core.hand_detector import HandDetector
from core.image_processor import ImageProcessor


class CameraWorker(QThread):
    """Lee la cámara y extrae landmarks en un hilo independiente de la UI.

    Señales:
        frame_ready(object, list): fotograma BGR ya preprocesado (y con los
            landmarks dibujados si los hay) y la lista de manos detectadas
            tal como la devuelve `HandDetector.detect` (lista vacía si no
            se detectó ninguna mano).
        camera_failed(str): la cámara no pudo abrirse; incluye un mensaje
            descriptivo para mostrar al usuario.
    """

    frame_ready = Signal(object, list)
    camera_failed = Signal(str)

    def __init__(self, device_id: int = 0, width: int = 640, height: int = 480, fps: int = 30, max_hands: int = 1, draw_landmarks: bool = True, max_pending_frames: int = 2, parent=None):
        super().__init__(parent)
        self.device_id = device_id
        self.width = width
        self.height = height
        self.fps = fps
        self.max_hands = max_hands
        self.draw_landmarks = draw_landmarks
        # Control de contrapresión: si la UI acumula más de este número de
        # frames sin consumir, el worker descarta los nuevos en lugar de
        # encolarlos (evita que la latencia crezca sin límite).
        self.max_pending_frames = max(1, max_pending_frames)
        self._pending_frames = 0
        self.dropped_frames = 0
        self._running = False
        # El objeto se libera solo cuando el hilo realmente termina; destruir
        # un QThread en ejecución aborta el proceso.
        self.finished.connect(self.deleteLater)

    def run(self) -> None:
        """Bucle principal del hilo: captura, detecta y emite hasta que se detenga."""
        capture = cv2.VideoCapture(self.device_id)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        capture.set(cv2.CAP_PROP_FPS, self.fps)
        if not capture.isOpened():
            capture.release()
            self.camera_failed.emit("No se pudo abrir la cámara.")
            return

        # El detector y el procesador se crean dentro del hilo para que
        # MediaPipe y OpenCV trabajen siempre desde el mismo hilo.
        image_processor = ImageProcessor((self.width, self.height))
        hand_detector = HandDetector(max_hands=self.max_hands)
        self._running = True
        try:
            while self._running:
                ok, frame = capture.read()
                if not ok:
                    # Breve pausa para no saturar la CPU si la cámara no entrega frames.
                    time.sleep(0.005)
                    continue
                if self._pending_frames >= self.max_pending_frames:
                    self.dropped_frames += 1
                    continue
                frame = cv2.flip(frame, 1)
                processed = image_processor.preprocess(frame)
                landmarks = hand_detector.detect(processed) or []
                display = hand_detector.draw_landmarks(processed, landmarks) if self.draw_landmarks and landmarks else processed
                self._pending_frames += 1
                self.frame_ready.emit(display, landmarks)
        finally:
            capture.release()
            hand_detector.cleanup()

    def frame_consumed(self) -> None:
        """Debe llamarse desde el slot de la UI al terminar de procesar un frame."""
        self._pending_frames = max(0, self._pending_frames - 1)

    def stop(self, timeout_ms: int = 2000) -> bool:
        """Solicita detener el bucle y espera a que el hilo termine.

        Args:
            timeout_ms: tiempo máximo de espera antes de continuar sin el hilo.

        Returns:
            True si el hilo terminó dentro del tiempo de espera.
        """
        self._running = False
        if not self.isRunning():
            return True
        if self.wait(timeout_ms):
            return True
        # Puede estar aún abriendo la cámara o cargando MediaPipe (varios
        # segundos). Se concede un margen mayor antes de desistir, porque
        # destruir un QThread en ejecución aborta el proceso.
        return self.wait(15000)
