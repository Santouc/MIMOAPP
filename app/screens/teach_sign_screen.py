"""Pantalla de enseñanza de señas del proyecto MIMO.

Este módulo define la pantalla "Enseñar seña", encargada de todo el flujo con
el que un usuario agrega nuevas muestras de una seña al dataset del sistema:

1. Muestra una página de advertencias con recomendaciones de captura
   (iluminación, fondo, cantidad de muestras, etc.).
2. Permite configurar la sesión: seleccionar la seña a enseñar y el tipo de
   captura (estática, un solo cuadro; o dinámica, una secuencia de cuadros).
3. Abre la cámara web y, mediante MediaPipe (a través de ``HandDetector``),
   detecta los 21 puntos de referencia (landmarks) de la mano en tiempo real.
4. Captura muestras: en modo estático cada pulsación de la tecla T guarda un
   cuadro normalizado; en modo dinámico la tecla T inicia y detiene la
   grabación de una secuencia completa, que luego se remuestrea a una
   longitud fija de 20 cuadros.
5. Al terminar la sesión, las muestras se guardan como "capturas pendientes"
   y se presenta un resumen con una vista previa del promedio de landmarks.
6. El usuario decide aceptar (integrar al dataset oficial y reentrenar el
   modelo automáticamente) o rechazar (descartar) la sesión pendiente.

Adicionalmente ofrece un modo de "vista previa general" que abre la cámara
sin guardar muestras, útil para probar la detección y registrar correcciones
de interpretación de transcripciones.
"""

import cv2
import numpy as np
from PySide6.QtCore import QTimer, Qt, Signal
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
from app.widgets import LandmarkPreview, TranscriptionCorrectionDialog
from core.hand_detector import HandDetector
from core.image_processor import ImageProcessor
from core.preprocessing import normalize_single_hand


class TeachSignScreen(QWidget):
    """Pantalla completa para enseñar (capturar) muestras de una seña.

    Organiza el flujo de enseñanza en cuatro páginas dentro de un
    ``QStackedWidget``:

    - Página de advertencias: recomendaciones previas a la captura.
    - Página de configuración: elección de la seña y del tipo de captura.
    - Página de cámara: video en vivo con detección de landmarks y controles
      por teclado (T captura/graba, C corrige interpretación, Q termina).
    - Página de resumen: vista previa del promedio de las muestras capturadas
      y botones para aceptar o rechazar la sesión pendiente.

    Señales:
        back_requested: se emite cuando el usuario pide volver al inicio.
    """

    # Señal emitida hacia la ventana principal para regresar al menú inicial.
    back_requested = Signal()

    def __init__(self, context: AppContext, parent=None):
        """Inicializa la pantalla y todo el estado de la sesión de captura.

        Parámetros:
            context: contexto global de la aplicación (acceso a señas,
                capturas pendientes, entrenamiento y transcripción).
            parent: widget padre opcional, siguiendo la convención de Qt.
        """
        super().__init__(parent)
        self.context = context
        # Recursos de video: la cámara (cv2.VideoCapture) y el detector de
        # manos de MediaPipe se crean solo cuando se inicia una sesión.
        self.capture = None
        self.hand_detector = None
        self.image_processor = ImageProcessor()
        # Temporizador que dispara el procesamiento de un cuadro de cámara
        # aproximadamente cada 30 ms (~30 FPS).
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._process_frame)
        # Landmarks de la mano detectados en el cuadro actual (o predichos).
        self.current_landmarks = None
        # Muestras acumuladas en la sesión: cuadros normalizados (estáticas)
        # o secuencias remuestreadas (dinámicas).
        self.samples = []
        # Buffer temporal de cuadros mientras se graba una secuencia dinámica.
        self.frame_buffer = []
        self.is_dynamic_recording = False
        self.last_dynamic_discard_reason = ""
        # Estado del "modo predictivo": si la detección falla por pocos
        # cuadros, se reutilizan los últimos landmarks válidos para evitar
        # cortes bruscos (hasta max_missed_landmark_frames cuadros).
        self.last_valid_landmarks = None
        self.missed_landmark_frames = 0
        self.max_missed_landmark_frames = 12
        # Datos de la sesión pendiente creada al finalizar la captura,
        # a la espera de que el usuario la acepte o rechace.
        self.pending_summary = None
        self.pending_session_id = None
        # Indica si la cámara está en vista previa general (sin guardar).
        self.preview_mode = False
        # Construye la interfaz y carga la lista de señas disponibles.
        self._build_ui()
        self.refresh_signs()

    def refresh_signs(self) -> None:
        """Recarga el combo de señas desde el registro de la aplicación.

        Habilita el botón de captura solo si existe al menos una seña
        registrada y actualiza el mensaje de estado para guiar al usuario.
        """
        # Vuelve a poblar el combo con las señas registradas actualmente.
        self.sign_combo.clear()
        for sign in self.context.signs.list_signs():
            self.sign_combo.addItem(sign["name"], sign["id"])
        # Sin señas registradas no tiene sentido capturar, pero la vista
        # previa general siempre está disponible.
        self.start_button.setEnabled(self.sign_combo.count() > 0)
        self.preview_button.setEnabled(True)
        if self.sign_combo.count() == 0:
            self.status_label.setText("Puedes usar la vista previa general. Para capturar muestras, primero agrega una seña en Gestionar señas.")
        else:
            self.status_label.setText("Selecciona una seña para capturar muestras o abre la vista previa general para probar correcciones.")

    def stop_camera(self) -> None:
        """Detiene el temporizador y libera la cámara y el detector de manos.

        Es seguro llamarla varias veces: solo libera los recursos que sigan
        activos. Se usa al terminar/cancelar una sesión o al salir de la
        pantalla.
        """
        # Detiene el bucle de procesamiento de cuadros.
        self.timer.stop()
        # Libera la cámara física para que otras aplicaciones puedan usarla.
        if self.capture is not None:
            self.capture.release()
            self.capture = None
        # Libera los recursos internos de MediaPipe.
        if self.hand_detector is not None:
            self.hand_detector.cleanup()
            self.hand_detector = None

    def _build_ui(self) -> None:
        """Construye la estructura general de la pantalla.

        Crea un ``QStackedWidget`` con las cuatro páginas del flujo de
        enseñanza (advertencias, configuración, cámara y resumen) y las
        registra en orden.
        """
        root = QVBoxLayout(self)
        self.pages = QStackedWidget()
        root.addWidget(self.pages)

        # Cada página se construye en su propio método auxiliar.
        self.warning_page = self._build_warning_page()
        self.setup_page = self._build_setup_page()
        self.camera_page = self._build_camera_page()
        self.summary_page = self._build_summary_page()

        self.pages.addWidget(self.warning_page)
        self.pages.addWidget(self.setup_page)
        self.pages.addWidget(self.camera_page)
        self.pages.addWidget(self.summary_page)

    def _build_warning_page(self) -> QWidget:
        """Crea la página inicial de recomendaciones antes de capturar.

        Retorna:
            QWidget con el texto de buenas prácticas y botones para
            continuar hacia la configuración o volver al inicio.
        """
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setSpacing(14)
        # Título y cuerpo con las recomendaciones de captura.
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
        # Botones de navegación: continuar al paso de configuración o salir.
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
        """Crea la página de configuración de la sesión de enseñanza.

        Aquí el usuario elige la seña a la que se asociarán las muestras y
        el tipo de captura: "static" (un cuadro por muestra) o "dynamic"
        (una secuencia de cuadros por muestra).

        Retorna:
            QWidget con los combos de seña/tipo y los botones de acción.
        """
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setSpacing(12)
        title = QLabel("Configurar enseñanza")
        title.setObjectName("TitleLabel")
        # Selección de seña y de tipo de captura; el dato interno del combo
        # ("static"/"dynamic") define el flujo de captura posterior.
        self.sign_combo = QComboBox()
        self.type_combo = QComboBox()
        self.type_combo.addItem("Estática", "static")
        self.type_combo.addItem("Dinámica", "dynamic")
        self.status_label = QLabel()
        self.status_label.setObjectName("BodyLabel")
        # Botones principales: iniciar captura, vista previa sin guardar,
        # refrescar la lista de señas y volver al inicio.
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
        """Crea la página de cámara en vivo con sus controles de captura.

        Muestra el video con los landmarks dibujados, una etiqueta de estado
        con las instrucciones/conteo de muestras y botones equivalentes a
        los atajos de teclado (T, C, Q) más un botón de cancelar.

        Retorna:
            QWidget con la vista de cámara y la fila de botones de control.
        """
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setSpacing(10)
        self.camera_title = QLabel("Cámara")
        self.camera_title.setObjectName("TitleLabel")
        # Etiqueta donde se pinta cada cuadro de video como QPixmap.
        self.camera_label = QLabel("Cámara no iniciada")
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setMinimumHeight(420)
        self.camera_label.setObjectName("CameraLabel")
        # Etiqueta de estado: instrucciones y contadores de muestras/frames.
        self.capture_status_label = QLabel("Controles: T captura/graba. C corrige interpretación. Q termina la sesión.")
        self.capture_status_label.setObjectName("BodyLabel")
        # Botones espejo de los atajos de teclado disponibles en esta página.
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
        """Crea la página de resumen de la sesión pendiente.

        Presenta los datos de la sesión (seña, tipo, cantidad de muestras),
        una vista previa del promedio de landmarks y los botones para
        aceptar (integrar al dataset y reentrenar) o rechazar la captura.

        Retorna:
            QWidget con el resumen, la vista previa y los botones de decisión.
        """
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setSpacing(12)
        title = QLabel("Resumen de capturas")
        title.setObjectName("TitleLabel")
        self.summary_label = QLabel()
        self.summary_label.setObjectName("BodyLabel")
        self.summary_label.setWordWrap(True)
        # Widget que dibuja el esqueleto promedio de la mano capturada
        # (imagen fija para estáticas o animación para dinámicas).
        self.preview = LandmarkPreview()
        # Botones de decisión final sobre la sesión pendiente.
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
        """Inicia una sesión de captura de muestras para la seña elegida.

        Valida que haya una seña seleccionada, reinicia todo el estado de la
        sesión, abre la cámara con resolución 640x480 a 30 FPS, crea el
        detector de manos (una sola mano) y arranca el bucle de cuadros.
        Si la cámara no puede abrirse, informa el error y aborta.
        """
        self.preview_mode = False
        # Sin seña seleccionada no hay a qué asociar las muestras.
        sign_id = self.sign_combo.currentData()
        if not sign_id:
            QMessageBox.warning(self, "Sin seña", "Primero selecciona una seña.")
            return
        # Reinicia por completo el estado de la sesión anterior.
        self.samples = []
        self.frame_buffer = []
        self.is_dynamic_recording = False
        self.last_dynamic_discard_reason = ""
        self.last_valid_landmarks = None
        self.missed_landmark_frames = 0
        self.current_landmarks = None
        self.pending_summary = None
        self.pending_session_id = None
        # Abre la cámara predeterminada y fija resolución y FPS deseados.
        self.capture = cv2.VideoCapture(0)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.capture.set(cv2.CAP_PROP_FPS, 30)
        if not self.capture.isOpened():
            self.capture = None
            QMessageBox.critical(self, "Cámara no disponible", "No se pudo abrir la cámara.")
            return
        # Detector de MediaPipe limitado a una sola mano por consistencia.
        self.hand_detector = HandDetector(max_hands=1)
        # Cambia a la página de cámara y arranca el bucle (~30 FPS).
        self.pages.setCurrentWidget(self.camera_page)
        self.timer.start(30)
        self._update_capture_status()

    def _start_preview_camera(self) -> None:
        """Abre la cámara en modo vista previa general (sin guardar muestras).

        Sigue el mismo procedimiento que ``_start_camera`` pero con
        ``preview_mode`` activo: se puede observar la detección de landmarks
        y usar la corrección de interpretación, sin acumular capturas.
        """
        self.preview_mode = True
        # Limpia cualquier estado remanente de sesiones anteriores.
        self.samples = []
        self.frame_buffer = []
        self.is_dynamic_recording = False
        self.last_dynamic_discard_reason = ""
        self.last_valid_landmarks = None
        self.missed_landmark_frames = 0
        self.current_landmarks = None
        self.pending_summary = None
        self.pending_session_id = None
        # Abre la cámara con la misma configuración que la captura normal.
        self.capture = cv2.VideoCapture(0)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.capture.set(cv2.CAP_PROP_FPS, 30)
        if not self.capture.isOpened():
            self.capture = None
            QMessageBox.critical(self, "Cámara no disponible", "No se pudo abrir la cámara.")
            return
        self.hand_detector = HandDetector(max_hands=1)
        # Ajusta el título para dejar claro que no se están guardando datos.
        self.camera_title.setText("Vista previa general")
        self.pages.setCurrentWidget(self.camera_page)
        self.timer.start(30)
        self._update_capture_status()

    def _process_frame(self) -> None:
        """Procesa un cuadro de la cámara en cada tick del temporizador.

        Pasos del pipeline por cuadro:
        1. Lee el cuadro y lo refleja horizontalmente (efecto espejo).
        2. Lo preprocesa con ``ImageProcessor`` (mejora para la detección).
        3. Detecta landmarks de la mano y aplica el suavizado predictivo.
        4. Dibuja los landmarks sobre la imagen a mostrar.
        5. Si hay una grabación dinámica activa, agrega el cuadro al buffer.
        6. Muestra la imagen resultante en la interfaz.
        """
        # Si la cámara fue liberada o el cuadro falla, no hay nada que hacer.
        if self.capture is None:
            return
        ok, frame = self.capture.read()
        if not ok:
            return
        # Espejo horizontal para que el movimiento se sienta natural.
        frame = cv2.flip(frame, 1)
        # Preprocesado de imagen y detección de landmarks de la mano.
        processed = self.image_processor.preprocess(frame)
        landmarks = self.hand_detector.detect(processed) if self.hand_detector else None
        # Suavizado predictivo: tolera pérdidas breves de detección.
        self.current_landmarks = self._get_predictive_landmarks(landmarks)
        # Dibuja el esqueleto de la mano sobre la copia que se mostrará.
        display = processed.copy()
        if landmarks:
            display = self.hand_detector.draw_landmarks(display, landmarks)
        # Durante una grabación dinámica, cada cuadro se acumula en el buffer.
        if self.type_combo.currentData() == "dynamic" and self.is_dynamic_recording:
            self._record_dynamic_frame()
        self._show_frame(display)

    def _show_frame(self, frame) -> None:
        """Convierte un cuadro BGR de OpenCV y lo pinta en la interfaz.

        Parámetros:
            frame: imagen BGR (numpy) tal como la entrega OpenCV.
        """
        # OpenCV entrega BGR; Qt espera RGB para construir la QImage.
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channels = rgb.shape
        image = QImage(rgb.data, width, height, channels * width, QImage.Format_RGB888)
        # Escala la imagen al tamaño de la etiqueta manteniendo proporción.
        pixmap = QPixmap.fromImage(image).scaled(
            self.camera_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.camera_label.setPixmap(pixmap)

    def _capture_current(self) -> None:
        """Maneja la tecla T: captura una muestra o alterna la grabación.

        Comportamiento según el contexto:
        - En vista previa general no se guarda nada; solo se recuerda al
          usuario los controles disponibles.
        - En modo estático: normaliza los landmarks actuales y los agrega
          como una nueva muestra.
        - En modo dinámico: inicia la grabación de una secuencia si estaba
          detenida, o la detiene y guarda la secuencia si estaba activa.
        """
        # En vista previa no se capturan muestras.
        if self.preview_mode:
            self.capture_status_label.setText("Vista previa general. Usa C para corregir interpretaciones o Q para volver.")
            return
        capture_type = self.type_combo.currentData()
        # Sin mano detectada no hay landmarks que capturar.
        if self.current_landmarks is None:
            self.capture_status_label.setText("No hay mano detectada. Ajusta iluminación/posición e intenta de nuevo.")
            return
        if capture_type == "static":
            # Muestra estática: un solo cuadro de landmarks normalizados.
            normalized = normalize_single_hand(self.current_landmarks)[0].tolist()
            self.samples.append(normalized)
        else:
            # Muestra dinámica: T alterna entre iniciar y detener la secuencia.
            if self.is_dynamic_recording:
                self._stop_dynamic_recording()
            else:
                self._start_dynamic_recording()
        self._update_capture_status()

    def _finish_session(self) -> None:
        """Maneja la tecla Q: cierra la sesión y crea la captura pendiente.

        En vista previa simplemente cierra la cámara. En una sesión real:
        detiene una posible grabación dinámica en curso, valida que existan
        muestras, libera la cámara, registra la sesión como pendiente en el
        almacenamiento y navega a la página de resumen para su revisión.
        """
        # La vista previa no genera capturas: solo se cierra la cámara.
        if self.preview_mode:
            self._cancel_camera()
            return
        capture_type = self.type_combo.currentData()
        # Si quedó una grabación dinámica abierta, se cierra primero.
        if capture_type == "dynamic" and self.is_dynamic_recording:
            self._stop_dynamic_recording()
        # Sin muestras no hay nada que revisar; se explica el motivo si la
        # última secuencia dinámica fue descartada.
        if not self.samples:
            detail = f"\n\n{self.last_dynamic_discard_reason}" if self.last_dynamic_discard_reason else ""
            QMessageBox.information(self, "Sin muestras", f"No capturaste muestras para revisar.{detail}")
            return
        # Libera la cámara antes de pasar a la etapa de revisión.
        self.stop_camera()
        sign_id = self.sign_combo.currentData()
        # Guarda las muestras como sesión pendiente (aún no forman parte
        # del dataset oficial hasta que el usuario las acepte).
        try:
            self.pending_summary = self.context.captures.create_pending_session(sign_id, capture_type, self.samples)
        except Exception as error:
            QMessageBox.critical(self, "Error guardando capturas", str(error))
            self.pages.setCurrentWidget(self.setup_page)
            return
        self.pending_session_id = self.pending_summary["session_id"]
        self._show_summary()

    def _show_summary(self) -> None:
        """Muestra la página de resumen con la vista previa del promedio.

        Configura la vista previa según el tipo de captura (secuencia
        animada para dinámicas, pose fija para estáticas) y arma el texto
        con seña, tipo y cantidad de muestras de la sesión pendiente.
        """
        average = self.pending_summary["average_landmarks"]
        capture_type = self.pending_summary["capture_type"]
        # La vista previa se alimenta distinto según el tipo de captura.
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
        """Acepta la sesión pendiente e integra las muestras al dataset.

        Tras aceptar, lanza automáticamente el reentrenamiento del modelo
        correspondiente al tipo de captura (con diálogo de progreso) y
        muestra el resultado del entrenamiento antes de volver a la
        configuración.
        """
        if not self.pending_session_id:
            return
        capture_type = self.pending_summary["capture_type"]
        # Mueve las muestras pendientes al dataset oficial.
        try:
            self.context.captures.accept_pending_session(self.pending_session_id)
        except Exception as error:
            QMessageBox.critical(self, "Error al aceptar", str(error))
            return
        # Reentrena el modelo con los nuevos datos e informa el resultado.
        training_result = self._train_with_progress(capture_type)
        QMessageBox.information(self, "Capturas aceptadas", self._format_training_message(training_result))
        self._reset_to_setup()

    def _reject_pending(self) -> None:
        """Rechaza y elimina la sesión pendiente sin tocar el dataset."""
        if not self.pending_session_id:
            return
        # Elimina la sesión pendiente del almacenamiento.
        try:
            self.context.captures.reject_pending_session(self.pending_session_id)
        except Exception as error:
            QMessageBox.critical(self, "Error al rechazar", str(error))
            return
        QMessageBox.information(self, "Capturas rechazadas", "La sesión pendiente fue descartada.")
        self._reset_to_setup()

    def _cancel_camera(self) -> None:
        """Cancela la sesión de cámara actual y regresa a la configuración.

        Libera la cámara, descarta las muestras acumuladas y restaura el
        título de la página de cámara.
        """
        self.stop_camera()
        # Descarta el estado de captura acumulado en esta sesión.
        self.samples = []
        self.frame_buffer = []
        self.is_dynamic_recording = False
        self.last_dynamic_discard_reason = ""
        self.preview_mode = False
        self.camera_title.setText("Cámara")
        self.pages.setCurrentWidget(self.setup_page)

    def _reset_to_setup(self) -> None:
        """Restablece todo el estado de la pantalla y vuelve a configuración.

        Se llama después de aceptar o rechazar una sesión pendiente: limpia
        muestras, buffers, sesión pendiente y vista previa, refresca la
        lista de señas y muestra la página de configuración.
        """
        # Limpia todo el estado de la sesión anterior.
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
        """Actualiza la etiqueta de estado de la página de cámara.

        El mensaje depende del contexto: vista previa (sin capturas),
        modo estático (contador de muestras) o modo dinámico (estado de la
        grabación, secuencias guardadas y frames del buffer actual).
        """
        if self.preview_mode:
            self.capture_status_label.setText("Vista previa general. No se guardan muestras. Usa C para corregir interpretación y Q para volver.")
            return
        capture_type = self.type_combo.currentData()
        if capture_type == "static":
            # Modo estático: se informa cuántas muestras se llevan capturadas.
            self.capture_status_label.setText(
                f"Modo estático. Presiona T para capturar una muestra. Muestras: {len(self.samples)}"
            )
        else:
            # Modo dinámico: se muestra si está grabando y los contadores.
            recording_text = "GRABANDO" if self.is_dynamic_recording else "detenida"
            self.capture_status_label.setText(
                f"Modo dinámico ({recording_text}). T inicia/detiene una secuencia completa. "
                f"Secuencias: {len(self.samples)} | Frames actuales: {len(self.frame_buffer)}"
            )

    def _start_dynamic_recording(self) -> None:
        """Inicia la grabación de una nueva secuencia dinámica.

        Vacía el buffer de cuadros, limpia el motivo de descarte anterior,
        activa la bandera de grabación e intenta registrar de inmediato el
        primer cuadro con la mano actual.
        """
        self.frame_buffer = []
        self.last_dynamic_discard_reason = ""
        self.is_dynamic_recording = True
        # Registra el primer cuadro sin esperar al siguiente tick del timer.
        self._record_dynamic_frame()

    def _stop_dynamic_recording(self) -> None:
        """Detiene la grabación dinámica y guarda la secuencia si es válida.

        Las secuencias con menos de 5 cuadros se descartan por ser demasiado
        cortas para representar un movimiento. Las válidas se remuestrean a
        una longitud fija de 20 cuadros y se agregan a las muestras.
        """
        self.is_dynamic_recording = False
        # Secuencia demasiado corta: se descarta e informa el motivo.
        if len(self.frame_buffer) < 5:
            self.frame_buffer = []
            self.last_dynamic_discard_reason = "La última secuencia dinámica fue descartada porque duró menos de 5 frames."
            self.capture_status_label.setText(self.last_dynamic_discard_reason)
            return
        # Normaliza la duración a 20 cuadros y guarda la secuencia.
        self.samples.append(self._resample_sequence(self.frame_buffer, 20))
        self.frame_buffer = []

    def _record_dynamic_frame(self) -> None:
        """Agrega el cuadro actual de landmarks al buffer de la secuencia.

        Se llama en cada tick del temporizador mientras la grabación
        dinámica está activa. Si no hay mano detectada, el cuadro se omite.
        """
        if self.current_landmarks is None:
            return
        # Normaliza los landmarks (independiente de posición/escala) y los
        # acumula en la secuencia en curso.
        normalized = normalize_single_hand(self.current_landmarks)[0].tolist()
        self.frame_buffer.append(normalized)
        self._update_capture_status()

    def _get_predictive_landmarks(self, landmarks):
        """Aplica un suavizado predictivo sobre la detección de la mano.

        Si el detector pierde la mano por unos pocos cuadros (hasta
        ``max_missed_landmark_frames``), devuelve los últimos landmarks
        válidos en lugar de ``None`` para evitar cortes en la grabación.

        Parámetros:
            landmarks: resultado crudo del detector (lista de manos o None).

        Retorna:
            Lista de 21 landmarks de la primera mano, o None si la pérdida
            de detección superó la tolerancia.
        """
        current = self._extract_first_hand(landmarks)
        # Detección exitosa: se actualiza el estado y se reinicia el contador.
        if current is not None:
            self.last_valid_landmarks = current
            self.missed_landmark_frames = 0
            return current
        # Pérdida breve: se reutilizan los últimos landmarks válidos.
        if self.last_valid_landmarks is not None and self.missed_landmark_frames < self.max_missed_landmark_frames:
            self.missed_landmark_frames += 1
            return self.last_valid_landmarks
        # Pérdida prolongada: se considera que ya no hay mano en escena.
        self.last_valid_landmarks = None
        return None

    def _extract_first_hand(self, landmarks):
        """Extrae la primera mano válida del resultado del detector.

        Parámetros:
            landmarks: estructura devuelta por ``HandDetector.detect``.

        Retorna:
            La lista de 21 landmarks de la primera mano, o None si el
            resultado está vacío o no tiene el formato esperado.
        """
        if not landmarks or not isinstance(landmarks, list):
            return None
        first = landmarks[0]
        # Una mano válida debe tener exactamente 21 puntos de referencia.
        if isinstance(first, list) and len(first) == 21:
            return first
        return None

    def _pad_or_truncate(self, sequence: list, target_length: int) -> list:
        """Ajusta una secuencia a una longitud fija por corte o repetición.

        Si la secuencia es más larga que ``target_length`` se trunca; si es
        más corta se rellena repitiendo el último elemento.

        Parámetros:
            sequence: lista de cuadros de landmarks.
            target_length: longitud objetivo de la secuencia.

        Retorna:
            Lista con exactamente ``target_length`` elementos (o vacía si la
            secuencia original estaba vacía).
        """
        if len(sequence) > target_length:
            return sequence[:target_length]
        if len(sequence) < target_length:
            if not sequence:
                return []
            # Rellena repitiendo el último cuadro hasta alcanzar el objetivo.
            sequence = sequence + [sequence[-1]] * (target_length - len(sequence))
        return sequence

    def _resample_sequence(self, sequence: list, target_length: int) -> list:
        """Remuestrea una secuencia de landmarks a una longitud fija.

        Usa interpolación lineal independiente para cada coordenada (x, y, z)
        de cada uno de los 21 puntos de la mano, de modo que secuencias de
        distinta duración queden normalizadas temporalmente sin perder la
        forma del movimiento.

        Parámetros:
            sequence: lista de cuadros, cada uno con 21 puntos (x, y, z).
            target_length: cantidad de cuadros deseada (20 en este flujo).

        Retorna:
            Lista de ``target_length`` cuadros interpolados.
        """
        array = np.asarray(sequence, dtype=np.float32)
        # Casos triviales: longitud exacta o un solo cuadro que se repite.
        if len(array) == target_length:
            return array.tolist()
        if len(array) == 1:
            return np.repeat(array, target_length, axis=0).tolist()
        # Posiciones temporales de origen y destino para la interpolación.
        source_positions = np.linspace(0, len(array) - 1, num=len(array))
        target_positions = np.linspace(0, len(array) - 1, num=target_length)
        resampled = np.empty((target_length, 21, 3), dtype=np.float32)
        # Interpola cada coordenada de cada punto a lo largo del tiempo.
        for point_index in range(21):
            for coord_index in range(3):
                resampled[:, point_index, coord_index] = np.interp(
                    target_positions,
                    source_positions,
                    array[:, point_index, coord_index],
                )
        return resampled.tolist()

    def _format_training_message(self, training_result) -> str:
        """Arma el mensaje informativo con el resultado del entrenamiento.

        Parámetros:
            training_result: objeto con los datos del entrenamiento
                (mensaje, bandera de éxito, muestras, clases y accuracy).

        Retorna:
            Texto listo para mostrar en un cuadro de diálogo.
        """
        message = "Las muestras fueron integradas al dataset oficial.\n\n"
        message += training_result.message
        # Si el modelo llegó a entrenarse, se agregan las métricas obtenidas.
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
        """Maneja la tecla C: registra una corrección de interpretación.

        Abre un diálogo donde el usuario indica cómo debe interpretarse un
        texto transcrito; si lo confirma, la corrección se guarda para que
        el sistema la aplique durante la traducción en vivo.
        """
        dialog = TranscriptionCorrectionDialog(parent=self)
        # Si el usuario cancela el diálogo, no se aprende nada.
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        # Guarda la asociación texto crudo -> interpretación deseada.
        raw_text, interpreted_text = dialog.values()
        self.context.transcription.learn_phrase(raw_text, interpreted_text)
        QMessageBox.information(self, "Interpretación aprendida", "El sistema recordará esta corrección durante la traducción en vivo.")

    def _train_with_progress(self, capture_type: str):
        """Ejecuta el entrenamiento del modelo mostrando un diálogo de espera.

        Muestra un ``QProgressDialog`` modal e indeterminado (sin botón de
        cancelar) mientras se entrena el modelo del tipo indicado, y lo
        cierra siempre al terminar, incluso ante errores.

        Parámetros:
            capture_type: "static" o "dynamic", modelo a reentrenar.

        Retorna:
            El resultado del entrenamiento devuelto por el servicio.
        """
        # Diálogo modal indeterminado que bloquea la interfaz mientras dura
        # el entrenamiento.
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
        # El "finally" garantiza que el diálogo se cierre aunque el
        # entrenamiento lance una excepción.
        try:
            return self.context.training.train(capture_type)
        finally:
            progress.close()
            QApplication.processEvents()

    def keyPressEvent(self, event) -> None:
        """Gestiona los atajos de teclado de la página de cámara.

        Solo actúa cuando la página visible es la de cámara:
        - T: captura una muestra o alterna la grabación dinámica.
        - C: abre el diálogo de corrección de interpretación.
        - Q: termina la sesión de captura.
        Cualquier otra tecla se delega al comportamiento estándar de Qt.
        """
        # Los atajos solo aplican mientras la cámara está en pantalla.
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
