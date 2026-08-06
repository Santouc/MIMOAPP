"""Pantalla de traducción en vivo del proyecto MIMO.

Este módulo define la pantalla principal de traducción de lenguaje de señas
en tiempo real. Su flujo de trabajo general es el siguiente:

1. Captura de video con OpenCV desde la cámara web (640x480 a ~30 FPS).
2. Preprocesamiento de cada fotograma con ``ImageProcessor``.
3. Detección de los 21 puntos de referencia (landmarks) de la mano mediante
   MediaPipe, encapsulado en ``HandDetector``.
4. Estabilización temporal de los landmarks (suavizado exponencial y
   tolerancia a fotogramas perdidos) para reducir el ruido del detector.
5. Clasificación de la seña con dos modelos de TensorFlow:
   - ``SignClassifier``: señas ESTÁTICAS (posturas fijas de la mano).
   - ``DynamicSignClassifier``: señas DINÁMICAS (secuencias de movimiento).
6. Consenso sobre buffers de predicciones recientes para evitar falsos
   positivos y parpadeos en la salida.
7. Conversión de la seña aceptada en texto mediante el servicio de
   transcripción del contexto de la aplicación (letras -> palabras/frases).
8. Dibujo de un overlay informativo sobre el video (texto transcrito, letra
   actual y estado del tracking) y presentación del fotograma en la interfaz.
9. Notificación de los cambios de transcripción a las extensiones instaladas.

La interfaz gráfica se construye con PySide6 (Qt) y utiliza un ``QTimer``
para procesar fotogramas de forma periódica sin bloquear el hilo de la UI.
"""

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

# --- Constantes de calibración del análisis de movimiento ---
# Umbral de movimiento promedio (en coordenadas normalizadas de MediaPipe)
# por debajo del cual se considera que la mano está "quieta".
DYNAMIC_MOTION_THRESHOLD = 0.012
# Cantidad de fotogramas consecutivos de quietud tras la cual se reinicia
# la secuencia dinámica (la mano quieta no puede formar una seña dinámica).
DYNAMIC_STILL_FRAMES_RESET = 6
# Fotogramas mínimos de quietud requeridos antes de aceptar una predicción
# estática (evita clasificar mientras la mano todavía se está moviendo).
STATIC_MIN_STILL_FRAMES = 6


class TranslateScreen(QWidget):
    """Pantalla de traducción de señas en tiempo real.

    Orquesta todo el pipeline de traducción: captura de cámara, detección
    de landmarks, estabilización, clasificación estática/dinámica, consenso,
    transcripción a texto y presentación visual. También expone controles
    para el usuario (iniciar/detener cámara, borrar/limpiar texto, enseñar
    interpretaciones, cambiar idioma y volver al inicio) y acciones
    aportadas por extensiones.

    Attributes:
        back_requested: Señal Qt emitida cuando el usuario solicita volver
            a la pantalla de inicio.
    """

    # Señal emitida hacia la ventana principal para regresar al menú inicial.
    back_requested = Signal()

    def __init__(self, context: AppContext, parent=None):
        """Inicializa el estado interno de la pantalla y construye la UI.

        Args:
            context: Contexto global de la aplicación con acceso a rutas de
                modelos, servicio de transcripción y gestor de extensiones.
            parent: Widget padre opcional (convención de Qt).
        """
        super().__init__(parent)
        self.context = context
        # Recursos de captura y detección: se crean al iniciar la cámara
        # y se liberan al detenerla (por eso comienzan en None).
        self.capture = None
        self.hand_detector = None
        self.image_processor = ImageProcessor()
        self.static_classifier = None
        self.dynamic_classifier = None
        # Temporizador de Qt que dispara el procesamiento de cada fotograma
        # sin bloquear el hilo de la interfaz gráfica.
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._process_frame)
        # Buffers de consenso:
        # - sign_buffer: últimas predicciones estáticas (seña, confianza).
        # - dynamic_sequence: ventana deslizante de landmarks para el
        #   clasificador dinámico (20 fotogramas = una secuencia completa).
        # - dynamic_buffer: últimas predicciones dinámicas (seña, confianza).
        self.sign_buffer = deque(maxlen=10)
        self.dynamic_sequence = deque(maxlen=20)
        self.dynamic_buffer = deque(maxlen=5)
        # Estado de análisis de movimiento: fotogramas consecutivos con la
        # mano quieta y referencia de landmarks del fotograma anterior.
        self.still_frames = 0
        self._last_motion_reference = None
        # Marca de tiempo de la última seña aceptada por la transcripción,
        # usada para detectar cuándo reiniciar el estado dinámico.
        self._last_seen_accepted_at = 0.0
        # Estado actual de la transcripción y último estado notificado a
        # las extensiones (para evitar notificaciones duplicadas).
        self.transcription_state = None
        self._last_notified_transcription = ("", "")
        # Confianza mínima para que una predicción participe en el consenso.
        self.confidence_threshold = 0.7
        # Estado de la estabilización de landmarks: última detección válida,
        # contador de fotogramas sin detección, tolerancia máxima, factor de
        # suavizado exponencial y etiqueta de estado del tracking.
        self.last_landmarks = None
        self.missed_frames = 0
        self.max_missed_frames = 4
        self.smoothing_alpha = 0.65
        self.tracking_status = "PERDIDO"
        # Variables para el cálculo de FPS (fotogramas por segundo).
        self.last_time = time.time()
        self.frame_count = 0
        self.fps = 0.0
        self._build_ui()

    def stop_camera(self) -> None:
        """Detiene la captura de video y libera los recursos asociados.

        Detiene el temporizador de procesamiento, libera la cámara de
        OpenCV y limpia el detector de manos de MediaPipe. Es seguro
        llamarla aunque la cámara ya esté detenida.
        """
        self.timer.stop()
        # Liberar la cámara de OpenCV si estaba abierta.
        if self.capture is not None:
            self.capture.release()
            self.capture = None
        # Cerrar los recursos internos de MediaPipe.
        if self.hand_detector is not None:
            self.hand_detector.cleanup()
            self.hand_detector = None

    def _build_ui(self) -> None:
        """Construye todos los widgets y layouts de la pantalla.

        Crea el área de video, las etiquetas de resultados (transcripción
        final, letras crudas, estado de transcripción y estado general),
        la fila de botones de control, los botones de extensiones y el
        selector de idioma. Solo se ejecuta una vez, desde el constructor.
        """
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        # Título de la pantalla y área donde se dibuja el video de la cámara.
        title = QLabel("Traducir en vivo")
        title.setObjectName("TitleLabel")
        self.video_label = QLabel("Presiona 'Iniciar traducción' para abrir la cámara")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumHeight(430)
        self.video_label.setObjectName("CameraLabel")

        # Etiquetas de resultados: texto final interpretado, letras crudas
        # detectadas, estado de la transcripción y estado general del sistema.
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

        # Fila de botones de control: cada botón se conecta a su acción
        # correspondiente mediante señales y slots de Qt.
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
        # Contenedor para los botones aportados dinámicamente por las
        # extensiones instaladas en la aplicación.
        self.extension_buttons_layout = QHBoxLayout()
        self.extension_actions = []
        buttons.addLayout(self.extension_buttons_layout)
        self.refresh_extension_actions()
        buttons.addStretch(1)
        # Selector de idioma de la transcripción, poblado con los idiomas
        # disponibles en el servicio de transcripción del contexto.
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

        # Ensamblado final del layout vertical de la pantalla.
        layout.addWidget(title)
        layout.addWidget(self.video_label, 1)
        layout.addWidget(QLabel("Transcripción final:"))
        layout.addWidget(self.translation_label)
        layout.addWidget(self.raw_text_label)
        layout.addWidget(self.transcription_status_label)
        layout.addWidget(self.status_label)
        layout.addLayout(buttons)

    def _start_camera(self) -> None:
        """Inicia la captura de video y arranca el ciclo de traducción.

        Carga los clasificadores, reinicia la transcripción y los buffers,
        abre la cámara con OpenCV (configurada a 640x480 y 30 FPS), crea el
        detector de manos de MediaPipe y pone en marcha el temporizador que
        procesa un fotograma cada 30 ms. Si la cámara no puede abrirse, se
        informa al usuario con un cuadro de diálogo crítico.
        """
        # Preparación previa: modelos, transcripción y buffers en limpio.
        self._load_classifiers()
        self.context.transcription.reset()
        self._reset_buffers()
        # Captura de cámara con OpenCV: dispositivo 0 (cámara por defecto)
        # con resolución y tasa de fotogramas solicitadas.
        self.capture = cv2.VideoCapture(0)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.capture.set(cv2.CAP_PROP_FPS, 30)
        if not self.capture.isOpened():
            self.capture = None
            QMessageBox.critical(self, "Cámara no disponible", "No se pudo abrir la cámara.")
            return
        # Detector de landmarks de MediaPipe limitado a una sola mano y
        # arranque del temporizador de procesamiento (~33 FPS máximo).
        self.hand_detector = HandDetector(max_hands=1)
        self.timer.start(30)
        self.status_label.setText(self._model_status_text())

    def _load_classifiers(self) -> None:
        """Carga (o recarga) los modelos de clasificación desde disco.

        Instancia el clasificador estático (posturas fijas) y el dinámico
        (secuencias de 20 fotogramas) usando las rutas de modelos y
        etiquetas definidas en el contexto de la aplicación. Si algún
        modelo no existe, el clasificador correspondiente queda con
        ``model`` en None y simplemente no se utiliza durante la traducción.
        """
        # Clasificador de señas estáticas (una postura = una seña).
        self.static_classifier = SignClassifier(
            model_path=str(self.context.paths.static_model_path),
            labels_path=str(self.context.paths.static_labels_path),
        )
        # Clasificador de señas dinámicas (secuencia de movimiento = seña).
        self.dynamic_classifier = DynamicSignClassifier(
            model_path=str(self.context.paths.dynamic_model_path),
            labels_path=str(self.context.paths.dynamic_labels_path),
            sequence_length=20,
        )

    def _process_frame(self) -> None:
        """Procesa un fotograma completo del pipeline de traducción.

        Es el corazón de la pantalla: se ejecuta periódicamente desde el
        ``QTimer`` y encadena captura, detección, estabilización,
        clasificación, transcripción y presentación. Si la cámara no está
        activa o el fotograma no pudo leerse, retorna sin hacer nada.
        """
        if self.capture is None:
            return
        # Captura del fotograma desde la cámara con OpenCV.
        ok, frame = self.capture.read()
        if not ok:
            return
        # Espejar horizontalmente para que el video se comporte como un
        # espejo (más natural para el usuario) y preprocesar la imagen.
        frame = cv2.flip(frame, 1)
        processed = self.image_processor.preprocess(frame)
        # Detección de landmarks con MediaPipe y estabilización temporal
        # para reducir el ruido entre fotogramas.
        raw_landmarks = self.hand_detector.detect(processed) if self.hand_detector else None
        landmarks = self._stabilize_landmarks(raw_landmarks)

        # Clasificación: se ejecutan en paralelo la rama estática y la
        # dinámica; cada una decide internamente si emite una predicción.
        static_sign = None
        dynamic_sign = None
        if landmarks is not None:
            processed = self.hand_detector.draw_landmarks(processed, landmarks)
            self._update_motion_state(landmarks)
            static_sign = self._process_static_prediction(landmarks)
            dynamic_sign = self._process_dynamic_prediction(landmarks)

        # Selección de la seña final y envío al servicio de transcripción,
        # que la convierte progresivamente en texto.
        final_sign = self._select_final_sign(static_sign, dynamic_sign)
        self.transcription_state = self.context.transcription.process_sign(final_sign)
        # Si la transcripción aceptó una seña nueva, se reinicia el estado
        # dinámico para no arrastrar la secuencia de la seña anterior.
        accepted_at = self.context.transcription.last_accepted_at
        if accepted_at != self._last_seen_accepted_at:
            self._last_seen_accepted_at = accepted_at
            self._reset_dynamic_state()
        # Actualización de la interfaz: etiquetas de texto, overlay sobre
        # el video y presentación del fotograma en pantalla.
        self._update_results(final_sign)
        self._draw_overlay(processed, final_sign)
        self._show_frame(processed)

    def _update_motion_state(self, landmarks) -> None:
        """Actualiza el contador de fotogramas con la mano quieta.

        Mide el desplazamiento promedio de la mano respecto del fotograma
        anterior y, según supere o no el umbral ``DYNAMIC_MOTION_THRESHOLD``,
        incrementa o reinicia ``still_frames``. Este contador permite
        distinguir entre señas estáticas (mano quieta) y dinámicas
        (mano en movimiento).

        Args:
            landmarks: Lista de manos detectadas; solo se analiza la primera.
        """
        hand_landmarks = landmarks[0] if landmarks else None
        # Solo se analizan manos completas (21 puntos de MediaPipe).
        if hand_landmarks is None or len(hand_landmarks) != 21:
            return
        # Comparar contra el fotograma anterior: si el movimiento es menor
        # al umbral, la mano se considera quieta y se acumula el contador.
        motion = self._measure_motion(hand_landmarks)
        if motion is not None and motion < DYNAMIC_MOTION_THRESHOLD:
            self.still_frames += 1
        else:
            self.still_frames = 0

    def _process_static_prediction(self, landmarks):
        """Ejecuta la rama de clasificación de señas ESTÁTICAS.

        Clasifica los landmarks actuales con el modelo estático y acumula
        la predicción en ``sign_buffer``. Solo devuelve un resultado cuando
        la mano lleva suficientes fotogramas quieta y el buffer tiene un
        mínimo de muestras, en cuyo caso se aplica el consenso.

        Args:
            landmarks: Landmarks estabilizados de la(s) mano(s) detectada(s).

        Returns:
            La seña estática consensuada, o None si aún no hay una
            predicción confiable.
        """
        # Sin modelo estático cargado no hay nada que clasificar.
        if self.static_classifier is None or self.static_classifier.model is None:
            return None
        # Clasificar el fotograma actual y acumular en el buffer de consenso.
        sign, confidence = self.static_classifier.classify(landmarks)
        self.sign_buffer.append((sign, confidence))
        # Requisitos para emitir un resultado: mano quieta el tiempo mínimo
        # y buffer con suficientes muestras para un consenso significativo.
        if self.still_frames < STATIC_MIN_STILL_FRAMES:
            return None
        if len(self.sign_buffer) < 5:
            return None
        return self._get_consensus_sign(self.sign_buffer)

    def _process_dynamic_prediction(self, landmarks):
        """Ejecuta la rama de clasificación de señas DINÁMICAS.

        Acumula los landmarks de la mano en una ventana deslizante
        (``dynamic_sequence``). Cuando la ventana se llena (20 fotogramas),
        clasifica la secuencia completa con el modelo dinámico y acumula
        la predicción en ``dynamic_buffer``. Solo devuelve un resultado
        cuando dicho buffer también se llena, aplicando consenso.

        Args:
            landmarks: Landmarks estabilizados de la(s) mano(s) detectada(s).

        Returns:
            La seña dinámica consensuada, o None si aún no hay una
            predicción confiable.
        """
        # Sin modelo dinámico cargado no hay nada que clasificar.
        if self.dynamic_classifier is None or self.dynamic_classifier.model is None:
            return None
        # Si se perdió la mano, la secuencia deja de tener continuidad y
        # se descarta todo el estado dinámico acumulado.
        if not landmarks or len(landmarks) == 0:
            self._reset_dynamic_state(clear_reference=True)
            return None
        hand_landmarks = landmarks[0]
        if len(hand_landmarks) != 21:
            return None
        # Si la mano lleva demasiado tiempo quieta no puede tratarse de una
        # seña dinámica: se vacían la secuencia y el buffer de predicciones.
        if self.still_frames >= DYNAMIC_STILL_FRAMES_RESET:
            if self.dynamic_sequence or self.dynamic_buffer:
                self.dynamic_sequence.clear()
                self.dynamic_buffer.clear()
            return None
        # Acumular el fotograma y esperar a completar la ventana de 20.
        self.dynamic_sequence.append(hand_landmarks)
        if len(self.dynamic_sequence) < self.dynamic_sequence.maxlen:
            return None
        # Clasificar la secuencia completa y acumular la predicción en el
        # buffer de consenso dinámico.
        sign, confidence = self.dynamic_classifier.classify_sequence(list(self.dynamic_sequence))
        self.dynamic_buffer.append((sign, confidence))
        if len(self.dynamic_buffer) < self.dynamic_buffer.maxlen:
            return None
        return self._get_consensus_sign(self.dynamic_buffer)

    def _measure_motion(self, hand_landmarks) -> float | None:
        """Calcula cuánto se movió la mano respecto del fotograma anterior.

        Compara las coordenadas (x, y) de los 21 puntos actuales contra la
        referencia guardada del fotograma previo y devuelve la distancia
        euclidiana promedio por punto, en coordenadas normalizadas.

        Args:
            hand_landmarks: Los 21 puntos de la mano en el fotograma actual.

        Returns:
            Desplazamiento promedio, o None si no hay referencia previa
            comparable o los datos no son convertibles a arreglo numérico.
        """
        # Convertir a arreglo NumPy quedándose solo con las coordenadas x, y.
        try:
            current = np.asarray(hand_landmarks, dtype=np.float32)[:, :2]
        except (TypeError, ValueError):
            return None
        # Guardar la referencia para el próximo fotograma antes de comparar.
        previous = self._last_motion_reference
        self._last_motion_reference = current
        if previous is None or previous.shape != current.shape:
            return None
        # Distancia euclidiana promedio entre puntos homólogos.
        return float(np.mean(np.linalg.norm(current - previous, axis=1)))

    def _reset_dynamic_state(self, clear_reference: bool = False) -> None:
        """Reinicia el estado de la clasificación dinámica.

        Vacía la secuencia de landmarks y el buffer de predicciones
        dinámicas y reinicia el contador de quietud.

        Args:
            clear_reference: Si es True, también descarta la referencia de
                movimiento (útil cuando la mano se pierde por completo).
        """
        self.dynamic_sequence.clear()
        self.dynamic_buffer.clear()
        self.still_frames = 0
        if clear_reference:
            self._last_motion_reference = None

    def _get_consensus_sign(self, buffer) -> str | None:
        """Obtiene la seña más votada dentro de un buffer de predicciones.

        Filtra las predicciones cuya confianza no supera el umbral y las
        etiquetadas como "unknown", y devuelve la seña más frecuente entre
        las restantes. Este mecanismo de consenso evita que una única
        predicción ruidosa se traduzca en una letra incorrecta.

        Args:
            buffer: Iterable de tuplas (seña, confianza).

        Returns:
            La seña con más votos válidos, o None si ninguna predicción
            supera el filtro.
        """
        # Solo participan del voto las predicciones confiables y conocidas.
        signs = [item[0] for item in buffer if item[1] > self.confidence_threshold and item[0] != "unknown"]
        if not signs:
            return None
        # La seña ganadora es la más frecuente dentro del buffer.
        return Counter(signs).most_common(1)[0][0]

    def _stabilize_landmarks(self, landmarks):
        """Estabiliza los landmarks detectados para reducir el ruido.

        Aplica dos técnicas de estabilización temporal:
        1. Tolerancia a fotogramas perdidos: si MediaPipe no detecta la mano
           durante unos pocos fotogramas, se reutiliza la última detección
           válida (estado "RECUPERANDO") en lugar de perder el tracking.
        2. Suavizado exponencial: cada mano nueva se empareja con la mano
           previa más cercana (comparando la posición de la muñeca) y sus
           coordenadas se mezclan con las anteriores usando el factor
           ``smoothing_alpha``, atenuando el temblor del detector.

        Además actualiza ``tracking_status`` ("OK", "RECUPERANDO" o
        "PERDIDO") para informar al usuario.

        Args:
            landmarks: Detección cruda de MediaPipe (lista de manos) o None.

        Returns:
            Los landmarks estabilizados, o None si el tracking se perdió.
        """
        # Caso sin detección: intentar sostener el tracking con la última
        # detección válida durante un máximo de fotogramas tolerados.
        if landmarks is None or len(landmarks) == 0:
            if self.last_landmarks is not None and self.missed_frames < self.max_missed_frames:
                self.missed_frames += 1
                self.tracking_status = "RECUPERANDO"
                return self.last_landmarks
            # Se agotó la tolerancia: el tracking se declara perdido y se
            # descarta todo el estado dinámico acumulado.
            self.last_landmarks = None
            self.missed_frames = 0
            self.tracking_status = "PERDIDO"
            self._reset_dynamic_state(clear_reference=True)
            return None

        # Primera detección tras una pérdida: no hay referencia previa con
        # la cual suavizar, así que se acepta tal cual.
        if self.last_landmarks is None:
            self.last_landmarks = landmarks
            self.missed_frames = 0
            self.tracking_status = "OK"
            return landmarks

        # Suavizado exponencial: cada mano actual se empareja con la mano
        # previa más cercana y se mezclan sus coordenadas.
        try:
            stabilized = []
            used_previous = set()
            for current_hand in landmarks:
                # Las manos con formato inesperado se dejan sin suavizar.
                if not isinstance(current_hand, list) or len(current_hand) != 21:
                    stabilized.append(current_hand)
                    continue
                current = np.array(current_hand, dtype=np.float32)
                # Buscar la mano previa más cercana comparando la posición
                # de la muñeca (punto 0), sin reutilizar manos ya asignadas.
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
                # Media ponderada: alpha da más peso a la posición actual y
                # (1 - alpha) conserva parte de la posición anterior.
                smoothed = self.smoothing_alpha * current + (1.0 - self.smoothing_alpha) * previous
                stabilized.append(smoothed.tolist())
            self.last_landmarks = stabilized
            self.missed_frames = 0
            self.tracking_status = "OK"
            return stabilized
        except Exception:
            # Ante cualquier error inesperado en el suavizado se prefiere
            # devolver la detección cruda antes que interrumpir el pipeline.
            self.last_landmarks = landmarks
            self.missed_frames = 0
            self.tracking_status = "OK"
            return landmarks

    def _select_final_sign(self, static_sign, dynamic_sign) -> str | None:
        """Selecciona la seña definitiva entre las dos ramas de clasificación.

        La seña dinámica tiene prioridad sobre la estática: si el modelo
        dinámico reconoció un gesto en movimiento, ese resultado prevalece;
        de lo contrario se usa la seña estática (si existe).

        Args:
            static_sign: Resultado de la rama estática, o None.
            dynamic_sign: Resultado de la rama dinámica, o None.

        Returns:
            La seña elegida, o None si ninguna rama produjo resultado.
        """
        return dynamic_sign or static_sign

    def _notify_extensions(self) -> None:
        """Notifica a las extensiones los cambios de la transcripción.

        Compara el estado actual (letras crudas y texto de salida) contra
        el último notificado y, solo si hubo cambios, envía el nuevo estado
        al gestor de extensiones. Así se evita inundar a las extensiones
        con notificaciones repetidas en cada fotograma.
        """
        state = self.transcription_state
        raw_text = state.raw_text if state else ""
        output_text = state.output_text if state else ""
        # Si el texto no cambió desde la última notificación, no se reenvía.
        snapshot = (raw_text, output_text)
        if snapshot == self._last_notified_transcription:
            return
        self._last_notified_transcription = snapshot
        self.context.extensions.notify_transcription(state)

    def _update_results(self, final_sign) -> None:
        """Actualiza las etiquetas de resultados de la interfaz.

        Refleja en pantalla el texto interpretado, las letras crudas, el
        estado de la transcripción y una línea de estado general (modelos
        cargados, estado del tracking y FPS). También dispara la
        notificación a extensiones.

        Args:
            final_sign: Seña seleccionada en el fotograma actual (no se
                muestra directamente aquí, pero forma parte del ciclo).
        """
        state = self.transcription_state
        self._notify_extensions()
        # Valores de respaldo ("-") cuando aún no hay texto transcrito.
        output_text = state.output_text if state and state.output_text else "-"
        raw_text = state.raw_text if state and state.raw_text else "-"
        transcription_status = state.status if state else "Esperando seña estable"
        self.translation_label.setText(output_text)
        self.raw_text_label.setText(f"Letras: {raw_text}")
        self.transcription_status_label.setText(f"Transcripción: {transcription_status}")
        self.status_label.setText(f"{self._model_status_text()} | Tracking: {self.tracking_status} | FPS: {self._calculate_fps():.1f}")

    def _draw_overlay(self, frame, final_sign) -> None:
        """Dibuja el overlay informativo sobre el fotograma de video.

        Superpone tres líneas de texto con OpenCV: el texto transcrito, las
        letras crudas junto con la seña actual, y el estado del tracking
        con un color semafórico (verde = OK, amarillo = recuperando,
        rojo = perdido).

        Args:
            frame: Fotograma BGR sobre el cual se dibuja (se modifica in situ).
            final_sign: Seña reconocida en el fotograma actual, o None.
        """
        # Preparar los textos a superponer con valores de respaldo.
        output_text = self.transcription_state.output_text if self.transcription_state and self.transcription_state.output_text else "-"
        raw_text = self.transcription_state.raw_text if self.transcription_state and self.transcription_state.raw_text else "-"
        translation_text = f"Texto: {output_text}"
        letter_text = f"Letras: {raw_text} | Actual: {final_sign or '-'}"
        # Color del estado de tracking en formato BGR de OpenCV.
        tracking_color = (0, 255, 0) if self.tracking_status == "OK" else (0, 255, 255) if self.tracking_status == "RECUPERANDO" else (0, 0, 255)
        # Dibujar las tres líneas del overlay en la esquina superior izquierda.
        cv2.putText(frame, translation_text, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.putText(frame, letter_text, (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
        cv2.putText(frame, f"Tracking: {self.tracking_status}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.65, tracking_color, 2)

    def _show_frame(self, frame) -> None:
        """Muestra el fotograma procesado en el widget de video de la UI.

        Convierte la imagen del formato BGR de OpenCV al RGB que espera Qt,
        la envuelve en un ``QImage``/``QPixmap`` y la escala al tamaño del
        widget conservando la relación de aspecto.

        Args:
            frame: Fotograma BGR ya procesado y con el overlay dibujado.
        """
        # Conversión de espacio de color BGR (OpenCV) a RGB (Qt).
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channels = rgb.shape
        # Construir la imagen Qt y escalarla al área de video disponible.
        image = QImage(rgb.data, width, height, channels * width, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(image).scaled(
            self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.video_label.setPixmap(pixmap)

    def _calculate_fps(self) -> float:
        """Calcula los fotogramas por segundo (FPS) del pipeline.

        Cuenta los fotogramas procesados y recalcula el promedio cada vez
        que transcurre al menos un segundo.

        Returns:
            El último valor de FPS calculado.
        """
        self.frame_count += 1
        current_time = time.time()
        elapsed = current_time - self.last_time
        # Recalcular el promedio una vez por segundo.
        if elapsed >= 1.0:
            self.fps = self.frame_count / elapsed
            self.last_time = current_time
            self.frame_count = 0
        return self.fps

    def _model_status_text(self) -> str:
        """Genera el texto descriptivo de disponibilidad de los modelos.

        Verifica si el modelo estático y el dinámico están cargados y con
        etiquetas disponibles, y devuelve un mensaje en español acorde a
        la combinación encontrada.

        Returns:
            Cadena de estado lista para mostrar al usuario.
        """
        # Un modelo se considera listo si existe, cargó y tiene etiquetas.
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
        """Restablece todos los buffers y contadores a su estado inicial.

        Vacía los buffers de consenso, reinicia el estado de tracking y de
        FPS, descarta el estado de transcripción y devuelve las etiquetas
        de la interfaz a sus valores por defecto. Se usa al iniciar o
        detener la cámara.
        """
        # Buffers de clasificación y estado de la estabilización.
        self.sign_buffer.clear()
        self.dynamic_sequence.clear()
        self.dynamic_buffer.clear()
        self.last_landmarks = None
        self.missed_frames = 0
        self.tracking_status = "PERDIDO"
        # Reinicio del medidor de FPS.
        self.last_time = time.time()
        self.frame_count = 0
        self.fps = 0.0
        # Estado de transcripción y etiquetas de la interfaz en limpio.
        self.transcription_state = None
        self.translation_label.setText("-")
        self.raw_text_label.setText("Letras: -")
        self.transcription_status_label.setText("Transcripción: esperando")

    def _backspace_transcription(self) -> None:
        """Elimina la última letra transcrita y refresca las etiquetas."""
        self.transcription_state = self.context.transcription.backspace()
        self._update_transcription_labels()

    def _clear_transcription(self) -> None:
        """Borra todo el texto transcrito y refresca las etiquetas."""
        self.transcription_state = self.context.transcription.clear()
        self._update_transcription_labels()

    def _teach_interpretation(self) -> None:
        """Permite al usuario enseñar una interpretación personalizada.

        Abre un diálogo de corrección con las letras crudas y el texto
        actual; si el usuario acepta, el servicio de transcripción aprende
        la asociación (letras -> frase corregida) para futuras
        transcripciones similares. Requiere que ya exista texto transcrito.
        """
        raw_text = self.context.transcription.get_raw_text()
        if not raw_text:
            QMessageBox.information(self, "Sin letras", "Primero transcribe algunas letras para enseñar una interpretación.")
            return
        # Mostrar el diálogo de corrección y esperar la decisión del usuario.
        current_text = self.context.transcription.get_output_text()
        dialog = TranscriptionCorrectionDialog(raw_text, current_text, self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        # Registrar la asociación aprendida en el servicio de transcripción.
        corrected_raw, corrected_text = dialog.values()
        self.transcription_state = self.context.transcription.learn_phrase(corrected_raw, corrected_text)
        self._update_transcription_labels()
        QMessageBox.information(self, "Interpretación aprendida", "El sistema recordará esta corrección para futuras transcripciones similares.")

    def _change_language(self) -> None:
        """Cambia el idioma de la transcripción según el combo de idiomas.

        Obtiene el código del idioma seleccionado, se lo comunica al
        servicio de transcripción y, si el cambio fue exitoso, refresca las
        etiquetas y notifica al usuario en la línea de estado.
        """
        code = self.language_combo.currentData()
        if not code or not self.context.transcription.set_language(code):
            return
        self.transcription_state = self.context.transcription.process_sign(None)
        self._update_transcription_labels()
        self.transcription_status_label.setText(f"Transcripción: idioma cambiado a {self.language_combo.currentText()}")

    def refresh_extension_actions(self) -> None:
        """Reconstruye los botones aportados por las extensiones.

        Elimina los botones de extensiones existentes y crea uno nuevo por
        cada acción que las extensiones instaladas exponen para la pantalla
        de traducción. Se invoca al construir la UI y cuando cambia el
        conjunto de extensiones activas.
        """
        # Eliminar los botones previos del layout de extensiones.
        while self.extension_buttons_layout.count():
            item = self.extension_buttons_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        # Crear un botón por cada acción disponible de las extensiones.
        self.extension_actions = self.context.extensions.translate_actions(self)
        for action in self.extension_actions:
            extension_button = QPushButton(action.label)
            extension_button.clicked.connect(action.callback)
            self.extension_buttons_layout.addWidget(extension_button)

    def _update_transcription_labels(self) -> None:
        """Refresca las etiquetas de transcripción y notifica extensiones.

        Variante ligera de ``_update_results`` para acciones que modifican
        el texto fuera del ciclo de video (borrar, limpiar, enseñar,
        cambiar idioma): actualiza solo las etiquetas relacionadas con la
        transcripción.
        """
        state = self.transcription_state
        self._notify_extensions()
        self.translation_label.setText(state.output_text if state and state.output_text else "-")
        self.raw_text_label.setText(f"Letras: {state.raw_text if state and state.raw_text else '-'}")
        self.transcription_status_label.setText(f"Transcripción: {state.status if state else 'esperando'}")

    def _stop_and_reset(self) -> None:
        """Detiene la cámara y deja la pantalla en su estado inicial.

        Libera los recursos de captura, reinicia la transcripción y los
        buffers, y actualiza las etiquetas para reflejar que la cámara
        está detenida.
        """
        self.stop_camera()
        self.context.transcription.reset()
        self._reset_buffers()
        self.video_label.setText("Cámara detenida")
        self.status_label.setText("Estado: cámara detenida")

    def keyPressEvent(self, event) -> None:
        """Gestiona los atajos de teclado de la pantalla.

        La tecla C abre el diálogo de enseñanza de interpretaciones.
        Además, cada acción de extensión puede definir su propia tecla de
        atajo; si coincide con la tecla presionada, se ejecuta su callback.
        Cualquier otra tecla se delega al comportamiento estándar de Qt.

        Args:
            event: Evento de teclado entregado por Qt.
        """
        # Atajo fijo: C abre el diálogo de "Enseñar interpretación".
        if event.key() == Qt.Key_C:
            self._teach_interpretation()
            return
        # Atajos dinámicos declarados por las extensiones instaladas.
        for action in self.extension_actions:
            if action.key and event.key() == getattr(Qt, f"Key_{action.key.upper()}", None):
                action.callback()
                return
        super().keyPressEvent(event)

    def _go_back(self) -> None:
        """Detiene la cámara y emite la señal para volver al inicio."""
        self._stop_and_reset()
        self.back_requested.emit()
