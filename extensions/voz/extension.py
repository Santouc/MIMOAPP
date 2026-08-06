"""Extensión de voz para MIMO.

Convierte a audio la frase transcrita usando síntesis de voz (pyttsx3).
Funciona de dos maneras:

1. Automática: cuando el usuario deja de deletrear durante unos segundos
   (pausa), la extensión dice en voz alta la frase acumulada.
2. Manual: mediante el botón "Repetir frase" o la tecla V en la pantalla
   de traducción.

La síntesis corre en un hilo de trabajo separado para no congelar la
interfaz gráfica mientras se reproduce el audio.
"""

import queue
import threading

# pyttsx3 es opcional: si no está instalado, la extensión se desactiva
# de forma segura y muestra un mensaje al usuario cuando intenta usarla.
try:
    import pyttsx3
except ImportError:
    pyttsx3 = None

from services.extension_service import TranslateAction

# Metadatos que el ExtensionService lee para mostrar la extensión en la UI.
NAME = "Voz"
VERSION = "2.0"
DESCRIPTION = "Dice automáticamente la frase transcrita al detectar una pausa, y a pedido con el botón o la tecla V."

# Segundos de pausa (sin cambios en la transcripción) antes de hablar solo.
AUTO_SPEAK_DELAY_SECONDS = 2.0


class SpeechService:
    """Motor de texto a voz con cola de reproducción en segundo plano.

    Encapsula pyttsx3 detrás de una cola: los textos a decir se encolan y
    un hilo trabajador los reproduce uno por uno, evitando bloquear el
    hilo principal de la aplicación.
    """

    def __init__(self, rate: int = 165, volume: float = 1.0):
        """Inicializa el servicio de voz.

        Args:
            rate: Velocidad de habla en palabras por minuto.
            volume: Volumen de reproducción entre 0.0 y 1.0.
        """
        self.rate = rate
        self.volume = volume
        # Solo está disponible si pyttsx3 se pudo importar.
        self.available = pyttsx3 is not None
        # Cola de frases pendientes; None actúa como señal de apagado.
        self._queue: queue.Queue[str | None] = queue.Queue()
        # Hilo trabajador que consume la cola (se crea bajo demanda).
        self._worker: threading.Thread | None = None
        # Bandera activa mientras se está reproduciendo audio.
        self._speaking = threading.Event()

    def speak(self, text: str) -> bool:
        """Encola un texto para que sea dicho en voz alta.

        Args:
            text: Frase a reproducir.

        Returns:
            True si el texto se encoló, False si estaba vacío o el motor
            de voz no está disponible.
        """
        clean = str(text).strip()
        if not clean or not self.available:
            return False
        # Asegurar que el hilo trabajador esté corriendo antes de encolar.
        self._ensure_worker()
        self._queue.put(clean)
        return True

    def is_speaking(self) -> bool:
        """Indica si hay audio reproduciéndose o frases pendientes en cola."""
        return self._speaking.is_set() or not self._queue.empty()

    def stop(self) -> None:
        """Vacía la cola de frases pendientes sin cortar la frase actual."""
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

    def shutdown(self) -> None:
        """Detiene el servicio: limpia la cola y apaga el hilo trabajador."""
        self.stop()
        # Enviar la señal de apagado (None) para que el hilo termine.
        if self._worker is not None and self._worker.is_alive():
            self._queue.put(None)

    def _ensure_worker(self) -> None:
        """Crea y arranca el hilo trabajador si no existe o ya terminó."""
        if self._worker is not None and self._worker.is_alive():
            return
        # daemon=True permite que el programa cierre aunque el hilo siga vivo.
        self._worker = threading.Thread(target=self._run_worker, daemon=True)
        self._worker.start()

    def _run_worker(self) -> None:
        """Bucle del hilo trabajador: consume la cola y reproduce cada frase.

        Se crea un motor pyttsx3 nuevo por frase porque reutilizar la misma
        instancia entre reproducciones causa problemas en algunos sistemas.
        """
        while True:
            text = self._queue.get()
            # None es la señal de apagado enviada por shutdown().
            if text is None:
                return
            self._speaking.set()
            try:
                # Configurar el motor: velocidad, volumen y voz en español.
                engine = pyttsx3.init()
                engine.setProperty("rate", self.rate)
                engine.setProperty("volume", self.volume)
                spanish_voice = self._find_spanish_voice(engine)
                if spanish_voice:
                    engine.setProperty("voice", spanish_voice)
                # Reproducir el texto de forma bloqueante dentro de este hilo.
                engine.say(text)
                engine.runAndWait()
                engine.stop()
            except Exception:
                # Cualquier fallo del motor de voz se ignora para no romper la app.
                pass
            finally:
                self._speaking.clear()

    def _find_spanish_voice(self, engine) -> str | None:
        """Busca entre las voces instaladas una que sea en español.

        Compara el identificador y nombre de cada voz contra marcadores
        comunes ("spanish", "es-", "es_") y nombres de voces conocidas de
        Windows en español (Sabina, Helena).

        Returns:
            El id de la voz en español, o None si no se encontró ninguna.
        """
        try:
            for voice in engine.getProperty("voices"):
                identifier = f"{voice.id} {voice.name}".lower()
                if "spanish" in identifier or "es-" in identifier or "es_" in identifier or "sabina" in identifier or "helena" in identifier:
                    return voice.id
        except Exception:
            pass
        return None


class Extension:
    """Punto de entrada de la extensión, instanciada por ExtensionService.

    Implementa los "hooks" que el sistema de extensiones reconoce:
    - setup: inicialización con el contexto de la app.
    - transcription_changed: se llama cada vez que cambia la transcripción.
    - translate_actions: aporta botones/atajos a la pantalla de traducción.
    - shutdown: limpieza al desactivar la extensión o cerrar la app.
    """

    def setup(self, context) -> None:
        """Prepara la extensión con el contexto global de la aplicación.

        Args:
            context: AppContext con acceso a los servicios (transcripción, etc.).
        """
        self.context = context
        self.speech = SpeechService()
        # Lock para proteger el estado compartido entre el hilo de la UI
        # y los timers de habla automática.
        self._lock = threading.Lock()
        # Timer pendiente que disparará el habla automática tras la pausa.
        self._auto_timer: threading.Timer | None = None
        # Último texto crudo (letras) y de salida (frase) observados.
        self._pending_raw = ""
        self._pending_output = ""
        # Texto crudo de la última frase ya dicha, para no repetirla.
        self._last_spoken_raw = ""

    def transcription_changed(self, state) -> None:
        """Reacciona a cada cambio en la transcripción en vivo.

        Reinicia el temporizador de pausa: si la transcripción deja de
        cambiar durante AUTO_SPEAK_DELAY_SECONDS, se dirá la frase en voz
        alta automáticamente.

        Args:
            state: Estado de transcripción con raw_text (letras) y
                output_text (frase final).
        """
        raw_text = state.raw_text if state else ""
        output_text = state.output_text if state else ""
        with self._lock:
            # Cancelar el timer anterior: la transcripción sigue cambiando.
            if self._auto_timer is not None:
                self._auto_timer.cancel()
                self._auto_timer = None
            # Si se borró la transcripción, reiniciar todo el estado.
            if not raw_text:
                self._pending_raw = ""
                self._pending_output = ""
                self._last_spoken_raw = ""
                return
            self._pending_raw = raw_text
            self._pending_output = output_text
            # No programar habla si la frase ya fue dicha o no hay salida.
            if raw_text == self._last_spoken_raw or not output_text:
                return
            # Programar el habla automática tras la pausa configurada.
            self._auto_timer = threading.Timer(AUTO_SPEAK_DELAY_SECONDS, self._auto_speak, args=(raw_text,))
            self._auto_timer.daemon = True
            self._auto_timer.start()

    def _auto_speak(self, raw_snapshot: str) -> None:
        """Dice la frase pendiente si la transcripción no cambió desde el timer.

        Args:
            raw_snapshot: Texto crudo capturado al programar el timer; si ya
                no coincide con el actual significa que hubo cambios y no
                se debe hablar todavía.
        """
        with self._lock:
            # Abortar si la transcripción cambió o la frase ya se dijo.
            if raw_snapshot != self._pending_raw or raw_snapshot == self._last_spoken_raw:
                return
            text = self._pending_output
            if not text:
                return
            self._last_spoken_raw = raw_snapshot
        # Hablar fuera del lock para no bloquear otros hilos.
        self.speech.speak(text)

    def translate_actions(self, screen) -> list[TranslateAction]:
        """Aporta la acción "Repetir frase" a la pantalla de traducción.

        Args:
            screen: La pantalla de traducción, usada para mostrar mensajes
                de estado al usuario.

        Returns:
            Lista con una TranslateAction asociada a la tecla V.
        """
        def speak_translation() -> None:
            # Obtener la frase final actual desde el servicio de transcripción.
            text = self.context.transcription.get_output_text()
            if not text:
                screen.transcription_status_label.setText("Transcripción: no hay frase para decir")
                return
            if not self.speech.available:
                screen.transcription_status_label.setText("Transcripción: motor de voz no disponible (instala pyttsx3)")
                return
            self.speech.speak(text)
            screen.transcription_status_label.setText(f'Transcripción: diciendo "{text}"')

        return [TranslateAction(label="Repetir frase (V)", callback=speak_translation, key="V")]

    def shutdown(self) -> None:
        """Libera recursos: cancela el timer pendiente y apaga la voz."""
        with self._lock:
            if self._auto_timer is not None:
                self._auto_timer.cancel()
                self._auto_timer = None
        self.speech.shutdown()
