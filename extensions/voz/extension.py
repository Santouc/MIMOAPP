import queue
import threading

try:
    import pyttsx3
except ImportError:
    pyttsx3 = None

from services.extension_service import TranslateAction

NAME = "Voz"
VERSION = "2.0"
DESCRIPTION = "Dice automáticamente la frase transcrita al detectar una pausa, y a pedido con el botón o la tecla V."

AUTO_SPEAK_DELAY_SECONDS = 2.0


class SpeechService:
    def __init__(self, rate: int = 165, volume: float = 1.0):
        self.rate = rate
        self.volume = volume
        self.available = pyttsx3 is not None
        self._queue: queue.Queue[str | None] = queue.Queue()
        self._worker: threading.Thread | None = None
        self._speaking = threading.Event()

    def speak(self, text: str) -> bool:
        clean = str(text).strip()
        if not clean or not self.available:
            return False
        self._ensure_worker()
        self._queue.put(clean)
        return True

    def is_speaking(self) -> bool:
        return self._speaking.is_set() or not self._queue.empty()

    def stop(self) -> None:
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

    def shutdown(self) -> None:
        self.stop()
        if self._worker is not None and self._worker.is_alive():
            self._queue.put(None)

    def _ensure_worker(self) -> None:
        if self._worker is not None and self._worker.is_alive():
            return
        self._worker = threading.Thread(target=self._run_worker, daemon=True)
        self._worker.start()

    def _run_worker(self) -> None:
        while True:
            text = self._queue.get()
            if text is None:
                return
            self._speaking.set()
            try:
                engine = pyttsx3.init()
                engine.setProperty("rate", self.rate)
                engine.setProperty("volume", self.volume)
                spanish_voice = self._find_spanish_voice(engine)
                if spanish_voice:
                    engine.setProperty("voice", spanish_voice)
                engine.say(text)
                engine.runAndWait()
                engine.stop()
            except Exception:
                pass
            finally:
                self._speaking.clear()

    def _find_spanish_voice(self, engine) -> str | None:
        try:
            for voice in engine.getProperty("voices"):
                identifier = f"{voice.id} {voice.name}".lower()
                if "spanish" in identifier or "es-" in identifier or "es_" in identifier or "sabina" in identifier or "helena" in identifier:
                    return voice.id
        except Exception:
            pass
        return None


class Extension:
    def setup(self, context) -> None:
        self.context = context
        self.speech = SpeechService()
        self._lock = threading.Lock()
        self._auto_timer: threading.Timer | None = None
        self._pending_raw = ""
        self._pending_output = ""
        self._last_spoken_raw = ""

    def transcription_changed(self, state) -> None:
        raw_text = state.raw_text if state else ""
        output_text = state.output_text if state else ""
        with self._lock:
            if self._auto_timer is not None:
                self._auto_timer.cancel()
                self._auto_timer = None
            if not raw_text:
                self._pending_raw = ""
                self._pending_output = ""
                self._last_spoken_raw = ""
                return
            self._pending_raw = raw_text
            self._pending_output = output_text
            if raw_text == self._last_spoken_raw or not output_text:
                return
            self._auto_timer = threading.Timer(AUTO_SPEAK_DELAY_SECONDS, self._auto_speak, args=(raw_text,))
            self._auto_timer.daemon = True
            self._auto_timer.start()

    def _auto_speak(self, raw_snapshot: str) -> None:
        with self._lock:
            if raw_snapshot != self._pending_raw or raw_snapshot == self._last_spoken_raw:
                return
            text = self._pending_output
            if not text:
                return
            self._last_spoken_raw = raw_snapshot
        self.speech.speak(text)

    def translate_actions(self, screen) -> list[TranslateAction]:
        def speak_translation() -> None:
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
        with self._lock:
            if self._auto_timer is not None:
                self._auto_timer.cancel()
                self._auto_timer = None
        self.speech.shutdown()
