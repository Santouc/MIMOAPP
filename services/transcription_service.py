"""Servicio de transcripción de MIMO.

Este módulo convierte la secuencia de señas detectadas por el modelo de
reconocimiento (una letra o comando por seña) en texto legible. Es el
"cerebro lingüístico" de la aplicación y se encarga de:

1. Estabilización temporal: una seña solo se acepta si se mantiene frente a
   la cámara durante un tiempo mínimo (``min_hold_seconds``), y para repetir
   la misma seña se exige una pausa (``cooldown_seconds``). Así se evitan
   letras duplicadas por parpadeos del detector.
2. Mapeo de tokens: cada seña aceptada se traduce mediante un ``token_map``
   (por ejemplo, la seña ESPACIO se convierte en " ", BORRAR elimina el
   último carácter y LIMPIAR vacía la transcripción).
3. Interpretación: el texto crudo (letras pegadas, sin espacios ni tildes)
   se transforma en frases con palabras reales. Para ello se combinan:
   - Reglas del usuario (``word_map``, ``correction_map``) guardadas en JSON.
   - Una "memoria" de correcciones aprendidas (frase cruda -> interpretación)
     que el usuario enseña desde la interfaz.
   - Un léxico de palabras frecuentes construido con la librería ``wordfreq``
     (si está instalada), que permite segmentar cadenas como "HOLACOMOESTAS"
     en "hola cómo estás" mediante programación dinámica.
4. Reparación difusa: los segmentos no reconocidos se intentan corregir con
   distancia de Levenshtein contra el vocabulario (tolerancia de un error).

La configuración persiste en dos archivos JSON gestionados por
``PathService``: las reglas del usuario (idioma y mapas personalizados) y la
memoria de frases aprendidas.
"""

import json
import re
import time
import unicodedata
from dataclasses import dataclass

# wordfreq es opcional: aporta el léxico de palabras frecuentes por idioma.
# Si no está instalada, el servicio sigue funcionando pero la segmentación
# automática de palabras queda limitada a los mapas del usuario y la memoria.
try:
    import wordfreq
except ImportError:
    wordfreq = None

from .path_service import PathService

# --- Constantes de configuración del léxico y la segmentación ---
# LEXICON_SIZE: cuántas palabras más frecuentes del idioma se cargan de wordfreq.
LEXICON_SIZE = 50000
# MIN_ZIPF: frecuencia mínima (escala Zipf) para admitir una palabra en el léxico;
# filtra palabras demasiado raras que generarían segmentaciones extrañas.
MIN_ZIPF = 3.0
# MAX_WORD_LENGTH: longitud máxima de palabra considerada al segmentar.
MAX_WORD_LENGTH = 24
# WORD_INSERTION_PENALTY: penalización por iniciar una palabra nueva durante la
# segmentación; favorece pocas palabras largas frente a muchas cortas.
WORD_INSERTION_PENALTY = 3.0
# UNKNOWN_CHAR_COST: costo por cada carácter que queda fuera del vocabulario.
UNKNOWN_CHAR_COST = 2.2
# Idioma por defecto de la aplicación.
DEFAULT_LANGUAGE = "es"
# Idiomas soportados. "single_letters" enumera las letras que pueden funcionar
# como palabra de una sola letra en cada idioma (p. ej. "y", "a", "o" en
# español); el resto de letras sueltas se descartan del léxico para no
# fragmentar la segmentación.
LANGUAGES = {
    "es": {"name": "Español", "single_letters": {"A", "E", "O", "U", "Y"}},
    "en": {"name": "English", "single_letters": {"A", "I"}},
    "pt": {"name": "Português", "single_letters": {"A", "E", "O"}},
    "fr": {"name": "Français", "single_letters": {"A", "Y"}},
    "it": {"name": "Italiano", "single_letters": {"A", "E", "O"}},
    "de": {"name": "Deutsch", "single_letters": set()},
}


@dataclass
class TranscriptionState:
    """Instantánea del estado de la transcripción para la interfaz.

    Atributos:
        raw_text: texto crudo acumulado (letras aceptadas tal cual).
        output_text: texto ya interpretado (palabras separadas, tildes, etc.).
        current_sign: seña que se está detectando en este instante (o None).
        status: mensaje descriptivo para mostrar al usuario
            (por ejemplo "Mantén A" o "Registrado: A").
    """

    raw_text: str
    output_text: str
    current_sign: str | None
    status: str


class TranscriptionService:
    """Convierte señas detectadas en texto interpretado y lo persiste.

    Mantiene el estado de la seña actual (para la estabilización temporal),
    la lista de tokens crudos aceptados y las reglas/memoria cargadas desde
    disco. Expone métodos de alto nivel para la interfaz: procesar una seña,
    borrar, limpiar, aprender frases y cambiar de idioma.
    """

    def __init__(self, paths: PathService, min_hold_seconds: float = 0.75, cooldown_seconds: float = 0.9):
        """Inicializa el servicio cargando reglas y memoria desde disco.

        Args:
            paths: servicio de rutas que indica dónde viven los JSON de
                reglas y de memoria de correcciones.
            min_hold_seconds: segundos que una seña debe mantenerse estable
                antes de aceptarse como carácter.
            cooldown_seconds: segundos de espera para poder registrar dos
                veces seguidas la misma seña (evita duplicados).
        """
        self.paths = paths
        self.min_hold_seconds = min_hold_seconds
        self.cooldown_seconds = cooldown_seconds
        # Rutas de persistencia: reglas configurables y memoria aprendida.
        self.rules_path = self.paths.transcription_rules_path
        self.memory_path = self.paths.transcription_memory_path
        # Se cargan las reglas del usuario, se resuelve el idioma activo y se
        # combinan las reglas por defecto con las personalizadas.
        self._user_rules = self._read_user_rules()
        self.language = self._normalize_language(self._user_rules.get("language"))
        self.rules = self._merge_rules(self._default_rules(), self._user_rules)
        self.memory = self._load_memory()
        # Estado de la transcripción en curso.
        self.raw_tokens: list[str] = []
        # Estado de estabilización: seña actual, cuándo empezó a mantenerse,
        # última seña aceptada y si la "sostenida" actual ya fue registrada.
        self.current_sign: str | None = None
        self.current_started_at = 0.0
        self.last_accepted_sign: str | None = None
        self.last_accepted_at = 0.0
        self.accepted_current_hold = False
        # Cachés perezosas del léxico (wordfreq) y del vocabulario combinado.
        self._lexicon: dict[str, tuple[str, float]] | None = None
        self._vocabulary_cache: dict[str, tuple[str, float]] | None = None

    def reset(self) -> None:
        """Reinicia por completo la transcripción y el estado de detección."""
        self.raw_tokens.clear()
        self.current_sign = None
        self.current_started_at = 0.0
        self.last_accepted_sign = None
        self.last_accepted_at = 0.0
        self.accepted_current_hold = False

    def process_sign(self, sign: str | None, timestamp: float | None = None) -> TranscriptionState:
        """Procesa la seña detectada en un fotograma y decide si aceptarla.

        Implementa la máquina de estabilización: la seña debe mantenerse
        ``min_hold_seconds`` para registrarse, cada sostenida cuenta una sola
        vez y repetir la misma seña exige respetar ``cooldown_seconds``.

        Args:
            sign: etiqueta predicha por el modelo (o None/"unknown" si no
                hay seña confiable en el fotograma).
            timestamp: marca de tiempo del fotograma; si se omite se usa la
                hora actual (útil para pruebas deterministas).

        Returns:
            TranscriptionState con el texto actualizado y un mensaje de
            estado orientado al usuario.
        """
        now = timestamp if timestamp is not None else time.time()
        normalized_sign = self._normalize_sign(sign)

        # Sin seña válida: se reinicia la sostenida en curso y se espera.
        if normalized_sign is None:
            self.current_sign = None
            self.current_started_at = 0.0
            self.accepted_current_hold = False
            return self._state(None, "Esperando seña estable")

        # Cambio de seña: comienza a contarse una nueva sostenida desde cero.
        if normalized_sign != self.current_sign:
            self.current_sign = normalized_sign
            self.current_started_at = now
            self.accepted_current_hold = False
            return self._state(normalized_sign, f"Detectando {normalized_sign}")

        # Misma seña: comprobar si ya se mantuvo el tiempo mínimo requerido.
        held_seconds = now - self.current_started_at
        if held_seconds < self.min_hold_seconds:
            return self._state(normalized_sign, f"Mantén {normalized_sign}")

        # La sostenida actual ya produjo un carácter; no se vuelve a contar.
        if self.accepted_current_hold:
            return self._state(normalized_sign, f"{normalized_sign} ya registrada")

        # Anti-rebote: repetir la misma letra exige esperar el cooldown.
        if normalized_sign == self.last_accepted_sign and now - self.last_accepted_at < self.cooldown_seconds:
            return self._state(normalized_sign, f"Espera para repetir {normalized_sign}")

        # Todas las condiciones se cumplen: la seña se acepta como token.
        self._accept_token(normalized_sign, now)
        return self._state(normalized_sign, f"Registrado: {normalized_sign}")

    def backspace(self) -> TranscriptionState:
        """Elimina el último carácter crudo (acción manual del usuario)."""
        if self.raw_tokens:
            self.raw_tokens.pop()
        return self._state(self.current_sign, "Último carácter eliminado")

    def clear(self) -> TranscriptionState:
        """Vacía la transcripción sin alterar el estado de detección."""
        self.raw_tokens.clear()
        return self._state(self.current_sign, "Transcripción limpiada")

    def learn_phrase(self, raw_text: str, interpreted_text: str) -> TranscriptionState:
        """Guarda en memoria la asociación frase cruda -> interpretación.

        Es el mecanismo de aprendizaje del usuario: si la interpretación
        automática fue incorrecta, puede enseñarle al sistema la forma
        correcta y esta se aplicará en el futuro (con tolerancia difusa).

        Args:
            raw_text: texto crudo tal como se deletreó (se normaliza a clave).
            interpreted_text: interpretación correcta escrita por el usuario.
        """
        raw_key = self._memory_key(raw_text)
        clean_interpretation = interpreted_text.strip()
        # Solo se aprende si ambos extremos son no vacíos tras normalizar.
        if raw_key and clean_interpretation:
            self.memory[raw_key] = clean_interpretation
            self._save_memory()
            # Se invalida la caché para que el vocabulario incluya la frase nueva.
            self._vocabulary_cache = None
        return self._state(self.current_sign, "Interpretación aprendida")

    def learn_current_phrase(self, interpreted_text: str) -> TranscriptionState:
        """Aprende la interpretación para la transcripción cruda actual."""
        return self.learn_phrase(self.get_raw_text(), interpreted_text)

    def get_raw_text(self) -> str:
        """Devuelve el texto crudo: la concatenación de los tokens aceptados."""
        return "".join(self.raw_tokens)

    def get_output_text(self) -> str:
        """Devuelve el texto interpretado (palabras, tildes, formato de frase)."""
        return self._interpret(self.get_raw_text())

    def available_languages(self) -> list[tuple[str, str]]:
        """Lista los idiomas soportados como pares (código, nombre legible)."""
        return [(code, info["name"]) for code, info in LANGUAGES.items()]

    def set_language(self, code: str) -> bool:
        """Cambia el idioma activo y lo persiste en las reglas del usuario.

        Al cambiar de idioma se reconstruyen las reglas combinadas y se
        invalidan las cachés de léxico y vocabulario, porque dependen del
        idioma cargado en wordfreq.

        Returns:
            True si el código era válido (aunque ya estuviera activo),
            False si no corresponde a un idioma soportado.
        """
        normalized = self._normalize_language(code)
        if normalized not in LANGUAGES:
            return False
        if normalized == self.language:
            return True
        # Actualizar idioma, persistirlo y regenerar reglas y cachés.
        self.language = normalized
        self._user_rules["language"] = normalized
        self._save_user_rules()
        self.rules = self._merge_rules(self._default_rules(), self._user_rules)
        self._lexicon = None
        self._vocabulary_cache = None
        return True

    def _normalize_language(self, code) -> str:
        """Normaliza un código de idioma; si es inválido usa el idioma por defecto."""
        normalized = str(code or "").strip().lower()
        return normalized if normalized in LANGUAGES else DEFAULT_LANGUAGE

    def _accept_token(self, sign: str, timestamp: float) -> None:
        """Registra una seña aceptada: la mapea a token y actualiza el anti-rebote.

        El mapeo puede producir un carácter normal (se agrega al texto crudo)
        o un comando especial (BORRAR/LIMPIAR) que modifica la lista y no
        agrega nada.
        """
        mapped = self._map_token(sign)
        if mapped:
            self.raw_tokens.append(mapped)
        # Registrar la aceptación para el cooldown y marcar la sostenida usada.
        self.last_accepted_sign = sign
        self.last_accepted_at = timestamp
        self.accepted_current_hold = True

    def _state(self, current_sign: str | None, status: str) -> TranscriptionState:
        """Construye la instantánea de estado que consume la interfaz."""
        return TranscriptionState(
            raw_text=self.get_raw_text(),
            output_text=self.get_output_text(),
            current_sign=current_sign,
            status=status,
        )

    def _read_user_rules(self) -> dict:
        """Lee el JSON de reglas del usuario; si no existe lo crea con valores por defecto.

        Si el archivo está corrupto o ilegible se devuelve un diccionario
        vacío para no bloquear el arranque de la aplicación.
        """
        self.rules_path.parent.mkdir(parents=True, exist_ok=True)
        # Primera ejecución: se genera el archivo con idioma y reglas en español.
        if not self.rules_path.exists():
            initial = {"language": DEFAULT_LANGUAGE, **self._spanish_defaults()}
            self.rules_path.write_text(json.dumps(initial, ensure_ascii=False, indent=2), encoding="utf-8")
            return initial
        try:
            loaded = json.loads(self.rules_path.read_text(encoding="utf-8-sig"))
            if isinstance(loaded, dict):
                return loaded
        except (OSError, json.JSONDecodeError):
            pass
        return {}

    def _save_user_rules(self) -> None:
        """Persiste en disco las reglas del usuario en formato JSON legible."""
        self.rules_path.parent.mkdir(parents=True, exist_ok=True)
        self.rules_path.write_text(json.dumps(self._user_rules, ensure_ascii=False, indent=2), encoding="utf-8")

    def _load_memory(self) -> dict[str, str]:
        """Carga la memoria de correcciones (clave cruda -> frase interpretada).

        Las claves se renormalizan al cargarlas y se descartan entradas
        vacías o inválidas. Si el archivo no existe, se crea vacío.
        """
        self.memory_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.memory_path.exists():
            self.memory_path.write_text("{}", encoding="utf-8")
            return {}
        try:
            loaded = json.loads(self.memory_path.read_text(encoding="utf-8-sig"))
            if isinstance(loaded, dict):
                # Se normalizan claves y valores, filtrando entradas vacías.
                return {self._memory_key(key): str(value).strip() for key, value in loaded.items() if self._memory_key(key) and str(value).strip()}
        except (OSError, json.JSONDecodeError):
            pass
        return {}

    def _save_memory(self) -> None:
        """Persiste la memoria de correcciones aprendidas en disco."""
        self.memory_path.parent.mkdir(parents=True, exist_ok=True)
        self.memory_path.write_text(json.dumps(self.memory, ensure_ascii=False, indent=2), encoding="utf-8")

    def _merge_rules(self, default_rules: dict, loaded: dict) -> dict:
        """Combina las reglas por defecto con las del usuario.

        Las entradas del usuario tienen prioridad sobre las por defecto.
        Caso especial: si el idioma activo no es español, se filtran de los
        mapas del usuario aquellas entradas que sean idénticas a los valores
        por defecto en español (fueron sembradas al crear el archivo y no
        deben "contaminar" otros idiomas). El ``token_map`` sí se conserva
        siempre porque los comandos (ESPACIO, BORRAR...) son universales.
        """
        if not isinstance(loaded, dict):
            return default_rules
        merged = default_rules.copy()
        spanish_defaults = self._spanish_defaults()
        # Se fusiona mapa por mapa, dando prioridad a lo definido por el usuario.
        for key in ("token_map", "word_map", "correction_map", "word_frequencies"):
            user_map = loaded.get(key)
            if not isinstance(user_map, dict):
                continue
            if self.language != DEFAULT_LANGUAGE and key != "token_map":
                # Excluir los valores sembrados en español al usar otro idioma.
                defaults_for_key = spanish_defaults.get(key, {})
                user_map = {k: v for k, v in user_map.items() if defaults_for_key.get(k) != v}
            merged[key] = {**default_rules.get(key, {}), **user_map}
        return merged

    def _default_rules(self) -> dict:
        """Construye las reglas base del sistema para el idioma activo.

        El ``token_map`` traduce señas de comando a acciones: ESPACIO/SPACE
        insertan un espacio, BORRAR/DELETE emiten el marcador <BACKSPACE> y
        LIMPIAR/CLEAR el marcador <CLEAR>. Si el idioma es español, se suman
        además los mapas de palabras, correcciones y frecuencias por defecto.
        """
        base = {
            "token_map": {
                "ESPACIO": " ",
                "SPACE": " ",
                "BORRAR": "<BACKSPACE>",
                "DELETE": "<BACKSPACE>",
                "LIMPIAR": "<CLEAR>",
                "CLEAR": "<CLEAR>",
            },
            "word_map": {},
            "correction_map": {},
            "word_frequencies": {},
        }
        # Solo el español recibe el vocabulario semilla por defecto.
        if self.language == DEFAULT_LANGUAGE:
            base.update(self._spanish_defaults())
        return base

    def _spanish_defaults(self) -> dict:
        """Reglas semilla para español.

        - word_map: deletreos frecuentes -> palabra con tildes/formato final.
        - correction_map: errores de deletreo habituales -> forma correcta
          (se aplican antes de buscar en el word_map).
        - word_frequencies: pesos relativos para priorizar palabras comunes.
        """
        return {
            "word_map": {
                "HOLA": "hola",
                "CHAO": "chao",
                "ADIOS": "adiós",
                "GRACIAS": "gracias",
                "PORFAVOR": "por favor",
                "SI": "sí",
                "NO": "no",
                "YO": "yo",
                "TU": "tú",
                "USTED": "usted",
                "ME": "me",
                "MI": "mi",
                "NOMBRE": "nombre",
                "ES": "es",
                "LLAMO": "llamo",
                "COMO": "cómo",
                "ESTAS": "estás",
                "ESTOY": "estoy",
                "BIEN": "bien",
                "MAL": "mal",
                "QUIERO": "quiero",
                "NECESITO": "necesito",
                "AYUDA": "ayuda",
                "AGUA": "agua",
                "COMIDA": "comida",
                "BAÑO": "baño",
                "BANO": "baño",
                "DOLOR": "dolor",
                "CASA": "casa",
                "ESCUELA": "escuela",
                "TRABAJO": "trabajo",
            },
            "correction_map": {
                "YAMO": "LLAMO",
                "MEYAMO": "MELLAMO",
                "GRASIAS": "GRACIAS",
                "PORFABOR": "PORFAVOR",
                "ADIO": "ADIOS",
            },
            "word_frequencies": {
                "HOLA": 100,
                "ME": 90,
                "LLAMO": 95,
                "ES": 80,
                "QUIERO": 85,
                "NECESITO": 85,
                "AYUDA": 90,
                "GRACIAS": 90,
                "AGUA": 75,
                "COMIDA": 70,
                "BAÑO": 70,
            },
        }

    def _normalize_sign(self, sign: str | None) -> str | None:
        """Normaliza la etiqueta de la seña a mayúsculas; None si no es válida.

        Las etiquetas vacías o "unknown" (sin predicción confiable) se
        descartan devolviendo None.
        """
        if not sign or sign == "unknown":
            return None
        value = str(sign).strip()
        if not value:
            return None
        return value.upper()

    def _map_token(self, sign: str) -> str | None:
        """Traduce una seña aceptada según el token_map y ejecuta comandos.

        Returns:
            El carácter a agregar al texto crudo, o None si la seña era un
            comando (<BACKSPACE>/<CLEAR>) que ya se ejecutó sobre la lista.
        """
        token_map = self.rules.get("token_map", {})
        # Si la seña no aparece en el mapa, se usa tal cual (letra normal).
        mapped = token_map.get(sign, sign)
        # Comando de borrado: elimina el último carácter y no agrega nada.
        if mapped == "<BACKSPACE>":
            if self.raw_tokens:
                self.raw_tokens.pop()
            return None
        # Comando de limpieza: vacía toda la transcripción.
        if mapped == "<CLEAR>":
            self.raw_tokens.clear()
            return None
        return mapped

    def _interpret(self, raw_text: str) -> str:
        """Convierte el texto crudo completo en la frase interpretada.

        Orden de resolución:
        1. Coincidencia (exacta o difusa) de la frase completa en memoria.
        2. Se divide por espacios conservándolos; cada bloque se busca en
           memoria y, si no hay coincidencia, se interpreta como cadena
           compacta de letras (segmentación en palabras).
        3. El resultado se formatea como oración (mayúscula inicial).
        """
        if not raw_text:
            return ""
        # Prioridad máxima: la frase completa ya fue enseñada por el usuario.
        exact_memory = self._match_memory(raw_text)
        if exact_memory:
            return exact_memory
        interpreted_parts = []
        # Se separan bloques y espacios (los espacios se conservan tal cual).
        for part in re.split(r"(\s+)", raw_text):
            if not part:
                continue
            if part.isspace():
                interpreted_parts.append(part)
                continue
            # Cada bloque intenta primero la memoria y luego la segmentación.
            memory_match = self._match_memory(part)
            if memory_match:
                interpreted_parts.append(memory_match)
            else:
                interpreted_parts.append(self._interpret_compact_letters(part))
        return self._format_sentence("".join(interpreted_parts))

    def _interpret_compact_letters(self, value: str) -> str:
        """Interpreta un bloque de letras sin espacios (p. ej. "HOLACOMOESTAS").

        Cadena de intentos, de mayor a menor prioridad:
        1. Aplicar el correction_map (arregla deletreos típicos).
        2. Buscar la clave corregida en la memoria aprendida.
        3. Buscar en el word_map (deletreo -> palabra con formato).
        4. Segmentar con programación dinámica usando el vocabulario.
        5. Como último recurso, devolver el texto en minúsculas.
        """
        key = self._rule_key(value)
        if not key:
            return value
        # Correcciones de deletreo conocidas antes de cualquier búsqueda.
        corrected_key = self._apply_corrections(key)
        memory_match = self._match_memory(corrected_key)
        if memory_match:
            return memory_match
        word_map = self._normalized_word_map()
        if corrected_key in word_map:
            return word_map[corrected_key]
        # Segmentación en palabras: cada segmento se muestra según sea
        # conocido (vocabulario) o desconocido (reparación difusa/minúsculas).
        segments = self._segment(corrected_key)
        if segments:
            return " ".join(self._display_segment(segment, known) for segment, known in segments)
        return corrected_key.lower()

    def _build_lexicon(self) -> dict[str, tuple[str, float]]:
        """Construye (y cachea) el léxico de palabras frecuentes con wordfreq.

        Cada entrada mapea la clave normalizada (mayúsculas, sin tildes) a la
        tupla (forma de despliegue, frecuencia Zipf). Se filtran palabras no
        alfabéticas, demasiado largas, poco frecuentes y letras sueltas que
        no funcionan como palabra en el idioma activo. Si wordfreq no está
        instalada, el léxico queda vacío.
        """
        if self._lexicon is not None:
            return self._lexicon
        lexicon: dict[str, tuple[str, float]] = {}
        single_letters = LANGUAGES[self.language]["single_letters"]
        if wordfreq is not None:
            # Recorrer las N palabras más frecuentes del idioma activo.
            for word in wordfreq.top_n_list(self.language, LEXICON_SIZE):
                if not word.isalpha() or len(word) > MAX_WORD_LENGTH:
                    continue
                key = self._rule_key(word)
                if not key:
                    continue
                # Las letras sueltas solo se admiten si son palabra válida.
                if len(key) == 1 and key not in single_letters:
                    continue
                zipf = wordfreq.zipf_frequency(word, self.language)
                if zipf < MIN_ZIPF:
                    continue
                # Ante claves duplicadas (p. ej. con/ sin tilde) gana la más frecuente.
                existing = lexicon.get(key)
                if existing is None or zipf > existing[1]:
                    lexicon[key] = (word, zipf)
        self._lexicon = lexicon
        return lexicon

    def _vocabulary(self) -> dict[str, tuple[str, float]]:
        """Vocabulario combinado: léxico + word_map del usuario + memoria.

        Las entradas del usuario reciben pesos altos artificiales (7.5 el
        word_map y 8.0 la memoria) para que la segmentación las prefiera por
        encima de las palabras genéricas de wordfreq. El resultado se cachea
        hasta que cambien las reglas, el idioma o la memoria.
        """
        if self._vocabulary_cache is not None:
            return self._vocabulary_cache
        vocabulary = dict(self._build_lexicon())
        # Palabras personalizadas del usuario, con prioridad sobre el léxico.
        for key, display in self._normalized_word_map().items():
            if key and key.isalnum():
                vocabulary[key] = (display, 7.5)
        # Frases aprendidas: máxima prioridad dentro del vocabulario.
        for key, phrase in self.memory.items():
            if key and key.isalnum():
                vocabulary[key] = (phrase, 8.0)
        self._vocabulary_cache = vocabulary
        return vocabulary

    def _segment(self, key: str) -> list[tuple[str, bool]]:
        """Divide una cadena compacta en palabras mediante programación dinámica.

        Algoritmo (similar a Viterbi sobre posiciones de la cadena):
        - ``best[i]`` guarda la mejor forma de cubrir los primeros ``i``
          caracteres como tupla (puntaje, inicio del último trozo, es_palabra).
        - Desde cada posición alcanzable se prueban dos transiciones:
          a) cerrar una palabra del vocabulario que empiece allí, cuyo
             puntaje premia la frecuencia Zipf y penaliza insertar palabra;
          b) consumir un carácter "desconocido", con costo por carácter y
             penalización extra al abrir un tramo desconocido nuevo.
        - Al final se reconstruye el camino óptimo hacia atrás y se fusionan
          los tramos desconocidos consecutivos en uno solo.

        Returns:
            Lista de pares (segmento, es_conocido) en orden de aparición.
        """
        vocabulary = self._vocabulary()
        # Sin vocabulario no hay forma de segmentar: todo es desconocido.
        if not vocabulary:
            return [(key, False)] if key else []
        length = len(key)
        # best[i] = (puntaje acumulado, inicio del último trozo, es palabra conocida)
        best: list[tuple[float, int, bool] | None] = [None] * (length + 1)
        best[0] = (0.0, 0, True)
        for index in range(length):
            entry = best[index]
            if entry is None:
                continue
            base_score = entry[0]
            limit = min(length, index + MAX_WORD_LENGTH)
            # Transición (a): probar todas las palabras del vocabulario que
            # comienzan en esta posición.
            for end in range(index + 1, limit + 1):
                candidate = key[index:end]
                vocab_entry = vocabulary.get(candidate)
                if vocab_entry is None:
                    continue
                # Puntaje relativo: frecuencia Zipf menos una referencia (9.0)
                # y menos la penalización por insertar una palabra nueva.
                score = base_score + (vocab_entry[1] - 9.0) - WORD_INSERTION_PENALTY
                previous = best[end]
                if previous is None or score > previous[0]:
                    best[end] = (score, index, True)
            # Transición (b): tratar el carácter actual como desconocido.
            # Abrir un tramo desconocido tras una palabra cuesta más que
            # extender uno ya abierto.
            chunk_start_penalty = WORD_INSERTION_PENALTY if entry[2] else 0.0
            unknown_score = base_score - UNKNOWN_CHAR_COST - chunk_start_penalty
            previous = best[index + 1]
            if previous is None or unknown_score > previous[0]:
                best[index + 1] = (unknown_score, index, False)
        if best[length] is None:
            return [(key, False)]
        # Reconstrucción del camino óptimo desde el final hacia el inicio.
        pieces: list[tuple[str, bool]] = []
        position = length
        while position > 0:
            _, start, known = best[position]
            pieces.append((key[start:position], known))
            position = start
        pieces.reverse()
        # Fusionar tramos desconocidos contiguos en un único segmento.
        merged: list[tuple[str, bool]] = []
        for text, known in pieces:
            if not known and merged and not merged[-1][1]:
                merged[-1] = (merged[-1][0] + text, False)
            else:
                merged.append((text, known))
        return merged

    def _display_segment(self, segment: str, known: bool) -> str:
        """Convierte un segmento en su forma visible para el usuario.

        Los segmentos conocidos usan la forma de despliegue del vocabulario
        (con tildes y minúsculas correctas). Los desconocidos intentan una
        reparación difusa y, si falla, se muestran en minúsculas tal cual.
        """
        if known:
            entry = self._vocabulary().get(segment)
            if entry:
                return entry[0]
            return segment.lower()
        # Segmento desconocido: intentar corregirlo por similitud.
        repaired = self._fuzzy_repair(segment)
        if repaired:
            return repaired
        return segment.lower()

    def _fuzzy_repair(self, segment: str) -> str | None:
        """Intenta reparar un segmento desconocido con distancia de Levenshtein.

        Solo actúa sobre segmentos de 5+ caracteres (los cortos son ambiguos)
        y acepta únicamente candidatos razonablemente frecuentes (Zipf >= 3.5)
        a exactamente un error de edición. Devuelve la palabra reparada o
        None si no hay candidato convincente.
        """
        if len(segment) < 5:
            return None
        vocabulary = self._vocabulary()
        best_word = None
        best_zipf = 0.0
        for candidate, (display, zipf) in vocabulary.items():
            # Descartes rápidos: longitud muy distinta o palabra poco común.
            if abs(len(candidate) - len(segment)) > 1:
                continue
            if zipf < 3.5:
                continue
            # Se exige exactamente un error de edición; empata el más frecuente.
            if self._levenshtein(segment, candidate) == 1:
                if zipf > best_zipf:
                    best_word = display
                    best_zipf = zipf
        return best_word

    def _match_memory(self, raw_text: str) -> str | None:
        """Busca una frase aprendida que coincida con el texto (exacta o difusa).

        Primero intenta la coincidencia exacta de clave. Si no existe, busca
        la entrada de memoria más cercana por Levenshtein, con umbrales que
        escalan con la longitud (aprox. un error tolerado por cada cinco
        caracteres). Devuelve la interpretación aprendida o None.
        """
        key = self._memory_key(raw_text)
        if not key:
            return None
        # Coincidencia exacta: la más confiable.
        if key in self.memory:
            return self.memory[key]
        # Búsqueda difusa: elegir la entrada más cercana en distancia de edición.
        best_key = None
        best_distance = 999
        for candidate in self.memory:
            # Descartar candidatos con longitud demasiado diferente.
            if abs(len(candidate) - len(key)) > max(2, len(key) // 3):
                continue
            distance = self._levenshtein(key, candidate)
            if distance < best_distance:
                best_key = candidate
                best_distance = distance
        # Aceptar solo si la distancia queda dentro del umbral proporcional.
        if best_key is not None and best_distance <= max(1, len(key) // 5):
            return self.memory[best_key]
        return None

    def _apply_corrections(self, key: str) -> str:
        """Aplica el correction_map a una clave normalizada.

        Si hay una corrección exacta para toda la clave, se usa directamente.
        En caso contrario, se aplican reemplazos de subcadenas ordenados de
        la corrección más larga a la más corta (para evitar que una regla
        corta "rompa" una coincidencia más específica).
        """
        corrections = {self._rule_key(k): self._rule_key(v) for k, v in self.rules.get("correction_map", {}).items()}
        if key in corrections:
            return corrections[key]
        # Reemplazos parciales, priorizando las reglas más largas.
        for source, target in sorted(corrections.items(), key=lambda item: len(item[0]), reverse=True):
            key = key.replace(source, target)
        return key

    def _normalized_word_map(self) -> dict[str, str]:
        """Devuelve el word_map con las claves normalizadas (sin tildes ni símbolos)."""
        return {self._rule_key(key): str(value) for key, value in self.rules.get("word_map", {}).items() if self._rule_key(key)}

    def _format_sentence(self, text: str) -> str:
        """Compacta espacios múltiples y capitaliza la primera letra de la frase."""
        clean = re.sub(r"\s+", " ", text).strip()
        if not clean:
            return ""
        return clean[0].upper() + clean[1:]

    def _memory_key(self, value: str) -> str:
        """Clave canónica para la memoria (misma normalización que las reglas)."""
        return self._rule_key(value)

    def _rule_key(self, value: str) -> str:
        """Normaliza un texto a clave canónica: mayúsculas, sin tildes, solo A-Z0-9.

        Ejemplo: "por favor" -> "PORFAVOR", "adiós" -> "ADIOS". Así los
        deletreos se comparan de forma robusta sin importar acentos,
        espacios ni signos de puntuación.
        """
        # Descomposición Unicode (NFD) para separar letras de sus tildes.
        normalized = unicodedata.normalize("NFD", str(value).upper())
        # Eliminar marcas diacríticas (categoría "Mn": marcas no espaciadas).
        without_accents = "".join(character for character in normalized if unicodedata.category(character) != "Mn")
        # Conservar únicamente letras y dígitos ASCII.
        return re.sub(r"[^A-Z0-9]", "", without_accents)

    def _levenshtein(self, left: str, right: str) -> int:
        """Calcula la distancia de Levenshtein (mínimo de inserciones,
        eliminaciones y sustituciones para transformar una cadena en otra).

        Implementación clásica de programación dinámica que solo conserva
        la fila anterior de la matriz para usar memoria O(n).
        """
        # Atajos triviales para cadenas iguales o vacías.
        if left == right:
            return 0
        if not left:
            return len(right)
        if not right:
            return len(left)
        # Fila inicial: transformar la cadena vacía en prefijos de "right".
        previous = list(range(len(right) + 1))
        for i, left_char in enumerate(left, start=1):
            current = [i]
            for j, right_char in enumerate(right, start=1):
                # Costos de las tres operaciones posibles en esta celda.
                insert_cost = current[j - 1] + 1
                delete_cost = previous[j] + 1
                replace_cost = previous[j - 1] + (left_char != right_char)
                current.append(min(insert_cost, delete_cost, replace_cost))
            previous = current
        return previous[-1]
