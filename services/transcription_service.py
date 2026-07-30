import json
import re
import time
import unicodedata
from dataclasses import dataclass

try:
    import wordfreq
except ImportError:
    wordfreq = None

from .path_service import PathService

LEXICON_SIZE = 50000
MIN_ZIPF = 3.0
MAX_WORD_LENGTH = 24
WORD_INSERTION_PENALTY = 3.0
UNKNOWN_CHAR_COST = 2.2
DEFAULT_LANGUAGE = "es"
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
    raw_text: str
    output_text: str
    current_sign: str | None
    status: str


class TranscriptionService:
    def __init__(self, paths: PathService, min_hold_seconds: float = 0.75, cooldown_seconds: float = 0.9):
        self.paths = paths
        self.min_hold_seconds = min_hold_seconds
        self.cooldown_seconds = cooldown_seconds
        self.rules_path = self.paths.transcription_rules_path
        self.memory_path = self.paths.transcription_memory_path
        self._user_rules = self._read_user_rules()
        self.language = self._normalize_language(self._user_rules.get("language"))
        self.rules = self._merge_rules(self._default_rules(), self._user_rules)
        self.memory = self._load_memory()
        self.raw_tokens: list[str] = []
        self.current_sign: str | None = None
        self.current_started_at = 0.0
        self.last_accepted_sign: str | None = None
        self.last_accepted_at = 0.0
        self.accepted_current_hold = False
        self._lexicon: dict[str, tuple[str, float]] | None = None
        self._vocabulary_cache: dict[str, tuple[str, float]] | None = None

    def reset(self) -> None:
        self.raw_tokens.clear()
        self.current_sign = None
        self.current_started_at = 0.0
        self.last_accepted_sign = None
        self.last_accepted_at = 0.0
        self.accepted_current_hold = False

    def process_sign(self, sign: str | None, timestamp: float | None = None) -> TranscriptionState:
        now = timestamp if timestamp is not None else time.time()
        normalized_sign = self._normalize_sign(sign)

        if normalized_sign is None:
            self.current_sign = None
            self.current_started_at = 0.0
            self.accepted_current_hold = False
            return self._state(None, "Esperando seña estable")

        if normalized_sign != self.current_sign:
            self.current_sign = normalized_sign
            self.current_started_at = now
            self.accepted_current_hold = False
            return self._state(normalized_sign, f"Detectando {normalized_sign}")

        held_seconds = now - self.current_started_at
        if held_seconds < self.min_hold_seconds:
            return self._state(normalized_sign, f"Mantén {normalized_sign}")

        if self.accepted_current_hold:
            return self._state(normalized_sign, f"{normalized_sign} ya registrada")

        if normalized_sign == self.last_accepted_sign and now - self.last_accepted_at < self.cooldown_seconds:
            return self._state(normalized_sign, f"Espera para repetir {normalized_sign}")

        self._accept_token(normalized_sign, now)
        return self._state(normalized_sign, f"Registrado: {normalized_sign}")

    def backspace(self) -> TranscriptionState:
        if self.raw_tokens:
            self.raw_tokens.pop()
        return self._state(self.current_sign, "Último carácter eliminado")

    def clear(self) -> TranscriptionState:
        self.raw_tokens.clear()
        return self._state(self.current_sign, "Transcripción limpiada")

    def learn_phrase(self, raw_text: str, interpreted_text: str) -> TranscriptionState:
        raw_key = self._memory_key(raw_text)
        clean_interpretation = interpreted_text.strip()
        if raw_key and clean_interpretation:
            self.memory[raw_key] = clean_interpretation
            self._save_memory()
            self._vocabulary_cache = None
        return self._state(self.current_sign, "Interpretación aprendida")

    def learn_current_phrase(self, interpreted_text: str) -> TranscriptionState:
        return self.learn_phrase(self.get_raw_text(), interpreted_text)

    def get_raw_text(self) -> str:
        return "".join(self.raw_tokens)

    def get_output_text(self) -> str:
        return self._interpret(self.get_raw_text())

    def available_languages(self) -> list[tuple[str, str]]:
        return [(code, info["name"]) for code, info in LANGUAGES.items()]

    def set_language(self, code: str) -> bool:
        normalized = self._normalize_language(code)
        if normalized not in LANGUAGES:
            return False
        if normalized == self.language:
            return True
        self.language = normalized
        self._user_rules["language"] = normalized
        self._save_user_rules()
        self.rules = self._merge_rules(self._default_rules(), self._user_rules)
        self._lexicon = None
        self._vocabulary_cache = None
        return True

    def _normalize_language(self, code) -> str:
        normalized = str(code or "").strip().lower()
        return normalized if normalized in LANGUAGES else DEFAULT_LANGUAGE

    def _accept_token(self, sign: str, timestamp: float) -> None:
        mapped = self._map_token(sign)
        if mapped:
            self.raw_tokens.append(mapped)
        self.last_accepted_sign = sign
        self.last_accepted_at = timestamp
        self.accepted_current_hold = True

    def _state(self, current_sign: str | None, status: str) -> TranscriptionState:
        return TranscriptionState(
            raw_text=self.get_raw_text(),
            output_text=self.get_output_text(),
            current_sign=current_sign,
            status=status,
        )

    def _read_user_rules(self) -> dict:
        self.rules_path.parent.mkdir(parents=True, exist_ok=True)
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
        self.rules_path.parent.mkdir(parents=True, exist_ok=True)
        self.rules_path.write_text(json.dumps(self._user_rules, ensure_ascii=False, indent=2), encoding="utf-8")

    def _load_memory(self) -> dict[str, str]:
        self.memory_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.memory_path.exists():
            self.memory_path.write_text("{}", encoding="utf-8")
            return {}
        try:
            loaded = json.loads(self.memory_path.read_text(encoding="utf-8-sig"))
            if isinstance(loaded, dict):
                return {self._memory_key(key): str(value).strip() for key, value in loaded.items() if self._memory_key(key) and str(value).strip()}
        except (OSError, json.JSONDecodeError):
            pass
        return {}

    def _save_memory(self) -> None:
        self.memory_path.parent.mkdir(parents=True, exist_ok=True)
        self.memory_path.write_text(json.dumps(self.memory, ensure_ascii=False, indent=2), encoding="utf-8")

    def _merge_rules(self, default_rules: dict, loaded: dict) -> dict:
        if not isinstance(loaded, dict):
            return default_rules
        merged = default_rules.copy()
        spanish_defaults = self._spanish_defaults()
        for key in ("token_map", "word_map", "correction_map", "word_frequencies"):
            user_map = loaded.get(key)
            if not isinstance(user_map, dict):
                continue
            if self.language != DEFAULT_LANGUAGE and key != "token_map":
                defaults_for_key = spanish_defaults.get(key, {})
                user_map = {k: v for k, v in user_map.items() if defaults_for_key.get(k) != v}
            merged[key] = {**default_rules.get(key, {}), **user_map}
        return merged

    def _default_rules(self) -> dict:
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
        if self.language == DEFAULT_LANGUAGE:
            base.update(self._spanish_defaults())
        return base

    def _spanish_defaults(self) -> dict:
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
        if not sign or sign == "unknown":
            return None
        value = str(sign).strip()
        if not value:
            return None
        return value.upper()

    def _map_token(self, sign: str) -> str | None:
        token_map = self.rules.get("token_map", {})
        mapped = token_map.get(sign, sign)
        if mapped == "<BACKSPACE>":
            if self.raw_tokens:
                self.raw_tokens.pop()
            return None
        if mapped == "<CLEAR>":
            self.raw_tokens.clear()
            return None
        return mapped

    def _interpret(self, raw_text: str) -> str:
        if not raw_text:
            return ""
        exact_memory = self._match_memory(raw_text)
        if exact_memory:
            return exact_memory
        interpreted_parts = []
        for part in re.split(r"(\s+)", raw_text):
            if not part:
                continue
            if part.isspace():
                interpreted_parts.append(part)
                continue
            memory_match = self._match_memory(part)
            if memory_match:
                interpreted_parts.append(memory_match)
            else:
                interpreted_parts.append(self._interpret_compact_letters(part))
        return self._format_sentence("".join(interpreted_parts))

    def _interpret_compact_letters(self, value: str) -> str:
        key = self._rule_key(value)
        if not key:
            return value
        corrected_key = self._apply_corrections(key)
        memory_match = self._match_memory(corrected_key)
        if memory_match:
            return memory_match
        word_map = self._normalized_word_map()
        if corrected_key in word_map:
            return word_map[corrected_key]
        segments = self._segment(corrected_key)
        if segments:
            return " ".join(self._display_segment(segment, known) for segment, known in segments)
        return corrected_key.lower()

    def _build_lexicon(self) -> dict[str, tuple[str, float]]:
        if self._lexicon is not None:
            return self._lexicon
        lexicon: dict[str, tuple[str, float]] = {}
        single_letters = LANGUAGES[self.language]["single_letters"]
        if wordfreq is not None:
            for word in wordfreq.top_n_list(self.language, LEXICON_SIZE):
                if not word.isalpha() or len(word) > MAX_WORD_LENGTH:
                    continue
                key = self._rule_key(word)
                if not key:
                    continue
                if len(key) == 1 and key not in single_letters:
                    continue
                zipf = wordfreq.zipf_frequency(word, self.language)
                if zipf < MIN_ZIPF:
                    continue
                existing = lexicon.get(key)
                if existing is None or zipf > existing[1]:
                    lexicon[key] = (word, zipf)
        self._lexicon = lexicon
        return lexicon

    def _vocabulary(self) -> dict[str, tuple[str, float]]:
        if self._vocabulary_cache is not None:
            return self._vocabulary_cache
        vocabulary = dict(self._build_lexicon())
        for key, display in self._normalized_word_map().items():
            if key and key.isalnum():
                vocabulary[key] = (display, 7.5)
        for key, phrase in self.memory.items():
            if key and key.isalnum():
                vocabulary[key] = (phrase, 8.0)
        self._vocabulary_cache = vocabulary
        return vocabulary

    def _segment(self, key: str) -> list[tuple[str, bool]]:
        vocabulary = self._vocabulary()
        if not vocabulary:
            return [(key, False)] if key else []
        length = len(key)
        best: list[tuple[float, int, bool] | None] = [None] * (length + 1)
        best[0] = (0.0, 0, True)
        for index in range(length):
            entry = best[index]
            if entry is None:
                continue
            base_score = entry[0]
            limit = min(length, index + MAX_WORD_LENGTH)
            for end in range(index + 1, limit + 1):
                candidate = key[index:end]
                vocab_entry = vocabulary.get(candidate)
                if vocab_entry is None:
                    continue
                score = base_score + (vocab_entry[1] - 9.0) - WORD_INSERTION_PENALTY
                previous = best[end]
                if previous is None or score > previous[0]:
                    best[end] = (score, index, True)
            chunk_start_penalty = WORD_INSERTION_PENALTY if entry[2] else 0.0
            unknown_score = base_score - UNKNOWN_CHAR_COST - chunk_start_penalty
            previous = best[index + 1]
            if previous is None or unknown_score > previous[0]:
                best[index + 1] = (unknown_score, index, False)
        if best[length] is None:
            return [(key, False)]
        pieces: list[tuple[str, bool]] = []
        position = length
        while position > 0:
            _, start, known = best[position]
            pieces.append((key[start:position], known))
            position = start
        pieces.reverse()
        merged: list[tuple[str, bool]] = []
        for text, known in pieces:
            if not known and merged and not merged[-1][1]:
                merged[-1] = (merged[-1][0] + text, False)
            else:
                merged.append((text, known))
        return merged

    def _display_segment(self, segment: str, known: bool) -> str:
        if known:
            entry = self._vocabulary().get(segment)
            if entry:
                return entry[0]
            return segment.lower()
        repaired = self._fuzzy_repair(segment)
        if repaired:
            return repaired
        return segment.lower()

    def _fuzzy_repair(self, segment: str) -> str | None:
        if len(segment) < 5:
            return None
        vocabulary = self._vocabulary()
        best_word = None
        best_zipf = 0.0
        for candidate, (display, zipf) in vocabulary.items():
            if abs(len(candidate) - len(segment)) > 1:
                continue
            if zipf < 3.5:
                continue
            if self._levenshtein(segment, candidate) == 1:
                if zipf > best_zipf:
                    best_word = display
                    best_zipf = zipf
        return best_word

    def _match_memory(self, raw_text: str) -> str | None:
        key = self._memory_key(raw_text)
        if not key:
            return None
        if key in self.memory:
            return self.memory[key]
        best_key = None
        best_distance = 999
        for candidate in self.memory:
            if abs(len(candidate) - len(key)) > max(2, len(key) // 3):
                continue
            distance = self._levenshtein(key, candidate)
            if distance < best_distance:
                best_key = candidate
                best_distance = distance
        if best_key is not None and best_distance <= max(1, len(key) // 5):
            return self.memory[best_key]
        return None

    def _apply_corrections(self, key: str) -> str:
        corrections = {self._rule_key(k): self._rule_key(v) for k, v in self.rules.get("correction_map", {}).items()}
        if key in corrections:
            return corrections[key]
        for source, target in sorted(corrections.items(), key=lambda item: len(item[0]), reverse=True):
            key = key.replace(source, target)
        return key

    def _normalized_word_map(self) -> dict[str, str]:
        return {self._rule_key(key): str(value) for key, value in self.rules.get("word_map", {}).items() if self._rule_key(key)}

    def _format_sentence(self, text: str) -> str:
        clean = re.sub(r"\s+", " ", text).strip()
        if not clean:
            return ""
        return clean[0].upper() + clean[1:]

    def _memory_key(self, value: str) -> str:
        return self._rule_key(value)

    def _rule_key(self, value: str) -> str:
        normalized = unicodedata.normalize("NFD", str(value).upper())
        without_accents = "".join(character for character in normalized if unicodedata.category(character) != "Mn")
        return re.sub(r"[^A-Z0-9]", "", without_accents)

    def _levenshtein(self, left: str, right: str) -> int:
        if left == right:
            return 0
        if not left:
            return len(right)
        if not right:
            return len(left)
        previous = list(range(len(right) + 1))
        for i, left_char in enumerate(left, start=1):
            current = [i]
            for j, right_char in enumerate(right, start=1):
                insert_cost = current[j - 1] + 1
                delete_cost = previous[j] + 1
                replace_cost = previous[j - 1] + (left_char != right_char)
                current.append(min(insert_cost, delete_cost, replace_cost))
            previous = current
        return previous[-1]
