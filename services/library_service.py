from dataclasses import dataclass

from .sign_service import SignService


WESTERN_ALPHABET = [
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "J",
    "K",
    "L",
    "M",
    "N",
    "O",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "U",
    "V",
    "W",
    "X",
    "Y",
    "Z",
]


@dataclass
class LibraryImportResult:
    library_name: str
    created: list[str]
    existing: list[str]

    @property
    def created_count(self) -> int:
        return len(self.created)

    @property
    def existing_count(self) -> int:
        return len(self.existing)


class LibraryService:
    def __init__(self, signs: SignService):
        self.signs = signs

    def import_western_alphabet(self) -> LibraryImportResult:
        created = []
        existing = []
        for letter in WESTERN_ALPHABET:
            if self.signs.find_by_name(letter):
                existing.append(letter)
                continue
            self.signs.add_sign(letter, [])
            created.append(letter)
        self.signs.export_labels()
        return LibraryImportResult("Alfabeto occidental", created, existing)
