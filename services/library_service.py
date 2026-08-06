"""Servicio de bibliotecas de señas predefinidas.

Este módulo define `LibraryService`, encargado de importar conjuntos de señas
predefinidos al registro de la aplicación. Actualmente incluye una única
biblioteca: el alfabeto occidental (letras A-Z), útil para que el usuario no
tenga que crear cada letra manualmente antes de capturar muestras.
"""

from dataclasses import dataclass

from .sign_service import SignService


# Letras del alfabeto occidental que se crean al importar la biblioteca.
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
    """Resultado de importar una biblioteca de señas.

    Atributos:
        library_name: nombre descriptivo de la biblioteca importada.
        created: nombres de las señas que se crearon en esta importación.
        existing: nombres de las señas que ya existían y se omitieron.
    """

    library_name: str
    created: list[str]
    existing: list[str]

    @property
    def created_count(self) -> int:
        """Cantidad de señas nuevas creadas."""
        return len(self.created)

    @property
    def existing_count(self) -> int:
        """Cantidad de señas que ya existían (omitidas)."""
        return len(self.existing)


class LibraryService:
    """Importa bibliotecas de señas predefinidas al registro."""

    def __init__(self, signs: SignService):
        """Inicializa el servicio.

        Args:
            signs: instancia de `SignService` donde se registrarán las señas.
        """
        self.signs = signs

    def import_western_alphabet(self) -> LibraryImportResult:
        """Importa el alfabeto occidental (A-Z) como señas individuales.

        Recorre cada letra: si ya existe una seña con ese nombre la omite,
        y si no, la crea sin tipos asignados (los tipos se agregan luego
        automáticamente al capturar muestras). Al final reexporta las
        etiquetas para mantener sincronizados los archivos de labels.

        Returns:
            Un `LibraryImportResult` con el detalle de creadas y existentes.
        """
        created = []
        existing = []
        # Crear cada letra que no exista todavía; registrar las omitidas.
        for letter in WESTERN_ALPHABET:
            if self.signs.find_by_name(letter):
                existing.append(letter)
                continue
            self.signs.add_sign(letter, [])
            created.append(letter)
        # Sincronizar los archivos de etiquetas con el registro actualizado.
        self.signs.export_labels()
        return LibraryImportResult("Alfabeto occidental", created, existing)
