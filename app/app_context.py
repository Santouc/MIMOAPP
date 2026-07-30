from services import CaptureService, DocumentService, ExtensionService, LibraryService, PathService, SignService, TrainingService, TranscriptionService


class AppContext:
    def __init__(self):
        self.paths = PathService()
        self.paths.ensure_app_dirs()
        self.signs = SignService(self.paths)
        self.captures = CaptureService(self.paths, self.signs)
        self.documents = DocumentService(self.paths)
        self.training = TrainingService(self.paths)
        self.libraries = LibraryService(self.signs)
        self.transcription = TranscriptionService(self.paths)
        self.extensions = ExtensionService(self.paths)
        self.extensions.load_all(self)
