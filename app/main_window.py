"""Ventana principal de la aplicación MIMO (T.L.S).

Define :class:`MainWindow`, la ventana raíz de la aplicación de escritorio.
Sus responsabilidades son:

- Crear (o recibir) el :class:`AppContext` con todos los servicios.
- Instanciar todas las pantallas y apilarlas en un ``QStackedWidget``,
  de modo que solo una pantalla sea visible a la vez.
- Conectar las señales de navegación de cada pantalla para cambiar la
  pantalla visible (patrón de navegación centralizada).
- Aplicar la hoja de estilos global (tema oscuro) a toda la interfaz.
- Liberar recursos (cámaras y extensiones) al cerrar la ventana.
"""

from PySide6.QtWidgets import QMainWindow, QStackedWidget

from app.app_context import AppContext
from app.screens import DocumentScreen, ExtensionsScreen, HomeScreen, ManageSignsScreen, TeachSignScreen, TranslateScreen


class MainWindow(QMainWindow):
    """Ventana raíz que orquesta la navegación entre las pantallas."""

    def __init__(self, context: AppContext | None = None):
        """Construye la ventana principal y todas sus pantallas.

        Args:
            context: contexto de aplicación con los servicios ya creados.
                Si es ``None``, se crea uno nuevo automáticamente (útil para
                el arranque normal de la aplicación).
        """
        super().__init__()
        # Contexto compartido: si no se recibe uno, se construye aquí.
        self.context = context or AppContext()
        # Título y tamaño inicial de la ventana.
        self.setWindowTitle("T.L.S - Traductor de Lengua de Señas")
        self.resize(980, 680)

        # Widget apilado: contiene todas las pantallas y muestra solo una.
        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)

        # Creación de todas las pantallas de la aplicación. Las pantallas de
        # créditos y manual reutilizan DocumentScreen con contenido distinto.
        self.home_screen = HomeScreen()
        self.teach_screen = TeachSignScreen(self.context)
        self.manage_screen = ManageSignsScreen(self.context)
        self.translate_screen = TranslateScreen(self.context)
        self.extensions_screen = ExtensionsScreen(self.context)
        self.credits_screen = DocumentScreen("Créditos", self.context.documents.read_credits())
        self.manual_screen = DocumentScreen("Manual de uso", self.context.documents.read_manual())

        # Registro de todas las pantallas en el widget apilado.
        self.stack.addWidget(self.home_screen)
        self.stack.addWidget(self.teach_screen)
        self.stack.addWidget(self.manage_screen)
        self.stack.addWidget(self.translate_screen)
        self.stack.addWidget(self.extensions_screen)
        self.stack.addWidget(self.credits_screen)
        self.stack.addWidget(self.manual_screen)

        # Conexión de las señales de navegación del menú principal:
        # cada señal cambia la pantalla visible (algunas refrescan datos antes).
        self.home_screen.teach_requested.connect(self._show_teach_screen)
        self.home_screen.manage_requested.connect(self._show_manage_screen)
        self.home_screen.translate_requested.connect(self._show_translate_screen)
        self.home_screen.extensions_requested.connect(self._show_extensions_screen)
        self.home_screen.credits_requested.connect(lambda: self.stack.setCurrentWidget(self.credits_screen))
        self.home_screen.manual_requested.connect(lambda: self.stack.setCurrentWidget(self.manual_screen))
        self.home_screen.exit_requested.connect(self.close)

        # Todas las pantallas secundarias comparten la misma acción de
        # "volver": mostrar de nuevo la pantalla de inicio.
        for screen in (
            self.teach_screen,
            self.manage_screen,
            self.translate_screen,
            self.extensions_screen,
            self.credits_screen,
            self.manual_screen,
        ):
            screen.back_requested.connect(lambda: self.stack.setCurrentWidget(self.home_screen))

        # Aplica el tema visual global a toda la ventana.
        self._apply_styles()

    def _apply_styles(self) -> None:
        """Aplica la hoja de estilos (QSS) con el tema oscuro de la aplicación.

        La hoja define colores, tipografías, bordes y estados (hover,
        pressed, selección) para todos los widgets usados en las pantallas:
        etiquetas, botones, combos, tablas, editores de texto, etc.
        """
        self.setStyleSheet(
            """
            QMainWindow {
                background-color: #0f172a;
            }
            QWidget {
                background-color: #0f172a;
                color: #e5e7eb;
                font-family: Segoe UI, Arial, sans-serif;
                font-size: 15px;
            }
            QLabel#TitleLabel {
                color: #f8fafc;
                font-size: 34px;
                font-weight: 700;
            }
            QLabel#SubtitleLabel {
                color: #cbd5e1;
                font-size: 17px;
            }
            QLabel#BodyLabel {
                color: #cbd5e1;
                font-size: 17px;
            }
            QLabel#TranslationLabel {
                color: #f8fafc;
                font-size: 44px;
                font-weight: 700;
                background-color: #111827;
                border: 1px solid #334155;
                border-radius: 14px;
                padding: 16px;
            }
            QPushButton {
                background-color: #2563eb;
                border: none;
                border-radius: 10px;
                color: white;
                min-height: 42px;
                padding: 8px 18px;
                font-size: 16px;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #1d4ed8;
            }
            QPushButton:pressed {
                background-color: #1e40af;
            }
            QComboBox {
                background-color: #111827;
                border: 1px solid #334155;
                border-radius: 8px;
                color: #e5e7eb;
                padding: 6px 10px;
                font-size: 15px;
                min-height: 30px;
            }
            QComboBox QAbstractItemView {
                background-color: #111827;
                border: 1px solid #334155;
                color: #e5e7eb;
                selection-background-color: #2563eb;
            }
            QTextBrowser {
                background-color: #111827;
                border: 1px solid #334155;
                border-radius: 12px;
                color: #e5e7eb;
                padding: 8px;
                font-size: 16px;
                selection-background-color: #2563eb;
                selection-color: white;
            }
            QPlainTextEdit {
                background-color: #111827;
                border: 1px solid #334155;
                border-radius: 10px;
                color: #e5e7eb;
                padding: 12px;
                font-family: Consolas, Courier New, monospace;
                font-size: 13px;
            }
            QLineEdit {
                background-color: #111827;
                border: 1px solid #334155;
                border-radius: 10px;
                color: #e5e7eb;
                min-height: 38px;
                padding: 6px 10px;
            }
            QTableWidget {
                background-color: #111827;
                border: 1px solid #334155;
                border-radius: 10px;
                color: #e5e7eb;
                gridline-color: #334155;
            }
            QHeaderView::section {
                background-color: #1e293b;
                color: #f8fafc;
                border: none;
                padding: 8px;
                font-weight: 700;
            }
            QTableWidget::item:selected {
                background-color: #2563eb;
                color: white;
            }
            QCheckBox {
                spacing: 8px;
            }
            QComboBox {
                background-color: #111827;
                border: 1px solid #334155;
                border-radius: 10px;
                color: #e5e7eb;
                min-height: 38px;
                padding: 6px 10px;
            }
            QLabel#CameraLabel {
                background-color: #020617;
                border: 1px solid #334155;
                border-radius: 12px;
                color: #94a3b8;
            }
            """
        )

    def _show_manage_screen(self) -> None:
        """Actualiza la tabla de señas y muestra la pantalla de gestión."""
        self.manage_screen.refresh()
        self.stack.setCurrentWidget(self.manage_screen)

    def _show_teach_screen(self) -> None:
        """Recarga las señas disponibles y muestra la pantalla de enseñanza."""
        self.teach_screen.refresh_signs()
        self.stack.setCurrentWidget(self.teach_screen)

    def _show_translate_screen(self) -> None:
        """Refresca las acciones de extensiones y muestra la traducción en vivo."""
        self.translate_screen.refresh_extension_actions()
        self.stack.setCurrentWidget(self.translate_screen)

    def _show_extensions_screen(self) -> None:
        """Actualiza el listado de extensiones y muestra su pantalla."""
        self.extensions_screen.refresh()
        self.stack.setCurrentWidget(self.extensions_screen)

    def closeEvent(self, event) -> None:
        """Libera los recursos al cerrar la ventana.

        Detiene las cámaras que pudieran estar activas en las pantallas de
        enseñanza y traducción, y apaga las extensiones cargadas, antes de
        delegar el cierre normal en la clase base de Qt.
        """
        # Detiene cámaras activas para liberar el dispositivo de video.
        self.teach_screen.stop_camera()
        self.translate_screen.stop_camera()
        # Notifica a las extensiones que la aplicación se está cerrando.
        self.context.extensions.shutdown()
        super().closeEvent(event)
