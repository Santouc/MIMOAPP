"""
Punto de entrada de la aplicación de escritorio MIMO.

Este módulo es el encargado de arrancar el traductor de lenguaje de señas
en su versión de escritorio. Su única responsabilidad es crear la aplicación
Qt (PySide6), instanciar la ventana principal (MainWindow) y ceder el control
al bucle de eventos de Qt hasta que el usuario cierre la aplicación.

Uso desde la línea de comandos:
    py desktop_app.py
"""

# Módulo estándar para acceder a los argumentos de la línea de comandos
# (sys.argv) y para interactuar con el intérprete de Python.
import sys

# QApplication es el objeto central de toda aplicación Qt: gestiona el bucle
# de eventos, los recursos gráficos y la comunicación con el sistema operativo.
from PySide6.QtWidgets import QApplication

# MainWindow es la ventana principal del traductor, definida en el paquete
# "app". Contiene toda la interfaz gráfica y la lógica de alto nivel.
from app.main_window import MainWindow


def main() -> int:
    """
    Inicializa y ejecuta la aplicación de escritorio.

    Pasos que realiza:
        1. Crea la instancia de QApplication pasándole los argumentos de la
           línea de comandos (Qt puede interpretar algunos de ellos).
        2. Crea y muestra la ventana principal del traductor.
        3. Inicia el bucle de eventos de Qt, que se mantiene activo hasta
           que el usuario cierra la aplicación.

    Returns:
        int: Código de salida devuelto por el bucle de eventos de Qt
             (0 indica una finalización normal).
    """
    # Se crea la aplicación Qt; debe existir exactamente una instancia
    # de QApplication antes de crear cualquier widget.
    application = QApplication(sys.argv)
    # Se construye la ventana principal con toda la interfaz del traductor.
    window = MainWindow()
    # Se hace visible la ventana en pantalla.
    window.show()
    # Se ejecuta el bucle de eventos; esta llamada bloquea hasta el cierre
    # de la aplicación y devuelve el código de salida correspondiente.
    return application.exec()

# Bloque de arranque: solo se ejecuta cuando el archivo se corre directamente
# (no cuando se importa como módulo). SystemExit propaga el código de salida
# de main() al sistema operativo.
if __name__ == "__main__":
    raise SystemExit(main())
