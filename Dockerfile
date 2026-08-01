# =============================================================================
# T.L.S — Traductor de Lengua de Señas
# Imagen Docker para ejecutar la aplicación de escritorio en cualquier sistema
# con soporte X11 (Linux nativo, Windows vía WSLg, macOS vía XQuartz).
# =============================================================================

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    QT_X11_NO_MITSHM=1 \
    QT_QPA_PLATFORM=xcb

# --- Dependencias del sistema -------------------------------------------------
# libgl1/libglib2.0-0        → OpenCV y MediaPipe
# libxcb-* / libxkbcommon    → plugin de plataforma Qt (PySide6) para X11
# libegl1/libfontconfig1     → renderizado Qt
# espeak / espeak-ng         → motor local de texto a voz (pyttsx3)
# libasound2 / alsa-utils    → salida de audio ALSA
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libegl1 \
    libfontconfig1 \
    libdbus-1-3 \
    libxkbcommon-x11-0 \
    libxcb-cursor0 \
    libxcb-icccm4 \
    libxcb-image0 \
    libxcb-keysyms1 \
    libxcb-randr0 \
    libxcb-render-util0 \
    libxcb-shape0 \
    libxcb-shm0 \
    libxcb-sync1 \
    libxcb-xfixes0 \
    libxcb-xinerama0 \
    libxcb-xkb1 \
    espeak \
    espeak-ng \
    libespeak1 \
    alsa-utils \
    libasound2 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Instalar dependencias Python primero (aprovecha la caché de capas de Docker)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el código del proyecto
COPY . .

CMD ["python", "desktop_app.py"]
