#!/usr/bin/env bash
# =============================================================================
# T.L.S — Lanzador Docker para Linux
# Uso:  bash docker-run.sh
# =============================================================================
set -e

# 1. Verificar Docker
if ! command -v docker &> /dev/null; then
    echo "ERROR: Docker no está instalado. Instálalo desde https://docs.docker.com/engine/install/"
    exit 1
fi

# 2. Permitir que el contenedor use la pantalla local
if command -v xhost &> /dev/null; then
    xhost +local:docker > /dev/null
    echo "[OK] Acceso X11 concedido a Docker"
else
    echo "[AVISO] 'xhost' no encontrado. Si la ventana no aparece, instala x11-xserver-utils"
fi

# 3. Detectar cámara
if [ -e /dev/video0 ]; then
    echo "[OK] Cámara detectada en /dev/video0"
else
    echo "[AVISO] No se encontró /dev/video0. La app abrirá pero sin cámara."
    echo "        Revisa tus dispositivos con: ls /dev/video*"
fi

# 4. Construir y ejecutar
echo "[...] Construyendo imagen (la primera vez tarda varios minutos)"
docker compose up --build

# 5. Revocar acceso X11 al salir
if command -v xhost &> /dev/null; then
    xhost -local:docker > /dev/null
fi
