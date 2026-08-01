# Ejecutar T.L.S con Docker

Docker empaqueta la aplicación con **todas sus dependencias** (Python, PySide6,
TensorFlow, MediaPipe, OpenCV) en una imagen aislada. Así se evita instalar
nada manualmente y los problemas de "en mi máquina no funciona".

> **Importante:** T.L.S es una aplicación de **escritorio con cámara**, no un
> servidor web. El contenedor necesita acceso a la *pantalla* y a la *cámara*
> del sistema anfitrión. El soporte varía según el sistema operativo:

| Sistema | Ventana (GUI) | Cámara | Voz | Dificultad |
| ------- | ------------- | ------ | --- | ---------- |
| **Linux** | ✅ X11 nativo | ✅ `/dev/video0` | ✅ PulseAudio | Baja |
| **Windows 10/11** | ✅ vía WSLg | ⚠️ requiere `usbipd` | ⚠️ vía WSLg | Media |
| **macOS** | ⚠️ vía XQuartz | ❌ no soportada | ❌ | Alta |

Para Windows y macOS, si Docker resulta complejo, la alternativa más simple
sigue siendo la instalación nativa (`pip install -r requirements.txt`).

---

## Linux (caso recomendado)

### Requisitos

- Docker Engine + plugin Compose ([guía oficial](https://docs.docker.com/engine/install/))
- Servidor gráfico X11 (o XWayland, incluido en casi todas las distros)
- Una cámara web visible como `/dev/video*`

### Opción A — Script automático

```bash
git clone https://github.com/Santouc/MIMOAPP.git
cd MIMOAPP
bash docker-run.sh
```

El script concede acceso X11, verifica la cámara, construye la imagen y
ejecuta la app. La primera construcción tarda varios minutos (TensorFlow pesa
~600 MB).

### Opción B — Manual

```bash
# Permitir que el contenedor abra ventanas en tu pantalla
xhost +local:docker

# Construir y ejecutar
docker compose up --build
```

### Si tu cámara no es /dev/video0

Lista tus dispositivos y edita `docker-compose.yml`:

```bash
ls /dev/video*
```

```yaml
devices:
  - /dev/video2:/dev/video0   # la app siempre usa el índice 0 interno
```

### Wayland puro (sin XWayland)

Si tu sesión es Wayland sin XWayland, exporta antes:

```bash
export DISPLAY=:0
xhost +local:docker
```

La mayoría de las distros con Wayland (Ubuntu, Fedora) incluyen XWayland,
por lo que el flujo normal funciona sin cambios.

---

## Windows 10/11

Docker Desktop en Windows corre sobre **WSL2**, y **WSLg** (incluido en
Windows 11 y Windows 10 actualizado) provee pantalla X11 automáticamente.

### Pasos

1. Instala [Docker Desktop](https://docs.docker.com/desktop/install/windows-install/) con backend WSL2.
2. Desde PowerShell:

```powershell
git clone https://github.com/Santouc/MIMOAPP.git
cd MIMOAPP
docker compose up --build
```

La ventana aparece gracias a WSLg (el socket X11 de WSLg ya está montado en
`/tmp/.X11-unix` dentro de los contenedores).

### Cámara en Windows (limitación real)

Los contenedores Linux **no ven las cámaras USB de Windows** directamente.
Se necesita [usbipd-win](https://github.com/dorssel/usbipd-win) para "pasar"
la cámara USB a WSL2:

```powershell
winget install usbipd
usbipd list                      # anota el BUSID de tu cámara
usbipd bind --busid <BUSID>
usbipd attach --wsl --busid <BUSID>
```

Después verifica dentro de WSL que exista `/dev/video0` y vuelve a ejecutar
`docker compose up`. Si esto resulta demasiado engorroso, en Windows es más
práctico ejecutar la app de forma nativa con Python.

---

## macOS

1. Instala [Docker Desktop](https://docs.docker.com/desktop/install/mac-install/) y [XQuartz](https://www.xquartz.org/).
2. En XQuartz → Preferencias → Seguridad → habilita *"Allow connections from network clients"* y reinicia XQuartz.
3. En una terminal:

```bash
xhost +localhost
DISPLAY=host.docker.internal:0 docker compose up --build
```

> **Limitación:** Docker en macOS **no puede acceder a la cámara** del Mac
> (no existe passthrough de dispositivos de video). La app abrirá y podrás
> navegar la interfaz, pero la traducción en vivo no tendrá imagen.
> En macOS la recomendación es la instalación nativa con Python.

---

## Persistencia de datos

El `docker-compose.yml` monta `./data` del repositorio dentro del contenedor:

- Las señas que enseñes, los datasets y los modelos reentrenados **se guardan
  en tu disco**, no se pierden al cerrar el contenedor.
- Si ya tienes un `model.h5` entrenado en `data/models/`, el contenedor lo usa
  directamente.

## Comandos útiles

```bash
docker compose up            # ejecutar (sin reconstruir)
docker compose up --build    # reconstruir tras cambios en el código
docker compose down          # detener y eliminar el contenedor
docker image rm tls-app      # borrar la imagen (libera ~3 GB)
```

## Solución de problemas

| Síntoma | Causa probable | Solución |
| ------- | -------------- | -------- |
| `could not connect to display` | Falta permiso X11 | `xhost +local:docker` |
| `qt.qpa.plugin: Could not load the Qt platform plugin "xcb"` | Falta socket X11 | Verifica el volumen `/tmp/.X11-unix` y la variable `DISPLAY` |
| Cámara no disponible en la app | Dispositivo no montado | Revisa `ls /dev/video*` y la sección `devices` del compose |
| No se escucha la voz | PulseAudio no montado | La voz es opcional; revisa el volumen del socket `pulse` |
| Imagen tarda mucho en construir | TensorFlow ~600 MB | Normal la primera vez; las siguientes usan caché |
