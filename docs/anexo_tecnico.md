# Anexo técnico de la aplicación T.L.S

Este documento resume la arquitectura actual después de la limpieza del proyecto.

## 1) Entrada principal
La aplicación principal es la app de escritorio PySide6:
```powershell
py desktop_app.py
```

`desktop_app.py` crea `QApplication`, instancia `MainWindow` y abre el flujo visual.

## 2) Estructura activa
- `app/`: UI PySide6, navegación, pantallas y widgets.
- `services/`: rutas, registro de señas, capturas, documentos, librería A-Z y entrenamiento.
- `core/`: detección de manos, procesamiento de imagen, datasets y preprocesamiento.
- `ml/`: clasificadores estático/dinámico y scripts de entrenamiento base.
- `visualization/`: cálculo/promedio de landmarks para resúmenes visuales.
- `utils/`: configuración y logging usados por ML y scripts archivados.
- `data/`: datasets, modelos, capturas pendientes, previews y registro.
- `docs/`: documentación de uso y especificación técnica.

## 3) Código archivado
El código que no forma parte del flujo principal fue movido a `archive/`:
- `archive/legacy_console/`: `main.py` y `teaching.py` antiguos.
- `archive/api/`: API experimental.
- `archive/ml_extras/`: utilidades ML auxiliares no conectadas a la app actual.

Estos archivos se conservan como referencia para evitar pérdida destructiva, pero no son necesarios para ejecutar la app principal.

## 4) Flujo de enseñanza
`TeachSignScreen` permite:
- seleccionar una seña registrada
- elegir captura estática o dinámica
- capturar con cámara
- terminar sesión con resumen visual
- aceptar o rechazar capturas pendientes
- entrenar automáticamente al aceptar

Las capturas dinámicas usan grabación tipo toggle: `T` inicia y `T` detiene.

## 5) Gestión de señas
`ManageSignsScreen` permite:
- agregar señas manualmente
- agregar alfabeto occidental `A-Z`
- seleccionar múltiples señas
- eliminar señas seleccionadas con sus datos asociados
- resetear todos los datos generados por la app

## 6) Traducción en vivo
`TranslateScreen` abre la cámara, detecta landmarks y usa clasificadores entrenados.

La UI muestra una traducción final única, aunque internamente pueda evaluar modelo estático y dinámico.

## 7) Datos persistentes
- Registro: `data/signs/signs.json`
- Dataset estático: `data/datasets/dataset_static.json`
- Dataset dinámico: `data/datasets/dataset_dynamic.json`
- Modelo estático: `data/models/model.h5`
- Modelo dinámico: `data/models/model_dynamic.h5`
- Labels: `data/models/labels.json`, `data/models/labels_dynamic.json`
- Detector de manos: `data/models/hand_landmarker.task`

`hand_landmarker.task` debe conservarse porque es el asset base de MediaPipe para detección de manos.

## 8) Reset seguro
`CaptureService.reset_all_data()` limpia:
- registro de señas
- datasets
- labels
- modelos entrenados
- capturas pendientes
- previews
- backups de modelos invalidados

No elimina `hand_landmarker.task`.
