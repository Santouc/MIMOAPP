# Manual de uso: aplicación de escritorio T.L.S

Este manual describe el flujo principal de la aplicación PySide6.

## Requisitos previos
- Windows 64-bit y cámara web funcional.
- Python compatible con las dependencias del proyecto.
- Dependencias instaladas desde la raíz del proyecto:
```powershell
py -m pip install -r requirements.txt
```

## Ejecutar la app
Desde la raíz del proyecto:
```powershell
py desktop_app.py
```

## Flujo recomendado
1. En `Gestionar señas`, agrega una seña manual o usa `Agregar alfabeto occidental`.
2. En `Enseñar seña`, captura muestras estáticas o dinámicas.
3. Revisa el resumen visual.
4. Acepta la captura para integrarla al dataset y entrenar automáticamente.
5. En `Traducir en vivo`, prueba el reconocimiento.

## Gestionar señas
Permite:
- agregar señas manualmente
- agregar todo el alfabeto occidental `A-Z`
- seleccionar varias señas
- eliminar señas seleccionadas con sus datos asociados
- resetear todos los datos generados por la app

Al eliminar señas, los modelos se reentrenan automáticamente con las señas restantes, por lo que el reconocimiento sigue funcionando sin pasos extra.

## Enseñar seña
Controles de cámara:
- `T` en modo estático: captura una muestra.
- `T` en modo dinámico: inicia o detiene una secuencia.
- `Q`: termina la sesión y muestra el resumen.

El resumen muestra:
- dibujo fijo para capturas estáticas
- animación promedio para capturas dinámicas

Las señas dinámicas reconocen tanto la forma de la mano como la trayectoria que recorre por la pantalla. Dos señas con la misma forma pero distinto movimiento (por ejemplo, trazar una línea o un círculo) se distinguen correctamente. Al grabar muestras dinámicas conviene repetir el movimiento de forma consistente. El sistema además se entrena automáticamente con ejemplos de "no seña" (transiciones entre letras, mano quieta, brazo desplazándose) para no confundir movimientos casuales con señas reales.

## Traducción en vivo
La pantalla muestra una única traducción final. Internamente puede usar modelo estático y dinámico, pero la interfaz simplifica el resultado para el usuario.

Controles de teclado durante la traducción:
- `C`: abre el diálogo para corregir/enseñar la interpretación de la frase actual.
- Las extensiones pueden agregar sus propios atajos (por ejemplo, `V` para repetir la frase en voz alta si la extensión `voz` está activa).

Con la extensión `voz` activa, la frase se dice automáticamente en voz alta cuando
dejas de hacer señas por unos 2 segundos; no necesitas presionar nada.

### Idioma de la transcripción

El selector `Idioma` (abajo a la derecha) define el diccionario que se usa para
convertir las letras deletreadas en frases. Idiomas disponibles: español, inglés,
portugués, francés, italiano y alemán. El idioma elegido queda guardado para las
próximas sesiones (en `data/transcription/rules.json`). Las frases enseñadas con
`C` se conservan al cambiar de idioma.

## Extensiones

La aplicación soporta extensiones: módulos opcionales que agregan funcionalidades sin
modificar el código principal. Se cargan automáticamente al iniciar la app desde la
carpeta `extensions/` en la raíz del proyecto.

### Cómo funciona el sistema

Al iniciar, la app revisa cada subcarpeta de `extensions/`. Si contiene un archivo
`extension.py`, lo importa, crea una instancia de su clase `Extension` y llama a su
método `setup(context)`. Si una extensión falla al cargar, la app sigue funcionando
con normalidad: el error queda registrado internamente y las demás extensiones se
cargan igual.

### Crear una extensión paso a paso

1. Crea una carpeta nueva dentro de `extensions/` con el nombre de tu extensión
   (minúsculas, sin espacios):

```
extensions/
  contador_frases/
    extension.py
```

2. Dentro de `extension.py`, define los metadatos y la clase `Extension`:

```python
NAME = "Contador de frases"
VERSION = "1.0"
DESCRIPTION = "Cuenta cuántas frases se han dicho en la sesión."


class Extension:
    def setup(self, context) -> None:
        # Se ejecuta una vez al iniciar la app.
        self.context = context
        self.contador = 0

    def translate_actions(self, screen) -> list:
        # Opcional: agrega botones/atajos a la pantalla de traducción.
        from services.extension_service import TranslateAction

        def contar() -> None:
            texto = self.context.transcription.get_output_text()
            if texto:
                self.contador += 1
                screen.transcription_status_label.setText(
                    f"Transcripción: frase #{self.contador} contada"
                )

        return [TranslateAction(label="Contar frase (N)", callback=contar, key="N")]

    def shutdown(self) -> None:
        # Opcional: se ejecuta al cerrar la app.
        pass
```

3. Reinicia la aplicación. La extensión se carga sola y su botón aparece en la
   pantalla `Traducir en vivo`.

### Qué puede hacer una extensión

La clase `Extension` puede definir estos métodos (todos opcionales):

| Método | Cuándo se llama | Para qué sirve |
|---|---|---|
| `setup(context)` | Al iniciar la app | Guardar referencia a los servicios e inicializar recursos |
| `translate_actions(screen)` | Al construir la pantalla de traducción | Agregar botones y atajos de teclado |
| `transcription_changed(state)` | Cada vez que cambia la transcripción en vivo | Reaccionar automáticamente a nuevas letras o frases (`state.raw_text`, `state.output_text`) |
| `shutdown()` | Al cerrar la app | Liberar recursos (hilos, archivos, audio) |

A través de `context`, la extensión accede a los servicios de la app:

| Servicio | Acceso | Ejemplos de uso |
|---|---|---|
| Transcripción | `context.transcription` | `get_output_text()`, `get_raw_text()`, `learn_phrase(crudo, frase)` |
| Señas | `context.signs` | Consultar señas registradas |
| Entrenamiento | `context.training` | Reentrenar modelos |
| Rutas | `context.paths` | Rutas de datos del proyecto (`data_dir`, `models_dir`, etc.) |

Cada acción de `translate_actions` es un `TranslateAction` con:
- `label`: texto del botón que aparecerá en la pantalla.
- `callback`: función sin argumentos que se ejecuta al presionar el botón o el atajo.
- `key`: letra opcional para usar como atajo de teclado. No debe chocar con las
  teclas reservadas de la app (`C` corrige interpretación, `Q` termina sesión en
  entrenamiento, `T` captura en entrenamiento).

### Administrar extensiones

Desde el menú principal, el botón `Extensiones` abre la pantalla de gestión, que
muestra todas las extensiones detectadas con su nombre, versión, descripción y estado.
Desde ahí puedes:

- **Activar / Desactivar** cada extensión con un clic; el cambio aplica de inmediato
  (sin reiniciar) y queda guardado en `data/extensions.json` para próximas sesiones.
- **Actualizar lista** para volver a detectar extensiones nuevas copiadas a la carpeta.

También puedes administrarlas manualmente:

- **Instalar**: copia la carpeta de la extensión dentro de `extensions/` y presiona
  `Actualizar lista` (o reinicia la app).
- **Quitar del todo**: elimina la carpeta de la extensión.
- **Dependencias**: si la extensión necesita una librería de Python, instálala con
  `py -m pip install <libreria>` antes de usarla. Una buena extensión debe manejar
  la ausencia de sus dependencias sin romper la app (ver `extensions/voz/extension.py`
  como referencia: si `pyttsx3` no está instalado, muestra un aviso en vez de fallar).

### Solución de problemas

- **El botón de la extensión no aparece**: verifica que la carpeta esté dentro de
  `extensions/`, que el archivo se llame exactamente `extension.py` y que contenga
  una clase llamada `Extension`.
- **La extensión carga pero falla al usarla**: revisa la consola desde donde
  ejecutaste `py desktop_app.py`; los errores de callbacks se registran internamente
  en `context.extensions.errors`.
- **El atajo de teclado no responde**: la ventana de traducción debe tener el foco
  (haz clic sobre ella) y la letra no debe estar reservada por la app.

### Extensiones incluidas

| Extensión | Descripción | Botón / atajo |
|---|---|---|
| `voz` | Dice automáticamente la frase transcrita al detectar una pausa de ~2 segundos (texto a voz local con pyttsx3, sin internet) | `Repetir frase (V)` / tecla `V` para repetirla manualmente |

## Archivos importantes
- Entrada principal: `desktop_app.py`
- Registro: `data/signs/signs.json`
- Datasets: `data/datasets/dataset_static.json`, `data/datasets/dataset_dynamic.json`
- Modelos: `data/models/model.h5`, `data/models/model_dynamic.h5`
- Labels: `data/models/labels.json`, `data/models/labels_dynamic.json`
- Detector de manos: `data/models/hand_landmarker.task`
- Extensiones: `extensions/` (una carpeta por extensión, ver sección Extensiones)
- Memoria de transcripción: `data/transcription/memory.json`, reglas: `data/transcription/rules.json`

## Código archivado
Los scripts antiguos de consola se conservan en `archive/legacy_console/` para referencia y pruebas manuales:
```powershell
py archive/legacy_console/teaching.py
py archive/legacy_console/main.py
```

La app recomendada sigue siendo `desktop_app.py`.
