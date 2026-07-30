# Extensiones de T.L.S

Esta carpeta permite agregar funcionalidades a la app sin modificar el código principal.
Las extensiones se cargan automáticamente al iniciar la aplicación.

## Estructura de una extensión

Cada extensión es una carpeta con un archivo `extension.py`:

```
extensions/
  mi_extension/
    extension.py
```

## Contenido mínimo de `extension.py`

```python
NAME = "Mi extensión"
VERSION = "1.0"
DESCRIPTION = "Qué hace esta extensión."


class Extension:
    def setup(self, context) -> None:
        # Se llama una vez al iniciar la app.
        # `context` da acceso a los servicios:
        #   context.transcription  -> transcripción (get_output_text, learn_phrase, ...)
        #   context.signs          -> señas registradas
        #   context.training       -> entrenamiento de modelos
        #   context.paths          -> rutas de datos del proyecto
        self.context = context

    def translate_actions(self, screen) -> list:
        # Opcional: agrega botones y atajos de teclado a la pantalla
        # de traducción en vivo. `screen` es la pantalla (widgets Qt).
        from services.extension_service import TranslateAction

        def mi_accion() -> None:
            texto = self.context.transcription.get_output_text()
            print(f"Frase actual: {texto}")

        return [TranslateAction(label="Mi botón (M)", callback=mi_accion, key="M")]

    def shutdown(self) -> None:
        # Opcional: se llama al cerrar la app (liberar recursos).
        pass
```

Además de `setup`, `translate_actions` y `shutdown`, una extensión puede definir
`transcription_changed(state)`: se llama cada vez que cambia la transcripción en
vivo (`state.raw_text` son las letras, `state.output_text` la frase interpretada).
Útil para reaccionar automáticamente, como hace la extensión `voz`.

## Notas

- Todos los métodos de `Extension` son opcionales excepto que la clase debe existir.
- `TranslateAction.key` es una letra opcional para usar como atajo de teclado
  mientras la cámara está activa (no debe chocar con `C` ni `Q`, que usa la app).
- Si una extensión falla al cargar, la app sigue funcionando: el error queda
  registrado en `context.extensions.errors`.
- Puedes activar y desactivar extensiones desde el menú principal de la app
  (botón `Extensiones`); el estado se guarda en `data/extensions.json`.
- También puedes quitar una extensión eliminando su carpeta.

## Extensiones incluidas

- **voz**: dice automáticamente la frase transcrita al detectar una pausa de ~2
  segundos, con texto a voz local (pyttsx3). Agrega el botón `Repetir frase (V)`
  y el atajo `V` para repetirla manualmente.
