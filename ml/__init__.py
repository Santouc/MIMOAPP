"""
Módulo de TensorFlow para clasificación de señas

Paquete de aprendizaje automático (ML) del proyecto MIMO. Agrupa todo lo
relacionado con el reconocimiento de señas mediante redes neuronales Keras:

- clasificador.py: SignClassifier, clasificador de señas ESTÁTICAS
  (un frame de 21 landmarks 3D de MediaPipe por predicción).
- dynamic_classifier.py: DynamicSignClassifier, clasificador de señas
  DINÁMICAS (secuencias de frames con movimiento).
- train.py: script de entrenamiento del modelo estático.
- train_dynamic.py: script de entrenamiento del modelo dinámico.

Solo se reexporta SignClassifier como interfaz pública principal del paquete;
el resto de los módulos se importa de forma explícita cuando se necesita.
"""

# Reexportar la clase principal para permitir "from ml import SignClassifier"
from .clasificador import SignClassifier

# Interfaz pública del paquete
__all__ = ['SignClassifier']
