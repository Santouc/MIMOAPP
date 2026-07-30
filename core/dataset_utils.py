#!/usr/bin/env python3
"""
Módulo centralizado de utilidades para datasets
Carga, guardado y validación de datasets de landmarks
"""

import json
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


def load_dataset(filepath: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Carga dataset desde archivo JSON
    
    Args:
        filepath: Ruta al archivo dataset JSON
        
    Returns:
        Tuple (X, y, labels)
        - X: Array de landmarks shape (N, 21, 3)
        - y: Array de etiquetas shape (N,)
        - labels: Lista de nombres de clases
        
    Raises:
        FileNotFoundError: Si el archivo no existe
        ValueError: Si el formato del dataset es inválido
        
    Examples:
        >>> X, y, labels = load_dataset("data/datasets/dataset_static.json")
        >>> print(f"Loaded {X.shape[0]} samples, {len(labels)} classes")
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Dataset no encontrado: {filepath}")
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        
        # Validar estructura
        if 'X' not in dataset or 'y' not in dataset or 'labels' not in dataset:
            raise ValueError("Dataset inválido: faltan campos X, y o labels")
        
        X = np.array(dataset['X'], dtype=np.float32)
        y = np.array(dataset['y'], dtype=np.int32)
        labels = dataset['labels']
        
        # Asegurar shape correcto: (N, 21, 3)
        if len(X.shape) == 4 and X.shape[1] == 1:
            X = X.squeeze(axis=1)  # Remover dimensión extra
        
        if len(X.shape) != 3 or X.shape[1] != 21 or X.shape[2] != 3:
            raise ValueError(f"Shape inválido de X: {X.shape}, esperado (N, 21, 3)")
        
        logger.info(f"Dataset cargado: {X.shape[0]} muestras, {len(labels)} clases")
        return X, y, labels
        
    except json.JSONDecodeError as e:
        raise ValueError(f"Error decodificando JSON: {e}")
    except Exception as e:
        raise ValueError(f"Error cargando dataset: {e}")


def save_dataset(X: np.ndarray, y: np.ndarray, labels: List[str], 
                 filepath: str, metadata: Optional[Dict] = None) -> bool:
    """
    Guarda dataset en formato JSON
    
    Args:
        X: Array de landmarks shape (N, 21, 3)
        y: Array de etiquetas shape (N,)
        labels: Lista de nombres de clases
        filepath: Ruta donde guardar el archivo
        metadata: Diccionario con metadatos adicionales
        
    Returns:
        True si se guardó exitosamente
        
    Examples:
        >>> save_dataset(X, y, labels, "data/datasets/my_dataset.json")
        True
    """
    filepath = Path(filepath)
    
    # Crear directorio si no existe
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # Validar shapes
    if len(X.shape) != 3 or X.shape[1] != 21 or X.shape[2] != 3:
        raise ValueError(f"Shape inválido de X: {X.shape}, esperado (N, 21, 3)")
    
    if len(y) != X.shape[0]:
        raise ValueError(f"Shape mismatch: X tiene {X.shape[0]} muestras, y tiene {len(y)}")
    
    # Metadatos por defecto
    default_metadata = {
        "num_samples": int(X.shape[0]),
        "num_classes": len(labels),
        "landmark_format": "mediapipe_21_3d",
        "coordinate_system": "normalized"
    }
    
    if metadata:
        default_metadata.update(metadata)
    
    dataset = {
        "X": X.tolist(),
        "y": y.tolist(),
        "labels": labels,
        "metadata": default_metadata
    }
    
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Dataset guardado en {filepath}")
        return True
        
    except Exception as e:
        logger.error(f"Error guardando dataset: {e}")
        return False


def validate_dataset(X: np.ndarray, y: np.ndarray, labels: List[str]) -> Dict[str, any]:
    """
    Valida integridad de un dataset
    
    Args:
        X: Array de landmarks
        y: Array de etiquetas
        labels: Lista de nombres de clases
        
    Returns:
        Diccionario con resultados de validación
    """
    validation = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "stats": {}
    }
    
    # Validar shapes
    if len(X.shape) != 3 or X.shape[1] != 21 or X.shape[2] != 3:
        validation["valid"] = False
        validation["errors"].append(f"Shape inválido de X: {X.shape}")
    
    if len(y) != X.shape[0]:
        validation["valid"] = False
        validation["errors"].append(f"Shape mismatch: X={X.shape[0]}, y={len(y)}")
    
    # Validar rango de coordenadas
    if X.min() < 0 or X.max() > 10:  # Después de normalización debería estar en rango razonable
        validation["warnings"].append(f"Coordenadas fuera de rango: [{X.min():.3f}, {X.max():.3f}]")
    
    # Validar etiquetas
    unique_labels = set(y)
    if len(unique_labels) != len(labels):
        validation["warnings"].append(f"No todas las clases están presentes: {len(unique_labels)}/{len(labels)}")
    
    # Estadísticas
    validation["stats"] = {
        "num_samples": int(X.shape[0]),
        "num_classes": len(labels),
        "classes_present": int(len(unique_labels)),
        "coordinate_range": [float(X.min()), float(X.max())]
    }
    
    return validation


def merge_datasets(datasets: List[Tuple[np.ndarray, np.ndarray, List[str]]]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Fusiona múltiples datasets en uno solo
    
    Args:
        datasets: Lista de tuplas (X, y, labels)
        
    Returns:
        Tuple (X_merged, y_merged, labels_merged)
    """
    if not datasets:
        raise ValueError("No datasets proporcionados")
    
    # Usar labels del primer dataset
    _, _, labels = datasets[0]
    
    X_list = []
    y_list = []
    
    for X, y, _ in datasets:
        X_list.append(X)
        y_list.append(y)
    
    X_merged = np.concatenate(X_list, axis=0)
    y_merged = np.concatenate(y_list, axis=0)
    
    # Reindexar etiquetas para que sean consistentes
    y_merged = np.array([labels.index(label) for label in labels])
    
    logger.info(f"Datasets fusionados: {X_merged.shape[0]} muestras totales")
    return X_merged, y_merged, labels


def split_dataset(X: np.ndarray, y: np.ndarray, 
                  train_ratio: float = 0.8, 
                  val_ratio: float = 0.1,
                  test_ratio: float = 0.1,
                  stratify: bool = True) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Divide dataset en train/val/test
    
    Args:
        X: Array de landmarks
        y: Array de etiquetas
        train_ratio: Proporción para entrenamiento
        val_ratio: Proporción para validación
        test_ratio: Proporción para prueba
        stratify: Si hacer stratified split
        
    Returns:
        Diccionario con splits: {'train': (X_train, y_train), ...}
    """
    from sklearn.model_selection import train_test_split
    
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 0.01:
        raise ValueError("Las proporciones deben sumar 1.0")
    
    # Primero separar train del resto
    if stratify:
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=(val_ratio + test_ratio), stratify=y, random_state=42
        )
    else:
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=(val_ratio + test_ratio), random_state=42
        )
    
    # Luego separar val y test
    val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)
    
    if stratify:
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=(1 - val_ratio_adjusted), stratify=y_temp, random_state=42
        )
    else:
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=(1 - val_ratio_adjusted), random_state=42
        )
    
    return {
        'train': (X_train, y_train),
        'val': (X_val, y_val),
        'test': (X_test, y_test)
    }


if __name__ == "__main__":
    # Prueba básica
    print("=== Prueba de Dataset Utils ===")
    
    # Crear dataset de prueba
    X_test = np.random.rand(100, 21, 3).astype(np.float32)
    y_test = np.random.randint(0, 5, 100)
    labels_test = ["A", "B", "C", "D", "E"]
    
    # Guardar
    save_dataset(X_test, y_test, labels_test, "data/datasets/test_dataset.json")
    
    # Cargar
    X_loaded, y_loaded, labels_loaded = load_dataset("data/datasets/test_dataset.json")
    print(f"Loaded: {X_loaded.shape[0]} samples, {len(labels_loaded)} classes")
    
    # Validar
    validation = validate_dataset(X_loaded, y_loaded, labels_loaded)
    print(f"Validation: {validation['valid']}")
    print(f"Stats: {validation['stats']}")
    
    # Limpiar archivo de prueba
    Path("data/datasets/test_dataset.json").unlink()
    
    print("\n✅ Prueba completada exitosamente")
