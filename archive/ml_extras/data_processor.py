#!/usr/bin/env python3
"""
Pipeline de limpieza y augmentación de dataset para entrenamiento de señas
Transforma dataset crudo X ∈ ℝ^(N,21,3) → X_clean ∈ ℝ^(N',21,3)
"""

import numpy as np
from typing import Tuple, List, Dict
from collections import Counter
from pathlib import Path

from utils.logger import get_logger
from core.dataset_utils import load_dataset as load_dataset_centralized, save_dataset

logger = get_logger(__name__)

class DatasetProcessor:
    """Pipeline completo de procesamiento de dataset"""
    
    def __init__(self):
        """Inicializa el procesador de dataset"""
        self.processing_stats = {}
    
    def load_dataset(self, filepath: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Carga dataset usando módulo centralizado
        
        Args:
            filepath: Ruta al archivo dataset
            
        Returns:
            Tuple (X, y, labels)
        """
        try:
            return load_dataset_centralized(filepath)
        except Exception as e:
            logger.error(f"Error cargando dataset: {e}")
            return np.array([]), np.array([]), []
    
    def clean_dataset(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Remueve muestras inválidas (shape incorrecto, NaN, valores fuera de rango)
        
        Args:
            X: Array (N, 21, 3) de landmarks
            y: Array (N,) de etiquetas
            
        Returns:
            Tuple (X_clean, y_clean)
        """
        logger.info("Iniciando limpieza de dataset...")
        
        X_clean = []
        y_clean = []
        invalid_count = 0
        
        for i, (xi, yi) in enumerate(zip(X, y)):
            xi = np.array(xi)
            
            # Validar shape
            if xi.shape != (21, 3):
                invalid_count += 1
                continue
            
            # Validar NaN o infinitos
            if not np.all(np.isfinite(xi)):
                invalid_count += 1
                continue
            
            # Validar rango de coordenadas (MediaPipe usa [0, 1])
            if not np.all((xi >= 0) & (xi <= 1)):
                invalid_count += 1
                continue
            
            X_clean.append(xi)
            y_clean.append(yi)
        
        X_clean = np.array(X_clean, dtype=np.float32)
        y_clean = np.array(y_clean, dtype=np.int32)
        
        logger.info(f"Limpieza completada: {len(X)} → {len(X_clean)} muestras")
        logger.info(f"Muestras inválidas removidas: {invalid_count}")
        
        self.processing_stats['cleaning'] = {
            'original_samples': len(X),
            'clean_samples': len(X_clean),
            'invalid_removed': invalid_count
        }
        
        return X_clean, y_clean
    
    def remove_duplicates(self, X: np.ndarray, y: np.ndarray, threshold: float = 1e-3) -> Tuple[np.ndarray, np.ndarray]:
        """
        Remueve muestras casi idénticas (frames consecutivos duplicados)
        
        Args:
            X: Array (N, 21, 3) de landmarks
            y: Array (N,) de etiquetas
            threshold: Umbral de distancia euclidiana para considerar duplicados
            
        Returns:
            Tuple (X_unique, y_unique)
        """
        logger.info(f"Removiendo duplicados con threshold {threshold}...")
        
        X_unique = []
        y_unique = []
        duplicate_count = 0
        
        prev_sample = None
        
        for i, (xi, yi) in enumerate(zip(X, y)):
            if prev_sample is None:
                X_unique.append(xi)
                y_unique.append(yi)
                prev_sample = xi
                continue
            
            # Calcular distancia euclidiana con muestra anterior
            dist = np.linalg.norm(xi - prev_sample)
            
            if dist > threshold:
                X_unique.append(xi)
                y_unique.append(yi)
                prev_sample = xi
            else:
                duplicate_count += 1
        
        X_unique = np.array(X_unique, dtype=np.float32)
        y_unique = np.array(y_unique, dtype=np.int32)
        
        logger.info(f"Duplicados removidos: {duplicate_count}")
        logger.info(f"Muestras únicas: {len(X_unique)}")
        
        self.processing_stats['duplicates'] = {
            'duplicates_removed': duplicate_count,
            'unique_samples': len(X_unique)
        }
        
        return X_unique, y_unique
    
    def add_gaussian_noise(self, X: np.ndarray, sigma: float = 0.01) -> np.ndarray:
        """
        Añade ruido gaussiano a los landmarks
        
        Args:
            X: Array (N, 21, 3) de landmarks
            sigma: Desviación estándar del ruido
            
        Returns:
            Array con ruido añadido
        """
        noise = np.random.normal(0, sigma, X.shape)
        X_noisy = X + noise
        
        # Asegurar que los valores permanezcan en rango válido [0, 1]
        X_noisy = np.clip(X_noisy, 0, 1)
        
        return X_noisy
    
    def rotate_hand(self, X: np.ndarray, angle_range: float = 0.2) -> np.ndarray:
        """
        Aplica rotación aleatoria en el plano XY a las manos
        
        Args:
            X: Array (N, 21, 3) de landmarks
            angle_range: Rango de ángulo en radianes
            
        Returns:
            Array rotado
        """
        X_rotated = X.copy()
        
        for i in range(len(X)):
            angle = np.random.uniform(-angle_range, angle_range)
            
            # Matriz de rotación en el plano XY
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            R = np.array([
                [cos_a, -sin_a, 0],
                [sin_a,  cos_a, 0],
                [0,      0,     1]
            ])
            
            # Aplicar rotación
            X_rotated[i] = X[i] @ R
        
        # Asegurar rango válido
        X_rotated = np.clip(X_rotated, 0, 1)
        
        return X_rotated
    
    def scale_hand(self, X: np.ndarray, scale_range: Tuple[float, float] = (0.9, 1.1)) -> np.ndarray:
        """
        Aplica scaling jitter a las manos
        
        Args:
            X: Array (N, 21, 3) de landmarks
            scale_range: Rango de factor de escala (min, max)
            
        Returns:
            Array escalado
        """
        X_scaled = X.copy()
        
        for i in range(len(X)):
            factor = np.random.uniform(scale_range[0], scale_range[1])
            X_scaled[i] = X[i] * factor
        
        # Asegurar rango válido
        X_scaled = np.clip(X_scaled, 0, 1)
        
        return X_scaled
    
    def translate_hand(self, X: np.ndarray, shift_range: float = 0.02) -> np.ndarray:
        """
        Aplica traslación aleatoria a las manos
        
        Args:
            X: Array (N, 21, 3) de landmarks
            shift_range: Rango máximo de traslación
            
        Returns:
            Array trasladado
        """
        X_translated = X.copy()
        
        for i in range(len(X)):
            shift = np.random.uniform(-shift_range, shift_range, (1, 3))
            X_translated[i] = X[i] + shift
        
        # Asegurar rango válido
        X_translated = np.clip(X_translated, 0, 1)
        
        return X_translated
    
    def augment_dataset(self, X: np.ndarray, y: np.ndarray, 
                       augment_factor: int = 4) -> Tuple[np.ndarray, np.ndarray]:
        """
        Aplica augmentación completa al dataset
        
        Args:
            X: Array (N, 21, 3) de landmarks
            y: Array (N,) de etiquetas
            augment_factor: Factor de augmentación (muestras adicionales por original)
            
        Returns:
            Tuple (X_augmented, y_augmented)
        """
        logger.info(f"Iniciando augmentación con factor {augment_factor}...")
        
        X_aug = [X]
        y_aug = [y]
        
        # Generar variantes augmentadas
        for i in range(augment_factor):
            if i == 0:
                # Ruido gaussiano
                X_aug.append(self.add_gaussian_noise(X))
            elif i == 1:
                # Rotación
                X_aug.append(self.rotate_hand(X))
            elif i == 2:
                # Scaling
                X_aug.append(self.scale_hand(X))
            elif i == 3:
                # Traslation
                X_aug.append(self.translate_hand(X))
            
            y_aug.append(y)
        
        # Concatenar todas las variantes
        X_final = np.vstack(X_aug)
        y_final = np.hstack(y_aug)
        
        logger.info(f"Augmentación completada: {len(X)} → {len(X_final)} muestras")
        
        self.processing_stats['augmentation'] = {
            'original_samples': len(X),
            'augmented_samples': len(X_final),
            'augmentation_factor': augment_factor
        }
        
        return X_final, y_final
    
    def balance_dataset(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Balancea clases para tener distribución uniforme
        
        Args:
            X: Array (N, 21, 3) de landmarks
            y: Array (N,) de etiquetas
            
        Returns:
            Tuple (X_balanced, y_balanced)
        """
        logger.info("Balanceando clases...")
        
        counts = Counter(y)
        max_count = max(counts.values())
        
        logger.info(f"Distribución original: {dict(counts)}")
        logger.info(f"Balanceando a {max_count} muestras por clase")
        
        X_bal = []
        y_bal = []
        
        for cls in counts:
            idxs = np.where(y == cls)[0]
            
            # Repetir muestras hasta alcanzar max_count
            for i in range(max_count):
                xi = X[idxs[i % len(idxs)]]
                X_bal.append(xi)
                y_bal.append(cls)
        
        X_balanced = np.array(X_bal, dtype=np.float32)
        y_balanced = np.array(y_bal, dtype=np.int32)
        
        # Mezclar dataset balanceado
        shuffle_idx = np.random.permutation(len(X_balanced))
        X_balanced = X_balanced[shuffle_idx]
        y_balanced = y_balanced[shuffle_idx]
        
        balanced_counts = Counter(y_balanced)
        logger.info(f"Distribución balanceada: {dict(balanced_counts)}")
        
        self.processing_stats['balancing'] = {
            'original_distribution': dict(counts),
            'balanced_distribution': dict(balanced_counts),
            'final_samples': len(X_balanced)
        }
        
        return X_balanced, y_balanced
    
    def preprocess_batch(self, X: np.ndarray) -> np.ndarray:
        """
        Preprocesamiento batch usando módulo centralizado
        
        Args:
            X: Array (N, 21, 3) de landmarks
            
        Returns:
            Array normalizado (N, 21, 3)
        """
        try:
            # Usar módulo centralizado
            normalized = normalize_landmarks(X)
            return normalized
        except Exception as e:
            logger.error(f"Error en preprocess_batch: {e}")
            return X.astype(np.float32)
    
    def process_full_pipeline(self, input_path: str, output_path: str = "processed_dataset.json",
                            augment_factor: int = 4, duplicate_threshold: float = 1e-3) -> Dict:
        """
        Ejecuta pipeline completo de procesamiento
        
        Args:
            input_path: Ruta del dataset crudo
            output_path: Ruta del dataset procesado
            augment_factor: Factor de augmentación
            duplicate_threshold: Umbral para remover duplicados
            
        Returns:
            Diccionario con estadísticas del procesamiento
        """
        logger.info("=== INICIANDO PIPELINE COMPLETO DE PROCESAMIENTO ===")
        
        # 1. Cargar dataset
        X, y, labels = self.load_dataset(input_path)
        if len(X) == 0:
            raise ValueError("No se pudo cargar el dataset")
        
        # 2. Limpieza
        X, y = self.clean_dataset(X, y)
        
        # 3. Remover duplicados
        X, y = self.remove_duplicates(X, y, threshold=duplicate_threshold)
        
        # 4. Preprocesamiento
        X = self.preprocess_batch(X)
        
        # 5. Augmentación
        X, y = self.augment_dataset(X, y, augment_factor=augment_factor)
        
        # 6. Balanceo
        X, y = self.balance_dataset(X, y)
        
        # 7. Guardar dataset procesado
        self.save_processed_dataset(X, y, labels, output_path)
        
        # 8. Validación final
        self.validate_final_dataset(X, y, labels)
        
        logger.info("=== PIPELINE COMPLETADO ===")
        
        return self.processing_stats
    
    def save_processed_dataset(self, X: np.ndarray, y: np.ndarray, 
                            labels: List[str], filepath: str) -> bool:
        """
        Guarda dataset procesado
        
        Args:
            X: Array (N, 21, 3) de landmarks procesados
            y: Array (N,) de etiquetas
            labels: Lista de nombres de clases
            filepath: Ruta de guardado
            
        Returns:
            True si se guardó exitosamente
        """
        try:
            dataset = {
                "X": X.tolist(),
                "y": y.tolist(),
                "labels": labels,
                "metadata": {
                    "num_samples": len(X),
                    "num_classes": len(labels),
                    "landmark_format": "mediapipe_21_3d_normalized",
                    "coordinate_system": "wrist_centered_index_scaled",
                    "processing_stats": self.processing_stats,
                    "processing_timestamp": np.datetime64('now').astype(int)
                }
            }
            
            with open(filepath, 'w') as f:
                json.dump(dataset, f, indent=2)
            
            logger.info(f"Dataset procesado guardado en {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error guardando dataset procesado: {e}")
            return False
    
    def validate_final_dataset(self, X: np.ndarray, y: np.ndarray, labels: List[str]):
        """
        Valida calidad del dataset final
        
        Args:
            X: Array (N, 21, 3) de landmarks
            y: Array (N,) de etiquetas
            labels: Lista de nombres de clases
        """
        logger.info("=== VALIDACIÓN FINAL ===")
        
        # Shape validation
        logger.info(f"Shape X: {X.shape}")
        logger.info(f"Shape y: {y.shape}")
        logger.info(f"Clases únicas: {len(set(y))}")
        
        # Distribution validation
        counts = Counter(y)
        logger.info(f"Distribución final: {dict(counts)}")
        
        # Quality validation
        nan_count = np.sum(np.isnan(X))
        inf_count = np.sum(np.isinf(X))
        out_of_range = np.sum((X < 0) | (X > 1))
        
        logger.info(f"Valores NaN: {nan_count}")
        logger.info(f"Valores infinitos: {inf_count}")
        logger.info(f"Valores fuera de rango: {out_of_range}")
        
        # Statistics
        mean_val = np.mean(X)
        std_val = np.std(X)
        min_val = np.min(X)
        max_val = np.max(X)
        
        logger.info(f"Estadísticas - Mean: {mean_val:.4f}, Std: {std_val:.4f}")
        logger.info(f"Rango: [{min_val:.4f}, {max_val:.4f}]")
        
        # Balance check
        balance_ratio = min(counts.values()) / max(counts.values())
        logger.info(f"Ratio de balance: {balance_ratio:.3f} (1.0 = perfect balance)")
        
        if balance_ratio > 0.95:
            logger.info("✅ Dataset bien balanceado")
        else:
            logger.warning("⚠️ Dataset podría necesitar más balance")
        
        logger.info("=== VALIDACIÓN COMPLETADA ===")


def main():
    """Función principal para ejecutar pipeline de procesamiento"""
    print("=== Pipeline de Procesamiento de Dataset ===")
    print("Este sistema limpiará, augmentará y balanceará el dataset para entrenamiento.")
    print()
    
    # Crear procesador
    processor = DatasetProcessor()
    
    # Ejecutar pipeline completo
    try:
        stats = processor.process_full_pipeline(
            input_path="dataset.json",
            output_path="processed_dataset.json",
            augment_factor=4,
            duplicate_threshold=1e-3
        )
        
        print("\n✅ Pipeline completado exitosamente!")
        print("\nEstadísticas del procesamiento:")
        for stage, data in stats.items():
            print(f"  {stage}: {data}")
        
        print("\nDataset procesado guardado en: processed_dataset.json")
        print("Listo para entrenamiento!")
        
    except Exception as e:
        print(f"❌ Error en el pipeline: {e}")
        logger.error(f"Pipeline error: {e}")


if __name__ == "__main__":
    main()
