#!/usr/bin/env python3
"""
Sistema de evaluación de modelo con análisis detallado de errores
Cuantifica rendimiento y modos de fallo a nivel de clase
"""

import numpy as np
import json
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import logging
from collections import defaultdict

# Importaciones para métricas
try:
    from sklearn.metrics import (
        confusion_matrix, precision_recall_fscore_support,
        classification_report
    )
except ImportError:
    logging.warning("scikit-learn no disponible, usando implementaciones propias")
    confusion_matrix = None
    precision_recall_fscore_support = None

# Importaciones para visualización (opcional)
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False
    logging.warning("matplotlib/seaborn no disponible, visualización deshabilitada")

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelEvaluator:
    """Sistema completo de evaluación de modelo con análisis de errores"""
    
    def __init__(self, model, labels: List[str]):
        """
        Inicializa el evaluador
        
        Args:
            model: Modelo entrenado de TensorFlow/Keras
            labels: Lista de nombres de clases
        """
        self.model = model
        self.labels = labels
        self.num_classes = len(labels)
        self.class_to_idx = {label: idx for idx, label in enumerate(labels)}
        self.idx_to_class = {idx: label for label, idx in self.class_to_idx.items()}
        
        logger.info(f"ModelEvaluator inicializado con {self.num_classes} clases")
    
    def predict_batch(self, X: np.ndarray, batch_size: int = 64) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Realiza predicciones batch vectorizadas
        
        Args:
            X: Array (M, 21, 3) de landmarks
            batch_size: Tamaño de batch para predicción
            
        Returns:
            Tuple (y_pred, y_prob, confidence)
        """
        logger.info(f"Realizando predicciones en {len(X)} muestras...")
        
        # Predicción batch
        y_prob = self.model.predict(X, batch_size=batch_size, verbose=0)
        y_pred = y_prob.argmax(axis=1)
        confidence = y_prob.max(axis=1)
        
        logger.info(f"Predicciones completadas: shape {y_prob.shape}")
        
        return y_pred, y_prob, confidence
    
    def compute_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calcula matriz de confusión normalizada y no normalizada
        
        Args:
            y_true: Etiquetas verdaderas
            y_pred: Etiquetas predichas
            
        Returns:
            Tuple (cm, cm_norm)
        """
        if confusion_matrix is None:
            # Implementación propia si sklearn no disponible
            cm = self._compute_confusion_matrix_manual(y_true, y_pred)
        else:
            cm = confusion_matrix(y_true, y_pred, labels=list(range(self.num_classes)))
        
        # Normalización por filas (row-wise)
        cm_norm = cm / (cm.sum(axis=1, keepdims=True) + 1e-8)
        
        return cm, cm_norm
    
    def _compute_confusion_matrix_manual(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Implementación manual de matriz de confusión"""
        cm = np.zeros((self.num_classes, self.num_classes), dtype=int)
        
        for true, pred in zip(y_true, y_pred):
            cm[true, pred] += 1
        
        return cm
    
    def compute_per_class_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """
        Calcula métricas por clase (precision, recall, f1, support)
        
        Args:
            y_true: Etiquetas verdaderas
            y_pred: Etiquetas predichas
            
        Returns:
            Diccionario con métricas por clase
        """
        if precision_recall_fscore_support is None:
            # Implementación propia
            return self._compute_metrics_manual(y_true, y_pred)
        
        prec, rec, f1, support = precision_recall_fscore_support(
            y_true, y_pred, labels=list(range(self.num_classes)), zero_division=0
        )
        
        metrics = {}
        for i, label in enumerate(self.labels):
            metrics[label] = {
                "precision": float(prec[i]),
                "recall": float(rec[i]),
                "f1": float(f1[i]),
                "support": int(support[i])
            }
        
        return metrics
    
    def _compute_metrics_manual(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Implementación manual de métricas por clase"""
        metrics = {}
        
        for i, label in enumerate(self.labels):
            # True positives, false positives, false negatives
            tp = np.sum((y_true == i) & (y_pred == i))
            fp = np.sum((y_true != i) & (y_pred == i))
            fn = np.sum((y_true == i) & (y_pred != i))
            
            # Precision, Recall, F1
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            support = int(np.sum(y_true == i))
            
            metrics[label] = {
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "support": support
            }
        
        return metrics
    
    def detect_top_confusions(self, cm: np.ndarray, k: int = 3) -> Dict[str, List[Tuple[str, int]]]:
        """
        Detecta las principales confusiones por clase
        
        Args:
            cm: Matriz de confusión
            k: Número de confusiones principales a detectar
            
        Returns:
            Diccionario con confusiones principales por clase
        """
        confusions = {}
        
        for i, label in enumerate(self.labels):
            row = cm[i].copy()
            row[i] = 0  # Excluir predicciones correctas
            
            # Obtener índices de las k confusiones principales
            top_idx = row.argsort()[::-1][:k]
            
            class_confusions = []
            for j in top_idx:
                if row[j] > 0:  # Solo incluir confusiones reales
                    confused_label = self.labels[j]
                    confusion_count = int(row[j])
                    class_confusions.append((confused_label, confusion_count))
            
            confusions[label] = class_confusions
        
        return confusions
    
    def detect_confidence_errors(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                confidence: np.ndarray, low_thr: float = 0.6, 
                                high_thr: float = 0.8) -> Dict:
        """
        Detecta errores basados en confianza
        
        Args:
            y_true: Etiquetas verdaderas
            y_pred: Etiquetas predichas
            confidence: Confianza de las predicciones
            low_thr: Umbral de confianza baja
            high_thr: Umbral de confianza alta
            
        Returns:
            Diccionario con estadísticas de errores por confianza
        """
        correct = (y_pred == y_true)
        
        # Índices de diferentes tipos de errores
        low_conf_correct_idx = np.where(correct & (confidence < low_thr))[0]
        high_conf_wrong_idx = np.where(~correct & (confidence > high_thr))[0]
        low_conf_wrong_idx = np.where(~correct & (confidence < low_thr))[0]
        high_conf_correct_idx = np.where(correct & (confidence > high_thr))[0]
        
        return {
            "low_conf_correct_count": len(low_conf_correct_idx),
            "high_conf_wrong_count": len(high_conf_wrong_idx),
            "low_conf_wrong_count": len(low_conf_wrong_idx),
            "high_conf_correct_count": len(high_conf_correct_idx),
            "low_conf_correct_indices": low_conf_correct_idx.tolist(),
            "high_conf_wrong_indices": high_conf_wrong_idx.tolist(),
            "thresholds": {"low": low_thr, "high": high_thr}
        }
    
    def compute_per_class_error_rate(self, cm: np.ndarray) -> Dict[str, float]:
        """
        Calcula tasa de error por clase
        
        Args:
            cm: Matriz de confusión
            
        Returns:
            Diccionario con tasa de error por clase
        """
        per_class_err = {}
        
        for i, label in enumerate(self.labels):
            total_samples = cm[i].sum()
            correct_samples = cm[i, i]
            
            error_rate = 1.0 - (correct_samples / (total_samples + 1e-8))
            per_class_err[label] = float(error_rate)
        
        return per_class_err
    
    def interpret_results(self, metrics: Dict, confusions: Dict, confidence_errors: Dict) -> Dict:
        """
        Interpreta resultados y genera recomendaciones de acción
        
        Args:
            metrics: Métricas por clase
            confusions: Confusiones principales
            confidence_errors: Errores por confianza
            
        Returns:
            Diccionario con interpretación y recomendaciones
        """
        interpretations = {
            "issues": [],
            "recommendations": [],
            "flags": {}
        }
        
        # Detectar confusiones problemáticas (A↔E, B↔C)
        problematic_pairs = []
        for class_name, class_confusions in confusions.items():
            for confused_with, count in class_confusions:
                pair = tuple(sorted([class_name, confused_with]))
                if pair in [("A", "E"), ("B", "C")] and count > 5:
                    problematic_pairs.append((pair[0], pair[1], count))
        
        if problematic_pairs:
            interpretations["issues"].append(f"Confusiones problemáticas detectadas: {problematic_pairs}")
            interpretations["recommendations"].append("Colectar muestras específicas para clases confundidas")
            interpretations["flags"]["problematic_confusions"] = problematic_pairs
        
        # Detectar clases con precision baja y recall alto
        for class_name, class_metrics in metrics.items():
            if class_metrics["precision"] < 0.7 and class_metrics["recall"] > 0.8:
                interpretations["issues"].append(f"Sobre-predicción de clase {class_name}")
                interpretations["recommendations"].append(f"Aumentar diversidad de muestras negativas para {class_name}")
                interpretations["flags"]["overprediction"] = interpretations["flags"].get("overprediction", [])
                interpretations["flags"]["overprediction"].append(class_name)
        
        # Detectar clases con recall bajo y precision alto
        for class_name, class_metrics in metrics.items():
            if class_metrics["recall"] < 0.7 and class_metrics["precision"] > 0.8:
                interpretations["issues"].append(f"Sub-detección de clase {class_name}")
                interpretations["recommendations"].append(f"Agregar más muestras para clase {class_name}")
                interpretations["flags"]["underdetection"] = interpretations["flags"].get("underdetection", [])
                interpretations["flags"]["underdetection"].append(class_name)
        
        # Detectar errores de alta confianza
        if confidence_errors["high_conf_wrong_count"] > 10:
            interpretations["issues"].append(f"Errores de alta confianza: {confidence_errors['high_conf_wrong_count']}")
            interpretations["recommendations"].append("Inspeccionar manualmente muestras con alta confianza incorrecta")
            interpretations["flags"]["high_confidence_errors"] = confidence_errors["high_conf_wrong_count"]
        
        # Detectar decisiones borderline
        if confidence_errors["low_conf_correct_count"] > 50:
            interpretations["issues"].append(f"Muestras correctas con baja confianza: {confidence_errors['low_conf_correct_count']}")
            interpretations["recommendations"].append("Mejorar separabilidad en frontera de decisión")
            interpretations["flags"]["borderline_decisions"] = confidence_errors["low_conf_correct_count"]
        
        return interpretations
    
    def predict_with_threshold(self, X: np.ndarray, threshold: float = 0.6) -> Tuple[List[str], np.ndarray]:
        """
        Predice con umbral de confianza
        
        Args:
            X: Array (M, 21, 3) de landmarks
            threshold: Umbral de confianza mínima
            
        Returns:
            Tuple (predictions, confidence)
        """
        y_pred, y_prob, confidence = self.predict_batch(X)
        
        predictions = []
        for i in range(len(y_pred)):
            if confidence[i] < threshold:
                predictions.append("unknown")
            else:
                predictions.append(self.labels[y_pred[i]])
        
        return predictions, confidence
    
    def visualize_confusion_matrix(self, cm_norm: np.ndarray, save_path: Optional[str] = None):
        """
        Visualiza matriz de confusión normalizada
        
        Args:
            cm_norm: Matriz de confusión normalizada
            save_path: Ruta para guardar visualización
        """
        if not HAS_PLOTTING:
            logger.warning("Visualización no disponible (matplotlib/seaborn no instalados)")
            return
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm_norm, annot=True, fmt=".2f", 
                   xticklabels=self.labels, yticklabels=self.labels,
                   cmap="Blues", vmin=0, vmax=1)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix (Normalized)")
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Matriz de confusión guardada en {save_path}")
        
        plt.show()
    
    def evaluate_model(self, X: np.ndarray, y: np.ndarray, 
                      batch_size: int = 64, save_path: Optional[str] = None,
                      visualize: bool = False) -> Dict:
        """
        Evaluación completa del modelo
        
        Args:
            X: Array (M, 21, 3) de landmarks
            y: Array (M,) de etiquetas
            batch_size: Tamaño de batch para predicción
            save_path: Ruta para guardar resultados
            visualize: Si generar visualización
            
        Returns:
            Diccionario completo con resultados de evaluación
        """
        logger.info("=== INICIANDO EVALUACIÓN COMPLETA DEL MODELO ===")
        
        # 1. Predicciones
        y_pred, y_prob, confidence = self.predict_batch(X, batch_size)
        
        # 2. Accuracy general
        accuracy = float(np.mean(y_pred == y))
        logger.info(f"Accuracy general: {accuracy:.4f}")
        
        # 3. Matriz de confusión
        cm, cm_norm = self.compute_confusion_matrix(y, y_pred)
        
        # 4. Métricas por clase
        per_class_metrics = self.compute_per_class_metrics(y, y_pred)
        
        # 5. Confusiones principales
        top_confusions = self.detect_top_confusions(cm, k=3)
        
        # 6. Errores por confianza
        confidence_errors = self.detect_confidence_errors(y, y_pred, confidence)
        
        # 7. Tasa de error por clase
        per_class_error_rate = self.compute_per_class_error_rate(cm)
        
        # 8. Interpretación de resultados
        interpretations = self.interpret_results(per_class_metrics, top_confusions, confidence_errors)
        
        # 9. Construir reporte final
        evaluation_results = {
            "accuracy": accuracy,
            "per_class": [
                {
                    "class": class_name,
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1": metrics["f1"],
                    "support": metrics["support"],
                    "error_rate": per_class_error_rate[class_name]
                }
                for class_name, metrics in per_class_metrics.items()
            ],
            "confusion_matrix": cm.tolist(),
            "confusion_matrix_norm": cm_norm.tolist(),
            "top_confusions": top_confusions,
            "confidence_errors": confidence_errors,
            "per_class_error_rate": per_class_error_rate,
            "interpretations": interpretations,
            "metadata": {
                "num_samples": len(X),
                "num_classes": self.num_classes,
                "batch_size": batch_size,
                "evaluation_timestamp": np.datetime64('now').astype(int)
            }
        }
        
        # 10. Visualización (opcional)
        if visualize:
            self.visualize_confusion_matrix(cm_norm, save_path=f"{save_path}_confusion_matrix.png" if save_path else None)
        
        # 11. Guardar resultados
        if save_path:
            with open(save_path, 'w') as f:
                json.dump(evaluation_results, f, indent=2)
            logger.info(f"Resultados guardados en {save_path}")
        
        logger.info("=== EVALUACIÓN COMPLETADA ===")
        
        return evaluation_results
    
    def print_summary(self, results: Dict):
        """
        Imprime resumen de resultados de evaluación
        
        Args:
            results: Diccionario de resultados de evaluación
        """
        print("\n" + "="*60)
        print("RESUMEN DE EVALUACIÓN DEL MODELO")
        print("="*60)
        
        print(f"\nAccuracy General: {results['accuracy']:.4f}")
        
        print(f"\nMétricas por Clase:")
        print("-" * 60)
        print(f"{'Clase':<8} {'Precision':<10} {'Recall':<10} {'F1':<10} {'Support':<8} {'Error':<8}")
        print("-" * 60)
        
        for class_metrics in results['per_class']:
            print(f"{class_metrics['class']:<8} {class_metrics['precision']:<10.3f} "
                  f"{class_metrics['recall']:<10.3f} {class_metrics['f1']:<10.3f} "
                  f"{class_metrics['support']:<8} {class_metrics['error_rate']:<8.3f}")
        
        print(f"\nPrincipales Confusiones:")
        print("-" * 40)
        for class_name, confusions in results['top_confusions'].items():
            if confusions:
                for confused_with, count in confusions:
                    print(f"{class_name} → {confused_with}: {count}")
        
        print(f"\nErrores por Confianza:")
        print("-" * 30)
        ce = results['confidence_errors']
        print(f"Correctos con baja confianza: {ce['low_conf_correct_count']}")
        print(f"Incorrectos con alta confianza: {ce['high_conf_wrong_count']}")
        
        if results['interpretations']['issues']:
            print(f"\nProblemas Detectados:")
            print("-" * 20)
            for issue in results['interpretations']['issues']:
                print(f"⚠️ {issue}")
            
            print(f"\nRecomendaciones:")
            print("-" * 15)
            for rec in results['interpretations']['recommendations']:
                print(f"💡 {rec}")
        
        print("\n" + "="*60)


def main():
    """Función principal para demostrar evaluación"""
    print("=== Sistema de Evaluación de Modelo ===")
    print("Este sistema evalúa modelos entrenados y analiza errores detalladamente.")
    print()
    
    # Ejemplo de uso (requiere modelo y datos reales)
    print("Para usar el evaluador:")
    print("1. Cargar modelo entrenado")
    print("2. Cargar dataset de validación")
    print("3. Crear ModelEvaluator")
    print("4. Ejecutar evaluate_model()")
    print()
    print("Ejemplo:")
    print("evaluator = ModelEvaluator(model, labels)")
    print("results = evaluator.evaluate_model(X_val, y_val)")
    print("evaluator.print_summary(results)")


if __name__ == "__main__":
    main()
