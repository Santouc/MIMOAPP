#!/usr/bin/env python3
"""
API de Producción para Reconocimiento de Señas
Expone inferencia del modelo vía REST y WebSocket
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
import numpy as np
import json
import time
import logging
from contextlib import asynccontextmanager
import asyncio
from collections import deque
import threading
import uvicorn

# Importaciones de TensorFlow
try:
    import tensorflow as tf
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False
    raise ImportError("TensorFlow es requerido para la API")

# Importaciones de TFLite (opcional)
try:
    import tflite_runtime.interpreter as tflite
    HAS_TFLITE_RUNTIME = True
except ImportError:
    HAS_TFLITE_RUNTIME = False

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Variables globales del modelo
model = None
labels = []
model_config = {}

# Sistema de seguridad
security = HTTPBearer()
API_KEY = "sign-recognition-api-key-2024"  # En producción usar variables de entorno

# Sistema de monitoreo
request_stats = {
    "total_requests": 0,
    "successful_requests": 0,
    "failed_requests": 0,
    "avg_latency": 0.0,
    "predictions": deque(maxlen=1000)
}

# Conexiones WebSocket activas
active_connections: set[WebSocket] = set()


class LandmarkInput(BaseModel):
    """Modelo de entrada para landmarks"""
    landmarks: List[List[float]] = Field(..., description="21 landmarks con coordenadas [x,y,z]")
    
    @validator('landmarks')
    def validate_landmarks(cls, v):
        if len(v) != 21:
            raise ValueError("Se requieren exactamente 21 landmarks")
        
        for i, landmark in enumerate(v):
            if len(landmark) != 3:
                raise ValueError(f"Landmark {i} debe tener 3 coordenadas [x,y,z]")
            
            # Validar rango (MediaPipe usa [0,1])
            for j, coord in enumerate(landmark):
                if not (0.0 <= coord <= 1.0):
                    raise ValueError(f"Coordenada {j} del landmark {i} debe estar en [0,1]")
        
        return v


class PredictionResponse(BaseModel):
    """Modelo de respuesta de predicción"""
    label: str = Field(..., description="Etiqueta predicha")
    confidence: float = Field(..., description="Confianza de la predicción (0-1)")
    processing_time_ms: float = Field(..., description="Tiempo de procesamiento en milisegundos")


class StatsResponse(BaseModel):
    """Modelo de respuesta de estadísticas"""
    total_requests: int
    successful_requests: int
    failed_requests: int
    avg_latency_ms: float
    uptime_seconds: float
    active_websockets: int
    model_info: Dict[str, Any]


def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verificación de API key"""
    if credentials.credentials != API_KEY:
        raise HTTPException(status_code=403, detail="API key inválida")
    return credentials


def load_model():
    """Carga modelo y etiquetas"""
    global model, labels, model_config
    
    try:
        # Intentar cargar TFLite primero (más rápido)
        model_path = "model.tflite"
        labels_path = "model.tflite_labels.json"
        
        if Path(model_path).exists() and HAS_TFLITE_RUNTIME:
            logger.info("Cargando modelo TFLite...")
            model = tflite.Interpreter(model_path=model_path)
            model.allocate_tensors()
            
            input_details = model.get_input_details()
            output_details = model.get_output_details()
            
            model_config = {
                "type": "tflite",
                "input_shape": input_details[0]['shape'],
                "output_shape": output_details[0]['shape']
            }
            
            logger.info(f"Modelo TFLite cargado: {model_config}")
        
        # Fallback a TensorFlow
        elif Path("model.h5").exists():
            logger.info("Cargando modelo TensorFlow...")
            model = tf.keras.models.load_model("model.h5")
            
            model_config = {
                "type": "tensorflow",
                "input_shape": (1, 21, 3),
                "output_shape": (None, len(labels))
            }
            
            logger.info(f"Modelo TensorFlow cargado: {model_config}")
        
        else:
            raise FileNotFoundError("No se encontró modelo (model.h5 o model.tflite)")
        
        # Cargar etiquetas
        if Path(labels_path).exists():
            with open(labels_path, 'r') as f:
                labels = json.load(f)
        else:
            # Fallback a labels.json
            with open("labels.json", 'r') as f:
                labels = json.load(f)
        
        logger.info(f"Modelo cargado exitosamente con {len(labels)} clases: {labels}")
        
        # Warm-up del modelo
        warmup_landmarks = [[0.5, 0.5, 0.5] for _ in range(21)]
        preprocess_and_predict(warmup_landmarks)
        logger.info("Modelo warm-up completado")
        
    except Exception as e:
        logger.error(f"Error cargando modelo: {e}")
        raise


def preprocess_landmarks(landmarks: List[List[float]]) -> np.ndarray:
    """
    Preprocesamiento consistente con entrenamiento
    
    Args:
        landmarks: Lista de 21 landmarks [x,y,z]
        
    Returns:
        Array preprocesado (1, 21, 3)
    """
    X = np.array(landmarks, dtype=np.float32)
    
    # Normalización de traducción (centrar en muñeca)
    X = X - X[0]
    
    # Normalización de escala (relativo a hueso índice)
    scale = np.linalg.norm(X[9])
    if scale > 1e-6:
        X = X / scale
    
    return np.expand_dims(X, axis=0)


def preprocess_and_predict(landmarks: List[List[float]]) -> tuple:
    """
    Preprocesa y predice usando el modelo cargado
    
    Args:
        landmarks: Lista de 21 landmarks [x,y,z]
        
    Returns:
        Tuple (label, confidence, processing_time_ms)
    """
    start_time = time.time()
    
    # Preprocesamiento
    X = preprocess_landmarks(landmarks)
    
    # Inferencia según tipo de modelo
    if model_config["type"] == "tflite":
        # TFLite inference
        input_idx = model.get_input_details()[0]['index']
        output_idx = model.get_output_details()[0]['index']
        
        model.set_tensor(input_idx, X)
        model.invoke()
        probs = model.get_tensor(output_idx)[0]
    
    else:
        # TensorFlow inference
        probs = model(X, training=False).numpy()[0]
    
    # Post-procesamiento
    idx = int(np.argmax(probs))
    confidence = float(np.max(probs))
    label = labels[idx] if confidence > 0.6 else "unknown"
    
    processing_time = (time.time() - start_time) * 1000  # ms
    
    return label, confidence, processing_time


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Context manager para startup/shutdown"""
    # Startup
    logger.info("Iniciando API de Reconocimiento de Señas...")
    load_model()
    logger.info("API lista para recibir requests")
    
    yield
    
    # Shutdown
    logger.info("Apagando API...")


# Crear aplicación FastAPI
app = FastAPI(
    title="Sign Language Recognition API",
    description="API de producción para reconocimiento de señas de lenguaje de señas",
    version="1.0.0",
    lifespan=lifespan
)

# Middleware CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción especificar dominios permitidos
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Endpoint raíz"""
    return {
        "message": "Sign Language Recognition API",
        "version": "1.0.0",
        "endpoints": {
            "predict": "/predict",
            "stats": "/stats",
            "websocket": "/ws",
            "health": "/health"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "num_classes": len(labels),
        "model_type": model_config.get("type", "unknown"),
        "uptime_seconds": time.time()
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict_sign(
    data: LandmarkInput,
    credentials: HTTPAuthorizationCredentials = Depends(verify_api_key)
):
    """
    Predice seña desde landmarks
    
    - **landmarks**: 21 landmarks con coordenadas [x,y,z] en rango [0,1]
    - **Returns**: Etiqueta predicha y confianza
    """
    try:
        request_stats["total_requests"] += 1
        
        # Inferencia
        label, confidence, processing_time = preprocess_and_predict(data.landmarks)
        
        # Actualizar estadísticas
        request_stats["successful_requests"] += 1
        request_stats["predictions"].append({
            "label": label,
            "confidence": confidence,
            "timestamp": time.time()
        })
        
        # Actualizar latencia promedio
        total_successful = request_stats["successful_requests"]
        if total_successful > 0:
            current_avg = request_stats["avg_latency"]
            request_stats["avg_latency"] = (
                (current_avg * (total_successful - 1) + processing_time) / total_successful
            )
        
        return PredictionResponse(
            label=label,
            confidence=confidence,
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        request_stats["failed_requests"] += 1
        logger.error(f"Error en predicción: {e}")
        raise HTTPException(status_code=500, detail=f"Error en predicción: {str(e)}")


@app.get("/stats", response_model=StatsResponse)
async def get_stats(credentials: HTTPAuthorizationCredentials = Depends(verify_api_key)):
    """
    Obtiene estadísticas del sistema
    
    - **Returns**: Estadísticas de uso y rendimiento
    """
    uptime = time.time()  # Simplificado - en producción usar timestamp real de inicio
    
    return StatsResponse(
        total_requests=request_stats["total_requests"],
        successful_requests=request_stats["successful_requests"],
        failed_requests=request_stats["failed_requests"],
        avg_latency_ms=request_stats["avg_latency"],
        uptime_seconds=uptime,
        active_websockets=len(active_connections),
        model_info=model_config
    )


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint para streaming en tiempo real
    
    Cliente envía landmarks continuamente, servidor responde con predicciones
    """
    await websocket.accept()
    active_connections.add(websocket)
    
    logger.info(f"Nueva conexión WebSocket. Total: {len(active_connections)}")
    
    try:
        while True:
            # Recibir landmarks del cliente
            data = await websocket.receive_json()
            
            # Validar estructura
            if "landmarks" not in data:
                await websocket.send_json({"error": "Campo 'landmarks' requerido"})
                continue
            
            try:
                landmarks = data["landmarks"]
                landmark_input = LandmarkInput(landmarks=landmarks)
                
                # Predicción
                label, confidence, processing_time = preprocess_and_predict(landmarks)
                
                # Enviar respuesta
                response = {
                    "label": label,
                    "confidence": confidence,
                    "processing_time_ms": processing_time,
                    "timestamp": time.time()
                }
                
                await websocket.send_json(response)
                
            except Exception as e:
                await websocket.send_json({"error": f"Error procesando landmarks: {str(e)}"})
                
    except WebSocketDisconnect:
        active_connections.remove(websocket)
        logger.info(f"Conexión WebSocket cerrada. Total: {len(active_connections)}")
    except Exception as e:
        logger.error(f"Error en WebSocket: {e}")
        if websocket in active_connections:
            active_connections.remove(websocket)


@app.post("/batch_predict")
async def batch_predict(
    data: Dict[str, List[List[List[float]]]],
    credentials: HTTPAuthorizationCredentials = Depends(verify_api_key)
):
    """
    Predicción batch para múltiples landmarks
    
    - **landmarks**: Lista de conjuntos de landmarks
    - **Returns**: Lista de predicciones
    """
    if "landmarks_batch" not in data:
        raise HTTPException(status_code=400, detail="Campo 'landmarks_batch' requerido")
    
    landmarks_batch = data["landmarks_batch"]
    
    if len(landmarks_batch) > 10:  # Límite de batch
        raise HTTPException(status_code=400, detail="Máximo 10 landmarks por batch")
    
    results = []
    
    for i, landmarks in enumerate(landmarks_batch):
        try:
            landmark_input = LandmarkInput(landmarks=landmarks)
            label, confidence, processing_time = preprocess_and_predict(landmarks)
            
            results.append({
                "index": i,
                "label": label,
                "confidence": confidence,
                "processing_time_ms": processing_time
            })
            
        except Exception as e:
            results.append({
                "index": i,
                "error": str(e)
            })
    
    return {"results": results}


if __name__ == "__main__":
    # Ejecutar servidor de desarrollo
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
