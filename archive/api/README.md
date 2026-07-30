# Sign Language Recognition API

API de producción para reconocimiento de señas de lenguaje de señas con soporte REST y WebSocket.

## Características

- **REST API**: Inferencia stateless con autenticación
- **WebSocket**: Streaming en tiempo real para baja latencia
- **Multi-model**: Soporte TensorFlow y TensorFlow Lite
- **Seguridad**: API key authentication y validación de entrada
- **Monitoreo**: Estadísticas de uso y rendimiento
- **Docker**: Deployment containerizado
- **Clientes**: Ejemplos de uso en Python

## Arquitectura

```
Client (Web/Mobile)
    ↓
API Gateway (FastAPI)
    ↓
Inference Engine (TF/TFLite)
    ↓
Response (JSON / stream)
```

## Endpoints

### REST API

#### `POST /predict`
Predice seña desde landmarks

```json
{
  "landmarks": [[x,y,z], ..., (21)]
}
```

**Response:**
```json
{
  "label": "A",
  "confidence": 0.93,
  "processing_time_ms": 15.2
}
```

#### `POST /batch_predict`
Predicción batch (máximo 10 muestras)

```json
{
  "landmarks_batch": [
    [[x,y,z], ...],  # muestra 1
    [[x,y,z], ...],  # muestra 2
    ...
  ]
}
```

#### `GET /stats`
Estadísticas del sistema (requiere autenticación)

#### `GET /health`
Health check del sistema

### WebSocket

#### `WS /ws`
Streaming en tiempo real

**Client → Server:**
```json
{
  "landmarks": [[x,y,z], ..., (21)]
}
```

**Server → Client:**
```json
{
  "label": "A",
  "confidence": 0.93,
  "processing_time_ms": 15.2,
  "timestamp": 1640995200.0
}
```

## Instalación

### Local Development

1. **Instalar dependencias:**
```bash
pip install -r requirements.txt
```

2. **Preparar modelos:**
```bash
# Copiar modelos al directorio de la API
cp ../data/models/model.h5 .
cp ../data/models/labels.json .
# o para TFLite
cp ../data/models/model.tflite .
cp ../data/models/model.tflite_labels.json labels.json
```

3. **Ejecutar servidor:**
```bash
python main.py
# o con uvicorn directamente
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Docker

1. **Construir imagen:**
```bash
docker build -t sign-recognition-api .
```

2. **Ejecutar contenedor:**
```bash
docker run -p 8000:8000 -v $(pwd)/models:/app/models:ro sign-recognition-api
```

3. **Con Docker Compose:**
```bash
docker-compose up -d
```

## Uso

### Cliente REST

```python
from client_examples import SignRecognitionClient

# Crear cliente
client = SignRecognitionClient("http://localhost:8000", "tu-api-key")

# Predicción
landmarks = [[0.5, 0.5, 0.5] for _ in range(21)]
result = client.predict(landmarks)

print(f"Seña: {result['label']}, Confianza: {result['confidence']}")
```

### Cliente WebSocket

```python
from client_examples import WebSocketSignClient
import asyncio

async def main():
    client = WebSocketSignClient("ws://localhost:8000/ws")
    
    await client.connect()
    
    # Streaming continuo
    for landmarks in landmarks_generator:
        result = await client.send_landmarks(landmarks)
        print(f"Predicción: {result['label']}")

asyncio.run(main())
```

### JavaScript/HTML

```javascript
// REST API
const response = await fetch('http://localhost:8000/predict', {
    method: 'POST',
    headers: {
        'Authorization': 'Bearer tu-api-key',
        'Content-Type': 'application/json'
    },
    body: JSON.stringify({
        landmarks: [[x,y,z], ...] // 21 landmarks
    })
});

const result = await response.json();
console.log(`Seña: ${result.label}, Confianza: ${result.confidence}`);

// WebSocket
const ws = new WebSocket('ws://localhost:8000/ws');

ws.onopen = () => {
    console.log('Conectado al WebSocket');
};

ws.onmessage = (event) => {
    const result = JSON.parse(event.data);
    console.log(`Predicción: ${result.label}`);
};

// Enviar landmarks
ws.send(JSON.stringify({
    landmarks: [[x,y,z], ...]
}));
```

## Configuración

### Variables de Entorno

- `API_KEY`: Clave de autenticación (default: "sign-recognition-api-key-2024")
- `LOG_LEVEL`: Nivel de logging (default: "INFO")
- `MODEL_PATH`: Ruta del modelo (default: auto-detectar)
- `LABELS_PATH`: Ruta de etiquetas (default: auto-detectar)

### Modelos Soportados

1. **TensorFlow (.h5)**: Máxima precisión, mayor uso de memoria
2. **TensorFlow Lite (.tflite)**: Optimizado, menor latencia

## Performance

### Métricas Esperadas

- **Latencia REST**: 15-30ms
- **Latencia WebSocket**: 10-20ms
- **Throughput**: 100+ req/s (CPU), 500+ req/s (GPU)
- **Memoria**: 200-500MB (TF), 50-100MB (TFLite)

### Optimizaciones

- **Model warm-up**: Pre-carga del modelo al iniciar
- **Buffer reuse**: Reutilización de arrays numpy
- **Async processing**: I/O no bloqueante
- **Connection pooling**: Manejo eficiente de conexiones

## Seguridad

### Autenticación

- **API Key**: Bearer token en header `Authorization`
- **Rate Limiting**: Configurable (implementar slowapi si es necesario)
- **Input Validation**: Validación estricta de landmarks

### Recomendaciones Producción

1. **HTTPS**: Terminación SSL con nginx/traefik
2. **API Keys Rotación**: Cambiar claves periódicamente
3. **Rate Limiting**: Limitar requests por cliente
4. **Monitoring**: Logs y métricas centralizadas
5. **Load Balancing**: Múltiples instancias detrás de LB

## Monitoreo

### Métricas Disponibles

- Requests totales/exitosos/fallidos
- Latencia promedio
- Conexiones WebSocket activas
- Uso de memoria/CPU
- Errores por tipo

### Endpoints de Monitoreo

- `GET /health`: Health check básico
- `GET /stats`: Estadísticas detalladas (auth requerida)

## Troubleshooting

### Problemas Comunes

1. **Model loading fails**:
   - Verificar paths de modelo y etiquetas
   - Compatibilidad de versión TensorFlow

2. **High latency**:
   - Usar TFLite en lugar de TensorFlow
   - Revisar recursos CPU/GPU

3. **WebSocket disconnects**:
   - Timeout de conexión
   - Rate limiting del cliente

4. **Authentication errors**:
   - Verificar API key
   - Formato del header Authorization

### Debug

```bash
# Ver logs del contenedor
docker logs sign-recognition-api

# Health check
curl http://localhost:8000/health

# Test con curl
curl -X POST http://localhost:8000/predict \
  -H "Authorization: Bearer tu-api-key" \
  -H "Content-Type: application/json" \
  -d '{"landmarks": [[0.5,0.5,0.5] for _ in range(21)]}'
```

## Escalado

### Horizontal Scaling

```yaml
# docker-compose.yml (múltiples instancias)
services:
  api-1:
    build: .
    ports: ["8001:8000"]
  api-2:
    build: .
    ports: ["8002:8000"]
  nginx:
    image: nginx:alpine
    ports: ["80:80"]
    volumes: ["./nginx.conf:/etc/nginx/nginx.conf"]
```

### Vertical Scaling

- **GPU**: Usar imagen tensorflow/tensorflow:latest-gpu
- **CPU**: Aumentar límites de CPU en Docker
- **Memory**: Ajustar según tamaño del modelo

## Testing

### Ejecutar tests

```bash
# Demo cliente REST
python client_examples.py rest

# Demo cliente WebSocket
python client_examples.py websocket

# Benchmark de rendimiento
python client_examples.py benchmark
```

### Tests Automáticos

```bash
# Unit tests (si se implementan)
pytest tests/

# Integration tests
pytest integration_tests/
```

## Deployment

### Producción con Docker Compose

```bash
# Iniciar todos los servicios
docker-compose up -d

# Ver estado
docker-compose ps

# Ver logs
docker-compose logs -f api
```

### Kubernetes

```yaml
# k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: sign-recognition-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: sign-recognition-api
  template:
    metadata:
      labels:
        app: sign-recognition-api
    spec:
      containers:
      - name: api
        image: sign-recognition-api:latest
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
```

## Contribución

1. Fork del repositorio
2. Feature branch: `git checkout -b feature/nueva-funcionalidad`
3. Commit: `git commit -am 'Agregar nueva funcionalidad'`
4. Push: `git push origin feature/nueva-funcionalidad`
5. Pull Request

## Licencia

MIT License - ver archivo LICENSE
