#!/usr/bin/env python3
"""
Ejemplos de clientes para la API de Reconocimiento de Señas
REST API y WebSocket
"""

import requests
import asyncio
import websockets
import json
import time
import numpy as np
from typing import List, Dict, Any


class SignRecognitionClient:
    """Cliente para la API REST de reconocimiento de señas"""
    
    def __init__(self, base_url: str = "http://localhost:8000", api_key: str = "sign-recognition-api-key-2024"):
        """
        Inicializa cliente REST
        
        Args:
            base_url: URL base de la API
            api_key: API key para autenticación
        """
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    def predict(self, landmarks: List[List[float]]) -> Dict[str, Any]:
        """
        Realiza predicción de seña
        
        Args:
            landmarks: 21 landmarks con coordenadas [x,y,z]
            
        Returns:
            Respuesta de la API con label y confidence
        """
        try:
            response = requests.post(
                f"{self.base_url}/predict",
                json={"landmarks": landmarks},
                headers=self.headers,
                timeout=10
            )
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            print(f"Error en request: {e}")
            return {"error": str(e)}
    
    def batch_predict(self, landmarks_batch: List[List[List[float]]]) -> Dict[str, Any]:
        """
        Realiza predicción batch
        
        Args:
            landmarks_batch: Lista de conjuntos de landmarks
            
        Returns:
            Lista de predicciones
        """
        try:
            response = requests.post(
                f"{self.base_url}/batch_predict",
                json={"landmarks_batch": landmarks_batch},
                headers=self.headers,
                timeout=30
            )
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            print(f"Error en batch request: {e}")
            return {"error": str(e)}
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas del sistema
        
        Returns:
            Estadísticas de uso y rendimiento
        """
        try:
            response = requests.get(
                f"{self.base_url}/stats",
                headers=self.headers,
                timeout=5
            )
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            print(f"Error obteniendo stats: {e}")
            return {"error": str(e)}
    
    def health_check(self) -> Dict[str, Any]:
        """
        Verifica salud del sistema
        
        Returns:
            Estado del sistema
        """
        try:
            response = requests.get(
                f"{self.base_url}/health",
                timeout=5
            )
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            print(f"Error en health check: {e}")
            return {"error": str(e)}


class WebSocketSignClient:
    """Cliente WebSocket para streaming en tiempo real"""
    
    def __init__(self, uri: str = "ws://localhost:8000/ws"):
        """
        Inicializa cliente WebSocket
        
        Args:
            uri: URI del WebSocket
        """
        self.uri = uri
        self.websocket = None
        self.connected = False
    
    async def connect(self):
        """Conecta al WebSocket"""
        try:
            self.websocket = await websockets.connect(self.uri)
            self.connected = True
            print("Conectado al WebSocket")
            
        except Exception as e:
            print(f"Error conectando WebSocket: {e}")
            self.connected = False
    
    async def send_landmarks(self, landmarks: List[List[float]]) -> Dict[str, Any]:
        """
        Envía landmarks y recibe predicción
        
        Args:
            landmarks: 21 landmarks con coordenadas [x,y,z]
            
        Returns:
            Respuesta de predicción
        """
        if not self.connected:
            return {"error": "No conectado"}
        
        try:
            # Enviar landmarks
            await self.websocket.send(json.dumps({"landmarks": landmarks}))
            
            # Recibir respuesta
            response = await self.websocket.recv()
            return json.loads(response)
            
        except Exception as e:
            print(f"Error en comunicación WebSocket: {e}")
            return {"error": str(e)}
    
    async def stream_predictions(self, landmarks_generator, duration_seconds: int = 30):
        """
        Stream de predicciones continuo
        
        Args:
            landmarks_generator: Generador de landmarks
            duration_seconds: Duración del streaming
        """
        if not self.connected:
            print("No conectado al WebSocket")
            return
        
        start_time = time.time()
        prediction_count = 0
        
        try:
            async for landmarks in landmarks_generator:
                if time.time() - start_time > duration_seconds:
                    break
                
                result = await self.send_landmarks(landmarks)
                
                if "error" not in result:
                    prediction_count += 1
                    print(f"Predicción #{prediction_count}: {result['label']} "
                          f"(conf: {result['confidence']:.2f}, "
                          f"tiempo: {result['processing_time_ms']:.1f}ms)")
                else:
                    print(f"Error: {result['error']}")
                
                # Pequeña pausa para no sobrecargar
                await asyncio.sleep(0.1)
                
        except KeyboardInterrupt:
            print("Streaming interrumpido")
        
        print(f"Streaming completado: {prediction_count} predicciones")
    
    async def disconnect(self):
        """Desconecta del WebSocket"""
        if self.websocket:
            await self.websocket.close()
            self.connected = False
            print("Desconectado del WebSocket")


def generate_mock_landmarks(num_samples: int = 100):
    """
    Generador de landmarks de ejemplo para testing
    
    Args:
        num_samples: Número de muestras a generar
    
    Yields:
        Conjunto de 21 landmarks [x,y,z]
    """
    for i in range(num_samples):
        # Simular movimiento de mano con variación
        base_landmarks = np.random.rand(21, 3) * 0.8 + 0.1
        
        # Añadir movimiento sinusoidal suave
        phase = i * 0.1
        wave = np.sin(phase) * 0.1
        
        # Modificar algunas coordenadas para simular gesto
        base_landmarks[:, 1] += wave  # Movimiento en Y
        base_landmarks[:, 0] += np.cos(phase) * 0.05  # Movimiento en X
        
        # Asegurar rango [0,1]
        base_landmarks = np.clip(base_landmarks, 0, 1)
        
        yield base_landmarks.tolist()


async def demo_websocket_client():
    """Demostración del cliente WebSocket"""
    print("=== Demostración Cliente WebSocket ===")
    
    client = WebSocketSignClient()
    
    # Conectar
    await client.connect()
    
    if client.connected:
        # Streaming de predicciones
        print("Iniciando streaming de 30 segundos...")
        await client.stream_predictions(
            generate_mock_landmarks(300),  # 10 predicciones por segundo
            duration_seconds=30
        )
        
        # Desconectar
        await client.disconnect()


def demo_rest_client():
    """Demostración del cliente REST"""
    print("=== Demostración Cliente REST ===")
    
    client = SignRecognitionClient()
    
    # Health check
    print("Health check:")
    health = client.health_check()
    print(f"  Estado: {health.get('status', 'unknown')}")
    print(f"  Modelo cargado: {health.get('model_loaded', False)}")
    print()
    
    # Predicción individual
    print("Predicción individual:")
    landmarks = [[0.5, 0.5, 0.5] for _ in range(21)]
    result = client.predict(landmarks)
    print(f"  Resultado: {result}")
    print()
    
    # Predicción batch
    print("Predicción batch:")
    batch_landmarks = [[landmarks] for _ in range(3)]
    batch_result = client.batch_predict(batch_landmarks)
    print(f"  Resultados: {batch_result}")
    print()
    
    # Estadísticas
    print("Estadísticas del sistema:")
    stats = client.get_stats()
    if "error" not in stats:
        print(f"  Requests totales: {stats['total_requests']}")
        print(f"  Requests exitosos: {stats['successful_requests']}")
        print(f"  Latencia promedio: {stats['avg_latency_ms']:.2f}ms")
        print(f"  WebSockets activos: {stats['active_websockets']}")
    else:
        print(f"  Error: {stats['error']}")


def benchmark_api(num_requests: int = 100):
    """
    Benchmark de rendimiento de la API
    
    Args:
        num_requests: Número de requests a enviar
    """
    print(f"=== Benchmark API ({num_requests} requests) ===")
    
    client = SignRecognitionClient()
    landmarks = [[0.5, 0.5, 0.5] for _ in range(21)]
    
    times = []
    errors = 0
    
    for i in range(num_requests):
        start_time = time.time()
        result = client.predict(landmarks)
        end_time = time.time()
        
        if "error" not in result:
            times.append((end_time - start_time) * 1000)  # ms
        else:
            errors += 1
        
        if (i + 1) % 10 == 0:
            print(f"  Completados: {i + 1}/{num_requests}")
    
    if times:
        avg_time = np.mean(times)
        min_time = np.min(times)
        max_time = np.max(times)
        
        print(f"\nResultados:")
        print(f"  Requests exitosos: {num_requests - errors}")
        print(f"  Requests fallidos: {errors}")
        print(f"  Latencia promedio: {avg_time:.2f}ms")
        print(f"  Latencia mínima: {min_time:.2f}ms")
        print(f"  Latencia máxima: {max_time:.2f}ms")
        print(f"  Throughput: {num_requests / sum(times) * 1000:.2f} req/s")
    else:
        print("No se completaron requests exitosos")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        mode = sys.argv[1]
        
        if mode == "rest":
            demo_rest_client()
        elif mode == "websocket":
            asyncio.run(demo_websocket_client())
        elif mode == "benchmark":
            benchmark_api()
        else:
            print("Uso: python client_examples.py [rest|websocket|benchmark]")
    else:
        print("=== Clientes API de Reconocimiento de Señas ===")
        print("Uso:")
        print("  python client_examples.py rest        - Demo REST API")
        print("  python client_examples.py websocket    - Demo WebSocket")
        print("  python client_examples.py benchmark    - Benchmark rendimiento")
        print()
        print("Ejemplos de código:")
        print("# Cliente REST")
        print("client = SignRecognitionClient()")
        print("result = client.predict(landmarks)")
        print()
        print("# Cliente WebSocket")
        print("client = WebSocketSignClient()")
        print("await client.connect()")
        print("result = await client.send_landmarks(landmarks)")
