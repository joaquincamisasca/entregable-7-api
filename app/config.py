"""
Configuración centralizada, leída de variables de entorno (.env).
Ningún secreto queda escrito en el código.
"""

import os

from dotenv import load_dotenv

load_dotenv()

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

# "redis" (producción, AsyncRedisSaver) o "memory" (solo para pruebas
# locales sin Redis Stack; el estado del grafo se pierde al reiniciar).
CHECKPOINTER = os.getenv("CHECKPOINTER", "redis").lower()

# Cantidad de workers que consumen la cola en paralelo. Con 5, las 5
# peticiones concurrentes de la prueba de carga se procesan a la vez.
WORKER_CONCURRENCY = int(os.getenv("WORKER_CONCURRENCY", "5"))

# Cuánto tiempo se conserva en Redis el estado de cada trabajo.
JOB_TTL_SECONDS = int(os.getenv("JOB_TTL_SECONDS", str(60 * 60 * 24)))

# Techo de pasos de LangGraph por ejecución (red de seguridad extra,
# además del MAX_PASOS del Supervisor).
RECURSION_LIMIT = int(os.getenv("RECURSION_LIMIT", "25"))

# Observabilidad (Arize Phoenix)
PHOENIX_ENABLED = os.getenv("PHOENIX_ENABLED", "true").lower() == "true"
PHOENIX_COLLECTOR_ENDPOINT = os.getenv("PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:6006/v1/traces")
PHOENIX_PROJECT_NAME = os.getenv("PHOENIX_PROJECT_NAME", "orquestador-multiagente")
