"""
Persistencia de trabajos en Redis.

Dos estructuras:
1. Un hash por trabajo (`job:{id}`) con su estado y resultado:
   PENDING -> RUNNING -> (WAITING_APPROVAL -> RUNNING) -> DONE | FAILED
2. Una cola de trabajo (lista de Redis `cola:trabajos`). La API hace
   LPUSH y los workers hacen BRPOP, así que es FIFO y además PERSISTENTE:
   si la API se reinicia, lo que estaba encolado sigue ahí.

Todo el acceso usa el cliente asíncrono de redis (`redis.asyncio`), así
que ninguna operación bloquea el event loop de FastAPI.
"""

import json
from datetime import datetime, timezone
from typing import Any, Optional

PENDING = "PENDING"
RUNNING = "RUNNING"
WAITING_APPROVAL = "WAITING_APPROVAL"
DONE = "DONE"
FAILED = "FAILED"

ESTADOS_FINALES = {DONE, FAILED}

COLA = "cola:trabajos"

# Campos que se guardan como JSON dentro del hash
_CAMPOS_JSON = {"result", "interrupt"}


def _ahora() -> str:
    return datetime.now(timezone.utc).isoformat()


def _clave(job_id: str) -> str:
    return f"job:{job_id}"


class JobStore:
    """Estado de trabajos + cola, sobre un cliente `redis.asyncio` (con decode_responses=True)."""

    def __init__(self, redis, ttl_segundos: int = 60 * 60 * 24):
        self.redis = redis
        self.ttl = ttl_segundos

    async def crear(self, job_id: str, query: str, requiere_aprobacion: bool) -> dict:
        ahora = _ahora()
        datos = {
            "job_id": job_id,
            "status": PENDING,
            "query": query,
            "requiere_aprobacion": "1" if requiere_aprobacion else "0",
            "created_at": ahora,
            "updated_at": ahora,
        }
        await self.redis.hset(_clave(job_id), mapping=datos)
        await self.redis.expire(_clave(job_id), self.ttl)
        return await self.obtener(job_id)

    async def actualizar(self, job_id: str, **campos: Any) -> None:
        datos = {"updated_at": _ahora()}
        for nombre, valor in campos.items():
            if valor is None:
                continue
            datos[nombre] = json.dumps(valor, ensure_ascii=False) if nombre in _CAMPOS_JSON else str(valor)
        await self.redis.hset(_clave(job_id), mapping=datos)
        await self.redis.expire(_clave(job_id), self.ttl)

    async def obtener(self, job_id: str) -> Optional[dict]:
        datos = await self.redis.hgetall(_clave(job_id))
        if not datos:
            return None
        for nombre in _CAMPOS_JSON:
            if nombre in datos:
                datos[nombre] = json.loads(datos[nombre])
        datos["requiere_aprobacion"] = datos.get("requiere_aprobacion") == "1"
        return datos

    async def encolar(self, mensaje: dict) -> None:
        await self.redis.lpush(COLA, json.dumps(mensaje, ensure_ascii=False))

    async def desencolar(self, timeout: int = 1) -> Optional[dict]:
        """Espera hasta `timeout` segundos por un mensaje (BRPOP, sin bloquear el event loop)."""
        item = await self.redis.brpop(COLA, timeout=timeout)
        if item is None:
            return None
        _, valor = item
        return json.loads(valor)
