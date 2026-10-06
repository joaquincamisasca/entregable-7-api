"""
Worker: ejecuta los trabajos en segundo plano.

Patrón: N corrutinas (WORKER_CONCURRENCY) que leen de la cola de Redis
con BRPOP. Cada mensaje es:
- {"tipo": "run",    "job_id": ...}                    -> ejecuta el grafo desde cero
- {"tipo": "resume", "job_id": ..., "decision": {...}}  -> reanuda tras la aprobación humana

Viven dentro del mismo proceso que la API (arrancan en el `lifespan` de
FastAPI). Es la variante "thread/tarea asíncrona simple" que habilita la
consigna; como la cola está en Redis, pasar a un proceso aparte (Arq,
Celery) solo cambiaría quién consume esa misma cola.

Regla de oro: si el grafo falla por cualquier motivo, el trabajo pasa a
FAILED con el error guardado. Así el cliente que hace polling nunca
queda esperando para siempre.
"""

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Optional

from langchain_core.messages import HumanMessage
from langgraph.types import Command

from app import config
from app.job_store import DONE, FAILED, RUNNING, WAITING_APPROVAL, JobStore
from app.observability import trazar

logger = logging.getLogger(__name__)


def _respuesta_final(valores: dict) -> str:
    """La respuesta es la síntesis del Supervisor (último mensaje suyo)."""
    for mensaje in reversed(valores.get("messages", [])):
        if getattr(mensaje, "name", None) == "supervisor":
            return mensaje.content
    mensajes = valores.get("messages", [])
    return mensajes[-1].content if mensajes else ""


def _traza(valores: dict) -> list:
    """Traza legible de la delegación: quién dijo qué."""
    return [
        {"agente": getattr(m, "name", None) or "usuario", "contenido": m.content}
        for m in valores.get("messages", [])
    ]


class WorkerPool:
    def __init__(self, store: JobStore, grafo, concurrencia: int = config.WORKER_CONCURRENCY):
        self.store = store
        self.grafo = grafo
        self.concurrencia = concurrencia
        self._tareas: list[asyncio.Task] = []
        self._detener = asyncio.Event()

    def iniciar(self) -> None:
        for n in range(self.concurrencia):
            self._tareas.append(asyncio.create_task(self._bucle(n), name=f"worker-{n}"))
        logger.info(f"{self.concurrencia} workers iniciados.")

    async def detener(self) -> None:
        self._detener.set()
        for tarea in self._tareas:
            tarea.cancel()
        await asyncio.gather(*self._tareas, return_exceptions=True)
        self._tareas.clear()

    async def _bucle(self, n: int) -> None:
        while not self._detener.is_set():
            try:
                mensaje = await self.store.desencolar(timeout=1)
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001 - Redis caído: reintentar sin matar al worker
                logger.error(f"worker-{n}: error leyendo la cola ({e}); reintento en 1s")
                await asyncio.sleep(1)
                continue

            if mensaje is not None:
                await self.procesar(mensaje)

    @trazar("job.ejecucion")
    async def procesar(self, mensaje: dict) -> None:
        """Procesa un mensaje de la cola. Nunca deja escapar una excepción."""
        job_id = mensaje["job_id"]
        cfg = {"configurable": {"thread_id": job_id}, "recursion_limit": config.RECURSION_LIMIT}
        inicio = time.perf_counter()

        try:
            trabajo = await self.store.obtener(job_id)
            if trabajo is None:
                logger.warning(f"Trabajo {job_id} inexistente (¿expiró?); se descarta el mensaje.")
                return

            await self.store.actualizar(job_id, status=RUNNING, started_at=_ahora_si_falta(trabajo))

            if mensaje.get("tipo") == "resume":
                entrada = Command(resume=mensaje.get("decision", {}))
            else:
                entrada = {
                    "messages": [HumanMessage(content=trabajo["query"])],
                    "contribuciones": {},
                    "pasos": 0,
                    "publicacion": None,
                }

            await self.grafo.ainvoke(entrada, cfg)
            snapshot = await self.grafo.aget_state(cfg)

            if snapshot.next:
                # El grafo quedó pausado en el nodo de aprobación humana.
                interrupciones = [i.value for t in snapshot.tasks for i in t.interrupts]
                await self.store.actualizar(
                    job_id,
                    status=WAITING_APPROVAL,
                    interrupt=interrupciones[0] if interrupciones else {},
                )
                logger.info(f"Trabajo {job_id} esperando aprobación humana.")
                return

            valores = snapshot.values
            await self.store.actualizar(
                job_id,
                status=DONE,
                finished_at=_ahora(),
                duracion_s=round(time.perf_counter() - inicio, 3),
                result={
                    "respuesta": _respuesta_final(valores),
                    "contribuciones": valores.get("contribuciones", {}),
                    "pasos_supervisor": valores.get("pasos", 0),
                    "publicacion": valores.get("publicacion"),
                    "traza": _traza(valores),
                },
            )
            logger.info(f"Trabajo {job_id} terminado (DONE).")

        except Exception as e:  # noqa: BLE001 - cualquier fallo debe terminar en FAILED
            logger.exception(f"Trabajo {job_id} falló")
            try:
                await self.store.actualizar(
                    job_id,
                    status=FAILED,
                    error=f"{type(e).__name__}: {e}",
                    finished_at=_ahora(),
                )
            except Exception:  # noqa: BLE001
                logger.exception(f"No se pudo marcar {job_id} como FAILED en Redis")


def _ahora() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ahora_si_falta(trabajo: dict) -> Optional[str]:
    """started_at se registra solo la primera vez (no se pisa al reanudar)."""
    return None if trabajo.get("started_at") else _ahora()
