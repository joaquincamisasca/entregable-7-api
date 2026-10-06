"""
API REST (FastAPI) del orquestador multi-agente.

Endpoints:
- POST /tasks                 -> encola la tarea y devuelve job_id al instante (202)
- GET  /tasks/{job_id}        -> estado y resultado del trabajo (polling)
- POST /tasks/{job_id}/approve -> decisión humana para tareas críticas (HITL)
- GET  /health                -> chequeo de Redis

Ningún endpoint ejecuta el grafo ni llama a un LLM: solo leen/escriben
en Redis con el cliente asíncrono. El trabajo pesado lo hacen los
workers en segundo plano, así el event loop nunca se bloquea.

Para levantarla:  uvicorn app.main:app --reload
"""

import logging
import uuid
from contextlib import asynccontextmanager
from typing import Callable, Optional

from fastapi import FastAPI, HTTPException, Request, status
from pydantic import BaseModel, ConfigDict, Field

from app import config
from app.graph import build_graph
from app.hitl import es_tarea_critica
from app.job_store import WAITING_APPROVAL, JobStore
from app.observability import init_observability
from app.worker import WorkerPool

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Modelos de entrada/salida (Pydantic v2, con model_config)
# ---------------------------------------------------------------------------

class TareaEntrada(BaseModel):
    query: str = Field(..., min_length=3, description="Solicitud en lenguaje natural para el orquestador.")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "query": "Buscá los comentarios sobre el checkout, analizá el sentimiento y calculá el rating promedio."
            }
        }
    )


class TareaCreada(BaseModel):
    job_id: str
    status: str
    requiere_aprobacion: bool


class Aprobacion(BaseModel):
    aprobado: bool = Field(True, description="True para aprobar la acción crítica, False para rechazarla.")
    comentario: str = Field("", description="Comentario opcional del revisor humano.")

    model_config = ConfigDict(
        json_schema_extra={"example": {"aprobado": True, "comentario": "Revisado, OK para publicar."}}
    )


# ---------------------------------------------------------------------------
# Fábricas por defecto (en los tests se reemplazan por versiones simuladas)
# ---------------------------------------------------------------------------

def _crear_redis_por_defecto():
    from redis.asyncio import Redis

    return Redis.from_url(config.REDIS_URL, decode_responses=True)


@asynccontextmanager
async def _crear_checkpointer_por_defecto():
    if config.CHECKPOINTER == "memory":
        from langgraph.checkpoint.memory import InMemorySaver

        logger.warning("CHECKPOINTER=memory: el estado del grafo NO se persiste en Redis.")
        yield InMemorySaver()
        return

    from langgraph.checkpoint.redis.aio import AsyncRedisSaver

    async with AsyncRedisSaver.from_conn_string(config.REDIS_URL) as saver:
        logger.info("Checkpointer de LangGraph: AsyncRedisSaver (Redis).")
        yield saver


def _crear_fastapi(lifespan) -> FastAPI:
    """
    Crea la app SIN trazas HTTP automáticas.

    Las versiones recientes de FastAPI emiten por su cuenta un span por
    cada request cuando hay un tracer global (que Phoenix configura). Los
    GET de polling pasarían a ser trazas propias de ~4 ms y contaminarían
    el p95 y el costo por ejecución del dashboard. Queremos 1 traza = 1
    trabajo, así que se apaga ese tracing. Las versiones viejas de FastAPI
    no tienen esa opción (y tampoco emiten esos spans), por eso el intento
    tolerante.
    """
    opciones = dict(
        title="Orquestador Multi-Agente — API",
        description="API asíncrona sobre el orquestador del Módulo 6, con Redis, Phoenix y aprobación humana.",
        version="1.0.0",
        lifespan=lifespan,
    )
    try:
        return FastAPI(**opciones, telemetry={"tracing": False})
    except TypeError:
        return FastAPI(**opciones)


def create_app(
    redis_factory: Optional[Callable] = None,
    checkpointer_factory: Optional[Callable] = None,
    concurrencia: int = config.WORKER_CONCURRENCY,
) -> FastAPI:
    redis_factory = redis_factory or _crear_redis_por_defecto
    checkpointer_factory = checkpointer_factory or _crear_checkpointer_por_defecto

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        init_observability()
        redis = redis_factory()
        store = JobStore(redis, ttl_segundos=config.JOB_TTL_SECONDS)

        async with checkpointer_factory() as checkpointer:
            grafo = build_graph(checkpointer)
            workers = WorkerPool(store, grafo, concurrencia)
            workers.iniciar()

            app.state.store = store
            app.state.redis = redis
            try:
                yield
            finally:
                await workers.detener()

        await redis.aclose()

    app = _crear_fastapi(lifespan)

    @app.post("/tasks", response_model=TareaCreada, status_code=status.HTTP_202_ACCEPTED)
    async def crear_tarea(entrada: TareaEntrada, request: Request) -> TareaCreada:
        """Encola la tarea y devuelve el job_id inmediatamente (no espera al grafo)."""
        store: JobStore = request.app.state.store
        job_id = str(uuid.uuid4())
        critica = es_tarea_critica(entrada.query)

        await store.crear(job_id, entrada.query, requiere_aprobacion=critica)
        await store.encolar({"tipo": "run", "job_id": job_id})

        logger.info(f"Tarea {job_id} encolada (crítica={critica}).")
        return TareaCreada(job_id=job_id, status="PENDING", requiere_aprobacion=critica)

    @app.get("/tasks/{job_id}")
    async def obtener_tarea(job_id: str, request: Request) -> dict:
        """Estado actual del trabajo. Pensado para polling."""
        trabajo = await request.app.state.store.obtener(job_id)
        if trabajo is None:
            raise HTTPException(status_code=404, detail=f"No existe el trabajo {job_id}.")
        return trabajo

    @app.post("/tasks/{job_id}/approve", status_code=status.HTTP_202_ACCEPTED)
    async def aprobar_tarea(job_id: str, aprobacion: Aprobacion, request: Request) -> dict:
        """
        Decisión humana sobre una tarea pausada (WAITING_APPROVAL).
        Encola la reanudación del grafo y vuelve al instante.
        """
        store: JobStore = request.app.state.store
        trabajo = await store.obtener(job_id)
        if trabajo is None:
            raise HTTPException(status_code=404, detail=f"No existe el trabajo {job_id}.")
        if trabajo["status"] != WAITING_APPROVAL:
            raise HTTPException(
                status_code=409,
                detail=f"El trabajo está en estado {trabajo['status']}, no espera aprobación.",
            )

        # Se marca RUNNING antes de encolar: así un doble click no encola dos reanudaciones.
        await store.actualizar(job_id, status="RUNNING")
        await store.encolar({"tipo": "resume", "job_id": job_id, "decision": aprobacion.model_dump()})

        logger.info(f"Tarea {job_id}: decisión humana recibida (aprobado={aprobacion.aprobado}).")
        return {"job_id": job_id, "status": "RUNNING", "aprobado": aprobacion.aprobado}

    @app.get("/health")
    async def health(request: Request) -> dict:
        try:
            await request.app.state.redis.ping()
            return {"status": "ok", "redis": "ok"}
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=503, detail=f"Redis no disponible: {e}")

    return app


app = create_app()
