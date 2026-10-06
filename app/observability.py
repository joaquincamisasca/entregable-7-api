"""
Observabilidad con Arize Phoenix (OpenInference / OpenTelemetry).

Dos niveles de trazas:
1. Automático: el instrumentador de LangChain de OpenInference traza
   CADA nodo del grafo y CADA
   llamada al LLM (con prompts, respuestas, tokens de entrada/salida y
   latencia). Con los tokens, Phoenix calcula el costo por ejecución.
2. Manual: el decorador `trazar()` envuelve la ejecución de cada trabajo
   en un span propio ("job.run" / "job.resume") con el job_id como
   atributo, y `using_session(job_id)` agrupa en una misma sesión de
   Phoenix la ejecución inicial y la reanudación tras la aprobación humana.

Si Phoenix no está disponible (o PHOENIX_ENABLED=false, como en los
tests), todo esto se degrada a no-op: la API funciona igual, sin trazas.
"""

import functools
import logging
from contextlib import nullcontext

from opentelemetry import trace

from app import config

logger = logging.getLogger(__name__)

_TRACER_NAME = "orquestador.api"
_inicializado = False


def init_observability() -> bool:
    """Inicializa Phoenix una sola vez. Devuelve True si quedó activo."""
    global _inicializado
    if _inicializado:
        return True
    if not config.PHOENIX_ENABLED:
        logger.info("Observabilidad deshabilitada (PHOENIX_ENABLED=false).")
        return False

    try:
        from phoenix.otel import register
    except ImportError:
        logger.warning("arize-phoenix-otel no está instalado: se continúa sin trazas.")
        return False

    tracer_provider = register(
        project_name=config.PHOENIX_PROJECT_NAME,
        endpoint=config.PHOENIX_COLLECTOR_ENDPOINT,
        auto_instrument=False,  # a propósito: ver nota abajo
        batch=True,             # envía spans en lote, sin frenar la ejecución
        verbose=False,
    )

    # Se instrumenta SOLO LangChain/LangGraph (nodos, LLM, herramientas).
    # Con auto_instrument=True, Phoenix además traza cada request HTTP de
    # FastAPI: los GET de polling (~4 ms) se convertían en trazas propias y
    # contaminaban las métricas del dashboard (el p50 daba 4 ms y el p95
    # mezclaba polling con trabajos reales). Así, 1 traza = 1 trabajo.
    from openinference.instrumentation.langchain import LangChainInstrumentor

    LangChainInstrumentor().instrument(tracer_provider=tracer_provider)
    _inicializado = True
    logger.info(
        f"Phoenix activo: proyecto '{config.PHOENIX_PROJECT_NAME}', "
        f"collector {config.PHOENIX_COLLECTOR_ENDPOINT}"
    )
    return True


def sesion(job_id: str):
    """Agrupa todos los spans de un trabajo bajo la misma sesión de Phoenix."""
    try:
        from openinference.instrumentation import using_session

        return using_session(session_id=job_id)
    except ImportError:
        return nullcontext()


def trazar(nombre: str):
    """
    Decorador para funciones async: abre un span con el job_id como
    atributo. Si no hay Phoenix configurado, el tracer de OpenTelemetry
    es un no-op y no tiene costo.
    """

    def decorador(func):
        @functools.wraps(func)
        async def envoltura(self, mensaje: dict, *args, **kwargs):
            tracer = trace.get_tracer(_TRACER_NAME)
            job_id = mensaje.get("job_id", "desconocido")
            with sesion(job_id):
                with tracer.start_as_current_span(nombre) as span:
                    span.set_attribute("job.id", job_id)
                    span.set_attribute("session.id", job_id)
                    span.set_attribute("job.tipo", mensaje.get("tipo", ""))
                    span.set_attribute("openinference.span.kind", "CHAIN")
                    return await func(self, mensaje, *args, **kwargs)

        return envoltura

    return decorador
