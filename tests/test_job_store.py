"""Tests de la persistencia de trabajos en Redis (con fakeredis) y del worker."""

import fakeredis.aioredis
import pytest
from langgraph.checkpoint.memory import InMemorySaver

from app.graph import build_graph
from app.job_store import DONE, FAILED, PENDING, WAITING_APPROVAL, JobStore
from app.worker import WorkerPool


@pytest.fixture
def store():
    return JobStore(fakeredis.aioredis.FakeRedis(decode_responses=True), ttl_segundos=60)


async def test_crear_y_obtener_trabajo(store):
    await store.crear("j1", "consulta de prueba", requiere_aprobacion=True)
    trabajo = await store.obtener("j1")

    assert trabajo["status"] == PENDING
    assert trabajo["query"] == "consulta de prueba"
    assert trabajo["requiere_aprobacion"] is True


async def test_trabajo_inexistente_devuelve_none(store):
    assert await store.obtener("no-existe") is None


async def test_campos_json_se_guardan_y_recuperan(store):
    await store.crear("j2", "consulta", requiere_aprobacion=False)
    await store.actualizar("j2", status=DONE, result={"respuesta": "ok", "pasos": 3})
    trabajo = await store.obtener("j2")

    assert trabajo["status"] == DONE
    assert trabajo["result"] == {"respuesta": "ok", "pasos": 3}


async def test_trabajo_tiene_ttl(store):
    await store.crear("j3", "consulta", requiere_aprobacion=False)
    ttl = await store.redis.ttl("job:j3")
    assert 0 < ttl <= 60


async def test_cola_es_fifo(store):
    await store.encolar({"tipo": "run", "job_id": "a"})
    await store.encolar({"tipo": "run", "job_id": "b"})

    assert (await store.desencolar(timeout=1))["job_id"] == "a"
    assert (await store.desencolar(timeout=1))["job_id"] == "b"
    assert await store.desencolar(timeout=1) is None


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

class _GrafoQueFalla:
    async def ainvoke(self, *args, **kwargs):
        raise RuntimeError("Ollama no responde")


async def test_worker_marca_failed_si_el_grafo_lanza_excepcion(store):
    """Error común a evitar: sin esto, el cliente haría polling para siempre."""
    await store.crear("j-falla", "consulta", requiere_aprobacion=False)
    worker = WorkerPool(store, _GrafoQueFalla(), concurrencia=1)

    await worker.procesar({"tipo": "run", "job_id": "j-falla"})
    trabajo = await store.obtener("j-falla")

    assert trabajo["status"] == FAILED
    assert "Ollama no responde" in trabajo["error"]
    assert "finished_at" in trabajo


async def test_worker_completa_tarea_no_critica(store, modelo_falso):
    await store.crear("j-ok", "Analizá el checkout", requiere_aprobacion=False)
    worker = WorkerPool(store, build_graph(InMemorySaver()), concurrencia=1)

    await worker.procesar({"tipo": "run", "job_id": "j-ok"})
    trabajo = await store.obtener("j-ok")

    assert trabajo["status"] == DONE
    assert "Síntesis" in trabajo["result"]["respuesta"]
    assert set(trabajo["result"]["contribuciones"]) == {"investigador", "analista"}
    agentes = [p["agente"] for p in trabajo["result"]["traza"]]
    assert agentes.index("investigador") < agentes.index("analista")


async def test_worker_pausa_y_reanuda_tarea_critica(store, modelo_falso):
    await store.crear("j-hitl", "Analizá el checkout y publicá el reporte", requiere_aprobacion=True)
    worker = WorkerPool(store, build_graph(InMemorySaver()), concurrencia=1)

    await worker.procesar({"tipo": "run", "job_id": "j-hitl"})
    trabajo = await store.obtener("j-hitl")
    assert trabajo["status"] == WAITING_APPROVAL
    assert trabajo["interrupt"]["accion"] == "publicar_reporte"

    await worker.procesar({"tipo": "resume", "job_id": "j-hitl", "decision": {"aprobado": True}})
    trabajo = await store.obtener("j-hitl")
    assert trabajo["status"] == DONE
    assert trabajo["result"]["publicacion"]["publicado"] is True


async def test_worker_ignora_trabajo_expirado(store):
    worker = WorkerPool(store, _GrafoQueFalla(), concurrencia=1)
    await worker.procesar({"tipo": "run", "job_id": "ya-no-existe"})  # no debe lanzar
    assert await store.obtener("ya-no-existe") is None
