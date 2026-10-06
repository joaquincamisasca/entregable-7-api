"""
Tests de la API completa: FastAPI real + workers reales + grafo real,
con Redis simulado (fakeredis), InMemorySaver y el LLM falso.
"""

import time
from contextlib import asynccontextmanager

import fakeredis.aioredis
import pytest
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import InMemorySaver

from app.main import create_app


@asynccontextmanager
async def _checkpointer_en_memoria():
    yield InMemorySaver()


@pytest.fixture
def cliente(modelo_falso):
    app = create_app(
        redis_factory=lambda: fakeredis.aioredis.FakeRedis(decode_responses=True),
        checkpointer_factory=_checkpointer_en_memoria,
        concurrencia=5,
    )
    with TestClient(app) as c:
        yield c


def _esperar_estado(cliente, job_id, estados, timeout=20):
    limite = time.time() + timeout
    while time.time() < limite:
        trabajo = cliente.get(f"/tasks/{job_id}").json()
        if trabajo["status"] in estados:
            return trabajo
        time.sleep(0.05)
    raise AssertionError(f"El trabajo {job_id} no llegó a {estados}: quedó en {trabajo['status']}")


def test_health(cliente):
    assert cliente.get("/health").json() == {"status": "ok", "redis": "ok"}


def test_post_devuelve_job_id_al_instante(cliente):
    inicio = time.perf_counter()
    r = cliente.post("/tasks", json={"query": "Analizá el feedback del checkout"})
    demora = time.perf_counter() - inicio

    assert r.status_code == 202
    cuerpo = r.json()
    assert cuerpo["status"] == "PENDING"
    assert cuerpo["job_id"]
    assert demora < 1.0  # no espera a que corra el grafo


def test_flujo_completo_hasta_done(cliente):
    job_id = cliente.post("/tasks", json={"query": "Analizá el feedback del checkout"}).json()["job_id"]
    trabajo = _esperar_estado(cliente, job_id, {"DONE", "FAILED"})

    assert trabajo["status"] == "DONE"
    assert "Síntesis" in trabajo["result"]["respuesta"]


def test_cinco_peticiones_concurrentes_terminan_todas(cliente):
    ids = [
        cliente.post("/tasks", json={"query": f"Analizá el feedback del checkout #{i}"}).json()["job_id"]
        for i in range(5)
    ]
    assert len(set(ids)) == 5

    estados = [_esperar_estado(cliente, j, {"DONE", "FAILED"})["status"] for j in ids]
    assert estados == ["DONE"] * 5


def test_flujo_hitl_aprobado(cliente):
    r = cliente.post("/tasks", json={"query": "Analizá el checkout y publicá el reporte"})
    assert r.json()["requiere_aprobacion"] is True
    job_id = r.json()["job_id"]

    trabajo = _esperar_estado(cliente, job_id, {"WAITING_APPROVAL", "DONE", "FAILED"})
    assert trabajo["status"] == "WAITING_APPROVAL"
    assert trabajo["interrupt"]["accion"] == "publicar_reporte"

    r = cliente.post(f"/tasks/{job_id}/approve", json={"aprobado": True, "comentario": "OK"})
    assert r.status_code == 202

    trabajo = _esperar_estado(cliente, job_id, {"DONE", "FAILED"})
    assert trabajo["status"] == "DONE"
    assert trabajo["result"]["publicacion"]["publicado"] is True


def test_flujo_hitl_rechazado(cliente):
    job_id = cliente.post("/tasks", json={"query": "Analizá el checkout y publicá el reporte"}).json()["job_id"]
    _esperar_estado(cliente, job_id, {"WAITING_APPROVAL"})

    cliente.post(f"/tasks/{job_id}/approve", json={"aprobado": False, "comentario": "No"})
    trabajo = _esperar_estado(cliente, job_id, {"DONE", "FAILED"})

    assert trabajo["result"]["publicacion"]["publicado"] is False


def test_aprobar_tarea_que_no_espera_aprobacion_da_409(cliente):
    job_id = cliente.post("/tasks", json={"query": "Analizá el feedback del checkout"}).json()["job_id"]
    _esperar_estado(cliente, job_id, {"DONE"})

    r = cliente.post(f"/tasks/{job_id}/approve", json={"aprobado": True})
    assert r.status_code == 409


def test_trabajo_inexistente_da_404(cliente):
    assert cliente.get("/tasks/no-existe").status_code == 404
    assert cliente.post("/tasks/no-existe/approve", json={"aprobado": True}).status_code == 404


def test_query_vacia_da_422(cliente):
    assert cliente.post("/tasks", json={"query": ""}).status_code == 422
