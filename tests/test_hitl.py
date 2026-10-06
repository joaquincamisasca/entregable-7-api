"""Tests del flujo Human-in-the-loop sobre el grafo real (con InMemorySaver y LLM falso)."""

import pytest
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from app.graph import build_graph
from app.hitl import NOMBRE_NODO, es_tarea_critica

CRITICA = "Analizá los comentarios del checkout y publicá el reporte en el canal del equipo."
NO_CRITICA = "Analizá los comentarios del checkout y calculá el rating promedio."


@pytest.mark.parametrize(
    "texto, esperado",
    [
        (CRITICA, True),
        ("Envía el resumen al equipo", True),
        ("mandá el informe por mail", True),
        ("Compartí los resultados", True),
        ("PUBLICAR reporte", True),
        (NO_CRITICA, False),
        ("¿Qué opinan los usuarios del onboarding?", False),
    ],
)
def test_es_tarea_critica(texto, esperado):
    assert es_tarea_critica(texto) is esperado


def _entrada(texto):
    return {"messages": [HumanMessage(content=texto)], "contribuciones": {}, "pasos": 0, "publicacion": None}


async def test_tarea_no_critica_termina_sin_pausa(modelo_falso):
    grafo = build_graph(InMemorySaver())
    cfg = {"configurable": {"thread_id": "no-critica"}}

    await grafo.ainvoke(_entrada(NO_CRITICA), cfg)
    snapshot = await grafo.aget_state(cfg)

    assert snapshot.next == ()
    assert snapshot.values.get("publicacion") is None


async def test_tarea_critica_se_pausa_antes_de_publicar(modelo_falso):
    grafo = build_graph(InMemorySaver())
    cfg = {"configurable": {"thread_id": "critica"}}

    await grafo.ainvoke(_entrada(CRITICA), cfg)
    snapshot = await grafo.aget_state(cfg)

    assert snapshot.next == (NOMBRE_NODO,)
    payload = snapshot.tasks[0].interrupts[0].value
    assert payload["accion"] == "publicar_reporte"
    assert "Síntesis" in payload["reporte"]  # el humano revisa el reporte real
    assert snapshot.values.get("publicacion") is None  # todavía NO se publicó


async def test_aprobacion_publica_el_reporte(modelo_falso):
    grafo = build_graph(InMemorySaver())
    cfg = {"configurable": {"thread_id": "aprobada"}}

    await grafo.ainvoke(_entrada(CRITICA), cfg)
    await grafo.ainvoke(Command(resume={"aprobado": True, "comentario": "OK"}), cfg)
    snapshot = await grafo.aget_state(cfg)

    assert snapshot.next == ()
    assert snapshot.values["publicacion"]["publicado"] is True
    assert snapshot.values["publicacion"]["comentario"] == "OK"


async def test_rechazo_no_publica(modelo_falso):
    grafo = build_graph(InMemorySaver())
    cfg = {"configurable": {"thread_id": "rechazada"}}

    await grafo.ainvoke(_entrada(CRITICA), cfg)
    await grafo.ainvoke(Command(resume={"aprobado": False, "comentario": "Faltan datos"}), cfg)
    snapshot = await grafo.aget_state(cfg)

    assert snapshot.next == ()
    assert snapshot.values["publicacion"]["publicado"] is False
    assert "RECHAZADA" in snapshot.values["messages"][-1].content
