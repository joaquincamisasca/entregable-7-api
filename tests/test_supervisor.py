"""
Tests del nodo Supervisor con un LLM FALSO (sin necesitar ninguna API).

Cubren especialmente el fallo real observado con llama3.2 en Ollama:
`with_structured_output()` devolvió None en vez de una decisión, y el
Supervisor se rompía con AttributeError. Ahora debe caer al plan B por
reglas, y la rúbrica de validación debe impedir cierres prematuros.
"""

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableLambda

from app import supervisor
from app.supervisor import DecisionSupervisor, nodo_supervisor


class _LLMFalso:
    """Simula un chat model: with_structured_output() devuelve lo configurado."""

    def __init__(self, respuestas_decision):
        # Lista de valores a devolver en cada intento (DecisionSupervisor, None o Exception).
        self._respuestas = list(respuestas_decision)
        self.intentos_realizados = 0

    def with_structured_output(self, _esquema):
        def _invocar(_mensajes):
            self.intentos_realizados += 1
            valor = self._respuestas.pop(0) if self._respuestas else None
            if isinstance(valor, Exception):
                raise valor
            return valor

        return RunnableLambda(_invocar)

    def invoke(self, _mensajes):
        # Usado para la síntesis final.
        return AIMessage(content="Síntesis final de prueba.")


def _estado(contribuciones=None, pasos=0):
    return {
        "messages": [HumanMessage(content="Analizá el feedback del checkout.")],
        "contribuciones": contribuciones or {},
        "pasos": pasos,
    }


def test_llm_devuelve_none_cae_al_plan_b_y_delega_al_investigador(monkeypatch):
    """El caso real que rompió el grafo: el LLM devuelve None."""
    llm = _LLMFalso([None, None])
    monkeypatch.setattr(supervisor, "get_llm", lambda: llm)

    resultado = nodo_supervisor(_estado())

    assert resultado["next_agent"] == "investigador"
    assert resultado["instruccion_para_agente"].strip() != ""
    assert llm.intentos_realizados == supervisor.INTENTOS_DECISION  # reintentó antes del plan B


def test_llm_lanza_excepcion_tambien_cae_al_plan_b(monkeypatch):
    llm = _LLMFalso([ValueError("json inválido"), ValueError("json inválido")])
    monkeypatch.setattr(supervisor, "get_llm", lambda: llm)

    resultado = nodo_supervisor(_estado())

    assert resultado["next_agent"] == "investigador"


def test_reintento_exitoso_no_usa_plan_b(monkeypatch):
    """Si el segundo intento del LLM sí sirve, se usa su decisión."""
    valida = DecisionSupervisor(
        next_agent="investigador",
        instruccion="Buscá comentarios sobre checkout.",
        razon="Faltan datos.",
    )
    llm = _LLMFalso([None, valida])
    monkeypatch.setattr(supervisor, "get_llm", lambda: llm)

    resultado = nodo_supervisor(_estado())

    assert resultado["next_agent"] == "investigador"
    assert resultado["instruccion_para_agente"] == "Buscá comentarios sobre checkout."


def test_plan_b_delega_al_analista_con_los_datos_del_investigador(monkeypatch):
    llm = _LLMFalso([None, None])
    monkeypatch.setattr(supervisor, "get_llm", lambda: llm)

    estado = _estado(contribuciones={"investigador": "5 comentarios, ratings 5,2,3,5,1"})
    resultado = nodo_supervisor(estado)

    assert resultado["next_agent"] == "analista"
    assert "5 comentarios, ratings 5,2,3,5,1" in resultado["instruccion_para_agente"]


def test_rubrica_impide_cerrar_antes_de_tiempo(monkeypatch):
    """Si el LLM dice FINISH sin que haya contribuido nadie, se corrige."""
    finish_prematuro = DecisionSupervisor(next_agent="FINISH", instruccion="", razon="Ya está.")
    llm = _LLMFalso([finish_prematuro])
    monkeypatch.setattr(supervisor, "get_llm", lambda: llm)

    resultado = nodo_supervisor(_estado())

    assert resultado["next_agent"] == "investigador"


def test_rubrica_impide_saltear_al_investigador(monkeypatch):
    """Si el LLM manda al analista sin datos, se corrige al investigador."""
    salto = DecisionSupervisor(next_agent="analista", instruccion="Analizá.", razon="...")
    llm = _LLMFalso([salto])
    monkeypatch.setattr(supervisor, "get_llm", lambda: llm)

    resultado = nodo_supervisor(_estado())

    assert resultado["next_agent"] == "investigador"


def test_analista_siempre_recibe_los_datos_aunque_el_llm_los_omita(monkeypatch):
    decision = DecisionSupervisor(next_agent="analista", instruccion="Analizá el sentimiento.", razon="...")
    llm = _LLMFalso([decision])
    monkeypatch.setattr(supervisor, "get_llm", lambda: llm)

    estado = _estado(contribuciones={"investigador": "DATOS-DEL-INVESTIGADOR"})
    resultado = nodo_supervisor(estado)

    assert resultado["next_agent"] == "analista"
    assert "DATOS-DEL-INVESTIGADOR" in resultado["instruccion_para_agente"]


def test_cierra_con_sintesis_cuando_ambos_contribuyeron(monkeypatch):
    finish = DecisionSupervisor(next_agent="FINISH", instruccion="", razon="Completo.")
    llm = _LLMFalso([finish])
    monkeypatch.setattr(supervisor, "get_llm", lambda: llm)

    estado = _estado(contribuciones={"investigador": "datos", "analista": "resultados"})
    resultado = nodo_supervisor(estado)

    assert resultado["next_agent"] == "FINISH"
    assert resultado["messages"][-1].content == "Síntesis final de prueba."
    assert resultado["messages"][-1].name == "supervisor"


def test_corte_por_max_pasos_no_llama_al_llm(monkeypatch):
    def _no_debe_llamarse():
        raise AssertionError("No debería llamarse al LLM después de MAX_PASOS")

    monkeypatch.setattr(supervisor, "get_llm", _no_debe_llamarse)

    resultado = nodo_supervisor(_estado(pasos=supervisor.MAX_PASOS))

    assert resultado["next_agent"] == "FINISH"
    assert "máximo" in resultado["messages"][-1].content


def test_la_delegacion_queda_registrada_en_la_traza(monkeypatch):
    llm = _LLMFalso([None, None])
    monkeypatch.setattr(supervisor, "get_llm", lambda: llm)

    resultado = nodo_supervisor(_estado())

    assert resultado["messages"][-1].name == "supervisor"
    assert "investigador" in resultado["messages"][-1].content
