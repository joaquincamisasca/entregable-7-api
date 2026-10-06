"""
Grafo del orquestador multi-agente (Módulo 6) + nodo de aprobación humana.

    START -> supervisor --(next_agent)--> investigador -> supervisor
                 |                        analista     -> supervisor
                 +--> FINISH --(¿tarea crítica?)--> aprobacion_humana -> END
                                     |                (interrupt: pausa)
                                     +--- no -------------------------> END

Cambios respecto del Módulo 6:
1. `build_graph(checkpointer)` recibe el checkpointer desde afuera
   (en producción, AsyncRedisSaver; en los tests, InMemorySaver).
2. Si la tarea es crítica (pide publicar el reporte), antes de terminar
   pasa por `aprobacion_humana`, que pausa el grafo hasta que una
   persona apruebe o rechace.
"""

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, StateGraph

from app import supervisor
from app.agents.analyst_agent import crear_agente_analista
from app.agents.research_agent import crear_agente_investigador
from app.hitl import NOMBRE_NODO, es_tarea_critica, nodo_aprobacion_humana
from app.state import OrquestadorState
from app.supervisor import _solicitud_original, nodo_supervisor


def _nodo_investigador(state: OrquestadorState) -> dict:
    """Ejecuta al Agente de Investigación con SOLO la instrucción puntual del Supervisor."""
    agente = crear_agente_investigador(supervisor.get_llm())
    resultado = agente.invoke({"messages": [HumanMessage(content=state["instruccion_para_agente"])]})
    respuesta = resultado["messages"][-1].content

    return {
        "messages": [AIMessage(content=respuesta, name="investigador")],
        "contribuciones": {**state.get("contribuciones", {}), "investigador": respuesta},
    }


def _nodo_analista(state: OrquestadorState) -> dict:
    """Ejecuta al Agente de Análisis con SOLO la instrucción puntual del Supervisor."""
    agente = crear_agente_analista(supervisor.get_llm())
    resultado = agente.invoke({"messages": [HumanMessage(content=state["instruccion_para_agente"])]})
    respuesta = resultado["messages"][-1].content

    return {
        "messages": [AIMessage(content=respuesta, name="analista")],
        "contribuciones": {**state.get("contribuciones", {}), "analista": respuesta},
    }


def _enrutar(state: OrquestadorState) -> str:
    """
    Arista condicional desde el Supervisor. Si el Supervisor cierra la
    tarea y esta es crítica, se pasa obligatoriamente por la aprobación
    humana antes de terminar.
    """
    destino = state["next_agent"]
    if destino != "FINISH":
        return destino
    if es_tarea_critica(_solicitud_original(state)):
        return NOMBRE_NODO
    return END


def build_graph(checkpointer=None):
    """
    Arma y compila el grafo.

    Args:
        checkpointer: dónde se persiste el estado de cada ejecución
            (thread_id = job_id). Es obligatorio para el flujo HITL:
            sin checkpointer, el grafo no puede pausarse y retomarse.
    """
    builder = StateGraph(OrquestadorState)

    builder.add_node("supervisor", nodo_supervisor)
    builder.add_node("investigador", _nodo_investigador)
    builder.add_node("analista", _nodo_analista)
    builder.add_node(NOMBRE_NODO, nodo_aprobacion_humana)

    builder.set_entry_point("supervisor")

    builder.add_conditional_edges(
        "supervisor",
        _enrutar,
        {"investigador": "investigador", "analista": "analista", NOMBRE_NODO: NOMBRE_NODO, END: END},
    )
    builder.add_edge("investigador", "supervisor")
    builder.add_edge("analista", "supervisor")
    builder.add_edge(NOMBRE_NODO, END)

    return builder.compile(checkpointer=checkpointer)
