"""
Human-in-the-loop (HITL): aprobación humana obligatoria para tareas críticas.

¿Qué se considera "crítico" en este sistema? Toda tarea que, además de
investigar y analizar, pide PUBLICAR el reporte (por ejemplo, en el
canal del equipo). Publicar es una acción con efectos secundarios: una
vez hecha, no se puede "deshacer" ante quien ya la leyó. Por eso, antes
de ejecutarla, el grafo se detiene con `interrupt()` y espera a que una
persona revise el reporte y lo apruebe o rechace desde la API
(POST /tasks/{id}/approve).

Mientras espera, el estado del grafo queda guardado en el checkpointer
de Redis: la pausa puede durar segundos u horas, y la API puede incluso
reiniciarse en el medio sin perder la tarea.
"""

import logging
import unicodedata
from datetime import datetime, timezone

from langchain_core.messages import AIMessage
from langgraph.types import interrupt

from app.state import OrquestadorState

logger = logging.getLogger(__name__)

NOMBRE_NODO = "aprobacion_humana"
CANAL_PUBLICACION = "#reportes-producto"

# Verbos que indican una acción con efectos secundarios fuera del sistema.
# Se comparan sin tildes y en minúscula.
_VERBOS_CRITICOS = (
    "publica",   # publicá / publicar / publicalo
    "envia",     # enviá / enviar
    "manda",     # mandá / mandar
    "comparti",  # compartí / compartir
)


def _normalizar(texto: str) -> str:
    """Pasa a minúsculas y saca tildes, para comparar palabras de forma robusta."""
    sin_tildes = unicodedata.normalize("NFKD", texto)
    sin_tildes = "".join(c for c in sin_tildes if not unicodedata.combining(c))
    return sin_tildes.lower()


def es_tarea_critica(solicitud: str) -> bool:
    """
    Clasificador determinístico: ¿la solicitud pide una acción con efectos
    secundarios (publicar, enviar, compartir el reporte)?

    Es a propósito una regla simple y no una decisión del LLM: lo que
    dispara la aprobación humana tiene que ser predecible y auditable.
    """
    texto = _normalizar(solicitud)
    return any(verbo in texto for verbo in _VERBOS_CRITICOS)


def publicar_reporte(reporte: str, canal: str = CANAL_PUBLICACION) -> dict:
    """
    Herramienta con efectos secundarios (SIMULADA): publica el reporte en
    el canal del equipo. En un sistema real acá iría la llamada a Slack,
    al mail o a la base de datos de reportes.
    """
    logger.info(f"Publicando reporte en {canal} ({len(reporte)} caracteres)")
    return {
        "publicado": True,
        "canal": canal,
        "fecha": datetime.now(timezone.utc).isoformat(),
    }


def _ultimo_reporte(state: OrquestadorState) -> str:
    """El reporte a revisar es la síntesis final que dejó el Supervisor."""
    for mensaje in reversed(state["messages"]):
        if getattr(mensaje, "name", None) == "supervisor":
            return mensaje.content
    return state["messages"][-1].content if state["messages"] else ""


def nodo_aprobacion_humana(state: OrquestadorState) -> dict:
    """
    Pausa obligatoria antes de la acción crítica.

    `interrupt()` detiene el grafo y devuelve el payload a quien lo
    invocó (el worker, que lo guarda en Redis con estado
    WAITING_APPROVAL). Cuando llega la decisión humana, el worker
    reanuda el grafo con `Command(resume=decision)` y `interrupt()`
    devuelve esa decisión.

    Nota: al reanudar, LangGraph vuelve a ejecutar este nodo desde el
    principio. Por eso todo lo que está antes de `interrupt()` no tiene
    efectos secundarios (solo arma el payload).
    """
    reporte = _ultimo_reporte(state)

    decision = interrupt(
        {
            "accion": "publicar_reporte",
            "canal": CANAL_PUBLICACION,
            "motivo": "Publicar el reporte tiene efectos secundarios: requiere aprobación humana.",
            "reporte": reporte,
        }
    )

    if isinstance(decision, dict):
        aprobado = bool(decision.get("aprobado", False))
        comentario = str(decision.get("comentario", "") or "")
    else:
        aprobado = bool(decision)
        comentario = ""

    if not aprobado:
        logger.info("Publicación rechazada por el revisor humano.")
        texto = "Publicación RECHAZADA por el revisor humano."
        if comentario:
            texto += f" Comentario: {comentario}"
        return {
            "messages": [AIMessage(content=texto, name=NOMBRE_NODO)],
            "publicacion": {"publicado": False, "comentario": comentario},
        }

    resultado = publicar_reporte(reporte)
    resultado["comentario"] = comentario
    return {
        "messages": [
            AIMessage(
                content=f"Reporte aprobado y publicado en {resultado['canal']}.",
                name=NOMBRE_NODO,
            )
        ],
        "publicacion": resultado,
    }
