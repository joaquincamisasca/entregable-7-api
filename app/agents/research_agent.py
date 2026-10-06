"""
Agente Especialista de Investigación.

Su única responsabilidad es RECOLECTAR información con su herramienta —
nunca analizarla ni sacar conclusiones (eso es trabajo del Agente de
Análisis). Esta separación de responsabilidades es a propósito: cada
especialista tiene un scope acotado, así el Supervisor puede razonar con
claridad sobre a quién delegarle cada parte de la tarea.

La herramienta de búsqueda es SIMULADA (una base de datos ficticia en
memoria), tal como habilita la consigna como alternativa a Tavily —
esto mantiene el prototipo autocontenido y testeable sin depender de
ninguna API externa de búsqueda.
"""

from typing import List

from langchain_core.tools import tool

# ---------------------------------------------------------------------------
# Base de datos ficticia en memoria: comentarios de usuarios sobre
# distintas features del producto, cada uno con un rating de 1 a 5.
# ---------------------------------------------------------------------------

_COMENTARIOS_SIMULADOS = {
    "checkout": [
        {"texto": "El nuevo checkout es mucho más rápido, me encantó la experiencia.", "rating": 5},
        {"texto": "Tarda demasiado en cargar el paso de pago, es frustrante.", "rating": 2},
        {"texto": "Buena experiencia en general, aunque el botón de confirmar es confuso.", "rating": 3},
        {"texto": "Excelente, el proceso de compra quedó buenísimo, lo recomiendo.", "rating": 5},
        {"texto": "Tuve un error al confirmar la compra, muy mala experiencia.", "rating": 1},
    ],
    "onboarding": [
        {"texto": "El tutorial inicial es claro y rápido, muy bien explicado.", "rating": 5},
        {"texto": "Me perdí en el paso 3, la explicación es confusa.", "rating": 2},
        {"texto": "Buena primera impresión, aunque es un poco largo.", "rating": 3},
    ],
}


@tool
def buscar_comentarios(tema: str) -> dict:
    """
    Busca comentarios de usuarios (simulados) sobre un tema específico
    del producto, junto con el rating (1-5) que dejó cada uno.

    Usá esta herramienta cuando necesites recolectar opiniones o
    feedback de usuarios sobre una feature puntual, antes de poder
    analizarlas.

    Args:
        tema: tema a buscar (ej: "checkout", "onboarding")

    Returns:
        - Si hay resultados: {"tema": str, "comentarios": [{"texto": str, "rating": int}, ...]}
        - Si no hay resultados para ese tema: {"error": str, "temas_disponibles": [...]}
    """
    tema_normalizado = tema.strip().lower()

    for clave, comentarios in _COMENTARIOS_SIMULADOS.items():
        if clave in tema_normalizado or tema_normalizado in clave:
            return {"tema": clave, "comentarios": comentarios}

    return {
        "error": f"No se encontraron comentarios sobre '{tema}'.",
        "temas_disponibles": list(_COMENTARIOS_SIMULADOS.keys()),
    }


TOOLS_INVESTIGADOR = [buscar_comentarios]

PROMPT_INVESTIGADOR = (
    "Sos el Agente de Investigación. Tu única función es RECOLECTAR "
    "información usando la herramienta buscar_comentarios, y devolver un "
    "resumen objetivo de lo que encontraste (texto y rating de cada "
    "comentario). NO analices el sentimiento ni saques conclusiones — "
    "eso es responsabilidad exclusiva del Agente de Análisis. Si la "
    "herramienta no encuentra resultados, decilo explícitamente e "
    "indicá los temas disponibles."
)


def crear_agente_investigador(llm):
    """
    Crea el sub-agente ReAct del Agente de Investigación con
    create_agent (LangChain 1.x, reemplazo de create_react_agent, que
    quedó deprecado en LangGraph 1.0): LLM + herramientas acotadas.

    Args:
        llm: chat model de LangChain a usar

    Returns:
        Grafo compilado del agente, listo para invocar con .invoke()
    """
    from langchain.agents import create_agent

    return create_agent(llm, tools=TOOLS_INVESTIGADOR, system_prompt=PROMPT_INVESTIGADOR)
