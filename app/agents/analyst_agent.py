"""
Agente Especialista de Análisis.

Recibe datos YA RECOLECTADOS (por ejemplo, por el Agente de
Investigación) y los procesa con sus herramientas: análisis de
sentimiento y cálculos estadísticos. Nunca sale a buscar información
por su cuenta — ese no es su rol.
"""

from typing import List

from langchain_core.tools import tool

# Heurística determinística y simple (conteo de palabras clave), elegida
# a propósito en vez de un análisis de sentimiento con LLM: hace que el
# resultado de esta herramienta sea reproducible y fácil de testear con
# asserts exactos, sin depender de la creatividad del modelo.
_PALABRAS_POSITIVAS = {
    "encantó", "excelente", "rápido", "recomiendo", "bien", "buena",
    "buenísimo", "genial", "claro",
}
_PALABRAS_NEGATIVAS = {
    "tarda", "frustrante", "error", "mala", "lento", "confuso",
    "confusa", "perdí", "problema",
}


@tool
def analizar_sentimiento(comentarios: List[str]) -> dict:
    """
    Analiza el sentimiento de una lista de comentarios de texto,
    clasificando cada uno como positivo, negativo o neutro (mixto) según
    las palabras clave que contiene.

    Args:
        comentarios: lista de textos a analizar (ej: comentarios de usuarios)

    Returns:
        {"positivos": int, "negativos": int, "neutros": int,
         "sentimiento_general": "positivo" | "negativo" | "mixto"}
    """
    positivos = negativos = neutros = 0

    for comentario in comentarios:
        texto = comentario.lower()
        tiene_positiva = any(p in texto for p in _PALABRAS_POSITIVAS)
        tiene_negativa = any(n in texto for n in _PALABRAS_NEGATIVAS)

        if tiene_positiva and not tiene_negativa:
            positivos += 1
        elif tiene_negativa and not tiene_positiva:
            negativos += 1
        else:
            neutros += 1

    if positivos > negativos:
        sentimiento_general = "positivo"
    elif negativos > positivos:
        sentimiento_general = "negativo"
    else:
        sentimiento_general = "mixto"

    return {
        "positivos": positivos,
        "negativos": negativos,
        "neutros": neutros,
        "sentimiento_general": sentimiento_general,
    }


@tool
def calcular_estadisticas(numeros: List[float]) -> dict:
    """
    Calcula estadísticas básicas (promedio, mínimo, máximo, cantidad)
    sobre una lista de valores numéricos, por ejemplo los ratings de una
    lista de comentarios de usuarios.

    Args:
        numeros: lista de valores numéricos

    Returns:
        - Si la lista no está vacía: {"promedio": float, "minimo": float,
          "maximo": float, "cantidad": int}
        - Si está vacía: {"error": str}
    """
    if not numeros:
        return {"error": "La lista de números está vacía; no hay nada que calcular."}

    return {
        "promedio": round(sum(numeros) / len(numeros), 2),
        "minimo": min(numeros),
        "maximo": max(numeros),
        "cantidad": len(numeros),
    }


TOOLS_ANALISTA = [analizar_sentimiento, calcular_estadisticas]

PROMPT_ANALISTA = (
    "Sos el Agente de Análisis. Recibís datos ya recolectados (por "
    "ejemplo, una lista de comentarios de usuarios con sus ratings) y tu "
    "trabajo es PROCESARLOS con tus herramientas: analizar_sentimiento y "
    "calcular_estadisticas. Nunca inventes datos que no te hayan sido "
    "provistos explícitamente en la instrucción — si te falta "
    "información para calcular algo, decilo en vez de asumir valores."
)


def crear_agente_analista(llm):
    """
    Crea el sub-agente ReAct del Agente de Análisis con
    create_agent (LangChain 1.x, reemplazo de create_react_agent, que
    quedó deprecado en LangGraph 1.0).

    Args:
        llm: chat model de LangChain a usar

    Returns:
        Grafo compilado del agente, listo para invocar con .invoke()
    """
    from langchain.agents import create_agent

    return create_agent(llm, tools=TOOLS_ANALISTA, system_prompt=PROMPT_ANALISTA)
