"""
Supervisor: el nodo router del orquestador.

En cada vuelta, mira la conversación acumulada hasta el momento (y qué
aportó cada especialista) y decide, con salida estructurada, quién actúa
a continuación — o si la tarea ya está completa.

Robustez (lección aprendida en la práctica con modelos locales chicos
como llama3.2, que a veces NO devuelven la salida estructurada pedida):

1. Reintento: si el LLM no devuelve una decisión válida, se reintenta.
2. Plan B por reglas: si igual falla, la decisión se toma con reglas
   determinísticas (¿quién falta contribuir?) en vez de romper el grafo.
3. Rúbrica de validación en código: pase lo que pase con el LLM, no se
   puede cerrar la tarea (FINISH) ni saltearse el orden hasta que los
   especialistas requeridos hayan aportado. Esto cumple el requisito de
   "Validación" de la consigna: el Supervisor valida los resultados de
   los especialistas antes de dar el END.

Incluye además dos salvaguardas contra el "Supervisor Infinito":
- Un contador de pasos (`pasos`) que fuerza el cierre después de un
  máximo de vueltas, sin importar qué decida el LLM.
- Un criterio de "Suficiencia" explícito en el prompt.
"""

import logging
import os
from typing import Literal, Optional

from langchain_core.messages import AIMessage
from pydantic import BaseModel, ConfigDict, Field

from app.state import OrquestadorState

logger = logging.getLogger(__name__)

GENERATION_PROVIDER = os.getenv("GENERATION_PROVIDER", "ollama")

# Techo absoluto de vueltas del Supervisor por invocación. Es la
# salvaguarda "dura": ni el mejor prompt garantiza que el LLM converja
# siempre, así que esto corta el ciclo pase lo que pase.
MAX_PASOS = int(os.getenv("MAX_PASOS_SUPERVISOR", "6"))

# Especialistas que DEBEN haber contribuido (en este orden) antes de que
# se pueda dar la tarea por completa. Es la rúbrica de validación.
ESPECIALISTAS_REQUERIDOS = ("investigador", "analista")

# Cuántas veces se le pide al LLM una decisión válida antes de pasar al
# plan B por reglas.
INTENTOS_DECISION = 2


def get_llm(
    provider: str = GENERATION_PROVIDER,
    model: Optional[str] = None,
    temperature: float = 0.0,
):
    """Crea el chat model de LangChain para el proveedor pedido (mismo patrón de entregables anteriores)."""
    if provider == "openai":
        from langchain_openai import ChatOpenAI

        model = model or os.getenv("OPENAI_MODEL", "gpt-4o")
        return ChatOpenAI(model=model, temperature=temperature, api_key=os.getenv("OPENAI_API_KEY"))

    elif provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        model = model or os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022")
        return ChatAnthropic(
            model=model, temperature=temperature, api_key=os.getenv("ANTHROPIC_API_KEY")
        )

    elif provider == "ollama":
        from langchain_ollama import ChatOllama

        model = model or os.getenv("OLLAMA_MODEL", "llama3.2")
        return ChatOllama(model=model, temperature=temperature)

    else:
        raise ValueError(f"Proveedor no soportado: {provider}")


class DecisionSupervisor(BaseModel):
    """Salida estructurada del Supervisor: a quién delegar y con qué instrucción."""

    next_agent: Literal["investigador", "analista", "FINISH"] = Field(
        description="Próximo nodo a ejecutar, o FINISH si la tarea ya está completa."
    )
    instruccion: str = Field(
        default="",
        description=(
            "Instrucción puntual y acotada para el agente elegido (solo "
            "lo que necesita para hacer SU parte, no todo el contexto). "
            "Vacío si next_agent es FINISH."
        ),
    )
    razon: str = Field(
        default="",
        description="Justificación breve de la decisión, para trazabilidad.",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "next_agent": "investigador",
                "instruccion": "Buscá comentarios de usuarios sobre 'checkout'.",
                "razon": "Todavía no se recolectó ningún dato sobre el tema pedido.",
            }
        }
    )


PROMPT_SUPERVISOR = (
    "Sos el Supervisor de un equipo de dos agentes especialistas:\n"
    "- 'investigador': RECOLECTA información (comentarios de usuarios). "
    "No analiza, solo busca y resume objetivamente.\n"
    "- 'analista': PROCESA información ya recolectada (análisis de "
    "sentimiento, cálculos estadísticos). Nunca sale a buscar datos.\n\n"
    "Dada la conversación y las contribuciones de cada agente hasta "
    "ahora, decidí quién debe actuar a continuación, o si la tarea ya "
    "está completa (FINISH).\n\n"
    "Rúbrica de validación: primero debe actuar el investigador; recién "
    "con sus datos, el analista. No elijas FINISH hasta que ambos hayan "
    "contribuido.\n\n"
    "Criterio de Suficiencia (estricto): una vez que ambos contribuyeron "
    "y sus resultados responden la solicitud original, elegí FINISH — no "
    "sigas pidiendo refinamientos por perfeccionismo.\n\n"
    "Contribuciones hasta ahora:\n{contribuciones}\n\n"
    "Si delegás, la 'instruccion' debe ser puntual: decile al agente "
    "exactamente qué necesitás de él, sin repetirle toda la "
    "conversación (el agente NO ve el historial completo, solo tu "
    "instrucción)."
)


def _formatear_contribuciones(contribuciones: dict) -> str:
    if not contribuciones:
        return "(ninguna todavía)"
    return "\n".join(f"- {agente}: {resumen}" for agente, resumen in contribuciones.items())


def _solicitud_original(state: OrquestadorState) -> str:
    """Devuelve el texto del primer mensaje del usuario (la solicitud original)."""
    for mensaje in state["messages"]:
        if getattr(mensaje, "type", None) == "human":
            return mensaje.content
    return ""


def _instruccion_por_defecto(agente: str, state: OrquestadorState) -> str:
    """Instrucción puntual de respaldo para un especialista, armada por reglas."""
    solicitud = _solicitud_original(state)
    contribuciones = state.get("contribuciones", {})

    if agente == "investigador":
        return (
            "Buscá los comentarios de usuarios relevantes para esta "
            f"solicitud y devolvé el texto y el rating de cada uno: {solicitud}"
        )

    if agente == "analista":
        datos = contribuciones.get("investigador", "(sin datos)")
        return (
            "Con los datos recolectados por el investigador, analizá el "
            "sentimiento general y calculá el rating promedio. Usá tus "
            f"herramientas.\n\nSolicitud original: {solicitud}\n\n"
            f"Datos recolectados:\n{datos}"
        )

    return ""


def _decision_por_reglas(state: OrquestadorState) -> DecisionSupervisor:
    """
    Plan B determinístico: si el LLM no logra dar una decisión válida,
    se elige al primer especialista requerido que todavía no contribuyó;
    si ya contribuyeron todos, se cierra.
    """
    contribuciones = state.get("contribuciones", {})

    for agente in ESPECIALISTAS_REQUERIDOS:
        if agente not in contribuciones:
            return DecisionSupervisor(
                next_agent=agente,
                instruccion=_instruccion_por_defecto(agente, state),
                razon=f"Decisión por reglas: '{agente}' todavía no contribuyó.",
            )

    return DecisionSupervisor(
        next_agent="FINISH",
        instruccion="",
        razon="Decisión por reglas: todos los especialistas ya contribuyeron.",
    )


def _decidir_con_llm(llm, mensajes: list) -> Optional[DecisionSupervisor]:
    """
    Le pide una decisión estructurada al LLM, con reintentos. Devuelve
    None si tras INTENTOS_DECISION intentos no obtuvo una decisión válida
    (modelos locales chicos a veces devuelven None o lanzan errores de
    parseo en vez de la salida estructurada pedida).
    """
    cadena = llm.with_structured_output(DecisionSupervisor)

    for intento in range(1, INTENTOS_DECISION + 1):
        try:
            decision = cadena.invoke(mensajes)
        except Exception as e:  # noqa: BLE001 - cualquier fallo del LLM debe caer al plan B
            logger.warning(f"Intento {intento}: el LLM falló al decidir ({type(e).__name__}: {e})")
            continue

        if isinstance(decision, DecisionSupervisor):
            return decision

        logger.warning(f"Intento {intento}: el LLM no devolvió una decisión estructurada válida.")

    return None


def _validar_decision(decision: DecisionSupervisor, state: OrquestadorState) -> DecisionSupervisor:
    """
    Rúbrica de validación (en código, independiente de lo que diga el LLM):

    1. Mientras falten especialistas requeridos, el próximo tiene que ser
       el primero que falta (así se respeta el orden investigador ->
       analista y no se puede cerrar antes de tiempo).
    2. Al analista siempre se le pasan los datos del investigador.
    3. Si se delega, la instrucción no puede quedar vacía.
    """
    contribuciones = state.get("contribuciones", {})
    faltantes = [a for a in ESPECIALISTAS_REQUERIDOS if a not in contribuciones]

    if faltantes and decision.next_agent != faltantes[0]:
        logger.warning(
            f"Decisión del LLM ('{decision.next_agent}') corregida a '{faltantes[0]}' "
            f"por la rúbrica de validación."
        )
        return DecisionSupervisor(
            next_agent=faltantes[0],
            instruccion=_instruccion_por_defecto(faltantes[0], state),
            razon=(
                f"Corregido por la rúbrica: primero debe actuar '{faltantes[0]}' "
                f"(el LLM había elegido '{decision.next_agent}')."
            ),
        )

    if decision.next_agent == "FINISH":
        return decision

    instruccion = decision.instruccion.strip()

    if not instruccion:
        instruccion = _instruccion_por_defecto(decision.next_agent, state)

    if decision.next_agent == "analista":
        datos = contribuciones.get("investigador")
        if datos and datos not in instruccion:
            instruccion += f"\n\nDatos recolectados por el investigador:\n{datos}"

    return decision.model_copy(update={"instruccion": instruccion})


def nodo_supervisor(state: OrquestadorState) -> dict:
    """
    Nodo del Supervisor: decide next_agent + instruccion, incrementa el
    contador de pasos, y corta por las dudas si se llegó a MAX_PASOS.
    """
    pasos = state.get("pasos", 0) + 1

    if pasos > MAX_PASOS:
        # Salvaguarda dura contra el "Supervisor Infinito": no importa
        # qué diga el LLM, después de MAX_PASOS vueltas se corta.
        return {
            "next_agent": "FINISH",
            "pasos": pasos,
            "messages": [
                AIMessage(
                    content=(
                        f"Se alcanzó el máximo de {MAX_PASOS} pasos sin que el "
                        f"Supervisor declarara la tarea completa. Cerrando con "
                        f"la información disponible hasta el momento."
                    ),
                    name="supervisor",
                )
            ],
        }

    llm = get_llm()
    contribuciones = state.get("contribuciones", {})

    prompt_con_contexto = PROMPT_SUPERVISOR.format(
        contribuciones=_formatear_contribuciones(contribuciones)
    )
    mensajes = [{"role": "system", "content": prompt_con_contexto}, *state["messages"]]

    decision = _decidir_con_llm(llm, mensajes)
    if decision is None:
        logger.warning("El LLM no logró decidir; se usa el plan B por reglas.")
        decision = _decision_por_reglas(state)

    decision = _validar_decision(decision, state)

    resultado = {
        "next_agent": decision.next_agent,
        "instruccion_para_agente": decision.instruccion,
        "pasos": pasos,
    }

    if decision.next_agent == "FINISH":
        # Si el Supervisor decide terminar, sintetiza la respuesta final
        # a partir de las contribuciones acumuladas.
        mensaje_sintesis = llm.invoke(
            [
                {
                    "role": "system",
                    "content": (
                        "Sintetizá una respuesta final clara para el usuario, "
                        "en español, basándote ÚNICAMENTE en las contribuciones "
                        "de los agentes especialistas:\n"
                        f"{_formatear_contribuciones(contribuciones)}"
                    ),
                },
                {"role": "user", "content": _solicitud_original(state)},
            ]
        )
        resultado["messages"] = [AIMessage(content=mensaje_sintesis.content, name="supervisor")]
    else:
        # Dejamos registrada en la traza la decisión de delegación, así
        # se ve el flujo Supervisor -> especialista de punta a punta.
        resultado["messages"] = [
            AIMessage(
                content=f"Delego en '{decision.next_agent}'. Motivo: {decision.razon or 'sin detalle'}",
                name="supervisor",
            )
        ]

    return resultado
