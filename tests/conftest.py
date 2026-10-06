"""
Configuración compartida de los tests.

- Desactiva Phoenix (no hace falta el collector para testear).
- Define un chat model FALSO y guionado que hace de Supervisor,
  Investigador y Analista sin llamar a ningún LLM real. Además simula el
  fallo real de llama3.2 (la decisión estructurada del Supervisor vuelve
  None), así que todos los tests ejercitan también el plan B por reglas.
"""

import os

os.environ["PHOENIX_ENABLED"] = "false"

from typing import Any, List, Optional  # noqa: E402

import pytest  # noqa: E402
from langchain_core.language_models.chat_models import BaseChatModel  # noqa: E402
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage  # noqa: E402
from langchain_core.outputs import ChatGeneration, ChatResult  # noqa: E402
from langchain_core.runnables import RunnableLambda  # noqa: E402

COMENTARIOS = [
    "El nuevo checkout es mucho más rápido, me encantó la experiencia.",
    "Tarda demasiado en cargar el paso de pago, es frustrante.",
    "Tuve un error al confirmar la compra, muy mala experiencia.",
]
RATINGS = [5, 2, 1]


class ChatGuionado(BaseChatModel):
    """Responde según el rol que indica el system prompt."""

    @property
    def _llm_type(self) -> str:
        return "chat-guionado"

    def bind_tools(self, tools, **kwargs):
        return self

    def with_structured_output(self, _esquema, **kwargs):
        return RunnableLambda(lambda _mensajes: None)

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        system = str(messages[0].content) if messages else ""
        tool_messages = [m for m in messages if isinstance(m, ToolMessage)]

        if "Agente de Investigación" in system:
            if not tool_messages:
                respuesta = AIMessage(
                    content="",
                    tool_calls=[{"name": "buscar_comentarios", "args": {"tema": "checkout"}, "id": "c1"}],
                )
            else:
                respuesta = AIMessage(content=f"Encontré 3 comentarios. Textos: {COMENTARIOS}. Ratings: {RATINGS}.")

        elif "Agente de Análisis" in system:
            if len(tool_messages) == 0:
                respuesta = AIMessage(
                    content="",
                    tool_calls=[{"name": "analizar_sentimiento", "args": {"comentarios": COMENTARIOS}, "id": "a1"}],
                )
            elif len(tool_messages) == 1:
                respuesta = AIMessage(
                    content="",
                    tool_calls=[{"name": "calcular_estadisticas", "args": {"numeros": RATINGS}, "id": "a2"}],
                )
            else:
                respuesta = AIMessage(content="Sentimiento mixto (1 positivo, 2 negativos). Rating promedio: 2.67.")

        else:
            respuesta = AIMessage(content="Síntesis: el checkout tiene sentimiento mixto y rating 2.67.")

        return ChatResult(generations=[ChatGeneration(message=respuesta)])


@pytest.fixture
def modelo_falso(monkeypatch):
    """Reemplaza el LLM real por el modelo guionado en todo el orquestador."""
    from app import supervisor

    modelo = ChatGuionado()
    monkeypatch.setattr(supervisor, "get_llm", lambda *a, **k: modelo)
    return modelo
