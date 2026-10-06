"""
Esquema de Estado compartido del orquestador (heredado del Módulo 6).

Campos (además de `messages`, que viene de MessagesState):

- next_agent: a quién le toca actuar ahora, según decide el Supervisor.
- instruccion_para_agente: instrucción puntual para el especialista
  elegido (evita la "Contaminación de Contexto": el especialista no ve
  todo el historial, solo esto).
- contribuciones: qué aportó cada agente.
- pasos: contador de vueltas del Supervisor (salvaguarda contra el
  "Supervisor Infinito").
- publicacion: NUEVO en este módulo. Resultado de la acción crítica
  (publicar el reporte), que solo se ejecuta después de la aprobación
  humana. Queda en None si la tarea no era crítica.
"""

from typing import Dict, Literal, Optional

from langgraph.graph import MessagesState

NombreAgente = Literal["investigador", "analista", "FINISH"]


class OrquestadorState(MessagesState):
    """Estado compartido entre el Supervisor, los especialistas y el nodo de aprobación humana."""

    next_agent: NombreAgente
    instruccion_para_agente: str
    contribuciones: Dict[str, str]
    pasos: int
    publicacion: Optional[dict]
