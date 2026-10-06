"""
Tests de las herramientas de ambos agentes especialistas (lógica pura,
sin necesitar ningún LLM).
"""

from app.agents.analyst_agent import analizar_sentimiento, calcular_estadisticas
from app.agents.research_agent import buscar_comentarios


def test_buscar_comentarios_encuentra_tema_existente():
    resultado = buscar_comentarios.invoke({"tema": "checkout"})
    assert "comentarios" in resultado
    assert len(resultado["comentarios"]) == 5


def test_buscar_comentarios_tema_inexistente_devuelve_sugerencias():
    resultado = buscar_comentarios.invoke({"tema": "facturación"})
    assert "error" in resultado
    assert "temas_disponibles" in resultado
    assert "checkout" in resultado["temas_disponibles"]


def test_analizar_sentimiento_clasifica_correctamente():
    comentarios = [
        "Me encantó, excelente experiencia.",
        "Tarda demasiado, muy frustrante.",
        "Todo bien.",
    ]
    resultado = analizar_sentimiento.invoke({"comentarios": comentarios})
    assert resultado["positivos"] == 2  # "encantó/excelente" y "bien"
    assert resultado["negativos"] == 1  # "tarda/frustrante"


def test_analizar_sentimiento_lista_vacia():
    resultado = analizar_sentimiento.invoke({"comentarios": []})
    assert resultado["positivos"] == 0
    assert resultado["negativos"] == 0
    assert resultado["neutros"] == 0


def test_calcular_estadisticas_devuelve_valores_correctos():
    resultado = calcular_estadisticas.invoke({"numeros": [5, 2, 3, 5, 1]})
    assert resultado["promedio"] == 3.2
    assert resultado["minimo"] == 1
    assert resultado["maximo"] == 5
    assert resultado["cantidad"] == 5


def test_calcular_estadisticas_lista_vacia_devuelve_error():
    resultado = calcular_estadisticas.invoke({"numeros": []})
    assert "error" in resultado
