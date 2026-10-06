"""
Registra en Phoenix un PRECIO DE REFERENCIA para el modelo local.

Por qué hace falta: Phoenix calcula el costo de cada ejecución como
    tokens x precio por token del modelo
y trae precios cargados solo para modelos de nube (gpt-4o, claude, etc.).
Un modelo que corre en tu compu con Ollama (llama3.2) no tiene precio, y
Phoenix muestra costo $0.00 aunque sí cuenta los tokens.

Este script le da a Phoenix un precio para `llama3.2`. Dos aclaraciones
honestas, que también van en el README:
- El costo REAL de correr Ollama en tu compu es $0 (más la luz).
- El valor que se carga es una referencia: aproximadamente lo que cobra
  un proveedor por servir un modelo de ese tamaño (~3B parámetros). Sirve
  para estimar cuánto costaría la misma carga en la nube y para que el
  dashboard muestre la métrica "costo por ejecución". Si cambiás de modelo
  (por ejemplo a gpt-4o-mini), Phoenix ya trae su precio real y este
  script no hace falta.

Uso (con Phoenix levantado):
    python scripts/setup_phoenix_pricing.py
    python scripts/setup_phoenix_pricing.py --modelo llama3.2 --entrada 0.06 --salida 0.06
"""

import argparse
import re
import sys

import httpx

MUTACION = """
mutation CrearModelo($input: CreateModelMutationInput!) {
  createModel(input: $input) {
    model { id name namePattern }
  }
}
"""

CONSULTA_MODELOS = """
query { generativeModels { edges { node { id name namePattern } } } }
"""


def main(url: str, modelo: str, entrada: float, salida: float) -> int:
    graphql = f"{url.rstrip('/')}/graphql"

    existentes = httpx.post(graphql, json={"query": CONSULTA_MODELOS}, timeout=30).json()
    nodos = [e["node"] for e in existentes.get("data", {}).get("generativeModels", {}).get("edges", [])]
    if any(n["name"] == modelo for n in nodos):
        print(f"El modelo '{modelo}' ya tiene precio registrado en Phoenix. No hay nada que hacer.")
        return 0

    variables = {
        "input": {
            "name": modelo,
            "provider": "ollama",
            "namePattern": f"^{re.escape(modelo)}.*$",
            "costs": [
                {"tokenType": "input", "kind": "PROMPT", "costPerMillionTokens": entrada},
                {"tokenType": "output", "kind": "COMPLETION", "costPerMillionTokens": salida},
            ],
        }
    }
    r = httpx.post(graphql, json={"query": MUTACION, "variables": variables}, timeout=30).json()

    if "errors" in r:
        print("No se pudo registrar el precio:", r["errors"][0]["message"])
        print("Alternativa manual: en Phoenix, Settings > Models > 'Add Model'.")
        return 1

    print(f"Precio de referencia registrado para '{modelo}': "
          f"US$ {entrada}/M tokens de entrada, US$ {salida}/M tokens de salida.")
    print("Las ejecuciones NUEVAS ya van a mostrar costo en el dashboard.")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Registra un precio de referencia para un modelo local en Phoenix")
    parser.add_argument("--url", default="http://localhost:6006")
    parser.add_argument("--modelo", default="llama3.2")
    parser.add_argument("--entrada", type=float, default=0.06, help="US$ por millón de tokens de entrada")
    parser.add_argument("--salida", type=float, default=0.06, help="US$ por millón de tokens de salida")
    args = parser.parse_args()
    sys.exit(main(args.url, args.modelo, args.entrada, args.salida))
