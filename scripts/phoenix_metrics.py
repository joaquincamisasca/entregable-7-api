"""
Lee de Phoenix las métricas de la prueba de carga: costo por ejecución y
latencia p50 / p95 / p99 de un proyecto. Son los MISMOS números que
calcula el dashboard (se consultan por su API GraphQL), así que sirven
para contrastar lo que ves en las capturas.

Uso (con Phoenix levantado, después de la prueba de carga):
    python scripts/phoenix_metrics.py
    python scripts/phoenix_metrics.py --proyecto orquestador-multiagente
"""

import argparse
import sys

import httpx

CONSULTA = """
query($nombre: String!) {
  projects(filter: {col: name, value: $nombre}) {
    edges { node {
      name
      traceCount
      p50: latencyMsQuantile(probability: 0.5)
      p95: latencyMsQuantile(probability: 0.95)
      p99: latencyMsQuantile(probability: 0.99)
      costSummary {
        total { cost tokens }
        prompt { cost tokens }
        completion { cost tokens }
      }
    } }
  }
}
"""


def main(url: str, proyecto: str) -> int:
    r = httpx.post(f"{url.rstrip('/')}/graphql", json={"query": CONSULTA, "variables": {"nombre": proyecto}}, timeout=30).json()
    if "errors" in r:
        print("Error al consultar Phoenix:", r["errors"][0]["message"])
        return 1

    edges = r["data"]["projects"]["edges"]
    if not edges:
        print(f"No existe el proyecto '{proyecto}' en Phoenix. ¿Corriste la API con PHOENIX_ENABLED=true?")
        return 1

    n = edges[0]["node"]
    ejecuciones = n["traceCount"] or 0
    costo = n["costSummary"]
    total = costo["total"]["cost"] or 0.0
    tokens = costo["total"]["tokens"] or 0

    print(f"Proyecto:              {n['name']}")
    print(f"Ejecuciones (trazas):  {ejecuciones}")
    print(f"Tokens totales:        {int(tokens)}  (entrada {int(costo['prompt']['tokens'] or 0)}, salida {int(costo['completion']['tokens'] or 0)})")
    print(f"Costo total:           US$ {total:.6f}")
    if ejecuciones:
        print(f"Costo por ejecución:   US$ {total / ejecuciones:.6f}")
    print(f"Latencia p50:          {n['p50'] / 1000:.2f} s")
    print(f"Latencia p95:          {n['p95'] / 1000:.2f} s")
    print(f"Latencia p99:          {n['p99'] / 1000:.2f} s")

    if total == 0 and tokens:
        print("\nOJO: hay tokens pero el costo es 0. Phoenix no tiene precio para tu modelo.")
        print("Corré primero:  python scripts/setup_phoenix_pricing.py  y repetí la prueba de carga.")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Métricas de la prueba de carga leídas de Phoenix")
    parser.add_argument("--url", default="http://localhost:6006")
    parser.add_argument("--proyecto", default="orquestador-multiagente")
    args = parser.parse_args()
    sys.exit(main(args.url, args.proyecto))
