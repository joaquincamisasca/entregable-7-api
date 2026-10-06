"""
Prueba de carga: 5 peticiones CONCURRENTES contra la API.

1. Dispara las 5 POST /tasks al mismo tiempo (asyncio.gather) y mide
   cuánto tarda la API en responder cada una (debe ser casi inmediato:
   la API solo encola).
2. Hace polling de cada trabajo hasta que termina (DONE/FAILED).
3. Informa la latencia de punta a punta de cada trabajo y calcula p50 y
   p95, y guarda todo en resultados/load_test.json.

Las métricas "oficiales" de la entrega (costo por ejecución y latencia
p95) se toman del dashboard de Phoenix; este reporte local sirve para
contrastarlas.

Uso (con la API levantada):
    python scripts/load_test.py
    python scripts/load_test.py --url http://localhost:8000 --n 5
"""

import argparse
import asyncio
import json
import math
import time
from datetime import datetime
from pathlib import Path

import httpx

CONSULTAS = [
    "Buscá los comentarios sobre el checkout, analizá el sentimiento y calculá el rating promedio.",
    "¿Qué opinan los usuarios del onboarding? Analizá el sentimiento y el rating promedio.",
    "Investigá el feedback del checkout y decime si el sentimiento general es positivo o negativo.",
    "Necesito el rating promedio y el sentimiento de los comentarios del onboarding.",
    "Analizá los comentarios del checkout: sentimiento general, rating mínimo, máximo y promedio.",
]

ESTADOS_FINALES = {"DONE", "FAILED"}


def percentil(valores: list, p: float) -> float:
    """Percentil por el método del rango más cercano (con 5 muestras, p95 = la más lenta)."""
    ordenados = sorted(valores)
    rango = max(1, math.ceil(p / 100 * len(ordenados)))
    return ordenados[rango - 1]


async def crear_tarea(cliente: httpx.AsyncClient, consulta: str) -> dict:
    inicio = time.perf_counter()
    r = await cliente.post("/tasks", json={"query": consulta})
    r.raise_for_status()
    return {**r.json(), "consulta": consulta, "latencia_post_s": time.perf_counter() - inicio, "t0": inicio}


async def esperar(cliente: httpx.AsyncClient, tarea: dict, timeout: float) -> dict:
    limite = time.perf_counter() + timeout
    while time.perf_counter() < limite:
        trabajo = (await cliente.get(f"/tasks/{tarea['job_id']}")).json()
        if trabajo["status"] in ESTADOS_FINALES | {"WAITING_APPROVAL"}:
            tarea["status"] = trabajo["status"]
            tarea["latencia_total_s"] = time.perf_counter() - tarea["t0"]
            tarea["error"] = trabajo.get("error")
            return tarea
        await asyncio.sleep(1)
    tarea["status"] = "TIMEOUT"
    tarea["latencia_total_s"] = timeout
    return tarea


async def main(url: str, n: int, timeout: float) -> None:
    consultas = [CONSULTAS[i % len(CONSULTAS)] for i in range(n)]

    async with httpx.AsyncClient(base_url=url, timeout=30) as cliente:
        print(f"Enviando {n} peticiones concurrentes a {url} ...")
        tareas = await asyncio.gather(*(crear_tarea(cliente, c) for c in consultas))

        for t in tareas:
            print(f"  POST {t['job_id'][:8]}  respondió en {t['latencia_post_s'] * 1000:6.1f} ms  ({t['status']})")

        print("\nEsperando a que terminen (polling cada 1s) ...")
        tareas = await asyncio.gather(*(esperar(cliente, t, timeout) for t in tareas))

    latencias = [t["latencia_total_s"] for t in tareas if t["status"] == "DONE"]

    print(f"\n{'job_id':10} {'estado':16} {'latencia total':>15}")
    for t in tareas:
        print(f"{t['job_id'][:8]:10} {t['status']:16} {t['latencia_total_s']:>13.1f} s")
        if t.get("error"):
            print(f"           error: {t['error']}")

    resumen = {
        "fecha": datetime.now().isoformat(timespec="seconds"),
        "url": url,
        "peticiones": n,
        "completadas": len(latencias),
        "latencia_post_max_ms": round(max(t["latencia_post_s"] for t in tareas) * 1000, 1),
        "latencia_p50_s": round(percentil(latencias, 50), 2) if latencias else None,
        "latencia_p95_s": round(percentil(latencias, 95), 2) if latencias else None,
        "trabajos": [
            {k: v for k, v in t.items() if k != "t0"} for t in tareas
        ],
    }

    print(f"\nCompletadas: {resumen['completadas']}/{n}")
    print(f"POST más lento: {resumen['latencia_post_max_ms']} ms (la API no espera al grafo)")
    print(f"Latencia p50: {resumen['latencia_p50_s']} s | p95: {resumen['latencia_p95_s']} s")

    salida = Path(__file__).resolve().parent.parent / "resultados" / "load_test.json"
    salida.parent.mkdir(exist_ok=True)
    salida.write_text(json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nResultados guardados en {salida}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prueba de carga concurrente contra la API")
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=1800, help="segundos máximos por trabajo")
    args = parser.parse_args()
    asyncio.run(main(args.url, args.n, args.timeout))
