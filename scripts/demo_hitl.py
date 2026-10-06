"""
Demo interactiva del flujo Human-in-the-loop, desde la terminal.

1. Crea una tarea CRITICA (pide publicar el reporte).
2. Espera a que el grafo se pause en WAITING_APPROVAL y te muestra el
   reporte que se va a publicar.
3. Te pregunta si lo aprobás; manda tu decisión a la API.
4. Muestra el resultado final (publicado o rechazado).

Uso (con la API levantada):
    python scripts/demo_hitl.py
    python scripts/demo_hitl.py --auto s      # aprueba sin preguntar
"""

import argparse
import sys
import time

import httpx

CONSULTA = "Buscá los comentarios sobre el checkout, analizá el sentimiento, calculá el rating promedio y publicá el reporte en el canal del equipo."


def esperar(cliente, job_id, estados, timeout):
    limite = time.time() + timeout
    while time.time() < limite:
        trabajo = cliente.get(f"/tasks/{job_id}").json()
        if trabajo["status"] in estados:
            return trabajo
        time.sleep(1)
    raise SystemExit(f"Timeout esperando {estados}")


def main(url: str, auto: str, timeout: float) -> None:
    with httpx.Client(base_url=url, timeout=30) as cliente:
        creada = cliente.post("/tasks", json={"query": CONSULTA}).json()
        job_id = creada["job_id"]
        print(f"Tarea creada: {job_id} (requiere aprobación: {creada['requiere_aprobacion']})")
        print("Esperando a que el sistema investigue, analice y llegue al punto crítico...")

        trabajo = esperar(cliente, job_id, {"WAITING_APPROVAL", "DONE", "FAILED"}, timeout)
        if trabajo["status"] == "FAILED":
            raise SystemExit(f"La tarea falló: {trabajo.get('error')}")
        if trabajo["status"] == "DONE":
            raise SystemExit("La tarea terminó sin pedir aprobación (¿no era crítica?).")

        pedido = trabajo["interrupt"]
        print("\n=== PAUSA: se requiere aprobación humana ===")
        print(f"Acción:  {pedido['accion']}  ->  {pedido['canal']}")
        print(f"Motivo:  {pedido['motivo']}")
        print(f"\nReporte a publicar:\n{pedido['reporte']}\n")

        respuesta = auto or input("¿Aprobás la publicación? (s/n): ")
        aprobado = respuesta.strip().lower().startswith("s")
        comentario = "Aprobado desde demo_hitl.py" if aprobado else "Rechazado desde demo_hitl.py"

        cliente.post(f"/tasks/{job_id}/approve", json={"aprobado": aprobado, "comentario": comentario})
        final = esperar(cliente, job_id, {"DONE", "FAILED"}, timeout)

        print(f"\nEstado final: {final['status']}")
        if final["status"] == "DONE":
            print(f"Publicación: {final['result']['publicacion']}")
        else:
            print(f"Error: {final.get('error')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Demo interactiva del Human-in-the-loop")
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--auto", default="", help="'s' aprueba, 'n' rechaza, sin preguntar")
    parser.add_argument("--timeout", type=float, default=900)
    args = parser.parse_args()
    sys.exit(main(args.url, args.auto, args.timeout))
