# API de producción para el Orquestador Multi-Agente

API REST (FastAPI) que expone el orquestador multi-agente del Módulo 6
(Supervisor + Investigador + Analista) con cuatro capacidades nuevas:

| Capacidad | Cómo |
|---|---|
| **Endpoints asíncronos** | `POST /tasks` encola y devuelve un `job_id` al instante; el trabajo corre en segundo plano |
| **Estado en Redis** | Estado de cada trabajo (`PENDING/RUNNING/WAITING_APPROVAL/DONE/FAILED`), cola de trabajos y checkpoints de LangGraph (`AsyncRedisSaver`) |
| **Observabilidad** | Trazas de cada nodo, llamada al LLM y herramienta en **Arize Phoenix** (OpenInference) |
| **Human-in-the-loop** | Las tareas críticas (publicar el reporte) pausan el grafo hasta que una persona apruebe en `POST /tasks/{id}/approve` |

**Módulo 7, Pre-entrega 7** - Programa de AI Engineering @ CodeHouse

---

## 🏗️ Arquitectura

```
 cliente ──POST /tasks──▶  FastAPI  ──LPUSH──▶  Redis (cola)
    ▲  │  (202 + job_id)      │                     │ BRPOP
    │  └──GET /tasks/{id}─────┤                     ▼
    │       (polling)         │              Workers (x5, asyncio)
    │                         ▼                     │ ainvoke / Command(resume)
    └──── estado ◀──────  Redis (job:{id}) ◀────────┤
                                                     ▼
                                    Grafo LangGraph (checkpointer = Redis)
                                    supervisor → investigador → analista → ...
                                                     │
                                       ¿tarea crítica? ─ sí ─▶ aprobacion_humana
                                                     │          (interrupt: PAUSA)
                                                     ▼
                                          trazas ──▶ Arize Phoenix
```

```
app/
├── main.py            # FastAPI: POST /tasks · GET /tasks/{id} · POST /tasks/{id}/approve · GET /health
├── graph.py           # orquestador del M6 + nodo de aprobación + checkpointer inyectado
├── worker.py          # pool de workers: corre la tarea y actualiza el estado en Redis
├── job_store.py       # estado de trabajos (hashes) y cola (lista) en Redis
├── hitl.py            # clasificador de tareas críticas + nodo con interrupt()
├── observability.py   # init de Phoenix + decorador de trazas
├── supervisor.py, state.py, agents/   # el orquestador del Módulo 6
└── config.py          # variables de entorno
scripts/
├── load_test.py             # 5 peticiones concurrentes + p50/p95
├── phoenix_metrics.py       # lee de Phoenix el costo por ejecución y el p95
├── setup_phoenix_pricing.py # precio de referencia para el modelo local
└── demo_hitl.py             # demo interactiva de la aprobación humana
tests/                 # 48 tests (sin Redis, Phoenix ni LLM reales)
screenshots/           # capturas del dashboard (ver "Evidencia")
docker-compose.yml     # Redis Stack + Phoenix
```

### Decisiones de diseño

- **No se bloquea el event loop.** Los endpoints solo hacen operaciones
  asíncronas contra Redis (`redis.asyncio`); nunca ejecutan el grafo ni
  llaman a un LLM. Medido con 5 trabajos corriendo, `/health` respondía en ~2 ms.
- **Cola en Redis, no en memoria.** La API hace `LPUSH` y los workers
  `BRPOP`. Si la API se reinicia, lo encolado sigue ahí. Los workers viven
  en el mismo proceso (variante "tarea asíncrona simple" que habilita la
  consigna); pasar a Arq o Celery solo cambiaría quién consume esa cola.
- **Un trabajo = un `thread_id`.** El `job_id` es el `thread_id` del
  checkpointer, así que cada ejecución tiene su propio estado persistido.
- **Qué es "crítico".** Una regla determinística (`es_tarea_critica`): la
  consulta pide *publicar / enviar / mandar / compartir* el reporte. Es a
  propósito una regla y no una decisión del LLM: lo que dispara la
  aprobación humana tiene que ser predecible y auditable.
- **Errores en segundo plano.** `Worker.procesar` captura **cualquier**
  excepción y deja el trabajo en `FAILED` con el error guardado, así el
  cliente nunca queda haciendo polling para siempre (hay un test que lo verifica).
- **1 traza = 1 trabajo.** Ver la nota sobre FastAPI en "Observabilidad".

---

## 🚀 Cómo levantar todo

Necesitás: **Python 3.11+**, **Docker Desktop** y **Ollama**.

> **¿Por qué Docker?** El checkpointer de LangGraph usa los módulos
> RediSearch y RedisJSON, que trae **Redis Stack** (o Redis 8+). Un Redis
> común falla con `unknown command 'FT.INFO'`. Además Redis no corre nativo
> en Windows.

### 1. Infraestructura (Redis Stack + Phoenix)

```bash
docker compose up -d
```

- Redis Stack en `localhost:6379`
- Phoenix (dashboard) en http://localhost:6006

### 2. Dependencias

```bash
python -m venv venv
venv\Scripts\activate          # Windows. Mac/Linux: source venv/bin/activate
pip install -r requirements.txt
copy .env.example .env         # Mac/Linux: cp .env.example .env
```

### 3. Ollama

```bash
ollama pull llama3.2
```

### 4. La API

```bash
uvicorn app.main:app --port 8000
```

Documentación interactiva (Swagger): http://localhost:8000/docs

### 5. Tests (no necesitan nada de lo anterior)

```bash
pytest -v
```

---

## 🧪 Cómo probarla

### Una tarea normal

En http://localhost:8000/docs → `POST /tasks` → *Try it out*, con:

```json
{"query": "Buscá los comentarios sobre el checkout, analizá el sentimiento y calculá el rating promedio."}
```

Devuelve `202` y un `job_id`. Con ese id, `GET /tasks/{job_id}` muestra
`PENDING → RUNNING → DONE`, y en `DONE` trae la respuesta y la traza de la
delegación.

### Human-in-the-loop

La forma más cómoda es la demo interactiva:

```bash
python scripts/demo_hitl.py
```

Crea una tarea que pide **publicar** el reporte, espera a que el grafo se
pause (`WAITING_APPROVAL`), te muestra el reporte y te pregunta si lo
aprobás. También se puede hacer a mano desde `/docs`:
`POST /tasks/{job_id}/approve` con `{"aprobado": true, "comentario": "OK"}`.

Mientras está en `WAITING_APPROVAL` **no se publicó nada**: la acción con
efectos secundarios solo se ejecuta después de la aprobación. Aprobar dos
veces devuelve `409`.

### Las 5 peticiones concurrentes

Con la API levantada, en otra terminal:

```bash
python scripts/load_test.py
```

Dispara 5 `POST /tasks` **al mismo tiempo**, hace polling hasta que
terminan y guarda en `resultados/load_test.json` la latencia de cada una
más el p50 y el p95.

> **Ollama y el paralelismo.** Ollama por defecto atiende pocas
> peticiones a la vez. Para que las 5 corran de verdad en paralelo,
> cerrá Ollama y arrancalo con `OLLAMA_NUM_PARALLEL=5` (en Windows: variable
> de entorno del sistema, y reiniciar Ollama). Sin eso las tareas se
> encolan dentro de Ollama y el p95 sube. Ambos resultados son válidos para
> la evidencia; solo cambian los números.

---

## 📊 Observabilidad y evidencia (Arize Phoenix)

Cada trabajo genera **una traza** en el proyecto `orquestador-multiagente`
con un árbol de spans: el trabajo, cada nodo del grafo (`supervisor`,
`investigador`, `analista`, `aprobacion_humana`), cada llamada al LLM (con
prompt, respuesta, **tokens de entrada y salida** y latencia) y cada
herramienta. Todos los spans de un trabajo comparten el `session.id = job_id`,
así que la ejecución inicial y la reanudación tras la aprobación humana se
ven agrupadas.

### Costo por ejecución

Phoenix calcula el costo como `tokens × precio del modelo`. Trae los
precios de los modelos de nube (gpt-4o, claude, etc.) pero **no de un modelo
local**: con `llama3.2` en Ollama, Phoenix cuenta los tokens y muestra
**costo $0.00**.

Por eso, antes de la prueba de carga, registrá un **precio de referencia**:

```bash
python scripts/setup_phoenix_pricing.py
```

> ⚠️ **Hay que ser claro con lo que significa ese número.** Correr Ollama
> en tu compu cuesta $0 (más la luz). El precio cargado (US$ 0,06 por millón
> de tokens, ajustable con `--entrada/--salida`) es una **referencia**: más o
> menos lo que cobraría un proveedor por servir un modelo de ese tamaño. El
> "costo por ejecución" del dashboard es entonces **una estimación de lo que
> costaría la misma carga en la nube**, no un gasto real. Si usás
> `GENERATION_PROVIDER=openai` o `anthropic`, Phoenix ya tiene el precio real
> y no hace falta este paso.

Los precios solo se aplican a las ejecuciones **posteriores** a
registrarlos: corré el script *antes* de la prueba de carga.

### Latencia p95 y lectura del dashboard

Para obtener los números exactos de la corrida de las 5 peticiones:

```bash
python scripts/phoenix_metrics.py
```

Imprime tokens, costo total, **costo por ejecución** y latencia
**p50/p95/p99**, leídos de la misma API que usa el dashboard. Sirve para
contrastar lo que muestran las capturas.

Qué mirar en http://localhost:6006 (proyecto `orquestador-multiagente`):

1. **Tabla de trazas**: una fila por trabajo, con latencia, tokens y costo.
   Ordená por latencia: con 5 trazas, el p95 corresponde a la más lenta.
2. **Una traza abierta** (`job.ejecucion`): el árbol de spans muestra dónde
   se concentra la latencia (qué nodo o llamada al LLM tarda más) y qué nodo
   consume más tokens.
3. **Resumen del proyecto**: costo total y percentiles de latencia.

### Capturas a guardar en `/screenshots`

Hacelas con tu propia corrida (no hay capturas de ejemplo en el repo a
propósito: son la evidencia de *tu* ejecución):

- `01_trazas.png` — la tabla de trazas con las 5 ejecuciones.
- `02_traza_detalle.png` — una traza abierta con el árbol de spans.
- `03_costo_por_ejecucion.png` — costo (tabla de trazas o resumen del proyecto).
- `04_latencia_p95.png` — latencia p95 (resumen, o la salida de `phoenix_metrics.py`).
- `05_hitl.png` *(opcional)* — una traza con la pausa de aprobación humana.

### Nota técnica: por qué solo se instrumenta LangChain

Las versiones recientes de FastAPI emiten por su cuenta un span por cada
request HTTP cuando hay un tracer configurado. Con eso, cada `GET` de
polling (~4 ms) pasaba a ser una traza más y **contaminaba las métricas**:
en una prueba con 5 trabajos el proyecto mostraba 25 trazas y un p50 de
4 ms. Se apagó con `FastAPI(telemetry={"tracing": False})` y se instrumenta
solo LangChain/LangGraph (`LangChainInstrumentor`); ahora 1 traza = 1 trabajo.

---

## 🛡️ Errores comunes que este proyecto evita

| Error común | Cómo se evita |
|---|---|
| Bloquear el event loop | Endpoints 100 % async con `redis.asyncio`; el grafo corre en workers, no en el endpoint |
| Ignorar errores en background tasks | `Worker.procesar` captura toda excepción → estado `FAILED` + error guardado en Redis (`tests/test_job_store.py`) |
| Doble aprobación / doble reanudación | `approve` valida el estado (409 si no espera aprobación) y marca `RUNNING` antes de encolar |
| Trabajos que se pierden al reiniciar | Cola y estado viven en Redis, no en memoria del proceso |
| Métricas contaminadas por el polling | Tracing HTTP de FastAPI apagado (ver nota anterior) |

---

## ⚠️ Qué se probó y qué no (honestidad sobre el alcance)

**Verificado con tests automáticos (48, ~2 s, sin servicios externos):**
el flujo completo de la API con workers reales (incluidas 5 tareas
concurrentes), el grafo real con pausa y reanudación del HITL (aprobar y
rechazar), la persistencia del estado y de la cola, el paso a `FAILED`, y
los percentiles. Los tests usan Redis simulado (`fakeredis`), un
checkpointer en memoria y un modelo de lenguaje falso y guionado.

**Verificado además contra servicios reales:** la API corriendo como
servidor con un **Redis real** (5 peticiones concurrentes respondidas en
~45 ms, ejecutadas en paralelo; `/health` ~2 ms con la carga corriendo), el
flujo HITL completo por HTTP, y las trazas, tokens, costo y percentiles en un
**Phoenix real**.

**No se pudo verificar en el entorno de desarrollo:**

- El checkpointer `AsyncRedisSaver` contra un **Redis Stack real** (el Redis
  de pruebas no traía los módulos de búsqueda y JSON). La integración se
  escribió contra la API de la librería y falla con un error claro si Redis
  no tiene los módulos; el resto del flujo se probó con el checkpointer en
  memoria (`CHECKPOINTER=memory`). Es lo primero a confirmar al levantar
  `docker compose`: si la API arranca y el log dice `Checkpointer de
  LangGraph: AsyncRedisSaver`, está funcionando.
- El orquestador con **`llama3.2` real**. En el Módulo 6 se vio que los
  modelos locales chicos a veces no devuelven la salida estructurada; el
  Supervisor tiene un plan B por reglas para eso (queda registrado en los
  logs de la API).
- El `docker-compose.yml`: está escrito con las imágenes oficiales pero no se
  pudo ejecutar Docker en el entorno de desarrollo.

---

## ✅ Checklist de la consigna

| Requisito | Dónde está |
|---|---|
| Endpoint async que encola y devuelve `job_id` sin bloquear | `app/main.py` → `POST /tasks` |
| Estado persistido en Redis, incluido `FAILED` ante excepción | `app/job_store.py`, `app/worker.py` |
| Checkpoints de LangGraph en Redis | `app/main.py` → `AsyncRedisSaver` |
| Trazas visibles en Phoenix | `app/observability.py` + capturas en `/screenshots` |
| Costo por ejecución y latencia p95 de 5 peticiones concurrentes | `scripts/load_test.py`, `scripts/phoenix_metrics.py` + capturas |
| Nodo HITL que pausa hasta recibir aprobación externa | `app/hitl.py`, `POST /tasks/{id}/approve` |
| README con pasos para Redis, la API y las 5 peticiones | Este archivo |
| `requirements.txt`, `.env.example`, Docker Compose | Raíz del repo |
