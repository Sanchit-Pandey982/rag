# Phase 10 Docker Handoff

## Copy-ready handoff prompt

You are taking over the Phase 10 Docker Compose work for a multi-tenant RAG chat platform in:

`C:\Users\Sanchit Pandey\OneDrive\Desktop\agent`

**Read this entire prompt before touching anything.** It is an end-to-end architecture briefing — runtime topology, backend layering, request lifecycles, data model, resilience phases, and the invariants that must not be broken — followed by operational rules and next actions. Sections 1–7 of this document hold the change log, the error history, the start/restart runbook, the end-to-end test script, and cleanup commands.

### A. The system in one paragraph

RAG Learning is a FastAPI + React platform where each registered user owns a private document corpus. Documents are chunked, embedded with Gemini, and stored in a Chroma collection that is always queried with a per-user filter. A question either retrieves grounded chunks and streams an answer sourced from them, or is refused with one exact sentence. Around this core there are twelve engineered phases: conversation summarization, hybrid (vector + BM25) retrieval, cross-encoder reranking, JWT sessions with Redis-backed refresh rotation and token blacklisting, response caching, a retry/circuit-breaker resilience layer, degraded-mode fallback answers, token/cost metering into MongoDB, document upload with background ingestion and crash recovery, Docker Compose with five services and health-gated startup, a LangGraph-style agent loop, and a stdlib-only tool registry (calculator + web search).

### B. Repository layout

| Path | Role |
|---|---|
| `phase1.py` | The RAG core. Dataclasses (`Document`, `RetrievedChunk`, `EvalCase`), Gemini wrappers (`embed_documents` ~L103, `embed_query` ~L140), corpus loader `load_txt_documents` (~L157), summarizer `summarize_conversation_history` (~L255), degraded-mode helpers (~L323–370), `build_chroma_client` (~L405), and `RAGSystem` (~L431) with `ingest_documents`, `retrieve`, `build_context`, `generate_answer`, `generate_answer_stream`, `_retrieve_chunks`, cache lookups, and the `run_once` / `run_once_stream` / `run_once_event_stream` trio (~L1042–1249). The bottom half (~L1500+) is evaluation/demo (`load_eval_cases`, `evaluate_cases`, threshold calibration, chunking experiments) — never imported at request time. Gemini client construction is lazy (`get_gemini_client`) so importing the module needs no credentials or network; do not move client construction to import time. |
| `app/main.py` | FastAPI app factory + `lifespan`. Startup order matters: validate `JWTService.from_environment()` → ping Redis → ping Mongo (`admin.command("ping")`) → build services on the shared clients → create `RAGSystem` (collection `learning_rag`, `reset=False`) → ingest `data/*.txt` under tenant `eval_user` **only if** the collection is empty → build `RAGService` → set `app.state.ready` → best-effort `requeue_stale_uploads` crash recovery. Shutdown closes Chroma client, Redis (`aclose`), and Mongo in nested `finally` blocks so every resource closes even after a partial failure. |
| `app/routes/` | Five routers: `auth.py` (prefix `/api/v1/auth`, custom `AuthRoute` route class), `chat.py` (prefix `/api/v1`), `documents.py` (`/api/v1/documents`), `conversations.py` (`/api/v1/conversations`), `usage.py` (`/api/v1/usage`). |
| `app/schemas/` | Pydantic request/response contracts. `ChatRequest` enforces the platform limits (`raw_query` 1–4000, `user_id` 1–256, `chat_history` list ≤20, `k` 1–10, `distance_threshold` 0–2, optional `conversation_id`); limits are enforced in `RAGSystem.retrieve` as well, so the schema is not the only gate. `ChatResponse` returns `answer`, `retrieval_query`, `retrieved_document_ids`, `chunks` (note: the REST field is `chunks`, the SSE event is `sources`). |
| `app/services/` | Business layer. `auth_service` (users), `refresh_token_service`, `token_blacklist_service` (Phase 4), `rate_limit_service`, `cache_service` (Phase 5), `degradation_service` (Phase 7), `retrieval.py:HybridRetriever` (Phase 2), `reranker.py:CrossEncoderReranker` (Phase 3a), `rag_services.py:RAGService` (facade over `RAGSystem`, owns the hybrid/rerank caches and flag resolution), `document_service` + `document_jobs` (upload → background ingest → requeue), `conversation_service` (Phase 3.5), `chat_orchestration.py` (the composition root for one chat turn), `usage_service` (Phase 8). |
| `app/dependencies/` | `auth.py` (`get_current_user`), `chat.py` (the 403 ownership check: `payload.user_id` must equal the JWT `sub`), `rate_limit.py` (login/refresh/upload throttles). |
| `app/security/` | `passwords.py` (Argon2 via `pwdlib`), `jwt.py` (`JWTService`, HS256, issuer/audience validation), `cookies.py` (the refresh cookie builder). |
| `app/agents/` | Phase 11: `state.py`, `config.py` (`DEFAULT_MAX_ITERATIONS=3`, `DEFAULT_CONFIDENCE_THRESHOLD=0.5`), `nodes.py` (`decide_route`, `use_tools`, confidence math, best-attempt tracking), `graph.py` (LangGraph `StateGraph` with lazy import, plus a dependency-free manual fallback loop that must stay behaviorally identical). |
| `app/tools/` | Phase 12, stdlib only: `base.py` (`BaseTool`), `calculator.py` (safe `ast` evaluation — never `eval`), `web_search.py` (`TAVILY_API_KEY`/`SERPAPI_API_KEY` backends, error dicts never raise), `registry.py` (`ToolRegistry`, calculator-first order, `resolve_registry`). |
| `app/utils/retry.py` | Phase 6: `retry_with_backoff` (transient-only: timeouts, connection errors, 429/5xx duck-typed via `code`/`status_code`) + `CircuitBreaker` + shared `get_breaker`/`reset_breakers` seams. |
| `app/models/user.py` | `normalize_username` and the user document shape. |
| `frontend/` | Vite + React. `src/api/chatApi.js` is the SSE client that validates the backend event order strictly; `src/components/chats/ChatWindow.jsx` renders streaming answers, the sources section, connection status, and the Phase 7 degraded banner; `src/history.js` manages tab-local conversation history; `App.jsx` composes the shell. |
| `data/*.txt` | Seed corpus (`rag_basics.txt`, `embeedings.txt`, `vector_bases.txt`). Baked into the backend image; `data/` is **not** bind-mounted, so rebuild the backend after editing it. |
| `eval_cases.json`, `langsmith_eval.py` | Retrieval/answer/refusal evaluation cases runnable against the live system. |
| `Dockerfile`, `chroma/Dockerfile`, `frontend/Dockerfile`, `docker-compose.yml`, `docker-compose.prod.yml`, `frontend/nginx.conf`, `frontend/nginx.prod.conf`, `.dockerignore`, `frontend/.dockerignore`, `DOCKER_SETUP.html` | The Phase 10 Docker layer (detailed in section 1 of this document). |

### C. Runtime topology (Docker)

A five-service Compose project named `rag-learning`:

```text
browser ──:80── frontend (nginx:1.27-alpine, serves /dist, proxies /api /health /ready)
                     │
                     └──http://backend:8000── backend (python:3.11-slim, uvicorn :8000, --reload in dev)
                     │                              │        │            │
                     │                          :6379      :27017        :8000
                     │                             │          │            │
                     │                          redis      mongo        chroma
                     │                          (7.2)       (7)       (1.5.9 + curl)
```

- Internal topology: only `backend` and `frontend` publish ports (8000 and 80). Redis, Mongo, and Chroma are on the Compose network only — never expose them. The backend reaches them via env: `REDIS_URL=redis://redis:6379/0`, `MONGODB_URI=mongodb://mongo:27017`, `MONGODB_DATABASE=agent`, `CHROMA_URL=http://chroma:8000`.
- Healthchecks: Redis (`redis-cli ping`), Mongo (`mongosh` ping), Chroma (`curl /api/v2/heartbeat` — why `chroma/Dockerfile` exists: upstream has neither python nor curl), backend (`urllib` GET `/health`, start_period 90s), frontend (`wget /health` through nginx). `depends_on: service_healthy` gates the backend on all three datastores and the frontend on the backend.
- Named volumes: `redis-data`, `mongo-data`, `chroma-data`, `uploads-data`. `docker compose down -v` deletes them all — destructive to chat/document data.
- Dev compose bind-mounts `./app` and `./phase1.py` into the backend (live code without rebuild) and sets `REFRESH_COOKIE_SECURE=false` for HTTP. The other env is fixed in `docker-compose.yml`: `UPLOAD_DIR=uploads`, plus the four datastore URLs.
- `docker-compose.prod.yml` overrides `backend.command` to drop `--reload`, forces `REFRESH_COOKIE_SECURE=true`, uses `volumes: !override` to drop the dev bind mounts (production must run the copied image code, not the host tree), and fronts nginx TLS on :443 with certs from `certs/` mounted read-only plus `frontend/nginx.prod.conf` (HTTP `/health` kept plain for the container probe, everything else 301s to HTTPS).
- `CHROMA_URL` is the seam: set (compose) → `chromadb.HttpClient` via `phase1.build_chroma_client`; unset → `PersistentClient` on `chroma_data/`, with `runtime_chroma_data/` fallback if the primary path isn't writable. Local dev behavior is unchanged when the variable is absent.
- Builds: backend image installs CPU torch first (`--index-url https://download.pytorch.org/whl/cpu`), then `requirements.txt` through a bounded 10-attempt retry loop with the optional `PIP_INDEX_URL` build arg (default official PyPI; e.g. `https://mirrors.aliyun.com/pypi/simple/` on slow networks). This retry loop already saved a real build — a transient tzdata hash mismatch failed attempt 1 and passed on attempt 2 from the pip cache.

### D. Backend layering and a chat request's full path

Layering, top to bottom: `nginx` (in Docker) or Vite dev proxy → FastAPI route (`AuthRoute` wrapper class on auth routes) → dependencies (auth, ownership, rate limit) → service → `phase1.RAGSystem` → Chroma/Gemini/Redis/Mongo.

For `POST /api/v1/chat/sse`, the exact lifecycle:

1. `prepare_sse_chat` — a **sync** dependency so FastAPI runs it in the threadpool and pre-stream errors keep real status codes (404 for a bad `conversation_id`, 403 for ownership, 401 upstream). It resolves conversation state / server history before any stream starts.
2. The route is a **sync generator endpoint** that yields `ServerSentEvent` items (FastAPI 0.141 native SSE). Returning an `EventSourceResponse` *instance* makes FastAPI try to iterate it and fails with `TypeError: 'EventSourceResponse' object is not iterable` — this broke curl with `transfer closed with outstanding read data remaining` and was fixed by yielding items instead.
3. `chat_orchestration` runs one turn: ownership check already done → history handling → `RAGService.run_once*` → answer → usage row (fail-open).
4. Event order on the wire, validated strictly by `chatApi.js`: `start → retrieval → token* → sources → done`, or `error` with a `stage` and a generic message. `done` may carry `{"metadata": {...}}` (cache hit/miss, degraded flag, usage) — uncached/byte-identical callers keep `done` as `{}`.

### E. Identity and session lifecycle (auth contract — don't redesign)

- **Register** `POST /api/v1/auth/register` → 201, Argon2 hashing, NIST floor of 8+ chars (the 256 ceiling is CPU-bounding, not policy). **Login** `POST /api/v1/auth/login` → permissive (`LoginRequest` must never distinguish bad format from bad credentials — always the uniform 401), and a fresh login revokes the old cookie session before issuing a new one.
- **Tokens**: access JWT — 15 minutes, Bearer header, held in memory only. Refresh JWT — 7 days, `HttpOnly` cookie `refresh_token`, `Path=/api/v1/auth`, `SameSite=Strict`, `Secure=true` by default (`REFRESH_COOKIE_SECURE=false` for local HTTP only).
- **Redis session store**: `auth:refresh:<jti> → user_id` written with `EXAT=<jwt exp>` + `NX`; rotation is a single atomic `GETDEL` (refresh = consume + reissue); logout is `DELETE`. **Redis command retries are deliberately disabled** — repeating a `GETDEL` after a lost reply is unsafe. Never trust a refresh JWT whose Redis entry is missing; never store refresh tokens in localStorage.
- **Token blacklist (Phase 4, always on)**: revoked access JTIs go to `auth:blacklist:<jti> → 1` with `EXAT` + `NX` (auto-expiry, no cleanup). `get_current_user` rejects listed tokens with 401 and maps Redis outage to **503 (fail-closed)**. Logout/refresh accept an optional Bearer token and blacklist its JTI when decodable; contracts stay unchanged (logout still 204, refresh still 401-on-bad-session).
- **Status semantics**: 401 → re-login; 403 → ownership mismatch; 503 → infra failure (retry, don't treat as bad credentials). Logout of an already-invalid cookie still returns 204 and clears the cookie.
- **`AuthRoute`**: strips request bodies from 422 responses and sets `Cache-Control: no-store` / `Pragma: no-cache` on every auth response — preserve both.
- Rate limiting lives on `rate_limit:` Redis keys (`RateLimitService`), applied via `enforce_login_rate_limit` / `enforce_refresh_rate_limit` / `enforce_upload_rate_limit` dependencies.

### F. The RAG core mechanics (`phase1.RAGSystem`)

- **Corpus → chunks**: `load_txt_documents` reads every `data/*.txt`, derives doc ids from filenames (`{user_id}:{doc_id}:{index}` chunk ids come later), and **normalizes the historical `embeedings` typo to doc id `embeddings`** (matches `eval_cases.json` — do not "fix" by renaming the effect). Chunking defaults: 512 chars, 64 overlap. Empty files produce empty documents — a corpus of 0-byte files leaves `/ready` at 503 forever; this actually happened once when the image was built before the content was saved.
- **Embeddings**: model `gemini-embedding-2`. `embed_documents` wraps **each chunk in its own `types.Content`** — passing a plain string list makes Gemini collapse it into one multi-part content and return one embedding, and the Chroma upsert then fails with unequal `ids`/`embeddings` (`ids: 2, metadatas: 2, embeddings: 1`). A count guard raises if embedding count ≠ chunk count. Query embedding (`embed_query`) is a single string, unaffected.
- **Generation**: model `gemini-3.6-flash`, called only through `generate_answer` / `generate_answer_stream`; both capture `usage_metadata` (strict `int` extraction so Mock attributes never fabricate rows) into a caller-owned `usage` collector (last-chunk-wins on clean stream completion).
- **Retrieval invariant**: every `collection.query` is filtered `where={"user_id": ...}`. Tenancy is enforced by the filter, never by trusting client data; chunk ids embed the user id as their prefix. **Never drop this filter.**
- **Limits**: `k` 1–10, `raw_query` 1–4000 chars, `chat_history` ≤20 messages (last 8 used), message content ≤8000 chars, `distance_threshold` 0–2 cosine. Defaults: chunk 512/64, models above.
- **Refusal contract**: an unanswerable/empty-retrieval turn returns exactly `I do not have enough information to answer that.` — and history is for reference resolution only, never a factual source.
- **Hybrid retrieval (Phase 2, flag-gated default-off `HYBRID_ENABLED`)**: `HybridRetriever` builds a per-user in-memory BM25 index (`rank_bm25`, lazy import) from `collection.get(where={"user_id": ...})`, over-fetches `min(10, max(2*k, HYBRID_OVERFETCH_MIN=20))` per leg, and fuses with RRF `1/(HYBRID_RRF_K=60 + rank)`. `RAGService` owns the long-lived cache and invalidates on ingest/delete; `RAGSystem._retrieve_chunks` falls back to vector-only on any BM25 failure.
- **Reranking (Phase 3a, flag-gated default-off `RERANKER_ENABLED`)**: `CrossEncoderReranker` (default `cross-encoder/ms-marco-MiniLM-L-6-v2`, `sentence-transformers`, lazy import on first use — **never at startup**) rescores the over-fetched candidate window (≤10 chunks) and keeps `min(k, RERANKER_TOP_K=5)`. Failure preserves retrieval order. Cross-encoder only; no Cohere backend.
- **Long-conversation summarization (Phase 1)**: past `SUMMARY_THRESHOLD` (20) completed messages, generation gets one leading `system` summary + the last 8 verbatim (`SUMMARY_KEEP_RECENT`) instead of the raw 20. The summary lives on the conversation (`summary`, `summary_message_count`, `summary_updated_at`) and is refreshed after `SUMMARY_MAX_AGE` (10) new messages. The LLM call lives in `phase1.summarize_conversation_history`; the service layer never prompts. Test seam: `chat_orchestration.default_summarize_fn`.
- **The `run_once*` trio** is the single entry point for one turn: `_retrieve_chunks` (hybrid/rerank inside) → response-cache lookup (Phase 5) → cache-hit replay or LLM: cache miss → generate → grounded answer stored only if successful (errors propagate; refusals/empties skipped). Per-request overrides (`use_hybrid`/`use_rerank`/`use_cache`/`use_degraded`/`use_agent`, each `True/False/None(env)`) exist on `run_once*`, `RAGService.run_once*`, and `chat_orchestration.run_chat_once`.

### G. The agent loop and tools (Phases 11–12, flag-gated default-off `AGENT_ENABLED`)

- `app/agents/` routes each query (`nodes.decide_route`: smalltalk → `direct` answer with no retrieval/LLM, everything else → `retrieve`), then runs retrieve → generate → check_confidence → respond_or_retry. Confidence = `1 − mean_distance/2` (refusal/empty/no-chunks → 0.0, direct answers → 1.0); weak answers retry with a wider `k` window up to `AGENT_MAX_ITERATIONS=3`, then the best-effort answer (or refusal) is returned with `gave_up=true`.
- `graph.build_graph` lazily imports `langgraph==1.2.11` (`StateGraph`) — never at startup — with an identical dependency-free manual loop on import/compile failure.
- Retrieval reuses `RAGSystem._retrieve_chunks` unchanged (tenant filter + hybrid RRF + rerank). Terminal generation failure honors the Phase 7 degraded-excerpt path when enabled. `nodes.track_best` keeps the highest-confidence attempt so retries strictly improve.
- Integration: `RAGService.run_once` owns it fail-open (agent exception → legacy `rag.run_once`); **streams keep the legacy path**. Results are additive `result["agent"]={route, attempts, confidence, gave_up, tools_used}` — `ChatResponse` ignores it.
- Tools (Phase 12, no new dependencies): after routing, a `use_tools` node tries the calculator first (math answered directly at confidence 1.0, no retrieval/LLM); web-search results become grounding chunks (`document_id="web_search"`, distance 0.5); tool misses/failures fall through to local retrieval, and retries always use the corpus. `run_agent(tools=...)` overrides the default registry (empty list disables tools); custom tools subclass `BaseTool` (`name`/`description`/`match`/`execute`).

### H. Resilience, caching, degradation, metering (Phases 4–8)

- **Retry + circuit breaker (Phase 6, always on)**: `retry_with_backoff` retries transient failures only (timeout, connection errors, 429/5xx duck-typed via `code`/`status_code` — never other 4xx, auth/duplicate-key errors, bugs, or cancellations) with exponential backoff + jitter (`RETRY_MAX_ATTEMPTS=3`, `RETRY_BASE_DELAY=0.5`, `RETRY_MAX_DELAY=8.0`). Applied to all Gemini calls (`circuit="gemini"`) and all Mongo service methods (`circuit="mongodb"`; the delegating `get_summary_state` is excluded to avoid nested-retry amplification). Service domain exceptions (`StoreUnavailable`) are seen through via the `__cause__` chain. Generators retry only **before the first yielded item** — never mid-stream. Retried inserts use fresh backend uuids per attempt (at-least-once, documented in the module). `CircuitBreaker`: opens after 5 consecutive transient failures (`CIRCUIT_BREAKER_THRESHOLD`), fails fast for 30s (`CIRCUIT_BREAKER_COOLDOWN_SECONDS`), then half-open probes; thread-safe, shared per dependency name via `get_breaker` (`reset_breakers` is the test seam).
- **Response cache (Phase 5, flag-gated default-off `CACHE_ENABLED`)**: `CacheService` stores answered payloads as `response_cache:<user>:<gen>:<sha256(user, query, sorted chunk ids)>` with `EX=CACHE_TTL_SECONDS` (3600). Invalidation is a per-user generation bump (`response_cache:gen:<user>` via `INCR`) by `RAGService` on ingest/delete — O(1), orphans expire via TTL. **Fail-open**: Redis trouble means uncached answers, never failed turns. Cache hits add `cache_hit` and SSE `done` carries `{"metadata": {"cache": "hit"|"miss"}}`; cached answers are never corrupted into refused turns.
- **Degraded mode (Phase 7, flag-gated default-off `DEGRADED_MODE_ENABLED`)**: when the LLM is unavailable *after* Phase 6 retries (exhausted transient error, or open `gemini` circuit — never bugs/4xx, never empty retrieval), `run_once*` serve the top `DEGRADED_MAX_CHUNKS=3` chunks as plain-text excerpts under the exact `I'm experiencing high load...` prefix. Pre-stream failures degrade; **mid-stream failures keep the `error` event — a partial answer must never be replaced**. Degraded answers are never cached (the LLM may have recovered), are counted in Redis (`llm:degraded:responses` via `DegradationTracker`, fail-open), and surface as `run_once["degraded"]` / `done.metadata.degraded` / a `ChatWindow.jsx` banner.
- **Token/cost metering (Phase 8, always on)**: `phase1` pipes `usage_metadata` into a caller-owned collector; `run_once` attaches `result["usage"]`, SSE `done` carries `metadata.usage` (verified live: prompt/completion/total counts with `model`), plain-text streams fill a route-owned collector; absent/invalid metadata keeps byte-identical outputs. `chat_orchestration` records rows **fail-open** (a ledger outage is logged, the answer still goes out) into Mongo `usage_logs` (`usage_id`, `user_id`, `timestamp`, token counts, `model`, `cost_usd` at `COST_PER_1K_INPUT=0.0001` / `COST_PER_1K_OUTPUT=0.0004` per 1K). `GET /api/v1/usage?days=7` returns UTC-calendar daily groups + totals (auth-scoped, 503 without service). Only generation calls are metered — rewrite/summary calls have no user context and are excluded by design.

### I. Data model

MongoDB (`agent` by default; `MONGODB_DATABASE=agent` in compose):

| Collection | Holds |
|---|---|
| `users` | Argon2-hashed credentials, normalized usernames, `user_id` (uuid), `disabled` flag; unique index via `ensure_indexes` |
| `conversations` | Per-user conversation metadata + Phase 1 summary fields |
| `messages` | Individual turns, per conversation, chronological |
| `documents` | Tenant-owned upload metadata: `document_id`, `original_filename`, `status` (`processing` → `ready`/`failed`), `chunk_count`, `title`, size |
| `usage_logs` | Phase 8 metric rows (one per completed chat turn) |

Chroma collection `learning_rag`: chunk ids `{user_id}:{document_id}:{index}`; every query filtered by `where={"user_id": ...}`. At handoff it held 13 seed chunks under `eval_user`.

Redis namespaces (one client, six logical stores): `auth:refresh:<jti>`, `auth:blacklist:<jti>`, `rate_limit:*`, `response_cache:<user>:<gen>:<hash>`, `response_cache:gen:<user>`, `llm:degraded:responses`.

### J. Endpoint reference (all verified against the running stack)

| Method & path | Contract notes |
|---|---|
| `GET /health` | Liveness, always 200 |
| `GET /ready` | 503 until Chroma has documents; then `{"status":"ready"}` |
| `POST /api/v1/auth/register` | 201, Argon2, username normalization |
| `POST /api/v1/auth/login` | Uniform 401 on bad credentials; issues access + refresh cookie |
| `POST /api/v1/auth/refresh` | Single-use `GETDEL` rotation; 401 on bad/used/missing session |
| `POST /api/v1/auth/logout` | 204 + cookie clear, even for an invalid cookie |
| `GET /api/v1/auth/me` | Snapshot of the current principal |
| `POST /api/v1/chat` | JSON `ChatResponse`: `answer`, `retrieval_query`, `retrieved_document_ids`, `chunks` |
| `POST /api/v1/chat/stream` | Plain-text `StreamingResponse` |
| `POST /api/v1/chat/sse` | Generator endpoint yielding `ServerSentEvent` items; strict `start → retrieval → token* → sources → done` |
| `POST /api/v1/conversations` / `GET ...` / `GET /{id}` / `GET /{id}/messages` | Persistent conversation listing/history |
| `POST /api/v1/documents/upload` | 202 + `processing` record; poll `GET /{document_id}` until `ready`/`failed` |
| `GET /api/v1/documents` / `GET /{id}` / `DELETE /{id}` | Tenant-scoped listing/detail/delete (delete also drops chunks) |
| `GET /api/v1/usage?days=7` | Daily UTC groups + rolled-up totals |

### K. Environment and configuration

`.env` (never printed/committed; root `.env.example` is additions-only and merged, never replaced): `JWT_SECRET_KEY` (≥32 UTF-8 bytes, HS256), `JWT_ISSUER`, `JWT_AUDIENCE`, `GEMINI_API_KEY` or `GOOGLE_API_KEY` (first-start ingestion embeds at boot and restart-loops without it), Mongo settings, plus optional `REDIS_URL`, `REFRESH_COOKIE_SECURE`, cache/retrieval flags, and `CHROMA_URL`. Phase flags (`HYBRID_ENABLED`, `RERANKER_ENABLED`, `CACHE_ENABLED`, `DEGRADED_MODE_ENABLED`, `AGENT_ENABLED`) default off. Frontend `frontend/.env.example` has only `API_PROXY_TARGET` (default `http://127.0.0.1:8000`) — Vite proxies `/api`, `/health`, `/ready` same-origin; never put Gemini keys in browser env vars.

### L. The fixes already made (context for the changes in sections 1–2)

- Backend image builds on `python:3.11-slim`, installs CPU torch first, uses the optional `PIP_INDEX_URL` build arg, and retries the requirements install (the retry loop has already rescued a real hash-mismatch failure — treat it as load-bearing).
- Chroma runs a thin `chromadb/chroma:1.5.9` layer with `curl`, because the upstream image ships neither python nor curl and the healthcheck needs an HTTP probe.
- `CHROMA_URL` switches `phase1.RAGSystem` from the embedded `PersistentClient` to `chromadb.HttpClient` (seam: `phase1.build_chroma_client`).
- `phase1.embed_documents` wraps each chunk in its own `types.Content` (Gemini embedding collapse bug — see section F) with a count guard.
- `/chat/sse` is a generator endpoint yielding `ServerSentEvent` items; pre-stream preparation lives in the sync dependency `prepare_sse_chat` so it runs in the threadpool and preserves 404s.
- The dev Compose file bind-mounts `app/` and `phase1.py`, but `data/` is baked into the backend image. Rebuild the backend after editing `data/*.txt`.
- The frontend has no registration, login, or upload UI. Register/login and upload through the API or Swagger, then paste the access token and registered `user_id` into the frontend settings (Retrieval settings → Access token; User context → user_id). The default frontend user is `eval_user`, which will not match a registered JWT (the JWT `sub` is a uuid, never the username — a mismatched `user_id` is exactly a 403).

### M. Verification state and known issues

- The Docker setup was validated from fresh images: all five containers healthy, backend `/health` and `/ready` → 200, frontend `/`, `/health`, `/ready` → 200, and a live full SSE sequence with usage metadata in `done`.
- A full smoke-test run was additionally completed (fresh test user, `.txt` upload → `ready` with chunks, grounded answer with `[Source 1]` citation, exact-refusal behavior for an unanswerable question, strict cross-user 403, and the `start → retrieval → token* → sources → done` order through nginx).
- The full backend suite is **413 tests, 1 pre-existing failure, 0 errors**. The failure is `test_completed_turns_load_in_chronological_order`, caused by the in-memory fake assigning tied timestamps (see section 6 for the fix approach). The 5 earlier SSE errors are fixed. Tests need **no** live Redis/Mongo/AI (InMemoryRedis double + fakes).
- Known pre-existing limitation: persistent conversations that send `conversation_id` can still fail in a live Mongo environment with `RuntimeError: Cannot use AsyncMongoClient in different event loop` because sync routes bridge with `asyncio.run` in worker threads. The current frontend `streamChat` does **not** send `conversation_id`, so its chat path is unaffected. Do not treat this as a Docker regression.
- Do not commit `.env`, `certs/*` (self-signed cert/key material exists locally — regenerate, never ship), Chroma data, uploads, or the pre-existing `architecture.html`. Do not overwrite the pre-existing `data/*.txt` corpus content. No commit has been made — review `git status` and `git diff` before staging anything (a `cookies.txt` smoke-test artifact may be present and should not be committed).

### N. Next actions

Verify the running stack (`docker compose ps`, all five healthy), then perform the frontend-to-backend end-to-end test in section 5 using a fresh test user and an uploaded `.txt` file (the full command sequence is there). Leave the stack running unless the user asks to stop it. If a datastore behaves oddly, consult section 2 (the error history) before changing anything — every container-level quirk seen so far already has a documented cause and fix.

---

## 1. What changed

### Docker and configuration

| File | Change |
|---|---|
| `Dockerfile` | Backend image: `python:3.11-slim`, CPU torch, optional `PIP_INDEX_URL`, bounded pip retry loop, no pip upgrade. |
| `.dockerignore` | Keeps `.env`, `.venv`, tests, frontend, certs, uploads, Chroma data, and docs out of the backend image. |
| `docker-compose.yml` | Five services, healthchecks, `depends_on: service_healthy`, internal service names, dev bind mounts, volumes, and ports 8000/80. |
| `docker-compose.prod.yml` | Production override: no `--reload`, `REFRESH_COOKIE_SECURE=true`, nginx TLS on 443, and `volumes: !override` to drop dev bind mounts. |
| `chroma/Dockerfile` | Thin layer over `chromadb/chroma:1.5.9` that installs `curl` for the heartbeat healthcheck. |
| `frontend/Dockerfile` | `node:22-alpine` build stage, then `nginx:1.27-alpine`. |
| `frontend/.dockerignore` | Keeps node modules, tests, and env files out of the frontend build. |
| `frontend/nginx.conf` | Proxies `/api`, `/health`, `/ready` to `backend:8000`; SSE buffering disabled; generous read/send timeouts. |
| `frontend/nginx.prod.conf` | HTTP `/health` for the container probe, HTTP-to-HTTPS redirect, TLS server on 443 with the same API proxy rules. |
| `.env.example` | Documents required runtime keys, optional `PIP_INDEX_URL`, cookie security, retrieval flags, and `CHROMA_URL`. |
| `.gitignore` | Ignores `certs/*` except `certs/.gitkeep`; existing rules ignore `.env`, Chroma data, and uploads. |
| `DOCKER_SETUP.html` | User-facing setup, architecture, startup, production TLS, and troubleshooting guide. |
| `AGENTS.md` | Added the Chroma/embedding and FastAPI SSE invariants and the data-rebuild note. |

### Application code

| File | Change |
|---|---|
| `phase1.py` | `build_chroma_client` supports `CHROMA_URL`; `embed_documents` now wraps each chunk in `types.Content` and raises if the embedding count does not match the chunk count. |
| `app/routes/chat.py` | `/chat/sse` yields `ServerSentEvent` items; `prepare_sse_chat` is a sync dependency that resolves conversation state before the stream starts. |
| `tests/test_chroma_mode.py` | Tests embedded/HTTP Chroma selection and the per-chunk embedding shape. |

### Pre-existing working-tree content, not authored here

- `data/*.txt` already contained the current corpus content. It was rebuilt into the backend image and ingested.
- `phase1.py` already contained the `build_chroma_client` work before the embedding fix.
- `architecture.html` is a pre-existing untracked file. Leave it alone.

---

## 2. Errors that occurred and how they were fixed

| Symptom | Root cause | Correction |
|---|---|---|
| Fresh pulls failed intermittently: `registry-1.docker.io: no such host`. | Transient Docker Hub DNS/network failure. | Retried the pulls. Redis 7.2 and Mongo 7 eventually pulled successfully. |
| Chroma container was unhealthy. | Upstream `chromadb/chroma:1.5.9` has neither Python nor `curl`. | Added `chroma/Dockerfile` with `curl` and used `/api/v2/heartbeat` for the healthcheck. |
| Backend `/ready` stayed 503 even though `/health` was 200. | The backend image contained 0-byte `data/*.txt` files because the image had been built before the current corpus content was saved. | Rebuilt the backend image; verified `data/*.txt` total 4232 bytes inside the image; startup ingested 13 chunks. |
| Ingestion failed with `ids: 2, metadatas: 2, embeddings: 1`. | `gemini-embedding-2` treats a plain `contents=["chunk1", "chunk2"]` as one multi-part content and returns one embedding. | Wrapped each chunk in `types.Content(parts=[types.Part.from_text(...)])`; added a count guard; re-ran ingestion. |
| Live `/chat/sse` returned HTTP 200 and then `curl: (18) transfer closed with outstanding read data remaining`; backend logged `TypeError: 'EventSourceResponse' object is not iterable`. | FastAPI 0.141 native SSE expects a generator endpoint that yields `ServerSentEvent` items. The route returned an `EventSourceResponse` instance. | Changed `/chat/sse` to a sync generator yielding `ServerSentEvent`; removed the response instance; verified direct and through nginx. |
| SSE tests failed with `RuntimeError: asyncio.run() cannot be called from a running event loop`. | FastAPI calls SSE endpoints in the event loop; `prepare_chat` was called directly in that path while it uses `asyncio.run`. | Moved pre-stream preparation into the sync dependency `prepare_sse_chat`, which FastAPI runs in the threadpool. The SSE test errors are gone. |
| `pip install` was very slow or timed out. | Docker Hub/PyPI network latency. | Added the optional `PIP_INDEX_URL` build arg and a bounded retry loop; used the Aliyun mirror locally. The default remains official PyPI. |
| `docker compose build/up` sometimes exited 1 through PowerShell/Tee even though the operation succeeded. | PowerShell `NativeCommandError` handling around native stderr/stdout. | Trust `docker compose ps`, container logs, and endpoint checks rather than the pipeline exit code. |
| Backend unable to ingest on a fresh start with no Gemini key. | Embeddings are generated during first-start ingestion. | `.env` must contain `GEMINI_API_KEY` or `GOOGLE_API_KEY`; without it, the backend can restart-loop. Empty data leaves `/health` at 200 but `/ready` at 503. |
| Production Compose could not be started. | `certs/server.crt` and `certs/server.key` are intentionally absent from Git. | Config was validated with `docker compose ... config --quiet`; TLS startup is documented but was not run. |

---

## 3. Current state at handoff

- All five containers healthy:
  - `rag-learning-backend-1` on `0.0.0.0:8000`
  - `rag-learning-frontend-1` on `0.0.0.0:80`
  - `rag-learning-chroma-1`, `rag-learning-mongo-1`, `rag-learning-redis-1` internal
- Backend `/health` → 200.
- Backend `/ready` → 200.
- Frontend `/`, `/health`, `/ready` → 200.
- Frontend nginx → backend `/ready` → `{"status":"ready"}`.
- Chroma `learning_rag` → 13 chunks under `eval_user`.
- Backend image → 2.58 GB.
- Frontend image → 74.1 MB.
- Chroma image → 845 MB.
- Live SSE through the frontend proxy returned the full `start → retrieval → token → sources → done` sequence.
- A full test-user upload → poll → chat run completed: upload reached `ready` with 8 chunks, `/chat/sse` returned 200, the answer was grounded in `data/rag_basics.txt` with 3 sources, and the `done` event included token usage metadata. The test document and user were deleted afterward.
- Temporary smoke-test users and their chunks were removed from MongoDB and Chroma; Chroma is back to the original 13 `eval_user` chunks, and the `users`/`documents`/`conversations`/`messages` collections are empty.
- `docker compose config --quiet` passes for dev and production overrides.
- `git diff --check` is clean aside from expected CRLF warnings on the pre-existing data files.
- No commit has been made.

---

## 4. Start or restart the project

Run PowerShell from the repository root.

### 4.1 Prerequisites

- Docker Desktop is running.
- `.env` exists with at least:
  - `GEMINI_API_KEY` or `GOOGLE_API_KEY`
  - `JWT_SECRET_KEY` (at least 32 UTF-8 bytes)
  - `JWT_ISSUER`
  - `JWT_AUDIENCE`
  - MongoDB settings needed by the app
- Do not print or commit `.env`.
- Use `REFRESH_COOKIE_SECURE=false` for local HTTP development; leave `true` for production.

Start Docker Desktop if needed, then:

```powershell
cd "C:\Users\Sanchit Pandey\OneDrive\Desktop\agent"

docker compose up -d --build
docker compose ps
```

Expected: backend, frontend, chroma, mongo, and redis all show `healthy`.

Check endpoints:

```powershell
curl.exe -sS -o NUL -w "backend /health: %{http_code}`n" http://localhost:8000/health
curl.exe -sS -o NUL -w "backend /ready : %{http_code}`n" http://localhost:8000/ready
curl.exe -sS -o NUL -w "frontend /     : %{http_code}`n" http://localhost/
curl.exe -sS -o NUL -w "frontend /ready: %{http_code}`n" http://localhost/ready
```

If `/ready` is 503:

1. Check `docker compose logs --tail=150 backend`.
2. Confirm `data/*.txt` is non-empty.
3. Confirm the Gemini key is present in `.env`.
4. If `data/` changed after the image was built, rebuild:

```powershell
docker compose build backend
docker compose up -d backend
```

If port 80 is occupied, change the frontend mapping in `docker-compose.yml` to `"8080:80"` and use `http://localhost:8080`.

### 4.2 Useful lifecycle commands

```powershell
docker compose logs -f backend
docker compose logs -f frontend
docker compose restart backend
docker compose down        # keeps named volumes
docker compose down -v     # deletes Redis, Mongo, Chroma, and upload volumes
```

`down -v` is destructive to local chat/document data. Use it only when a clean slate is intended.

### 4.3 Production override (optional)

The production Compose file is config-validated but was not started because TLS certificates are absent. To test locally, create self-signed certs:

```powershell
New-Item -ItemType Directory -Force certs | Out-Null
openssl req -x509 -newkey rsa:2048 -nodes `
  -keyout certs/server.key -out certs/server.crt -days 365 `
  -subj "/CN=localhost" `
  -addext "subjectAltName=DNS:localhost,IP:127.0.0.1"

docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d --build
```

Open `https://localhost` and accept the self-signed certificate warning. Do not commit the certificate or key files.

---

## 5. End-to-end test: frontend prompt to backend answer

### Important frontend limitation

The current React UI is a development shell. It does **not** contain registration, login, or document upload screens. It expects:

- an access token pasted into **Retrieval settings → Access token**
- a registered user's UUID in **User context → user_id**

The default user is `eval_user`. The pre-ingested 13 chunks are under `eval_user` for evaluation, but a normal login does not produce a token for that tenant. For a clean end-to-end demo, register a normal test user and upload a `.txt` as that user.

### 5.1 Register, login, and upload a document

Run this in PowerShell from the repository root:

```powershell
$Base = "http://localhost"
$Username = "phase10_$([DateTimeOffset]::UtcNow.ToUnixTimeSeconds())"
$Password = "Phase10Test!123"
$AuthBody = @{ username = $Username; password = $Password } | ConvertTo-Json

$User = Invoke-RestMethod -Method Post `
  -Uri "$Base/api/v1/auth/register" `
  -ContentType "application/json" `
  -Body $AuthBody

$Login = Invoke-RestMethod -Method Post `
  -Uri "$Base/api/v1/auth/login" `
  -ContentType "application/json" `
  -Body $AuthBody

$Headers = @{ Authorization = "Bearer $($Login.access_token)" }

Write-Host "USER_ID:      $($User.user_id)"
Write-Host "ACCESS_TOKEN: $($Login.access_token)"
```

The access token is a 15-minute test credential. Do not paste it into chat, commit it, or share it.

Upload the existing corpus file as the test user's document:

```powershell
$UploadJson = curl.exe -sS -X POST "$Base/api/v1/documents/upload" `
  -H "Authorization: Bearer $($Login.access_token)" `
  -F "file=@data/rag_basics.txt;type=text/plain"

$Upload = $UploadJson | ConvertFrom-Json

Write-Host "DOCUMENT_ID:  $($Upload.document_id)"
Write-Host "INITIAL STATE: $($Upload.status)"
```

If `data/rag_basics.txt` is unavailable, create a temporary `.txt` file with a simple RAG explanation and upload that instead.

Poll until the background ingestion finishes:

```powershell
$Doc = $null
for ($i = 0; $i -lt 60; $i++) {
  $Doc = Invoke-RestMethod -Method Get `
    -Uri "$Base/api/v1/documents/$($Upload.document_id)" `
    -Headers $Headers

  if ($Doc.status -eq "ready" -or $Doc.status -eq "failed") { break }
  Start-Sleep -Seconds 2
}

$Doc | Select-Object document_id, status, chunk_count, title | Format-List
```

Expected: `status` becomes `ready` and `chunk_count` is greater than zero. If `status` is `failed`, check `docker compose logs --tail=150 backend` and the Gemini key.

### 5.2 Use the frontend

1. Open `http://localhost`.
2. Open the gear icon (Retrieval settings).
3. Paste the `ACCESS_TOKEN` value into **Access token**.
4. Set **User context → user_id** to the `USER_ID` value from the script.
5. Click **Apply user context**.
   - This clears the current tab's conversations; that is expected.
6. In the composer, ask a question that the uploaded file answers, for example:

   `What is RAG?`

7. Press Enter.

Expected UI behavior:

- Status moves through connecting/searching/streaming.
- The assistant answer streams token by token.
- A sources section appears with the uploaded document.
- The answer is grounded in the uploaded text.

If it fails:

| Result | Meaning | Action |
|---|---|---|
| 401 | Missing, invalid, or expired access token. | Login again and paste the new token. |
| 403 | `user_id` does not match the JWT subject. | Use the exact `user_id` returned by registration. |
| Refusal text: `I do not have enough information to answer that.` | No documents for this user, or the upload is not ready. | Check `GET /api/v1/documents`, wait for `ready`, and confirm `user_id`. |
| 502/504 | nginx/backend network or timeout. | Check `docker compose ps` and backend logs. |
| `The response could not be completed.` | Backend SSE error event. | Inspect backend logs for the failing stage. |

The frontend does not auto-refresh tokens. The access token expires after 15 minutes; log in again when it expires.

### 5.3 API-level SSE verification (same path as the browser)

This confirms the backend stream independently of the React UI:

```powershell
$ChatBody = @{
  raw_query   = "What is RAG?"
  user_id     = $User.user_id
  chat_history = @()
} | ConvertTo-Json -Depth 5

$BodyFile = Join-Path $env:TEMP "phase10-chat.json"
[System.IO.File]::WriteAllText($BodyFile, $ChatBody)

curl.exe -sS -N -X POST "$Base/api/v1/chat/sse" `
  -H "Authorization: Bearer $($Login.access_token)" `
  -H "Content-Type: application/json" `
  --data-binary "@$BodyFile"
```

Expected event sequence:

```text
event: start
data: {"raw_query": "What is RAG?"}

event: retrieval
data: {"retrieval_query": "...", "retrieved_document_ids": ["..."]}

event: token
data: {"text": "..."}

event: sources
data: {"sources": [...]}

event: done
data: {}
```

A user with no documents receives the exact refusal token:

`I do not have enough information to answer that.`

### 5.4 Non-streaming JSON check (optional)

```powershell
$Result = Invoke-RestMethod -Method Post `
  -Uri "$Base/api/v1/chat" `
  -Headers $Headers `
  -ContentType "application/json" `
  -Body $ChatBody

$Result | ConvertTo-Json -Depth 8
```

Expected fields include `answer`, `retrieval_query`, `retrieved_document_ids`, and `chunks`.

### 5.5 Cleanup

Delete the uploaded document from the test user's tenant:

```powershell
Invoke-RestMethod -Method Delete `
  -Uri "$Base/api/v1/documents/$($Doc.document_id)" `
  -Headers $Headers
```

Log out the access token:

```powershell
Invoke-RestMethod -Method Post `
  -Uri "$Base/api/v1/auth/logout" `
  -Headers $Headers
```

Optional local cleanup of the test user (only in development; verify the username first):

```powershell
docker compose exec -T mongo mongosh --quiet agent --eval `
  "db.users.deleteOne({username: '$Username'}); db.documents.deleteMany({user_id: '$($User.user_id)'}); db.conversations.deleteMany({user_id: '$($User.user_id)'}); db.messages.deleteMany({user_id: '$($User.user_id)'}); db.usage_logs.deleteMany({user_id: '$($User.user_id)'});"

docker compose exec -T backend sh -c "rm -rf /app/uploads/$($User.user_id)"
```

The test user can also be left in place; use a unique username each run. The API document delete should already remove the chunks, metadata record, and stored file; the optional Mongo/filesystem cleanup is a fallback. Do not run destructive cleanup against production data.

---

## 6. Known issues and out-of-scope items

### Persistent conversations with `conversation_id`

The `POST /api/v1/chat`, `/chat/stream`, and `/chat/sse` routes are synchronous and bridge to async Mongo services with `asyncio.run` in a worker thread. A live `AsyncMongoClient` is bound to the lifespan event loop, so a request with `conversation_id` can raise:

`RuntimeError: Cannot use AsyncMongoClient in different event loop.`

This is not caused by the Docker changes. The current frontend `streamChat` does not send `conversation_id`; its chat path uses client-side history. A future fix should make the conversation routes async (or bridge with `asyncio.run_coroutine_threadsafe` to the lifespan loop) and preserve pre-stream 404 behavior.

### One unit-test failure

`test_completed_turns_load_in_chronological_order` sets identical `created_at` values for the user and assistant messages in each fake turn, then expects the user message first. The in-memory fake's descending sort reverses ties. It is a pre-existing test-data issue, not a Docker or SSE regression. Fix by giving the assistant message a later timestamp or adding a deterministic secondary sort key; do not change tenant filtering.

### Production TLS

`docker-compose.prod.yml` is config-validated but was not started because `certs/server.crt` and `certs/server.key` are absent. The local self-signed instructions are in section 4.3.

### Frontend integration items

The “Integration roadmap” cards (document upload UI, persistent conversation storage, evaluation UI, concurrency controls) are placeholders. Do not assume they are implemented. The current working path is manual token/user setup plus API upload, then streaming chat.

### `eval_user` corpus

The 13 pre-ingested chunks live under `eval_user` for evaluation. The frontend default `user_id` is also `eval_user`, but that does not authenticate by itself. For a normal demo, upload a document as a registered user. Do not remove the tenant filter to make `eval_user` docs visible to other users.

---

## 7. Do not commit / preserve

- `.env` and any real secrets.
- `certs/server.crt` and `certs/server.key`.
- `chroma_data/`, `runtime_chroma_data/`, and Docker volumes.
- `uploads/`.
- The pre-existing untracked `architecture.html`.
- The current `data/*.txt` content (it is the corpus; do not overwrite it).
- Do not include temporary smoke-test users, uploaded test files, or test tokens.

A safe review sequence:

```powershell
git status --short
git diff --check
git diff -- .env.example .gitignore AGENTS.md app/routes/chat.py phase1.py
```

No commit was made during the Phase 10 work. Ask the user before committing, and stage only the intended Docker/configuration/code files.
