# Production Roadmap — RAG Learning

Status version: 2026-10-23. Baseline: the Phase 10–12 codebase described in `AGENTS.md`, validated by the smoke test in `HANDOFF_PHASE10.md` (5 healthy containers, full SSE verification, 413-test suite with 1 known pre-existing failure).

---

## Copy-ready handoff prompt

You are picking up the **production-hardening roadmap** for the RAG Learning project in:

`C:\Users\Sanchit Pandey\OneDrive\Desktop\agent`

Read `AGENTS.md` and `HANDOFF_PHASE10.md` first — they contain the architecture contracts and the Docker/runbook context you must not break. This document (`PRODUCTION.md`) is the work plan; sections 1–9 below are the assessment, invariants, and four implementation waves with acceptance criteria.

**Current live state (2026-10-23):**
- Docker stack: all five services (`backend`, `frontend`, `chroma`, `mongo`, `redis`) healthy. Start with `docker compose up -d` if stopped (no rebuild needed unless `data/*.txt`, frontend source, or Docker files change). Dev stack runs with `--reload` and bind-mounted `app/` + `phase1.py`.
- Data caution: the Mongo volume was **recreated empty** during a recent session (`docker compose down -v` side effect). The only account is a local test account `smoketest_user` / `dummypass123` (user_id `63504450-df73-4e09-ba3c-52dfec5e3379`) with corpus doc `rag_basics.txt` ingested (8 chunks, `ready`) under its tenant, re-ingested seed chunks under `eval_user`. Do not treat this as production data; the Wave 0 backup drill exists exactly because of this incident.
- Frontend: no registration/login/upload UI (by design at this stage); work manually via Swagger (`http://localhost:8000/docs`) and paste the access token + user_id into UI settings. Access tokens live 15 minutes and die on page refresh (React state only).
- Tests: 413 tests, exactly **1 pre-existing failure** (`test_completed_turns_load_in_chronological_order`, tied-timestamp issue in the in-memory fake), 0 errors.
- Known bug in scope (Wave 1.2): persistent conversations with `conversation_id` fail against live Mongo (`AsyncMongoClient` across event loops). The current frontend does not send `conversation_id`, so normal chat is unaffected.
- `certs/` contains only `.gitkeep` (no TLS certs; Wave 3 automates this).

**What you must do first:** Wave 0 (`PRODUCTION.md` §3) — fix the failing test, repo hygiene (`cookies.txt` at repo root is a smoke-test artifact with a live refresh cookie: delete it), CI skeleton, and the config-contract docs. Then Wave 1 items 1.1 (frontend auth UI) and 1.2 (async chat routes / `conversation_id` fix). Waves, sequencing rule, and definitions of done are in §8. The 10 invariants in §2 are non-negotiable — codify them in review checklists, never in skeleton "TODO" comments.

**Housekeeping rules:** never commit `.env`, `certs/*`, `chroma_data/`, `uploads/`, `cookies.txt`, or smoke-test artifacts; never print secrets or live tokens into chat or files; keep `AGENTS.md` updated when a contract or new flag lands (it is the cross-session source of truth); run the backend suite with `LANGSMITH_TRACING=false; LANGCHAIN_TRACING_V2=false` and no live infra. Commit after review, in logical units per wave item.

---

## 1. Honest current-state assessment

What exists already (do not rebuild these — they are production-shaped):

| Capability | Where |
|---|---|
| Multi-tenant vector isolation (`where={"user_id": ...}`, chunk ids `{user}:{doc}:{idx}`) | `phase1.py:RAGSystem` |
| Hybrid retrieval (BM25 + RRF), cross-encoder reranking, per-request overrides | `app/services/retrieval.py`, `app/services/reranker.py` |
| Strict refusal contract (`I do not have enough information to answer that.`, exact string) and distance-threshold discipline | `phase1.py`, `app/schemas/chat.py` |
| JWT access/refresh pairs, Redis single-use refresh rotation (`GETDEL`), token blacklist (fail-closed), rate limiting | `app/services/refresh_token_service.py`, `token_blacklist_service.py`, `rate_limit_service.py` |
| Response caching with per-user generation invalidation (fail-open) | `app/services/cache_service.py` |
| Retry + circuit breaker (transient-only, generator-safe) | `app/utils/retry.py` |
| Degraded-mode excerpts (never cached, never mid-stream swap) | `phase1.py`, `degradation_service.py` |
| Token/cost ledger + `GET /api/v1/usage` | `app/services/usage_service.py`, `app/routes/usage.py` |
| Upload → background ingest → poll → crash requeue | `app/routes/documents.py`, `app/services/document_jobs.py` |
| Long-conversation summarization with stored summary | `phase1.summarize_conversation_history`, `conversation_service.py` |
| Agent loop (LangGraph + manual fallback), stdlib tool registry | `app/agents/`, `app/tools/` |
| Health-gated 5-service Docker stack, prod TLS override | `docker-compose*.yml`, `frontend/nginx*.conf` |
| SSE contract validated strictly client-side; connection status UI | `frontend/src/api/chatApi.js` |
| Eval harness (retrieval recall / answer / refusal accuracy) | `eval_cases.json`, `langsmith_eval.py`, `phase1` demo |

What is **not** production-grade yet: frontend auth, a known correctness bug in persistent conversations, single-process assumptions, ops observability, datastore durability, CI/CD, and RAG quality operations.

---

## 2. Non-negotiable invariants (carry forward through every change)

1. Retrieval always filtered by `where={"user_id": ...}`; chunk ids stay `{user_id}:{doc_id}:{index}`.
2. Unanswerable ⇒ exactly `I do not have enough information to answer that.` History is reference-only, never a factual source.
3. SSE order `start → retrieval → token* → sources → done` (or `error`); the frontend validates it strictly; uncached callers keep byte-identical outputs.
4. Limits enforced at both schema and `RAGSystem.retrieve` (k 1–10, raw_query 1–4000, history ≤20/8, content ≤8000, threshold 0–2 cosine).
5. Redis command retries stay disabled on `GETDEL`; refresh entry and token must agree on `user_id`; blacklist maps Redis outage to 503 (fail-closed).
6. `AuthRoute` keeps stripping bodies from 422s and the no-store/no-cache headers.
7. `embed_documents` keeps one `types.Content` per chunk; `embeedings.txt` keeps normalizing to doc id `embeddings`.
8. Gemini client construction stays lazy (import-time safety).
9. Flag-gated features default off (`HYBRID_ENABLED`, `RERANKER_ENABLED`, `CACHE_ENABLED`, `DEGRADED_MODE_ENABLED`, `AGENT_ENABLED`) — new flags follow the same pattern.
10. Never commit `.env`, `certs/*`, `chroma_data/`, `uploads/`, or smoke-test artifacts (`cookies.txt` etc.).

---

## 3. Wave 0 — Hygiene (days)

Do these first; they unblock everything else.

### 0.1 Fix the known unit-test failure
- Symptom: `test_completed_turns_load_in_chronological_order` — the in-memory Mongo fake reverses tied `created_at` timestamps.
- Fix: give the assistant message a strictly later timestamp in the test data, or add a deterministic secondary sort key in the fake. Do **not** touch tenant filtering or real-collection sorting.
- Acceptance: `unittest discover -s tests` → 413 passed, 0 failures.

### 0.2 Repo hygiene
- Delete/ignore `cookies.txt` (smoke-test artifact with a live refresh cookie).
- `.gitignore` review pass: confirm `certs/*` (keep `.gitkeep`), `.env`, chroma data, uploads, temp files.
- Stage and commit the current Phase 10 work after a `git diff --check` review (the change list is in `HANDOFF_PHASE10.md` §7).

### 0.3 CI skeleton (GitHub Actions or equivalent)
- Job 1 (every push): backend `unittest` with no live infra.
- Job 2 (every push): frontend `node --test`, Playwright e2e with headless Edge.
- Job 3 (every push): `docker compose build` for backend + frontend images (no push), `docker compose config --quiet` for dev and prod variants; optionally `hadolint` + `trivy image --severity HIGH,CRITICAL`.
- Job 4 (nightly, secret-required): corpus eval — boot ephemeral compose with a test-only Gemini key, run `langsmith_eval.py`, fail on recall/answer/regression thresholds.
- Acceptance: green pipeline on the reference commit.

### 0.4 Config contract tidy-up
- Make every runtime knob discoverable in one place: extend the env docs so each flag (`CACHE_TTL_SECONDS`, `RETRY_*`, `CIRCUIT_*`, `SUMMARY_*`, `AGENT_*`, `SEARCH_*`) has its default, valid range, and failure semantics documented next to the code that reads it.

---

## 4. Wave 1 — Real users (1–2 weeks)

Goal: a user can sign up, upload, chat, and stay logged in from the browser with zero manual token handling.

### 1.1 Frontend auth UI (highest user-facing value)
- Build `Register` / `Login` screens calling `POST /api/v1/auth/register` / `login`; store the **access token in memory only** (behavior stays contract-correct; never localStorage).
- Login state machine: on 401 mid-session, attempt **at most one** refresh **before** an SSE stream starts, per the flow specced in `PHASE_3_4_AUTH.md` (serialize concurrent refreshes: a "loser" gets 401 + cleared cookie and must re-login; never auto-refresh on 403; never retry side-effecting ops; never replay a partial stream).
- Document upload screen: `POST /api/v1/documents/upload` + polling `GET /{id}` until `ready`/`failed`, with visible per-document status.
- Persistence reality check: on refresh the user must re-login (in-memory token per the auth contract) and re-attach to their conversation list via `GET /api/v1/conversations` (works after 1.2).
- Acceptance: Playwright e2e — register → login → upload → grounded answer → sources visible; a page refresh forces re-login instead of 401 noise.

### 1.2 Fix persistent conversations (the known correctness bug)
- Bug: chat routes are sync and bridge to async services with `asyncio.run` in worker threads; the lifespan-scoped `AsyncMongoClient` is bound to a different loop ⇒ `RuntimeError: Cannot use AsyncMongoClient in a different event loop` when `conversation_id` is sent.
- Approach options, in preference order:
  1. Make `POST /api/v1/chat`, `/chat/stream`, `/chat/sse` **fully async routes**; make `prepare_sse_chat`'s pre-stream resolution async too, preserving the pre-stream 404/403 contract.
  2. Where sync code must call into async (`phase1` LLM/stream calls), bridge with `asyncio.run_coroutine_threadsafe` back to the lifespan loop — never bare `asyncio.run` in a worker thread.
- A side effect of going async: `chat_orchestration` usage-ledger writes (currently fail-open) move onto the request loop; keep them fail-open with a bounded timeout so a Mongo spike can never stall the stream.
- Acceptance: with `conversation_id`, a 6-turn conversation loads prior messages server-side (summarization path exercised), and the 404-on-unknown-conversation pre-stream behavior is unchanged. Add the regression test.

### 1.3 Minimal per-user abusive-use guard
- Daily chat-turn quota via Redis (`rate_limit:` namespace, `INCR` + `EXPIRE` at UTC midnight), 429 with `Retry-After`; exempt the admin/eval tenant.
- Acceptance: a scripted burst gets throttled without affecting other users.

---

## 5. Wave 2 — Correctness at scale (2–4 weeks)

Goal: 2+ backend replicas behind a load balancer behave identically; ops can see what the system is doing.

### 2.1 Multi-replica readiness
- **Document locks**: replace the in-process `DocumentLocks` mutex with Redis locks (`SET lock:doc:{id} NX PX 60000` + token-verified release). Delete-vs-ingest races become cross-process-safe.
- **BM25 caches** (`HybridRetriever` per-user in-memory index): rebuildable from Chroma by design, so replicas converge after an **invalidation signal**. Add a Redis pub/sub channel `rag:cache:invalidate` published on ingest/delete; subscribers drop the affected user's hybrid/rerank/cache entries and rebuild lazily. No replica ever serves a stale index for longer than a message hop.
- **Reranker model**: lazy-load per process by intent (each replica loads its own cross-encoder); document the memory cost per replica (~2 GB with torch) in sizing notes.
- Acceptance: run 2 replicas against shared stores; ingest on A, immediate query on B reflects the document.

### 2.2 Real ingestion worker
- Extract ingestion from `BackgroundTasks` (dies with the process) into a **separate worker process** consuming a Redis Streams queue (`documents:jobs`), with consumer groups so a crashed job is redelivered from the pending-entries list — this makes `requeue_stale_uploads` obsolete (or reduces it to a startup no-op check).
- Job payload: `{user_id, document_id, file_path}`; the worker performs the exact current pipeline (validate → save → chunk → embed → upsert → status flip) and publishes `rag:cache:invalidate`.
- Acceptance: kill the worker mid-ingest; on restart the document completes exactly once (idempotent via delete-first re-ingest, which already exists).
- Compose change: new `worker` service built from the same backend image, different entrypoint; depends_on redis only.

### 2.3 Observability
- `/metrics` (Prometheus format, off by default via `METRICS_ENABLED`):
  - counters: requests by route+status, cache hit/miss, degraded answers, refusal answers, agent tool routes;
  - gauges: **breaker state** per circuit (`gemini`, `mongodb`), Chroma doc count, active ingestion jobs;
  - histograms: `retrieve` latency split by stage (vector / hybrid / rerank), first-token latency, full-turn latency, Gemini tokens per turn.
- Structured JSON logs (`python-json-logger`) with request-id middleware; SSE stage transitions at DEBUG, failures at ERROR with `stage` attached.
- Alert thresholds (first pass): breaker half-open for >5 min; degraded counter rising steeply; ingestion success rate <95%; p95 chat latency over budget.
- Acceptance: a dashboard answers "how healthy is chat right now" without reading logs.

### 2.4 Mongo event-loop fallback if 1.2 stalls
- If full-async migration must be split: a dedicated **request-loop Mongo client** (client created per event loop, cached by `id(asyncio.get_running_loop())`) so no client is ever shared across loops. Fallback, not goal — it multiplies connection pools.

---

## 6. Wave 3 — Durability, security, delivery (3–5 weeks)

Goal: the system survives a machine loss and a bad deploy.

### 6.1 Datastores
- **Mongo**: 3-node replica set in compose-prod (`keyFile` auth), `w=majority` for the collections backing auth; scheduled `mongodump` to a backup volume; documented restore drill (`down -v` → restore → verify users/chunks).
- **Redis**: already append-only in dev; in prod keep `appendonly yes` + a replica. Document that its loss costs only refresh sessions and cache warmth (all fail-open by design) — a deliberate, cheap failure mode.
- **Chroma**: periodic snapshot of the data volume **plus** an export/rebuild script (`collection.get` → JSONL → re-embed). Point of truth = Mongo `documents` records + source files in `uploads-data`; Chroma is reconstructable from them.
- **Uploads**: named volume + off-host copy schedule; `uploads-data` is the only non-reconstructable input.
- Acceptance: a scripted "delete all four volumes" drill restores service from backups within the stated RTO with zero document/user loss.

### 6.2 Transport & deployment security
- TLS by automation, not hand-certs: certbot sidecar (webroot challenge) or swap nginx for Traefik/Caddy with ACME in compose-prod.
- Pin image digests (`mongo@sha256:...`), read-only root filesystem + dropped capabilities for backend/frontend/chroma, non-root `USER` in the backend image (currently root for the pip cache mount convenience).
- Secrets: no changes to the token design; add a secret-rotation runbook (JWT key rotation invalidates all sessions by design — document it).
- Auth extras (priority order): refresh-token **reuse detection** (a rotated-then-reused refresh JWT revokes the user's whole session family), password-reset flow (single-use JWT + email hook), audit log collection for auth events.

### 6.3 Environments & pipeline
- `.env.dev` / `.env.staging` / `.env.prod` (all gitignored; `.env.example` stays additions-only).
- CI builds multi-arch images, pushes to a registry on tag; deploy job applies compose-prod to staging; manual gate to prod.
- Acceptance: "green repo → running staging" with no human one-offs.

---

## 7. Wave 4 — RAG quality operations (ongoing)

Goal: retrieval quality stops depending on hand calibration and starts producing evidence.

### 7.1 Corpus capabilities
- Extend ingestion beyond `.txt` (PDF via `pypdf`, HTML stripping, DOCX via converter), storing normalized extracted text next to the source in uploads. Validate any default change with the existing `run_chunking_experiments` matrix before promoting (512/64 stays the baseline).

### 7.2 Index lifecycle
- Stamp chunk metadata with `embedding_model` + `embedding_version`. On bump: a background reindex job re-embeds tenant-by-tenant (throttled) with per-user cache invalidation; old-version chunks stay queryable until re-embed completes (dual-read migration) to avoid an outage window.
- Deduplication: content-hash metadata on chunks; ingest skips byte-identical re-uploads (content-derived `document_id` replaces today's delete-recreate churn).

### 7.3 Evaluation as a gate, not a demo
- CI job (from 0.3) runs `eval_cases.json` against golden thresholds (retrieval recall / answer keyword / refusal). A change to chunk size, `distance_threshold`, hybrid weights, or reranker top-k must pass **or** move the golden numbers with justification in the PR.
- Ship a calibration script artifact (threshold pull curve, per `calibrate_distance_threshold`) as a CI output so threshold changes stay data-driven.

### 7.4 User feedback loop
- Extend `messages` with a `rating` field; add 👍/👎 buttons in `ChatWindow.jsx` wired to an authenticated `POST /api/v1/messages/{id}/rating`.
- Weekly report: refusal rate, low-confidence agent iterations (`result["agent"]` already carries `attempts`/`confidence`), thumbs-down clusters → new eval cases. This is how `eval_cases.json` compounds.

### 7.5 Advanced retrieval (explicitly out of scope until evidence demands it)
- Semantic chunking, HyDE, query decomposition, multi-tenant ACL documents, or a second embedding model are pre-declared non-goals; each needs an eval-driven case from 7.4 before work starts.

---

## 8. Milestone summary

| Milestone | Ships | Definition of done |
|---|---|---|
| **M0 hygiene** | Wave 0 | 413/0 green suite, CI skeleton, secrets/certs hygiene |
| **M1 user-ready** | Wave 1 | Register→login→upload→chat end-to-end in browser with auto-refresh; persistent conversations fixed; chat quota live |
| **M2 scale-ready** | Wave 2 | 2 replicas behave identically; worker-based ingestion; dashboards + alerts |
| **M3 ops-ready** | Wave 3 | Backup/restore drill passes; TLS automated; pinned builds; staging deploys |
| **M4 quality-ready** | Wave 4 | Eval gates in CI; PDF/HTML corpora; reindex path; feedback loop closed |

Sequencing rule: **no wave starts before the previous one's definition of done is met.** Waves 1 and 2 are independent enough to parallelize; waves 0/3/4 are not.

## 9. Explicit non-goals
- Rewriting the auth token design (rotation/blacklist semantics stay).
- Any "global" retrieval mode (removing the tenant filter) — even for admin tooling.
- WebSockets (SSE covers it; the frontend already contract-validates it).
- Model-swapping abstraction layers — models change via config only (`EMBEDDING_MODEL`, generation model), exactly where they live today.
