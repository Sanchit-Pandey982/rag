# Runtime configuration contract

Single reference for every runtime knob. Values are read with
`os.getenv` helpers next to the feature that uses them — there is no
central settings object by design. Booleans accept `1/true/yes/on`
(case-insensitive); anything else is false. Numeric values that are
unparseable or outside the documented range fall back to the default
(they never crash a request), except where noted.

`.env` is never committed. Copy additions from `.env.example` into an
existing local `.env`; Compose reads it through `env_file`.

## Core / infrastructure

| Variable | Default | Valid values | Failure semantics |
|---|---|---|---|
| `GEMINI_API_KEY` / `GOOGLE_API_KEY` | unset | Google AI Studio key | Startup ingests `data/*.txt` and calls Gemini; without a key the backend cannot become `/ready` and real chat turns fail with the normal error/degraded path. `GOOGLE_API_KEY` wins if both are set. |
| `JWT_SECRET_KEY` | unset | ≥32 UTF-8 bytes | Lifespan validation fails fast; the app does not start. |
| `JWT_ISSUER` / `JWT_AUDIENCE` | unset | strings | Lifespan validation fails fast when missing. |
| `MONGODB_URI` | driver default | Mongo URI | Users/conversations/documents/usage are unavailable (5xx or 503); Compose sets `mongodb://mongo:27017`. |
| `MONGODB_DATABASE` | `agent` | database name | Uses that database; Compose pins `agent`. |
| `REDIS_URL` | client default | Redis URL | Auth refresh/blacklist fail closed (401/503); caching, rate limiting, quota and degradation counters fail open (logged). Needs Redis 7.2+ for `GETDEL`. |
| `CHROMA_URL` | unset | HTTP(S) Chroma base URL | Set → `chromadb.HttpClient` (Compose uses `http://chroma:8000`); unset → embedded `PersistentClient` on `chroma_data/`. An unreachable HTTP server fails readiness/chat. |
| `UPLOAD_DIR` | `uploads` | writable path | Relative paths resolve under the project root. Unwritable storage makes uploads fail (500) after validation. |
| `MAX_UPLOAD_BYTES` | `5242880` (5 MiB) | positive int | Oversized files get 413; invalid values use the default. |
| `REFRESH_COOKIE_SECURE` | `true` | `true` / `false` | `true` requires HTTPS (production). Local HTTP must set `false` explicitly or the browser will not send the refresh cookie. |

## Retrieval / generation flags

Flag-gated features default **off** and keep output byte-identical to
the legacy path when off. Per-request overrides (`use_hybrid`,
`use_rerank`, `use_cache`, `use_degraded`, `use_agent`) accept
`True`/`False`/`None` (None follows the env flag).

| Variable | Default | Valid range | Failure semantics |
|---|---|---|---|
| `HYBRID_ENABLED` | `false` | boolean | Off → vector-only retrieval. On → BM25+vector RRF; any BM25 failure falls back to vector-only. |
| `HYBRID_RRF_K` | `60` | >0 int | Invalid → 60. Larger values flatten rank fusion. |
| `HYBRID_OVERFETCH_MIN` | `20` | >0 int (effective window capped at 10 by Chroma) | Invalid → 20. |
| `RERANKER_ENABLED` | `false` | boolean | Off → retrieval order unchanged. On → cross-encoder rescoring; any load/score failure keeps retrieval order. |
| `RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | sentence-transformers model id | Invalid model is logged and retrieval order is kept. Model loads lazily on first reranked query. |
| `RERANKER_TOP_K` | `5` | >0 int, effective `min(k, value)` | Invalid → 5. |
| `CACHE_ENABLED` | `false` | boolean | Off → no cache keys, `done` stays `{}`. On → Redis outage is a transparent miss/skip. |
| `CACHE_TTL_SECONDS` | `3600` | >0 int | Invalid → 3600. Entries orphaned by ingest/delete expire after this TTL. |
| `DEGRADED_MODE_ENABLED` | `false` | boolean | Off → LLM outage is an error. On → pre-stream outage serves up to `DEGRADED_MAX_CHUNKS` excerpts under the fixed `I'm experiencing high load...` prefix; mid-stream failures still emit `error`. Degraded answers are never cached. |
| `DEGRADED_MAX_CHUNKS` | `3` | >0 int | Invalid → 3. |
| `AGENT_ENABLED` | `false` | boolean | Off → legacy pipeline. On → retrieve/generate/confidence loop; any agent exception falls back to legacy `run_once`. Streams always use the legacy path. |
| `AGENT_MAX_ITERATIONS` | `3` | 1–10 | Out-of-range/invalid clamps to the nearest bound. |
| `AGENT_CONFIDENCE_THRESHOLD` | `0.5` | 0.0–1.0 | Out-of-range/invalid clamps to the nearest bound. Below it the loop retries (bounded), then returns the best attempt with `gave_up=true`. |

## Summarization / persistence

| Variable | Default | Valid range | Failure semantics |
|---|---|---|---|
| `SUMMARY_THRESHOLD` | `20` | >0 int (completed messages) | Invalid → 20. Below it, plain recent history is sent. |
| `SUMMARY_MAX_AGE` | `10` | >0 int (new messages) | Invalid → 10. A stale summary is refreshed only after this many new messages; refresh failure keeps the previous summary (or plain history). |
| `SUMMARY_KEEP_RECENT` | `8` (code constant `SUMMARY_KEEP_RECENT`) | code constant | Always appended verbatim after the summary message. |

## Reliability

| Variable | Default | Valid range | Failure semantics |
|---|---|---|---|
| `RETRY_MAX_ATTEMPTS` | `3` | ≥1 int | Invalid → 3. Applies only to transient failures (timeouts, connection errors, 429/5xx). |
| `RETRY_BASE_DELAY` | `0.5` | ≥0 seconds | Invalid → 0.5. |
| `RETRY_MAX_DELAY` | `8.0` | >0 seconds | Invalid → 8.0. Backoff is exponential + jitter, capped here. |
| `CIRCUIT_BREAKER_THRESHOLD` | `5` | ≥1 int | Invalid → 5 consecutive transient failures open the `gemini`/`mongodb` circuit. |
| `CIRCUIT_BREAKER_COOLDOWN_SECONDS` | `30` | >0 seconds | Invalid → 30. After cooldown one half-open probe is allowed. |

## Metering / tools

| Variable | Default | Valid range | Failure semantics |
|---|---|---|---|
| `COST_PER_1K_INPUT` | `0.0001` | ≥0 USD | Invalid → default. Ledger writes are fail-open (answer still ships). |
| `COST_PER_1K_OUTPUT` | `0.0004` | ≥0 USD | Invalid → default. |
| `SEARCH_BACKEND` | `none` | `tavily` / `serpapi` / `none` | Unknown value behaves as `none` (no network call). |
| `TAVILY_API_KEY` / `SERPAPI_API_KEY` | unset | provider keys | Missing key → tool reports "not configured" and the turn falls through to local retrieval. Keys are never logged. |
| `SEARCH_MAX_RESULTS` | `5` | 1–10 | Clamped to range; invalid → 5. |
| `SEARCH_TIMEOUT_SECONDS` | `10` | 1–60 | Clamped; invalid → 10. Timeout/failure → error dict, never raises into chat. |
| `CALCULATOR_MAX_EXPR_LEN` | `200` | 1–1000 | Clamped; invalid → 200. |

## Fixed policy values (not env-tunable)

These are security/contract constants; change them in code with a test,
not in `.env`:

- Chat schema/RAG limits: `k` 1–10, `raw_query` 1–4000 chars,
  `chat_history` ≤20 messages (last 8 used), message content ≤8000
  chars, `distance_threshold` 0–2 (cosine).
- Chunking: 512 characters with 64 overlap.
- Models: `gemini-embedding-2` (embeddings), `gemini-3.6-flash`
  (generation). Change via the module constants, never a browser env var.
- Access token 15 min; refresh token 7 days; refresh cookie
  `Path=/api/v1/auth`, `SameSite=Strict`, `HttpOnly`.
- Per-IP fixed-window rate limits (Redis `rate_limit:` namespace):
  register 5/300 s, login 10/60 s, refresh 30/60 s, chat 30/60 s,
  upload 10/60 s. Exceeded → 429 with `Retry-After`. Redis outage →
  fail open for chat/upload, fail closed where session trust is involved.
- Per-user daily chat quota (Wave 1.3): see `CHAT_DAILY_QUOTA` in the
  section below once enabled.

## Eval gate (nightly CI only)

| Variable | Default | Meaning |
|---|---|---|
| `EVAL_MIN_RETRIEVAL_RECALL` | `0.80` | Minimum mean recall over answerable cases. |
| `EVAL_MIN_ANSWER_ACCURACY` | `0.80` | Minimum share of answerable cases containing all required terms. |
| `EVAL_MIN_REFUSAL_ACCURACY` | `0.80` | Minimum share of unanswerable cases returning the exact refusal. |

Run locally with `python langsmith_eval.py --local` (needs a Gemini key
and a corpus; inside Compose the backend already has both).
