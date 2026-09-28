# AGENTS.md

## Layout
- Backend entry: `app.main:app` (FastAPI). Routes: `app/routes/auth.py` (`/api/v1/auth/*`), `app/routes/chat.py` (`/api/v1/chat`, `/chat/stream`, `/chat/sse`). RAG core: `phase1.py` (`RAGSystem`). Eval: `eval_cases.json`, `langsmith_eval.py`.
- Frontend: `frontend/` (Vite + React). SSE client: `frontend/src/api/chatApi.js`. Vite proxies `/api`, `/health`, `/ready` — keep same-origin.

## Backend startup (order matters)
- Lifespan in `app/main.py`: validate `JWTService.from_environment()` first → `ping` Redis → `ping` Mongo → init Chroma RAG (ingests `data/*.txt` if `learning_rag` empty) → sets `app.state.ready`.
- `/health` is liveness (always 200). `/ready` is 503 until Chroma has docs.
- Chroma path: `chroma_data/`, falls back to `runtime_chroma_data/` if primary isn't writable. Both are gitignored; never commit them.
- `phase1.py` lazily creates the Gemini client — importing it needs no credentials/network. Don't move client construction to import time.

## Auth contract (don't redesign)
- Access JWT 15 min (Bearer, in-memory only). Refresh JWT 7 days, HttpOnly cookie `refresh_token`, `Path=/api/v1/auth`, `SameSite=Strict`, `Secure=true` by default.
- Redis holds sessions as `auth:refresh:<jti> -> user_id` with `EXAT=<jwt exp>` + `NX`; rotation is single `GETDEL`; logout is `DELETE`. Command retries are deliberately disabled (repeating `GETDEL` after a lost reply is unsafe). Never trust a refresh JWT without its Redis entry; never put refresh tokens in localStorage.
- Chat authorization (`app/dependencies/chat.py`): `payload.user_id` must equal JWT `sub`, else 403. A 401 means re-login; a 503 means infra failure (retry, don't treat as bad credentials). Logout of an already-invalid cookie still returns 204 + clears cookie.
- `AuthRoute` strips request bodies from 422s and sets `Cache-Control: no-store` / `Pragma: no-cache` on all auth responses — preserve both.

## RAG invariants
- Retrieval is always filtered by `where={"user_id": ...}`; chunk ids are `{user_id}:{doc_id}:{index}`. Never drop the filter.
- Unanswerable → return exactly `I do not have enough information to answer that.` History is for reference resolution only, never a factual source.
- Limits enforced at both schema and `RAGSystem.retrieve`: `k` 1–10, `raw_query` 1–4000 chars, `chat_history` ≤20 msgs (last 8 used), message content ≤8000, `distance_threshold` 0–2 (cosine). Defaults: chunk 512/64, models `gemini-embedding-2` / `gemini-3.6-flash`.
- `load_txt_documents` normalizes the `embeedings` filename typo to doc id `embeddings` (matches `eval_cases.json`) — don't "fix" by renaming the effect.
- SSE event order: `start → retrieval → token* → sources → done`, or `error` (with `stage` + generic message). Frontend validates these strictly.

## Env (PowerShell repo — use `; if ($?) {}` not `&&`)
- `.env.example` (root) holds **additions only** (`REDIS_URL`, `REFRESH_COOKIE_SECURE`) — merge into existing `.env`, never replace it. Required existing keys: `JWT_SECRET_KEY` (≥32 UTF-8 bytes, HS256), `JWT_ISSUER`, `JWT_AUDIENCE`, MongoDB/model + Gemini keys. Never print or hardcode secrets.
- Local dev only: `$env:REFRESH_COOKIE_SECURE='false'` (HTTP). Leave `true` in production.
- Frontend `frontend/.env.example` has only `API_PROXY_TARGET` (default `http://127.0.0.1:8000`). Never put Gemini keys in browser env vars.
- Needs Redis server 7.2+ (uses `GETDEL`, `SET ... EXAT ... NX`); installing `redis==8.1.0` client does not start a server.

## Commands
```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt
$env:REDIS_URL='redis://localhost:6379/0'; $env:REFRESH_COOKIE_SECURE='false'
.venv\Scripts\python.exe -m uvicorn app.main:app --reload
# Backend tests need NO live Redis/Mongo/AI (InMemoryRedis double + fakes):
$env:LANGSMITH_TRACING='false'; $env:LANGCHAIN_TRACING_V2='false'
.venv\Scripts\python.exe -m unittest discover -s tests -v
```
- Frontend (`frontend/`): `npm run dev` | `npm test` (`node --test tests/*.test.js`) | `npm run test:e2e` (Playwright, Edge `msedge` channel, starts its own dev server on 5173; `reuseExistingServer` only off CI).

## Frontend gotchas
- Refresh at most once per 401 and only before an SSE stream starts; serialize concurrent refreshes (loser gets 401 + cleared cookie, so last-writer can clobber the winner). Never auto-refresh on 403, never replay a partial stream, never blindly retry side-effecting ops. No auto refresh/retry implementation exists yet — see `PHASE_3_4_AUTH.md` for the intended flow.
