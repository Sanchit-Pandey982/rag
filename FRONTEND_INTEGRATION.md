# Frontend completion and remaining backend integration

This guide follows **architecture → request/data flow → engineering reasoning → implementation/code**. It documents the actual checkout, not an assumed phase plan. The earlier phase/feature list was not included in the request, so the frontend gaps in REACT_UI_ARCHITECTURE.md were used as the implementation scope.

The streaming flow now includes backend stage-aware error handling and a Gemini client timeout. See [STREAMING_ARCHITECTURE.md](STREAMING_ARCHITECTURE.md) for the current error, completion, cancellation, and test contracts.

## 1. Architecture

The application now has a runnable React/Vite shell around your existing interfaces.

```text
frontend/src/main.jsx
  App.jsx                         User context, retrieval settings, session navigation
    ConnectionStatus.jsx          GET /health and /ready
    ChatWindow.jsx                Per-conversation messages, composer, cancellation
      Message                     Markdown, errors, citations, retrieval details
      history.js                  Maps completed turns into existing ChatMessage objects
      api/chatApi.js              Existing streamChat interface and SSE decoder
        POST /api/v1/chat/sse      Existing FastAPI route, unchanged
```

App owns development user selection and retrieval settings. Each conversation gets its own mounted ChatWindow and independent local messages. Only the selected conversation is visible. Leaving a conversation or visiting the roadmap aborts its browser request; returning keeps the partial answer with a stopped label. Changing user context replaces all conversations. Refreshing clears the tab's history.

The sidebar's conversation IDs are browser UI keys. They are not backend session IDs, authentication tokens, or additions to ChatRequest.

### Implemented UI

| Capability | Behavior |
| --- | --- |
| Runnable app | React entrypoint, Vite configuration, locked dependencies, development/build/preview scripts |
| Responsive workspace | Conversation sidebar, main chat, retrieval settings drawer on smaller screens |
| Streaming | Visible search/generation/completion status, incremental Markdown answer, Stop |
| Follow-ups | Completed local turns enter the next request's chat_history |
| Sources | Expandable actual source metadata, original distance value, retrieval query and document IDs |
| Error recovery | HTTP validation details, invalid-stream errors, interrupted answer retention, explicit retry |
| Controls | Existing user_id, k, rewrite_query, distance_threshold fields |
| Session handling | Independent conversations in memory; changing user resets local chats |
| Diagnostics | Actual health/readiness checks; browser elapsed time per completed response |
| Integration roadmap | Explains backend-pending capabilities without fabricated uploads, metrics, or successful jobs |

### Preserved interfaces

ChatWindow retains user_id, chat_history, k, rewrite_query, and distance_threshold. It has three **optional UI-only props**: active (defaults to true), onTitle, and onBusyChange. They support navigation, conversation labels, and disabling settings during requests. They are never sent to the backend.

streamChat retains its original named arguments, including signal and onEvent. Its callback now receives decoded JSON objects. This intentional browser-side correction is necessary because FastAPI already JSON-encodes SSE data. Existing handler names handleSubmit, handleStreamEvent, appendToken, attachSources, and stopStreaming are retained.

No endpoint was renamed or added. No proposed future endpoint names are presented as existing contracts.

## 2. Request and data flow

### One chat request

1. The user types or selects a suggested question. Suggestions fill the composer; Send or Enter submits. Shift+Enter inserts a newline.
2. handleSubmit trims the question, validates its length and user context, and acquires a synchronous AbortController lock against duplicate submissions.
3. buildChatHistory combines supplied history with completed local question/answer pairs. Failed, cancelled, and empty answers are excluded. UI-only IDs, sources, statuses, and timers are removed.
4. The user question and empty assistant message appear immediately. A captured assistant message ID routes every subsequent event to its own placeholder.
5. streamChat POSTs the existing six fields. During development Vite forwards this request to FastAPI.
6. The unchanged route validates ChatRequest, calls RAGService.run_once_event_stream, and forwards RAGSystem events.
7. The browser decodes UTF-8 bytes, frames SSE messages, parses each JSON data payload, and dispatches it to handleStreamEvent.
8. React appends data.text, records retrieval data, renders data.sources, and marks completion only when done arrives.
9. The next question includes the completed visible turns in chat_history.

### Exact request contract

```json
{
  "raw_query": "How are embeddings used in a vector database?",
  "user_id": "eval_user",
  "chat_history": [
    { "role": "user", "content": "What is RAG?" },
    { "role": "assistant", "content": "The previous completed answer." }
  ],
  "k": 3,
  "rewrite_query": false,
  "distance_threshold": null
}
```

| Field | Existing backend constraint | UI treatment |
| --- | --- | --- |
| raw_query | 1–4,000 characters | Nonempty trimmed query; bounded composer |
| user_id | 1–256 characters | Required development user selector, defaults to eval_user |
| chat_history | At most 20 messages | Most recent 20 valid history messages |
| content inside history | 1–8,000 characters | Empty entries excluded; overlong history content truncated to 8,000 Unicode code points |
| k | Integer 1–10 | Integer slider, default 3 |
| rewrite_query | Boolean | Switch, default false |
| distance_threshold | null or number 0–2 | Optional filter, off by default |

The textarea's browser maxLength counts UTF-16 code units, making its 4,000 limit conservative for emoji. The original answer stays intact on screen even when its history representation is truncated.

### Existing SSE contract

| Event | Data | UI effect |
| --- | --- | --- |
| start | raw_query | Shows search-in-progress state |
| retrieval | retrieval_query, retrieved_document_ids | Records actual retrieval query and chunk result document IDs |
| token | text | Appends text to the assistant answer |
| sources | sources array | Renders expandable source cards |
| done | {} | Marks answer complete and ends reader consumption |
| error | stage, message | Marks application failure, keeps partial answer text, and ends reader consumption |

Each sources entry currently contains chunk_id, document_id, source, title, chunk_index, and distance. The UI displays these fields exactly; it does not infer document content, file download URLs, or confidence percentages. Duplicate document IDs can correspond to separate retrieved chunks.

The decoder tolerates fragmented UTF-8, CRLF boundaries split between reads, multiline data, heartbeat comments, and an unterminated final event. Unknown event types are ignored. Malformed known events and EOF without either done or error are surfaced as errors.

The backend emits an additive error event on application failure. It logs the stage and traceback on the backend and sends only the stage and a generic message. React handles this event separately from tokens; only done marks success.

## 3. Engineering reasoning

### Conversation state and multi-user isolation

React state keeps the first implementation understandable and avoids quietly persisting private chat content on a shared browser. Switching conversations preserves local state, while switching users discards it. A page refresh intentionally clears everything.

This is presentation isolation, not a security boundary. The browser can submit any user_id. Real tenant isolation must bind that existing field to a verified identity on the server and check ownership during retrieval and document access. Changing a dropdown cannot establish authorization.

### History correctness

Only successful pairs are included in local follow-up context. An interrupted generation should not silently become an authoritative assistant answer. Retry resends the original question as a new visible attempt and omits the failed pair from history. Completed earlier turns remain available.

The history cap protects the existing schema limit. Character truncation is a compatibility measure, not a substitute for a token-budgeted memory strategy. Production summarization, context selection, and storage belong in the backend phase.

### Streaming and concurrency

One browser request runs per active conversation. The synchronous ref closes the window before React state updates, preventing rapid double submissions. Stream callbacks update by assistant ID. Switching away aborts the browser request; unmount cleanup does the same.

AbortController cancels browser transport. It does not prove Gemini stopped generating, nor that server work stopped billing. Backend disconnect detection and cancellation propagation still need your implementation.

Receiving done is different from the connection closing. The UI treats unexpected EOF as incomplete and retains partial text. It does not auto-retry a generation that might already have consumed resources. The user initiates retry explicitly.

### Observability and readiness

ConnectionStatus requests the existing health and ready endpoints every 30 seconds and when refreshed manually, with a five-second timeout. It distinguishes reachable-but-unready from unavailable.

These endpoints describe the service globally. A ready collection does not prove the selected user has documents. The default eval_user matches app/main.py's current startup ingestion owner. At implementation time all three data/*.txt files were empty.

Browser elapsed time measures submission through receipt of done. It includes network, retrieval, and generation. It is not server model latency, token usage, or a billed cost estimate.

### Rendering model output

ReactMarkdown renders Markdown without allowing raw HTML. Links use the renderer's safe URL handling and open with noreferrer/noopener. Remote Markdown images are rendered as text placeholders to avoid automatic third-party requests. Source paths are displayed as metadata rather than fabricated download links.

The layout has labeled controls, keyboard submission, visible focus styles, reduced-motion support, and status announcements. Detailed token changes are not announced on every fragment.

## 4. Implementation and integration

### Run the frontend

From the repository root:

```powershell
cd frontend
npm ci
npm run dev
```

Open http://127.0.0.1:5173. This command runs only the frontend. It does not start Python or ingest documents. Without FastAPI, the workspace still renders and reports API unavailable.

The default development proxy target is http://127.0.0.1:8000. To use a different port, create frontend/.env.local using frontend/.env.example:

```dotenv
API_PROXY_TARGET=http://127.0.0.1:8011
```

Restart Vite after editing the target. The proxy forwards /api, /health, and /ready; it does not change the browser's existing relative endpoint URLs. Never put GEMINI_API_KEY in a VITE_ variable or any frontend source file.

Node 22.12+ is required by this package. The project uses the standard [Vite development/build workflow](https://vite.dev/guide/). Stream teardown follows React's [effect cleanup lifecycle](https://react.dev/reference/react/useEffect).

### Connect your current backend yourself

1. Provide the intended knowledge text in the existing data files.
2. Configure the backend's existing GEMINI_API_KEY in the backend environment.
3. Start your existing application with your Python environment, for example:

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

4. Confirm /health and /ready, then submit with eval_user.
5. Inspect the browser Network panel: POST /api/v1/chat/sse should return Content-Type: text/event-stream and the existing named events.
6. Ask a follow-up and inspect chat_history in the second request.
7. Try another user with no ingested records to verify expected retrieval behavior.

The startup code ingests data only when the entire collection is empty. Adding files after a nonempty collection exists will not automatically reindex them. Use your own ingestion workflow for that case; do not delete the existing vector store as a routine integration step.

### Remaining backend phases

These are integration work areas inferred from the current repository and your learning goals, not replacements for an unseen numbered phase plan.

| Work area | Backend work for you | Frontend connection point | Evidence before enabling product controls |
| --- | --- | --- | --- |
| Authentication and tenancy | Verify identity; enforce ownership for retrieval, ingestion, and document reads; bind or validate user_id | Replace development identity selection in App with authenticated context while keeping the current request contract compatible | User A cannot request user B's corpus or stored conversations |
| Document ingestion | Define authenticated upload, indexing status, document listing, and deletion contracts; process ingestion jobs | Add knowledge UI after contracts are agreed; source cards already use document_id and chunk_id | Upload progresses to searchable content; failures are actionable; deletes remove access |
| Conversation persistence | Design authorized conversation/message storage and its retrieval contract; make turn completion and retry semantics explicit | Lift/load ChatWindow messages through the future storage adapter; preserve message rendering and stream handlers | Reload restores the right user's history; failed attempts remain distinguishable |
| Streaming resilience | Stage-aware error events and Gemini HTTP timeout are implemented; immediate cancellation of upstream work, total pipeline deadlines, and resource limits remain separate work | Existing error, stopped, retry, and incomplete-stream states | Disconnect releases resources; concurrent requests remain isolated; errors are observable |
| Evaluation | Extend existing eval_cases.json and langsmith_eval.py workflows with suitable fixtures and metrics | Add results UI only after a real result query contract exists | Groundedness/retrieval regressions are measurable using deterministic corpus ownership |
| Observability | Correlate API request, rewrite, retrieval, generation, and stream lifecycle; record durations/errors with appropriate redaction | Connect future trace metadata to the existing retrieval details area | A failed user turn can be traced across layers; sensitive text isn't exposed indiscriminately |
| Deployment | Serve frontend assets and proxy existing API routes; preserve streaming; enforce credentials, limits, timeouts, and readiness | Same relative browser URLs can remain | Real incremental delivery, cancellation, readiness routing, and cross-user tests pass under load |

For each new capability, settle its backend request/response contract before adding frontend calls. The current chat contract has no conversation ID, file upload field, trace ID, evaluation-run ID, or authorization mechanism; none has been invented in chat requests.

### Concrete extension example: consume existing sources

This mapping is already implemented in ChatWindow:

```jsx
case "token":
  appendToken(assistantMessageId, data.text);
  break;
case "sources":
  attachSources(assistantMessageId, data.sources);
  break;
```

Each callback updates the assistant message captured for this request. If later source contracts gain excerpts or authorized links, add those fields to the renderer only after the backend emits them. The current sources event does not include chunk text.

### Deployment handoff

```powershell
cd frontend
npm run build
npm run preview
```

The build emits frontend/dist. Preview serves the static build for inspection; the dev-only proxy is not included. A real deployment must route /api, /health, and /ready to FastAPI through your infrastructure, or separately define an approved cross-origin deployment.

Configure the production proxy to avoid buffering event streams and choose an idle timeout appropriate to retrieval/generation. Verify this with actual streamed responses in the deployed environment. Keep backend secrets server-side. No production infrastructure was modified.

### Verification

```powershell
cd frontend
npm test
npm run test:e2e
npm run build
```

Unit tests exercise fragmented SSE, UTF-8, malformed data, premature EOF, response validation, abort, and history bounds. Browser tests mock existing API routes to cover rendered streaming, source details, follow-up payloads, settings, user isolation, retry, cancellation, unsafe Markdown, and responsive navigation.

Browser tests use installed Microsoft Edge via Playwright's msedge channel. On a host without Edge, install that browser or change the test channel to an installed Playwright Chromium browser. Tests do not call Gemini or start the Python backend.

Mocked browser tests validate the frontend contract and UI behavior. A successful real-model response, actual tenant authorization, backend cancellation, and production load behavior still require your backend integration checks.

Implementation verification: the production build, 11 unit tests, and 9 browser tests passed. Desktop and mobile screenshots were inspected. File hashes for the existing Python application, schemas, RAG pipeline, evaluation script, requirements, and root .gitignore matched their pre-implementation values.
