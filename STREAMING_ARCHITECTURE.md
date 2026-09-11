# Streaming architecture

The existing endpoints, request fields, and event names are preserved.

```text
USER
  -> React
  -> fetch() POST JSON
       -> HTTP status failure -> UI error
       -> ReadableStream
       -> SSE parser
       -> semantic event handler
            start     -> searching state
            retrieval -> retrieval metadata
            token     -> assistant state -> ReactMarkdown -> UI
            sources   -> source citations
            done      -> SUCCESS
            error     -> APPLICATION FAILURE

FastAPI /api/v1/chat/sse
  -> RAGService (existing request adapter)
  -> RAGSystem.run_once_event_stream()
       -> condense_question() when rewrite_query is enabled
       -> retrieve()
       -> generate_answer_stream() -> Gemini streaming
       -> sources
       -> done

  Any application exception
       -> log stage and detailed traceback on the backend
       -> error event with a generic client message

Separately:
  AbortController / network loss
       -> transport connection terminates
       -> client handles cancellation, read failure, or missing done/error
```

## Application events

`phase1.py` owns the pipeline and its error boundary. Its stages are `start`,
`query_rewrite`, `retrieval`, `generation`, `sources`, and `done`. A failure
produces one terminal error event and no subsequent sources or done event:

```text
event: error
data: {"stage":"generation","message":"The response could not be completed."}

```

`logger.exception()` retains the detailed exception and traceback in backend
logs. Neither the event payload nor the React error display includes those
details. Existing `start`, `retrieval`, `token`, `sources`, and `done` names
and payloads remain unchanged.

## Client completion and transport

`frontend/src/api/chatApi.js` checks `response.ok` before opening an SSE reader.
HTTP failures reject the request; validation errors retain their field messages,
while server failures use a generic message.

The reader tracks `doneReceived` and `errorReceived` separately. It dispatches
each decoded event to `ChatWindow.handleStreamEvent()`, including `error`.
Either terminal event ends reader consumption and releases the connection.
Successful resolution of `streamChat()` means a terminal event was handled;
only the `done` handler marks the assistant response successful.

An EOF without either terminal event is incomplete. A failed network read
also reaches the UI error path. Partial tokens remain visible, and failed
turns are excluded from follow-up history. Error messages are never appended
to the Markdown answer.

`ChatWindow` already creates an `AbortController` per request and passes its
signal to fetch. Stop, leaving a conversation, and unmounting abort that
request. Cancellation is displayed as stopped, without marking success or
creating a server application error. The backend's `except Exception` does
not catch `GeneratorExit` or `asyncio.CancelledError`.

Browser cancellation terminates the browser transport; it does not guarantee
that an already-running synchronous Gemini request stops immediately.

## Gemini timeout

`get_gemini_client()` configures `types.HttpOptions(timeout=GEMINI_TIMEOUT_MS)`
with `GEMINI_TIMEOUT_MS = 60_000`. The shared client applies this configuration
to rewriting, embedding, and generation. The SDK uses milliseconds (see the
[Google SDK source](https://github.com/googleapis/python-genai/blob/main/google/genai/_api_client.py)).
This configures the SDK's HTTP timeout, not a total wall-clock deadline for the
entire RAG pipeline. A timeout raised during the pipeline becomes an error
event for the active stage.

## Verification

Run from the project root:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Run from `frontend`:

```powershell
npm test
npm run test:e2e
npm run build
```

Backend tests exercise the actual pipeline and FastAPI SSE route with mocked
retrieval and generation. They cover normal completion, retrieval failure,
generation failure after partial text, generation timeout, rewrite and source
failures, cancellation, generator closure, and client timeout configuration.

Client tests cover fragmented SSE, HTTP status failures, terminal error events,
normal completion, user cancellation, premature EOF, and abrupt network read
failure. Browser tests verify Markdown rendering, citations, failure messages,
partial-answer retention, retry history, and the Stop control. Tests use
controlled responses without live Gemini requests or database changes.
