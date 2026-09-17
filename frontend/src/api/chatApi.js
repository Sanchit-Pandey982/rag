const CHAT_SSE_URL = "/api/v1/chat/sse";

// Existing public request interface; protocol decoding stays at the browser boundary.
export async function streamChat({
  raw_query, user_id, chat_history, k, rewrite_query, distance_threshold, access_token, onEvent, signal,
}) {
  const response = await fetch(CHAT_SSE_URL, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Accept: "text/event-stream",
      ...(access_token ? { Authorization: `Bearer ${access_token}` } : {}),
    },
    body: JSON.stringify({ raw_query, user_id, chat_history, k, rewrite_query, distance_threshold }),
    signal,
  });

  if (!response.ok) {
    let detail = "";
    // Validation messages are useful, but server exceptions stay on the backend.
    if (response.status === 422) {
      try {
        const body = await response.json();
        if (Array.isArray(body.detail)) {
          detail = body.detail.map((item) => `${item.loc?.join(".")}: ${item.msg}`).join("; ");
        }
      } catch {
        // Proxy errors may have no JSON body.
      }
    }
    throw new Error(`Chat request failed (${response.status}).${detail ? ` ${detail}` : " Check the API connection and try again."}`);
  }
  if (!response.headers.get("content-type")?.includes("text/event-stream")) {
    throw new Error("Expected an event stream. Check the /api proxy configuration.");
  }
  if (!response.body) throw new Error("Streaming response body is not available.");

  const reader = response.body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";
  let doneReceived = false;
  let errorReceived = false;

  function dispatch(rawEvent) {
    const parsed = parseSSEEvent(rawEvent);
    if (!parsed) {
      return;
    }
    if (parsed.event === "done") {
      doneReceived = true;
    }
    if (parsed.event === "error") {
      errorReceived = true;
    }
    // Application failures reach the semantic handler just like other events.
    // Transport and parsing failures reject streamChat and reach its catch block.
    onEvent(parsed);
  }

  function drain() {
    // Normalize the accumulated buffer: CRLF may be split across network reads.
    buffer = buffer.replace(/\r\n/g, "\n");
    while (!doneReceived && !errorReceived) {
      const boundary = buffer.indexOf("\n\n");
      if (boundary === -1) {
        break;
      }
      const rawEvent = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary + 2);
      dispatch(rawEvent);
    }
  }

  try {
    while (!doneReceived && !errorReceived) {
      signal?.throwIfAborted();
      const { value, done } = await reader.read();
      signal?.throwIfAborted();
      if (done) {
        buffer += decoder.decode();
        drain();
        if (!doneReceived && !errorReceived && buffer.trim()) dispatch(buffer);
        if (!doneReceived && !errorReceived) {
          throw new Error("The connection ended before the answer was complete. You can retry this question.");
        }
        break;
      }
      buffer += decoder.decode(value, { stream: true });
      drain();
    }
  } finally {
    try {
      await reader.cancel();
    } catch {
      // Abort can already have closed the reader.
    }
    reader.releaseLock();
  }
}

function parseSSEEvent(rawEvent) {
  let event = null;
  const dataLines = [];
  for (const line of rawEvent.split("\n")) {
    if (line.startsWith("event:")) event = line.slice(6).trim();
    else if (line.startsWith("data:")) dataLines.push(line.slice(5).replace(/^ /, ""));
  }
  if (!event || !dataLines.length) return null; // Includes SSE heartbeat comments.
  if (!["start", "retrieval", "token", "sources", "done", "error"].includes(event)) return null;
  let data;
  try { data = JSON.parse(dataLines.join("\n")); }
  catch { throw new Error(`The server sent invalid JSON in a ${event} event.`); }
  if (!data || typeof data !== "object" || Array.isArray(data)) throw new Error(`Invalid ${event} event data.`);
  if (event === "token" && typeof data.text !== "string") throw new Error("The server sent an invalid token event.");
  if (event === "sources" && (!Array.isArray(data.sources) || data.sources.some((source) =>
    !source || typeof source !== "object" ||
    !["chunk_id", "document_id", "source", "title"].every((key) => typeof source[key] === "string") ||
    !Number.isFinite(source.distance) || !Number.isInteger(source.chunk_index)
  ))) throw new Error("The server sent invalid source metadata.");
  if (event === "retrieval" && (typeof data.retrieval_query !== "string" ||
    !Array.isArray(data.retrieved_document_ids) || data.retrieved_document_ids.some((id) => typeof id !== "string"))) {
    throw new Error("The server sent invalid retrieval metadata.");
  }
  return { event, data };
}
