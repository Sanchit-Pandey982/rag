import test from "node:test";
import assert from "node:assert/strict";
import { streamChat } from "../src/api/chatApi.js";
import { buildChatHistory } from "../src/history.js";

const payload = { raw_query: "How?", user_id: "eval_user", chat_history: [], k: 3, rewrite_query: false, distance_threshold: null };
const encode = (event, data) => `event: ${event}\r\ndata: ${JSON.stringify(data)}\r\n\r\n`;
function mockStream(t, text, fragmentSize = 1) {
  const bytes = new TextEncoder().encode(text);
  let offset = 0;
  t.mock.method(globalThis, "fetch", async () => new Response(new ReadableStream({
    pull(controller) {
      if (offset >= bytes.length) { controller.close(); return; }
      controller.enqueue(bytes.slice(offset, offset += fragmentSize));
    },
  }), { headers: { "content-type": "text/event-stream; charset=utf-8" } }));
}

test("decodes fragmented UTF-8, split CRLF, multiline data, heartbeats and final residual done", async (t) => {
  const text = ": heartbeat\r\n\r\n" + encode("start", { raw_query: "How?" }) +
    'event: token\r\ndata: {\r\ndata: "text": "Hello 🌍 नमस्ते"}\r\n\r\n' +
    'event: done\r\ndata: {}';
  mockStream(t, text);
  const events = [];
  await streamChat({ ...payload, onEvent: (event) => events.push(event) });
  assert.deepEqual(events.map(({ event }) => event), ["start", "token", "done"]);
  assert.equal(events[1].data.text, "Hello 🌍 नमस्ते");
});

test("sends precisely the existing contract and stops at done even if the connection stays open", async (t) => {
  let cancelled = false;
  t.mock.method(globalThis, "fetch", async (url, options) => {
    assert.equal(url, "/api/v1/chat/sse");
    assert.deepEqual(JSON.parse(options.body), payload);
    assert.equal(options.method, "POST");
    return new Response(new ReadableStream({
      start(controller) { controller.enqueue(new TextEncoder().encode(encode("done", {}))); },
      cancel() { cancelled = true; },
    }), { headers: { "content-type": "text/event-stream" } });
  });
  await streamChat({ ...payload, onEvent() {} });
  assert.equal(cancelled, true);
});

test("reports premature EOF while preserving already dispatched tokens", async (t) => {
  mockStream(t, encode("token", { text: "Partial" }), 100);
  const events = [];
  await assert.rejects(streamChat({ ...payload, onEvent: (event) => events.push(event) }), /before the answer was complete/);
  assert.equal(events[0].data.text, "Partial");
});

test("rejects malformed JSON and invalid token payload", async (t) => {
  mockStream(t, "event: token\ndata: nope\n\n");
  await assert.rejects(streamChat({ ...payload, onEvent() {} }), /invalid JSON/);
});
test("rejects a token object without text", async (t) => {
  mockStream(t, encode("token", { answer: "wrong field" }));
  await assert.rejects(streamChat({ ...payload, onEvent() {} }), /invalid token/);
});
test("does not accept a null source that would crash citation rendering", async (t) => {
  mockStream(t, encode("sources", { sources: [null] }));
  await assert.rejects(streamChat({ ...payload, onEvent() {} }), /invalid source metadata/);
});
test("surfaces FastAPI validation details", async (t) => {
  t.mock.method(globalThis, "fetch", async () => new Response(JSON.stringify({
    detail: [{ loc: ["body", "user_id"], msg: "Field required" }],
  }), { status: 422 }));
  await assert.rejects(streamChat({ ...payload, onEvent() {} }), /422.*body.user_id: Field required/);
});
test("rejects an HTML fallback response", async (t) => {
  t.mock.method(globalThis, "fetch", async () => new Response("<html></html>", { headers: { "content-type": "text/html" } }));
  await assert.rejects(streamChat({ ...payload, onEvent() {} }), /proxy configuration/);
});
test("aborts without marking a partial answer complete", async (t) => {
  const controller = new AbortController();
  t.mock.method(globalThis, "fetch", async () => new Response(new ReadableStream({
    start(stream) {
      stream.enqueue(new TextEncoder().encode(encode("token", { text: "Partial" })));
      controller.signal.addEventListener("abort", () => stream.error(new DOMException("Aborted", "AbortError")));
    },
  }), { headers: { "content-type": "text/event-stream" } }));
  await assert.rejects(streamChat({ ...payload, signal: controller.signal, onEvent() { controller.abort(); } }), { name: "AbortError" });
});
test("history excludes incomplete turns, strips UI fields and keeps the most recent 20 messages", () => {
  const messages = Array.from({ length: 12 }, (_, index) => [
    { role: "user", content: `q${index}`, id: "local" },
    { role: "assistant", content: `a${index}`, status: "done", sources: [] },
  ]).flat();
  messages.push({ role: "user", content: "failed" }, { role: "assistant", content: "partial", status: "error" });
  const history = buildChatHistory([], messages);
  assert.equal(history.length, 20);
  assert.deepEqual(history[0], { role: "user", content: "q2" });
  assert.deepEqual(history.at(-1), { role: "assistant", content: "a11" });
});
test("history obeys message length limits and includes valid supplied context", () => {
  const history = buildChatHistory([{ role: "system", content: "context" }], [
    { role: "user", content: "q" },
    { role: "assistant", content: "🌍".repeat(8001), status: "done" },
  ]);
  assert.deepEqual(history[0], { role: "system", content: "context" });
  assert.equal([...history.at(-1).content].length, 8000);
});
