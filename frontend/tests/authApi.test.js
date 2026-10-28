import test from "node:test";
import assert from "node:assert/strict";
import {
  apiRequest,
  AuthError,
  getAccessToken,
  loginRequest,
  refreshSession,
  resetAuthSession,
  subscribe,
} from "../src/api/authApi.js";

const tokenBody = (token = "access-1") => new Response(JSON.stringify({
  access_token: token,
  token_type: "bearer",
  expires_in: 900,
}), { status: 200, headers: { "content-type": "application/json" } });

test.beforeEach(() => resetAuthSession());

test("login keeps the access token in module memory and posts credentials", async (t) => {
  t.mock.method(globalThis, "fetch", async (url, options) => {
    assert.equal(url, "/api/v1/auth/login");
    assert.equal(options.method, "POST");
    assert.equal(options.credentials, "include");
    assert.deepEqual(JSON.parse(options.body), {
      username: "Alice", password: "long-enough-password",
    });
    assert.equal(options.headers.Authorization, undefined);
    return tokenBody();
  });
  const payload = await loginRequest("Alice", "long-enough-password");
  assert.equal(payload.access_token, "access-1");
  assert.equal(getAccessToken(), "access-1");
});

test("concurrent refreshes share one rotation", async (t) => {
  let release;
  let calls = 0;
  t.mock.method(globalThis, "fetch", async (url) => {
    assert.equal(url, "/api/v1/auth/refresh");
    calls += 1;
    await new Promise((resolve) => { release = resolve; });
    return tokenBody("access-2");
  });

  const first = refreshSession();
  const second = refreshSession();
  await Promise.resolve();
  assert.equal(calls, 1, "both callers share the in-flight refresh");
  release();
  const [a, b] = await Promise.all([first, second]);
  assert.equal(a, "access-2");
  assert.equal(b, "access-2");
  assert.equal(calls, 1);
});

test("refresh failure clears the session and notifies subscribers", async (t) => {
  t.mock.method(globalThis, "fetch", async (url) => {
    if (url === "/api/v1/auth/login") return tokenBody();
    return new Response(JSON.stringify({ detail: "Invalid or expired refresh session." }), {
      status: 401, headers: { "content-type": "application/json" },
    });
  });
  await loginRequest("alice", "long-enough-password");
  const states = [];
  const unsubscribe = subscribe((snapshot) => states.push(snapshot.token));
  await assert.rejects(refreshSession(), /expired refresh session|session expired/);
  assert.equal(getAccessToken(), null);
  assert.deepEqual(states, [null]);
  unsubscribe();
});

test("a GET refreshes once and replays with the new token", async (t) => {
  const calls = [];
  t.mock.method(globalThis, "fetch", async (url, options) => {
    calls.push({ url, auth: options.headers.Authorization });
    if (url === "/api/v1/auth/login") return tokenBody("stale");
    if (url === "/api/v1/auth/refresh") return tokenBody("fresh");
    if (calls.filter((call) => call.url === "/api/v1/documents").length === 1) {
      return new Response(null, { status: 401 });
    }
    return new Response(JSON.stringify([{ document_id: "doc-1" }]), {
      status: 200, headers: { "content-type": "application/json" },
    });
  });
  await loginRequest("alice", "long-enough-password");
  const documents = await apiRequest("/api/v1/documents");
  assert.equal(documents[0].document_id, "doc-1");
  assert.deepEqual(
    calls.filter((call) => call.url === "/api/v1/documents").map((call) => call.auth),
    ["Bearer stale", "Bearer fresh"],
  );
});

test("a side-effecting request refreshes but is never replayed", async (t) => {
  const calls = [];
  t.mock.method(globalThis, "fetch", async (url, options) => {
    calls.push({ url, method: options.method });
    if (url === "/api/v1/auth/login") return tokenBody("stale");
    if (url === "/api/v1/auth/refresh") return tokenBody("fresh");
    return new Response(null, { status: 401 });
  });
  await loginRequest("alice", "long-enough-password");
  await assert.rejects(
    apiRequest("/api/v1/documents/upload", { method: "POST", body: new FormData() }),
    (error) => error instanceof AuthError && error.code === "retry",
  );
  assert.equal(calls.filter((call) => call.url === "/api/v1/documents/upload").length, 1);
  assert.equal(calls.filter((call) => call.url === "/api/v1/auth/refresh").length, 1);
});

test("403 is never treated as an expired session", async (t) => {
  let refreshes = 0;
  t.mock.method(globalThis, "fetch", async (url) => {
    if (url === "/api/v1/auth/refresh") refreshes += 1;
    return new Response(JSON.stringify({ detail: "Forbidden" }), {
      status: 403, headers: { "content-type": "application/json" },
    });
  });
  await assert.rejects(
    apiRequest("/api/v1/documents"),
    (error) => error.status === 403 && !(error instanceof AuthError),
  );
  assert.equal(refreshes, 0);
});

test("login errors never reflect the submitted password", async (t) => {
  t.mock.method(globalThis, "fetch", async () => new Response(
    JSON.stringify({ detail: "Invalid username or password." }),
    { status: 401, headers: { "content-type": "application/json" } },
  ));
  await assert.rejects(
    loginRequest("alice", "secret-password-123"),
    (error) => !error.message.includes("secret-password-123"),
  );
});
