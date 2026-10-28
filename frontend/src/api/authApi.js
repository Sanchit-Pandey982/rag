// Browser auth boundary. The access token lives in this module's memory
// only: never localStorage, never sessionStorage, never a cookie the JS
// can read. A refresh is single-flight, so concurrent requests share one
// rotation and never race each other for the same refresh cookie.

const AUTH_URL = "/api/v1/auth";

export class AuthError extends Error {
  constructor(message, { status = 0, retryAfter = null, code = "auth" } = {}) {
    super(message);
    this.name = "AuthError";
    this.status = status;
    this.retryAfter = retryAfter;
    this.code = code;
  }
}

export class ApiError extends Error {
  constructor(message, { status = 0, retryAfter = null } = {}) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.retryAfter = retryAfter;
  }
}

const REFRESH_MARGIN_MS = 60_000;

let session = { token: null, expiresAt: 0 };
let refreshPromise = null;
const listeners = new Set();

function notify() {
  const snapshot = getSessionSnapshot();
  for (const listener of listeners) listener(snapshot);
}

export function subscribe(listener) {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

export function getSessionSnapshot() {
  return { token: session.token, expiresAt: session.expiresAt };
}

export function getAccessToken() {
  return session.token;
}

function storeToken(payload) {
  const seconds = Number(payload?.expires_in);
  session = {
    token: typeof payload?.access_token === "string" ? payload.access_token : null,
    expiresAt: Number.isFinite(seconds) && seconds > 0
      ? Date.now() + seconds * 1000
      : 0,
  };
  notify();
}

function clearToken() {
  if (session.token === null) return;
  session = { token: null, expiresAt: 0 };
  notify();
}

// Explicit local sign-out after a server rejection. The subscriber in
// useAuth turns this into the login screen; the refresh cookie was already
// cleared server-side when the refresh endpoint returned 401.
export function expireSession() {
  clearToken();
}

// Test seam: resets module memory without touching real storage.
export function resetAuthSession() {
  session = { token: null, expiresAt: 0 };
  refreshPromise = null;
  notify();
}

async function errorDetail(response) {
  try {
    const body = await response.json();
    if (typeof body?.detail === "string") return body.detail;
    if (Array.isArray(body?.detail)) {
      return body.detail
        .map((item) => item?.msg)
        .filter((message) => typeof message === "string")
        .join("; ");
    }
  } catch {
    // Proxies may return HTML or an empty body.
  }
  return "";
}

function retryAfterSeconds(response) {
  const value = Number(response.headers.get("retry-after"));
  return Number.isFinite(value) && value > 0 ? Math.ceil(value) : null;
}

async function authRequest(path, { method = "POST", body, token } = {}) {
  const response = await fetch(`${AUTH_URL}${path}`, {
    method,
    credentials: "include",
    headers: {
      Accept: "application/json",
      ...(body ? { "Content-Type": "application/json" } : {}),
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
    },
    ...(body ? { body: JSON.stringify(body) } : {}),
  });
  if (response.ok) return response;
  const detail = await errorDetail(response);
  const retryAfter = retryAfterSeconds(response);
  const fallback = response.status === 429
    ? `Too many attempts.${retryAfter ? ` Try again in ${retryAfter} seconds.` : ""}`
    : response.status >= 500
      ? "The authentication service is unavailable. Please retry."
      : detail || "The request could not be completed.";
  throw new AuthError(
    response.status === 429 ? fallback : detail || fallback,
    { status: response.status, retryAfter },
  );
}

export async function loginRequest(username, password) {
  const response = await authRequest("/login", {
    body: { username, password },
  });
  const payload = await response.json();
  storeToken(payload);
  return payload;
}

export async function registerRequest(username, password) {
  const response = await authRequest("/register", {
    body: { username, password },
  });
  return response.json();
}

export async function currentUserRequest(token = session.token) {
  if (!token) throw new AuthError("Please sign in again.", { code: "signed-out" });
  const response = await fetch(`${AUTH_URL}/me`, {
    method: "GET",
    credentials: "include",
    headers: { Accept: "application/json", Authorization: `Bearer ${token}` },
  });
  if (!response.ok) {
    const detail = await errorDetail(response);
    throw new AuthError(detail || "Could not load your account.", {
      status: response.status,
    });
  }
  return response.json();
}

export async function refreshSession() {
  if (refreshPromise) return refreshPromise;
  const previousToken = session.token;
  refreshPromise = (async () => {
    let response;
    try {
      response = await fetch(`${AUTH_URL}/refresh`, {
        method: "POST",
        credentials: "include",
        headers: {
          Accept: "application/json",
          ...(previousToken
            ? { Authorization: `Bearer ${previousToken}` }
            : {}),
        },
      });
    } catch (error) {
      throw new AuthError(
        "The authentication service is unreachable. Please retry.",
        { code: "network" },
      );
    }
    if (!response.ok) {
      clearToken();
      const detail = await errorDetail(response);
      throw new AuthError(
        response.status >= 500
          ? "The authentication service is unavailable. Please retry."
          : detail || "Your session expired. Please sign in again.",
        { status: response.status, code: "expired" },
      );
    }
    const payload = await response.json();
    storeToken(payload);
    return session.token;
  })().finally(() => {
    refreshPromise = null;
  });
  return refreshPromise;
}

// Refresh shortly before expiry so side-effecting requests (upload,
// conversation creation) do not have to be replayed after a 401.
async function ensureFreshToken() {
  if (!session.token) return null;
  if (!session.expiresAt || session.expiresAt - Date.now() > REFRESH_MARGIN_MS) {
    return session.token;
  }
  return refreshSession();
}

export async function logoutRequest() {
  const token = session.token;
  try {
    await fetch(`${AUTH_URL}/logout`, {
      method: "POST",
      credentials: "include",
      headers: token ? { Authorization: `Bearer ${token}` } : {},
    });
  } finally {
    clearToken();
  }
}

function requestHeaders(headers, body, token) {
  const result = { Accept: "application/json", ...headers };
  if (body !== undefined && !(body instanceof FormData)) {
    result["Content-Type"] = "application/json";
  }
  if (token) result.Authorization = `Bearer ${token}`;
  return result;
}

function requestBody(body) {
  if (body === undefined || body instanceof FormData) return body;
  return JSON.stringify(body);
}

async function parseApiResponse(response) {
  if (response.status === 204) return null;
  if (response.ok) {
    try {
      return await response.json();
    } catch {
      return null;
    }
  }
  const detail = await errorDetail(response);
  const retryAfter = retryAfterSeconds(response);
  throw new ApiError(
    response.status >= 500
      ? "The service is unavailable. Please retry."
      : detail || `Request failed (${response.status}).`,
    { status: response.status, retryAfter },
  );
}

// JSON API helper for documents/conversations. GETs get one refresh+replay
// on 401. Side-effecting calls refresh but are never replayed blindly, and
// 403 is never treated as an expired session.
export async function apiRequest(path, {
  method = "GET",
  body,
  headers = {},
  signal,
} = {}) {
  const send = async (token) => fetch(path, {
    method,
    credentials: "include",
    headers: requestHeaders(headers, body, token),
    body: requestBody(body),
    signal,
  });

  let token = await ensureFreshToken();
  let response = await send(token);
  const safeToReplay = method === "GET" || method === "HEAD";

  if (response.status === 401) {
    if (!safeToReplay) {
      await refreshSession().catch(() => {});
      throw new AuthError(
        "Your session changed. Please retry the action.",
        { status: 401, code: "retry" },
      );
    }
    try {
      token = await refreshSession();
    } catch (error) {
      throw error instanceof AuthError
        ? error
        : new AuthError("Your session expired. Please sign in again.", {
            status: 401,
            code: "expired",
          });
    }
    response = await send(token);
    if (response.status === 401) {
      clearToken();
      throw new AuthError("Your session expired. Please sign in again.", {
        status: 401,
        code: "expired",
      });
    }
  }
  return parseApiResponse(response);
}
