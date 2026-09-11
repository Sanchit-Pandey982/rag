import { test, expect } from "@playwright/test";

const sse = (event, data) => `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`;
const response = sse("start", { raw_query: "Question" }) +
  sse("retrieval", { retrieval_query: "Standalone question", retrieved_document_ids: ["rag_basics"] }) +
  sse("token", { text: "A **grounded** answer with `code`." }) +
  sse("sources", { sources: [{ title: "RAG basics", source: "data/rag_basics.txt", document_id: "rag_basics", chunk_id: "chunk-1", chunk_index: 0, distance: 0.1234 }] }) +
  sse("done", {});

test.beforeEach(async ({ page }) => {
  await page.route("**/health", (route) => route.fulfill({ json: { status: "ok" } }));
  await page.route("**/ready", (route) => route.fulfill({ json: { status: "ready" } }));
});

test("streams a grounded answer, renders sources and sends follow-up history", async ({ page }) => {
  const requests = [];
  const errors = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await page.route("**/api/v1/chat/sse", (route) => {
    requests.push(route.request().postDataJSON());
    return route.fulfill({ contentType: "text/event-stream", body: response });
  });
  await page.goto("/");
  await expect(page.getByText("API ready", { exact: true })).toBeVisible();
  await page.getByRole("textbox", { name: "Ask your knowledge a question" }).fill("Explain RAG");
  await page.getByRole("button", { name: "Send question" }).click();
  await expect(page.locator(".chat-panel:not([hidden])").getByText("Response complete", { exact: true })).toBeVisible();
  await expect(page.locator(".markdown strong")).toHaveText("grounded");
  await page.getByText("RAG basics", { exact: true }).click();
  await expect(page.getByText("0.1234", { exact: true })).toBeVisible();
  await page.getByRole("textbox", { name: "Ask your knowledge a question" }).fill("And embeddings?");
  await page.getByRole("textbox", { name: "Ask your knowledge a question" }).press("Enter");
  await expect.poll(() => requests.length).toBe(2);
  expect(requests[0]).toEqual({ raw_query: "Explain RAG", user_id: "eval_user", chat_history: [], k: 3, rewrite_query: false, distance_threshold: null });
  expect(requests[1].chat_history).toEqual([{ role: "user", content: "Explain RAG" }, { role: "assistant", content: "A **grounded** answer with `code`." }]);
  expect(errors).toEqual([]);
});

test("changes retrieval settings, isolates conversations and clears them on a user change", async ({ page }) => {
  const requests = [];
  await page.route("**/api/v1/chat/sse", (route) => {
    requests.push(route.request().postDataJSON());
    return route.fulfill({ contentType: "text/event-stream", body: response });
  });
  await page.goto("/");
  await page.getByLabel("Retrieved chunks").fill("6");
  await page.getByRole("switch", { name: /Rewrite follow-ups/ }).check();
  await page.getByRole("switch", { name: /Distance filter/ }).check();
  await page.getByRole("button", { name: /Understand the basics/ }).click();
  await page.getByRole("button", { name: "Send question" }).click();
  await expect(page.locator(".chat-panel:not([hidden])").getByText("Response complete", { exact: true })).toBeVisible();
  expect(requests[0].k).toBe(6);
  expect(requests[0].rewrite_query).toBe(true);
  expect(requests[0].distance_threshold).toBe(1);
  await page.getByRole("button", { name: /New conversation/ }).first().click();
  await page.getByRole("textbox", { name: "Ask your knowledge a question" }).fill("New thread");
  await page.getByRole("button", { name: "Send question" }).click();
  await expect(page.locator(".chat-panel:not([hidden])").getByText("Response complete", { exact: true })).toBeVisible();
  expect(requests[1].chat_history).toEqual([]);
  await page.getByLabel(/User context/).fill("user_B");
  await page.getByRole("button", { name: /Apply user context/ }).click();
  await expect(page.getByRole("heading", { name: /Good questions/ })).toBeVisible();
  await expect(page.locator(".conversation-item")).toHaveCount(1);
  await page.getByRole("textbox", { name: "Ask your knowledge a question" }).fill("My question");
  await page.getByRole("button", { name: "Send question" }).click();
  await expect.poll(() => requests.length).toBe(3);
  expect(requests[2].user_id).toBe("user_B");
  expect(requests[2].chat_history).toEqual([]);
});

test("retains a partial answer, allows retry and excludes failed history", async ({ page }) => {
  let calls = 0;
  await page.route("**/api/v1/chat/sse", (route) => {
    calls += 1;
    if (calls === 1) return route.fulfill({ contentType: "text/event-stream", body: sse("token", { text: "Partial answer" }) });
    expect(route.request().postDataJSON().chat_history).toEqual([]);
    return route.fulfill({ contentType: "text/event-stream", body: response });
  });
  await page.goto("/");
  await page.getByRole("textbox", { name: "Ask your knowledge a question" }).fill("Question");
  await page.getByRole("button", { name: "Send question" }).click();
  await expect(page.getByRole("alert")).toContainText("before the answer was complete");
  await expect(page.getByText("Partial answer", { exact: true })).toBeVisible();
  await page.getByRole("button", { name: "Retry question" }).click();
  await expect(page.locator(".chat-panel:not([hidden])").getByText("Response complete", { exact: true })).toBeVisible();
});

for (const stage of ["retrieval", "generation"]) {
  test(`${stage} error shows application failure and keeps it out of answer text and history`, async ({ page }) => {
    let calls = 0;
    await page.route("**/api/v1/chat/sse", (route) => {
      calls += 1;
      if (calls > 1) {
        expect(route.request().postDataJSON().chat_history).toEqual([]);
        return route.fulfill({ contentType: "text/event-stream", body: response });
      }
      let body = sse("start", { raw_query: "Question" });
      if (stage === "generation") {
        body += sse("retrieval", { retrieval_query: "Question", retrieved_document_ids: [] });
        body += sse("token", { text: "Partial answer" });
      }
      body += sse("error", { stage, message: "The response could not be completed." });
      return route.fulfill({ contentType: "text/event-stream", body });
    });
    await page.goto("/");
    await page.getByRole("textbox", { name: "Ask your knowledge a question" }).fill("Question");
    await page.getByRole("button", { name: "Send question" }).click();
    await expect(page.getByRole("alert")).toHaveText("The response could not be completed.");
    await expect(page.getByText("Response complete", { exact: true })).toHaveCount(0);
    if (stage === "generation") {
      await expect(page.locator(".markdown")).toHaveText("Partial answer");
    } else {
      await expect(page.locator(".markdown")).toHaveCount(0);
    }
    await page.getByRole("button", { name: "Retry question" }).click();
    await expect(page.locator(".chat-panel:not([hidden])").getByText("Response complete", { exact: true })).toBeVisible();
  });
}

test("HTTP status failure shows a generic UI error", async ({ page }) => {
  await page.route("**/api/v1/chat/sse", (route) => route.fulfill({
    status: 500, json: { detail: "private database exception" },
  }));
  await page.goto("/");
  await page.getByRole("textbox", { name: "Ask your knowledge a question" }).fill("Question");
  await page.getByRole("button", { name: "Send question" }).click();
  await expect(page.getByRole("alert")).toContainText("Chat request failed (500)");
  await expect(page.getByRole("alert")).not.toContainText("private database exception");
  await expect(page.getByRole("button", { name: "Retry question" })).toBeVisible();
});

test("stop cancels a pending request and restores the composer", async ({ page }) => {
  let release;
  const pending = new Promise((resolve) => { release = resolve; });
  await page.route("**/api/v1/chat/sse", async (route) => {
    await pending;
    await route.fulfill({ contentType: "text/event-stream", body: response }).catch(() => {});
  });
  await page.goto("/");
  await page.getByRole("textbox", { name: "Ask your knowledge a question" }).fill("Question");
  await page.getByRole("button", { name: "Send question" }).click();
  await page.getByRole("button", { name: "Stop response" }).click();
  await expect(page.getByText("Response stopped", { exact: true })).toBeVisible();
  await expect(page.getByRole("button", { name: "Send question" })).toBeVisible();
  release();
});

test("renders safely when the API returns untrusted Markdown", async ({ page }) => {
  await page.route("**/api/v1/chat/sse", (route) => route.fulfill({
    contentType: "text/event-stream",
    body: sse("token", { text: '<script>window.hacked=true</script>\n\n[bad](javascript:alert(1))\n\n![tracking](https://example.com/pixel)' }) + sse("done", {}),
  }));
  await page.goto("/");
  await page.getByRole("textbox", { name: "Ask your knowledge a question" }).fill("Question");
  await page.getByRole("button", { name: "Send question" }).click();
  await expect(page.locator(".chat-panel:not([hidden])").getByText("Response complete", { exact: true })).toBeVisible();
  await expect(page.locator(".markdown img, .markdown script, .markdown a[href^='javascript:']")).toHaveCount(0);
  expect(await page.evaluate(() => window.hacked)).toBeUndefined();
});

test("mobile layout stays in the viewport and exposes settings and roadmap", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/");
  await expect(page.getByRole("heading", { name: /Good questions/ })).toBeVisible();
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  await page.screenshot({ path: "test-results/mobile-workspace.png", fullPage: true });
  await page.getByRole("button", { name: /New conversation/ }).first().click();
  await expect(page.getByLabel("Conversation", { exact: true })).toBeVisible();
  await expect(page.getByLabel("Conversation", { exact: true }).locator("option")).toHaveCount(2);
  await page.getByRole("button", { name: "Toggle retrieval settings" }).click();
  await expect(page.getByLabel(/User context/)).toBeVisible();
  await page.getByRole("button", { name: "Close retrieval settings" }).click();
  await page.getByRole("button", { name: "Integration roadmap" }).click();
  await expect(page.getByRole("heading", { name: /From a conversation/ })).toBeVisible();
  await page.screenshot({ path: "test-results/mobile-roadmap.png", fullPage: true });
});

test("desktop empty state and API unavailable state", async ({ page }) => {
  await page.setViewportSize({ width: 1440, height: 1000 });
  await page.route("**/health", (route) => route.abort());
  await page.goto("/");
  await expect(page.getByText("API unavailable", { exact: true })).toBeVisible();
  await page.screenshot({ path: "test-results/desktop-workspace.png", fullPage: true });
});

test("switching conversations aborts the old request without contaminating a new turn", async ({ page }) => {
  let release;
  const pending = new Promise((resolve) => { release = resolve; });
  let calls = 0;
  await page.route("**/api/v1/chat/sse", async (route) => {
    calls += 1;
    if (calls === 1) await pending;
    await route.fulfill({ contentType: "text/event-stream", body: response }).catch(() => {});
  });
  await page.goto("/");
  await page.getByRole("textbox", { name: "Ask your knowledge a question" }).fill("Old question");
  await page.getByRole("button", { name: "Send question" }).click();
  await expect.poll(() => calls).toBe(1);
  await page.getByRole("button", { name: /New conversation/ }).first().click();
  await page.getByRole("textbox", { name: "Ask your knowledge a question" }).fill("New question");
  await page.getByRole("button", { name: "Send question" }).click();
  await expect(page.locator(".chat-panel:not([hidden])").getByText("Response complete", { exact: true })).toBeVisible();
  release();
  await page.getByRole("button", { name: "Old question", exact: true }).click();
  await expect(page.getByText("Response stopped", { exact: true })).toBeVisible();
  await expect(page.locator(".chat-panel:not([hidden]) .markdown")).toHaveCount(0);
});

test("reports reachable but unready API separately from an outage", async ({ page }) => {
  await page.route("**/ready", (route) => route.fulfill({ status: 503, json: { detail: "service not ready" } }));
  await page.goto("/");
  await expect(page.getByText("API not ready", { exact: true })).toBeVisible();
});
