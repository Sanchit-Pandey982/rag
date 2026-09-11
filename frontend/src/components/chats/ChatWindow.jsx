import { useEffect, useId, useRef, useState } from "react";
import { ArrowDown, ArrowUp, BookOpen, Check, ChevronRight, Copy, FileText, Layers3, RotateCcw, Search, Square, Workflow } from "lucide-react";
import ReactMarkdown from "react-markdown";
import { streamChat } from "../../api/chatApi.js";
import { buildChatHistory } from "../../history.js";

const prompts = [
  { icon: BookOpen, title: "Understand the basics", query: "What is retrieval-augmented generation and how does it work?" },
  { icon: Search, title: "Connect the concepts", query: "How are embeddings used in a vector database?" },
  { icon: Workflow, title: "Explore the pipeline", query: "Explain the steps from a question to a grounded RAG answer." },
];
const statusLabels = {
  starting: "Connecting to your knowledge…", started: "Searching your knowledge…",
  retrieving: "Sources retrieved · Preparing answer…", streaming: "Writing an answer…",
  done: "Response complete", cancelled: "Response stopped", error: "Response interrupted",
};

function Message({ message, onRetry, isStreaming }) {
  const [copied, setCopied] = useState(false);
  const [copyError, setCopyError] = useState(false);
  const copyTimer = useRef(null);
  useEffect(() => () => clearTimeout(copyTimer.current), []);
  async function copyAnswer() {
    try {
      await navigator.clipboard.writeText(message.content);
      setCopied(true);
      setCopyError(false);
      clearTimeout(copyTimer.current);
      copyTimer.current = setTimeout(() => setCopied(false), 2000);
    } catch { setCopyError(true); }
  }
  if (message.role === "user") return <article className="message user-message"><span className="message-role">You</span><div className="user-bubble">{message.content}</div></article>;
  return <article className="message assistant-message">
    <div className="assistant-identity"><span className="assistant-avatar"><Layers3 size={16} /></span><strong>RAG assistant</strong><span>Grounded response</span></div>
    <div className="assistant-body">
      {message.content ? <div className="markdown"><ReactMarkdown skipHtml components={{
        a: ({ children, href }) => <a href={href} target="_blank" rel="noopener noreferrer">{children}</a>,
        img: ({ alt }) => <span>[Image: {alt || "omitted"}]</span>,
      }}>{message.content}</ReactMarkdown></div> : !["error", "cancelled", "done"].includes(message.status) ? <div className="thinking"><span /><span /><span /><span className="sr-only">Waiting for answer</span></div> : null}
      {message.status === "done" && !message.content && <p className="helper">The server completed without answer text.</p>}
      {message.retrieval && <details className="retrieval-details">
        <summary><Search size={13} /> Retrieval details <span>{message.retrieval.retrieved_document_ids.length} retrieved chunks</span></summary>
        <div><strong>Retrieval query</strong><p>{message.retrieval.retrieval_query}</p><strong>Document IDs</strong><p>{message.retrieval.retrieved_document_ids.join(", ") || "No matching documents"}</p></div>
      </details>}
      {message.sources?.length > 0 && <section className="source-section" aria-label="Answer sources">
        <h3><FileText size={13} /> SOURCES <span>{message.sources.length}</span></h3>
        <div className="source-grid">{message.sources.map((source, index) => <details className="source-card" key={`${source.chunk_id}-${index}`}>
          <summary><span className="source-number">{index + 1}</span><span><strong>{source.title || source.document_id}</strong><small>{source.source}</small></span><ChevronRight size={13} /></summary>
          <dl><dt>Document</dt><dd>{source.document_id}</dd><dt>Chunk ID</dt><dd>{source.chunk_id}</dd><dt>Chunk index</dt><dd>{source.chunk_index}</dd><dt>Distance</dt><dd>{source.distance.toFixed(4)}</dd></dl>
        </details>)}</div>
      </section>}
      {message.sources?.length === 0 && <p className="helper">No source citations were returned for this answer.</p>}
      {message.error && <div className="message-error" role="alert">{message.error}</div>}
      {message.status === "cancelled" && <p className="stopped-note">Stopped. Any partial answer above is incomplete.</p>}
      <div className="message-actions">
        {message.content && <button onClick={copyAnswer} className="text-button">{copied ? <Check size={13} /> : <Copy size={13} />}{copied ? "Copied" : "Copy answer"}</button>}
        {["error", "cancelled"].includes(message.status) && <button className="text-button" disabled={isStreaming} onClick={() => onRetry(message.query)}><RotateCcw size={13} /> Retry question</button>}
        {copyError && <span role="status" className="helper">Copy unavailable. Select and copy the answer manually.</span>}
        {message.elapsed != null && <span className="elapsed">{(message.elapsed / 1000).toFixed(1)}s · Browser elapsed</span>}
      </div>
    </div>
  </article>;
}

export default function ChatWindow({
  user_id, chat_history = [], k = 3, rewrite_query = false, distance_threshold = null,
  active = true, onTitle, onBusyChange,
}) {
  const [messages, setMessages] = useState([]);
  const queryId = useId();
  const [raw_query, setRawQuery] = useState("");
  const [isStreaming, setIsStreaming] = useState(false);
  const [streamStatus, setStreamStatus] = useState(null);
  const [showJump, setShowJump] = useState(false);
  const abortControllerRef = useRef(null);
  const scrollRef = useRef(null);
  const textareaRef = useRef(null);
  const followScrollRef = useRef(true);
  const previousUserRef = useRef(user_id);

  useEffect(() => () => abortControllerRef.current?.abort(), []);
  useEffect(() => {
    if (!active) abortControllerRef.current?.abort();
  }, [active]);
  useEffect(() => {
    if (previousUserRef.current !== user_id) {
      abortControllerRef.current?.abort();
      setMessages([]);
      setRawQuery("");
      setStreamStatus(null);
      previousUserRef.current = user_id;
    }
  }, [user_id]);
  useEffect(() => {
    if (active && followScrollRef.current && scrollRef.current) scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
  }, [messages, active, streamStatus]);
  useEffect(() => {
    const textarea = textareaRef.current;
    if (textarea) {
      textarea.style.height = "auto";
      textarea.style.height = `${Math.min(textarea.scrollHeight, 180)}px`;
    }
  }, [raw_query]);

  function updateMessage(id, patch) {
    setMessages((previous) => previous.map((message) => message.id === id ? { ...message, ...patch } : message));
  }

  async function handleSubmit(event, retryQuery) {
    event?.preventDefault();
    const query = (retryQuery ?? raw_query).trim();
    if (!query || [...query].length > 4000 || abortControllerRef.current || !user_id?.trim() || !active) return;

    const assistantMessageId = crypto.randomUUID();
    const controller = new AbortController();
    abortControllerRef.current = controller; // Synchronous lock also prevents rapid duplicate submits.
    const startedAt = performance.now();
    const history = buildChatHistory(chat_history, messages);
    if (!messages.length) onTitle?.(query.length > 48 ? query.slice(0, 48) + "…" : query);
    setMessages((previous) => [...previous,
      { id: crypto.randomUUID(), role: "user", content: query },
      { id: assistantMessageId, role: "assistant", content: "", sources: null, status: "starting", query },
    ]);
    if (retryQuery === undefined) setRawQuery("");
    setIsStreaming(true);
    onBusyChange?.(true);
    setStreamStatus("starting");
    followScrollRef.current = true;
    setShowJump(false);

    try {
      await streamChat({
        raw_query: query, user_id, chat_history: history, k, rewrite_query, distance_threshold,
        signal: controller.signal,
        onEvent: ({ event, data }) => {
          if (controller.signal.aborted) return;
          handleStreamEvent(event, data, assistantMessageId, startedAt);
        },
      });
    } catch (error) {
      if (controller.signal.aborted || error.name === "AbortError") {
        setStreamStatus("cancelled");
        updateMessage(assistantMessageId, { status: "cancelled" });
      } else {
        setStreamStatus("error");
        updateMessage(assistantMessageId, { status: "error", error: error.message || "Unable to reach the API. Check your connection and retry." });
      }
    } finally {
      if (abortControllerRef.current === controller) {
        abortControllerRef.current = null;
        setIsStreaming(false);
        onBusyChange?.(false);
      }
    }
  }

  function handleStreamEvent(event, data, assistantMessageId, startedAt) {
    switch (event) {
      case "start":
        setStreamStatus("started");
        updateMessage(assistantMessageId, { status: "started" });
        break;
      case "retrieval":
        setStreamStatus("retrieving");
        updateMessage(assistantMessageId, { retrieval: data, status: "retrieving" });
        break;
      case "token":
        setStreamStatus("streaming");
        appendToken(assistantMessageId, data.text);
        break;
      case "sources":
        attachSources(assistantMessageId, data.sources);
        break;
      case "done":
        setStreamStatus("done");
        updateMessage(assistantMessageId, { status: "done", elapsed: performance.now() - startedAt });
        break;
      case "error":
        setStreamStatus("error");
        updateMessage(assistantMessageId, {
          status: "error",
          error: "The response could not be completed.",
        });
        break;
    }
  }

  function appendToken(assistantMessageId, token) {
    setMessages((previous) => previous.map((message) => message.id === assistantMessageId
      ? { ...message, content: message.content + token, status: "streaming" } : message));
  }
  function attachSources(assistantMessageId, sources) { updateMessage(assistantMessageId, { sources }); }
  function stopStreaming() { abortControllerRef.current?.abort(); }

  return <div className="chat-window">
    <div className="message-scroll" ref={scrollRef} onScroll={() => {
      const el = scrollRef.current;
      followScrollRef.current = el.scrollHeight - el.scrollTop - el.clientHeight < 100;
      setShowJump(!followScrollRef.current);
    }}>
      {!messages.length ? <section className="empty-state">
        <div className="empty-symbol"><Layers3 size={29} strokeWidth={1.6} /><span /></div>
        <span className="eyebrow">YOUR KNOWLEDGE, CONNECTED</span>
        <h2>Good questions.<br /><span>Grounded answers.</span></h2>
        <p>Ask a question, explore an idea, or connect the dots.<br className="desktop-break" /> Your assistant finds the context in your documents.</p>
        <div className="prompt-grid">{prompts.map(({ icon: Icon, title, query }) => <button key={title} className="prompt-card" onClick={() => { setRawQuery(query); textareaRef.current?.focus(); }}>
          <Icon size={19} strokeWidth={1.7} /><strong>{title}</strong><span>{query}</span><ChevronRight size={15} />
        </button>)}</div>
        <div className="empty-footnote"><FileText size={13} /> References included when your API returns sources</div>
      </section> : <div className="message-list" aria-label="Conversation messages" aria-busy={isStreaming}>
        {messages.map((message) => <Message key={message.id} message={message} isStreaming={isStreaming} onRetry={(query) => handleSubmit(null, query)} />)}
      </div>}
    </div>
    {showJump && <button className="jump-button" onClick={() => { followScrollRef.current = true; scrollRef.current.scrollTop = scrollRef.current.scrollHeight; setShowJump(false); }}><ArrowDown size={14} /> Latest response</button>}
    <div className="composer-wrap">
      <div className={`stream-status ${isStreaming ? "is-streaming" : ""}`} role="status" aria-live="polite">{streamStatus ? <><span className="tiny-dot" />{statusLabels[streamStatus]}</> : <><span className="tiny-dot" /> Ready for your first question</>}</div>
      <form className="composer" onSubmit={handleSubmit}>
        <label className="sr-only" htmlFor={queryId}>Ask your knowledge a question</label>
        <textarea ref={textareaRef} id={queryId} placeholder="Ask your knowledge a question…" value={raw_query} rows={2} maxLength={4000}
          onChange={(event) => setRawQuery(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing) {
              event.preventDefault();
              handleSubmit(event);
            }
          }} />
        <div className="composer-bottom"><span className="composer-context"><Layers3 size={13} /> Your knowledge <span className="composer-divider">/</span> {k} chunks</span><div><span className="character-count">{raw_query.length.toLocaleString()} / 4,000</span>
          {isStreaming ? <button type="button" className="send-button stop-button" onClick={stopStreaming} aria-label="Stop response"><Square size={15} fill="currentColor" /></button>
            : <button className="send-button" type="submit" disabled={!raw_query.trim() || !user_id?.trim()} aria-label="Send question"><ArrowUp size={20} /></button>}
        </div></div>
      </form>
      <div className="composer-hint"><span><kbd>Enter</kbd> to send · <kbd>Shift + Enter</kbd> for a new line</span><span>Answers may contain mistakes.</span></div>
    </div>
  </div>;
}
