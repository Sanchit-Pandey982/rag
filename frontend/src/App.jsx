import { useCallback, useEffect, useState } from "react";
import { ArrowUpRight, BookOpen, ChevronRight, Database, Layers3, LogOut, MessageSquare, Plus, Settings2, ShieldCheck, Sparkles, X } from "lucide-react";
import { apiRequest, expireSession } from "./api/authApi.js";
import AuthScreen from "./components/auth/AuthScreen.jsx";
import ChatWindow from "./components/chats/ChatWindow.jsx";
import ConnectionStatus from "./components/ConnectionStatus.jsx";
import DocumentsPage from "./components/documents/DocumentsPage.jsx";
import useAuth from "./hooks/useAuth.js";

const emptyConversation = () => ({
  id: crypto.randomUUID(),
  conversationId: null,
  title: "New conversation",
  hydrate: false,
});

const fromServer = (conversation) => ({
  id: conversation.conversation_id,
  conversationId: conversation.conversation_id,
  title: conversation.title || "New conversation",
  hydrate: true,
});

const integrationItems = [
  ["Knowledge ingestion", "Upload .txt documents from the Documents page. Indexing progress and ready/failed status are visible without reloading."],
  ["Conversation storage", "Conversations and completed turns are stored server-side. Sign in again after a reload to reattach to your history."],
  ["Evaluation & observability", "A nightly CI job gates retrieval recall, answer accuracy, and refusals. Per-request dashboards remain future work."],
  ["Concurrency & deployment", "Server request limits and the daily chat quota are live. Multi-replica cache invalidation and a dedicated ingestion worker remain future work."],
];

export default function App() {
  const {
    user, accessToken, busy: authBusy, error: authError,
    clearError, login, register, logout, refreshAccessToken,
  } = useAuth();
  const [k, setK] = useState(3);
  const [rewrite_query, setRewriteQuery] = useState(false);
  const [distance_threshold, setDistanceThreshold] = useState(null);
  const [conversations, setConversations] = useState([]);
  const [activeId, setActiveId] = useState(null);
  const [conversationsLoading, setConversationsLoading] = useState(false);
  const [page, setPage] = useState("chat");
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [busyById, setBusyById] = useState({});
  const busy = Object.values(busyById).some(Boolean);
  const selectedId = activeId || conversations[0]?.id || null;
  const selected = conversations.find((item) => item.id === selectedId) || null;

  useEffect(() => {
    if (!user) {
      setConversations([]);
      setActiveId(null);
      setPage("chat");
      return undefined;
    }
    let cancelled = false;
    setConversationsLoading(true);
    apiRequest("/api/v1/conversations")
      .then((records) => {
        if (cancelled) return;
        const items = records.map(fromServer);
        if (!items.length) items.push(emptyConversation());
        setConversations(items);
        setActiveId(items[0].id);
      })
      .catch(() => {
        if (!cancelled) setConversations([emptyConversation()]);
      })
      .finally(() => {
        if (!cancelled) setConversationsLoading(false);
      });
    return () => { cancelled = true; };
  }, [user?.user_id]);

  const addConversation = useCallback(() => {
    const conversation = emptyConversation();
    setConversations((previous) => [conversation, ...previous]);
    setActiveId(conversation.id);
    setPage("chat");
  }, []);

  const ensureConversation = useCallback(async (localId, title) => {
    const existing = conversations.find((item) => item.id === localId);
    if (existing?.conversationId) return existing.conversationId;
    const created = await apiRequest("/api/v1/conversations", {
      method: "POST",
      body: { title: (title || "New conversation").slice(0, 256) },
    });
    setConversations((previous) => previous.map((item) => item.id === localId
      ? {
          ...item,
          conversationId: created.conversation_id,
          title: created.title || item.title,
        }
      : item));
    return created.conversation_id;
  }, [conversations]);

  const renameConversation = useCallback((localId, title) => {
    setConversations((previous) => previous.map((item) => item.id === localId
      ? { ...item, title }
      : item));
  }, []);

  const signOut = useCallback(async () => {
    setConversations([]);
    setActiveId(null);
    setPage("chat");
    await logout();
  }, [logout]);

  if (!user) {
    return <AuthScreen
      onLogin={login}
      onRegister={register}
      busy={authBusy}
      error={authError}
      clearError={clearError}
    />;
  }

  const initials = user.username.slice(0, 2).toUpperCase();

  return <div className="app-shell">
    <aside className="sidebar" aria-label="Workspace navigation">
      <a className="brand" href="#chat" onClick={() => setPage("chat")}>
        <span className="brand-mark"><Layers3 size={21} /></span>
        <span>RAG<span className="brand-light"> Learning</span><small>KNOWLEDGE WORKSPACE</small></span>
      </a>
      <button className="new-chat" onClick={addConversation}><Plus size={17} /> New conversation <span>+</span></button>
      <nav className="primary-nav" aria-label="Main navigation">
        <button className={page === "chat" ? "nav-item selected" : "nav-item"} onClick={() => setPage("chat")}><MessageSquare size={17} /> Chat workspace <ChevronRight size={15} /></button>
        <button className={page === "documents" ? "nav-item selected" : "nav-item"} onClick={() => setPage("documents")}><Database size={17} /> Documents</button>
        <button className={page === "integration" ? "nav-item selected" : "nav-item"} onClick={() => setPage("integration")}><BookOpen size={17} /> Integration roadmap</button>
      </nav>
      <div className="section-label">CONVERSATIONS <span>{conversations.length}</span></div>
      <div className="conversation-list">
        {conversations.map((conversation) => <button key={conversation.id}
          className={`conversation-item ${page === "chat" && selectedId === conversation.id ? "current" : ""}`}
          onClick={() => { setActiveId(conversation.id); setPage("chat"); }}>
          <MessageSquare size={14} /><span>{conversation.title}</span>
        </button>)}
      </div>
      <div className="sidebar-note"><ShieldCheck size={18} /><div><strong>Stored with your account</strong><p>Conversations are saved server-side. Reloading signs you out, then your history returns after sign-in.</p></div></div>
      <div className="user-card"><span className="avatar">{initials}</span><div><strong>{user.username}</strong><small>Signed in</small></div><button className="icon-button" aria-label="Sign out" title="Sign out" disabled={authBusy} onClick={signOut}><LogOut size={17} /></button><button className="icon-button" aria-label="Open retrieval settings" onClick={() => setSettingsOpen(true)}><Settings2 size={18} /></button></div>
    </aside>

    <main className="main-panel" id="chat">
      <header className="topbar">
        <div className="breadcrumb">Workspace <ChevronRight size={13} /><strong>{page === "chat" ? "Chat" : page === "documents" ? "Documents" : "Integration"}</strong></div>
        <div className="topbar-actions"><span className="environment-tag">Authenticated</span><button className="settings-toggle icon-button" aria-label="Toggle retrieval settings" aria-expanded={settingsOpen} onClick={() => setSettingsOpen(!settingsOpen)}><Settings2 size={19} /></button></div>
      </header>
      {page === "chat" ? <div className="workspace-heading"><div><h1>{selected?.title || (conversationsLoading ? "Loading conversations…" : "New conversation")}</h1><p>Explore your knowledge, one question at a time.</p></div><span className="model-badge"><Sparkles size={13} /> RAG assistant</span></div> : null}
      {page === "chat" && conversations.length > 1 && <div className="mobile-conversations"><label htmlFor="active-conversation">Conversation</label><select id="active-conversation" value={selectedId || ""} onChange={(event) => setActiveId(event.target.value)}>{conversations.map((conversation) => <option key={conversation.id} value={conversation.id}>{conversation.title}</option>)}</select></div>}
      <div className="chat-panels" hidden={page !== "chat"}>
        {conversations.map((conversation) => <div className="chat-panel" key={conversation.id} hidden={conversation.id !== selectedId}>
          <ChatWindow user_id={user.user_id} access_token={accessToken}
            conversation_id={conversation.conversationId} hydrate={conversation.hydrate}
            chat_history={[]} k={k} rewrite_query={rewrite_query} distance_threshold={distance_threshold}
            active={page === "chat" && selectedId === conversation.id}
            onEnsureConversation={(title) => ensureConversation(conversation.id, title)}
            refreshAccessToken={refreshAccessToken}
            onAuthExpired={expireSession}
            onBusyChange={(value) => setBusyById((previous) => ({ ...previous, [conversation.id]: value }))}
            onTitle={(title) => renameConversation(conversation.id, title)} />
        </div>)}
      </div>
      {page === "documents" && <DocumentsPage />}
      {page === "integration" && <section className="integration-page">
        <span className="eyebrow">THE NEXT CHAPTER</span><h1>From a conversation<br />to a production system.</h1>
        <p className="integration-intro">Streaming chat, account isolation, document ingestion, and persistent conversations are live. These capabilities are the next layer of the production plan.</p>
        <div className="roadmap-list">{integrationItems.map(([title, description], index) => <article key={title}><span className="roadmap-number">0{index + 1}</span><div><h2>{title}</h2><p>{description}</p></div><span className="pending-label">{index < 2 ? "Live" : "Next"}</span></article>)}</div>
        <div className="guide-note"><BookOpen size={20} /><div><strong>Continue with the integration guide</strong><p>Open FRONTEND_INTEGRATION.md in the project root for architecture, data flow, engineering reasoning, and implementation steps.</p></div></div>
      </section>}
      <footer className="workspace-footer"><span><span className="tiny-dot" /> Grounded in your knowledge</span><span>Verify answers against their sources <ArrowUpRight size={12} /></span></footer>
    </main>

    <aside className={`inspector ${settingsOpen ? "is-open" : ""}`} aria-label="Retrieval settings" onKeyDown={(event) => { if (event.key === "Escape") setSettingsOpen(false); }}>
      <div className="inspector-title"><span><Settings2 size={17} /> Retrieval settings</span><button className="close-settings icon-button" aria-label="Close retrieval settings" onClick={() => setSettingsOpen(false)}><X size={18} /></button></div>
      <div className="inspector-content">
        <div className="inspector-section"><span className="eyebrow">CONNECTION</span><ConnectionStatus /><p className="helper">Live status from your API. Checked every 30 seconds.</p></div>
        <div className="inspector-section account-section">
          <span className="eyebrow">ACCOUNT</span>
          <div className="account-row"><span className="avatar">{initials}</span><div><strong>{user.username}</strong><small>{user.user_id}</small></div></div>
          <button className="text-button" onClick={signOut} disabled={authBusy}><LogOut size={13} /> Sign out</button>
          <p className="helper">The access token stays in memory only. Reloading the page requires signing in again; your stored conversations return after login.</p>
        </div>
        <fieldset disabled={busy} className="retrieval-fields">
          <legend className="sr-only">Retrieval parameters</legend>
          <div className="inspector-section">
            <label className="field-label" htmlFor="chunk-count">Retrieved chunks <output>{k}</output></label>
            <input id="chunk-count" type="range" min="1" max="10" step="1" value={k} onChange={(event) => setK(Number(event.target.value))} />
            <div className="range-labels"><span>1 · Focused</span><span>10 · Broader</span></div>
            <p className="helper">Maximum chunks requested for each question.</p>
          </div>
          <div className="inspector-section">
            <label className="toggle-row" htmlFor="rewrite-query"><span>Rewrite follow-ups<small>Use conversation context</small></span><input id="rewrite-query" type="checkbox" role="switch" checked={rewrite_query} onChange={(event) => setRewriteQuery(event.target.checked)} /></label>
            <p className="helper">Turns a follow-up into a standalone retrieval question. Adds a model call.</p>
          </div>
          <div className="inspector-section">
            <label className="toggle-row" htmlFor="distance-filter"><span>Distance filter<small>Exclude weaker matches</small></span><input id="distance-filter" type="checkbox" role="switch" checked={distance_threshold !== null} onChange={(event) => setDistanceThreshold(event.target.checked ? 1 : null)} /></label>
            {distance_threshold !== null && <><label className="field-label distance-label" htmlFor="distance-threshold">Maximum distance <output>{distance_threshold.toFixed(2)}</output></label><input id="distance-threshold" type="range" min="0" max="2" step="0.05" value={distance_threshold} onChange={(event) => setDistanceThreshold(Number(event.target.value))} /></>}
            <p className="helper">Lower distance is more restrictive. This is not a confidence score.</p>
          </div>
        </fieldset>
        <div className="knowledge-note"><Database size={19} /><h3>Answers start with sources.</h3><p>Retrieved references appear below each answer. Your API controls which documents this user can access.</p></div>
      </div>
      <div className="inspector-bottom"><span className="tiny-dot" /> Settings apply to your next question</div>
    </aside>
  </div>;
}
