import { useState } from "react";
import { ArrowUpRight, BookOpen, ChevronRight, Database, Layers3, MessageSquare, Plus, Settings2, ShieldCheck, Sparkles, X } from "lucide-react";
import ChatWindow from "./components/chats/ChatWindow.jsx";
import ConnectionStatus from "./components/ConnectionStatus.jsx";

const newConversation = () => ({ id: crypto.randomUUID(), title: "New conversation" });

const integrationItems = [
  ["Identity & access", "Connect authenticated identity to user_id and enforce ownership in the API. The current user selector is a development control."],
  ["Knowledge ingestion", "Connect document upload, indexing progress, and document management after their backend contracts exist."],
  ["Conversation storage", "Persist and restore conversations through authenticated backend storage. Current conversations live in this tab until refresh."],
  ["Evaluation & observability", "Connect evaluation runs and server traces. Retrieval details shown in chat already come from existing SSE events."],
  ["Concurrency & deployment", "Add server request limits, cancellation propagation, and a production proxy configured for streaming."],
];

export default function App() {
  const [user_id, setUserId] = useState("eval_user");
  const [userDraft, setUserDraft] = useState("eval_user");
  const [k, setK] = useState(3);
  const [rewrite_query, setRewriteQuery] = useState(false);
  const [distance_threshold, setDistanceThreshold] = useState(null);
  const [conversations, setConversations] = useState(() => [newConversation()]);
  const [activeId, setActiveId] = useState(null);
  const [page, setPage] = useState("chat");
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [busyById, setBusyById] = useState({});
  const busy = Object.values(busyById).some(Boolean);
  const selectedId = activeId || conversations[0].id;
  const selected = conversations.find((item) => item.id === selectedId);

  function addConversation() {
    const conversation = newConversation();
    setConversations((previous) => [conversation, ...previous]);
    setActiveId(conversation.id);
    setPage("chat");
  }

  function changeUser(event) {
    event.preventDefault();
    const next = userDraft.trim();
    if (!next || next === user_id || busy) return;
    setUserId(next);
    setUserDraft(next);
    setConversations([newConversation()]);
    setActiveId(null);
    setPage("chat");
  }

  return <div className="app-shell">
    <aside className="sidebar" aria-label="Workspace navigation">
      <a className="brand" href="#chat" onClick={() => setPage("chat")}>
        <span className="brand-mark"><Layers3 size={21} /></span>
        <span>RAG<span className="brand-light"> Learning</span><small>KNOWLEDGE WORKSPACE</small></span>
      </a>
      <button className="new-chat" onClick={addConversation}><Plus size={17} /> New conversation <span>+</span></button>
      <nav className="primary-nav" aria-label="Main navigation">
        <button className={page === "chat" ? "nav-item selected" : "nav-item"} onClick={() => setPage("chat")}><MessageSquare size={17} /> Chat workspace <ChevronRight size={15} /></button>
        <button className={page === "integration" ? "nav-item selected" : "nav-item"} onClick={() => setPage("integration")}><BookOpen size={17} /> Integration roadmap</button>
      </nav>
      <div className="section-label">THIS SESSION <span>{conversations.length}</span></div>
      <div className="conversation-list">
        {conversations.map((conversation) => <button key={conversation.id}
          className={`conversation-item ${page === "chat" && selectedId === conversation.id ? "current" : ""}`}
          onClick={() => { setActiveId(conversation.id); setPage("chat"); }}>
          <MessageSquare size={14} /><span>{conversation.title}</span>
        </button>)}
      </div>
      <div className="sidebar-note"><ShieldCheck size={18} /><div><strong>Your session, your context</strong><p>Chats stay in this tab. Refreshing clears your conversation history.</p></div></div>
      <div className="user-card"><span className="avatar">{user_id.slice(0, 2).toUpperCase()}</span><div><strong>{user_id}</strong><small>Development user</small></div><button className="icon-button" aria-label="Open retrieval settings" onClick={() => setSettingsOpen(true)}><Settings2 size={18} /></button></div>
    </aside>

    <main className="main-panel" id="chat">
      <header className="topbar">
        <div className="breadcrumb">Workspace <ChevronRight size={13} /><strong>{page === "chat" ? "Chat" : "Integration"}</strong></div>
        <div className="topbar-actions"><span className="environment-tag">Development</span><button className="settings-toggle icon-button" aria-label="Toggle retrieval settings" aria-expanded={settingsOpen} onClick={() => setSettingsOpen(!settingsOpen)}><Settings2 size={19} /></button></div>
      </header>
      {page === "chat" ? <div className="workspace-heading"><div><h1>{selected?.title || "New conversation"}</h1><p>Explore your knowledge, one question at a time.</p></div><span className="model-badge"><Sparkles size={13} /> RAG assistant</span></div> : null}
      {page === "chat" && conversations.length > 1 && <div className="mobile-conversations"><label htmlFor="active-conversation">Conversation</label><select id="active-conversation" value={selectedId} onChange={(event) => setActiveId(event.target.value)}>{conversations.map((conversation) => <option key={conversation.id} value={conversation.id}>{conversation.title}</option>)}</select></div>}
      <div className="chat-panels" hidden={page !== "chat"}>
        {conversations.map((conversation) => <div className="chat-panel" key={conversation.id} hidden={conversation.id !== selectedId}>
          <ChatWindow user_id={user_id} chat_history={[]} k={k} rewrite_query={rewrite_query} distance_threshold={distance_threshold}
            active={page === "chat" && selectedId === conversation.id}
            onBusyChange={(value) => setBusyById((previous) => ({ ...previous, [conversation.id]: value }))}
            onTitle={(title) => setConversations((previous) => previous.map((item) => item.id === conversation.id ? { ...item, title } : item))} />
        </div>)}
      </div>
      {page === "integration" && <section className="integration-page">
        <span className="eyebrow">THE NEXT CHAPTER</span><h1>From a conversation<br />to a production system.</h1>
        <p className="integration-intro">Your streaming chat is connected to the existing API contract. These capabilities need backend integration before their product controls can go live.</p>
        <div className="roadmap-list">{integrationItems.map(([title, description], index) => <article key={title}><span className="roadmap-number">0{index + 1}</span><div><h2>{title}</h2><p>{description}</p></div><span className="pending-label">Backend pending</span></article>)}</div>
        <div className="guide-note"><BookOpen size={20} /><div><strong>Continue with the integration guide</strong><p>Open FRONTEND_INTEGRATION.md in the project root for architecture, data flow, engineering reasoning, and implementation steps.</p></div></div>
      </section>}
      <footer className="workspace-footer"><span><span className="tiny-dot" /> Grounded in your knowledge</span><span>Verify answers against their sources <ArrowUpRight size={12} /></span></footer>
    </main>

    <aside className={`inspector ${settingsOpen ? "is-open" : ""}`} aria-label="Retrieval settings" onKeyDown={(event) => { if (event.key === "Escape") setSettingsOpen(false); }}>
      <div className="inspector-title"><span><Settings2 size={17} /> Retrieval settings</span><button className="close-settings icon-button" aria-label="Close retrieval settings" onClick={() => setSettingsOpen(false)}><X size={18} /></button></div>
      <div className="inspector-content">
        <div className="inspector-section"><span className="eyebrow">CONNECTION</span><ConnectionStatus /><p className="helper">Live status from your API. Checked every 30 seconds.</p></div>
        <form className="inspector-section" onSubmit={changeUser}>
          <label className="field-label" htmlFor="user-id">User context <code>user_id</code></label>
          <input id="user-id" value={userDraft} maxLength={256} required disabled={busy} onChange={(event) => setUserDraft(event.target.value)} />
          <button className="text-button" disabled={busy || !userDraft.trim() || userDraft.trim() === user_id} type="submit">Apply user context <ChevronRight size={13} /></button>
          <p className="helper">Changing user clears this tab’s chats. This development control is not authentication.</p>
        </form>
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
