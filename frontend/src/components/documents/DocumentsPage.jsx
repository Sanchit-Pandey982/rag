import { useCallback, useEffect, useRef, useState } from "react";
import { AlertCircle, CheckCircle2, Clock3, Database, FileText, LoaderCircle, RefreshCw, UploadCloud } from "lucide-react";
import { apiRequest } from "../../api/authApi.js";

const ACTIVE_STATUSES = new Set(["processing", "uploaded"]);

const statusCopy = {
  ready: "Ready to search",
  processing: "Indexing",
  uploaded: "Queued",
  failed: "Processing failed",
};

function formatBytes(bytes) {
  if (!Number.isFinite(bytes)) return "";
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function StatusIcon({ status }) {
  if (status === "ready") return <CheckCircle2 size={14} />;
  if (status === "failed") return <AlertCircle size={14} />;
  if (ACTIVE_STATUSES.has(status)) return <LoaderCircle size={14} className="spin" />;
  return <Clock3 size={14} />;
}

export default function DocumentsPage() {
  const [documents, setDocuments] = useState([]);
  const [loading, setLoading] = useState(true);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const fileInput = useRef(null);

  const load = useCallback(async () => {
    try {
      const records = await apiRequest("/api/v1/documents");
      setDocuments(records);
      setError("");
    } catch (caught) {
      setError(caught.message || "Could not load your documents.");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  useEffect(() => {
    if (!documents.some((document) => ACTIVE_STATUSES.has(document.status))) return undefined;
    const timer = setInterval(load, 2500);
    return () => clearInterval(timer);
  }, [documents, load]);

  async function upload(event) {
    const file = event.target.files?.[0];
    if (!file) return;
    setError("");
    setNotice("");
    if (!file.name.toLowerCase().endsWith(".txt")) {
      setError("Only plain-text (.txt) documents are supported.");
      event.target.value = "";
      return;
    }
    const form = new FormData();
    form.append("file", file);
    setUploading(true);
    try {
      const record = await apiRequest("/api/v1/documents/upload", {
        method: "POST",
        body: form,
      });
      setDocuments((previous) => [record, ...previous.filter((item) => item.document_id !== record.document_id)]);
      setNotice(`${record.original_filename} uploaded. Indexing starts automatically.`);
    } catch (caught) {
      setError(caught.message || "The upload could not be completed.");
    } finally {
      setUploading(false);
      if (fileInput.current) fileInput.current.value = "";
    }
  }

  return <section className="documents-page" aria-labelledby="documents-title">
    <span className="eyebrow">KNOWLEDGE BASE</span>
    <h1 id="documents-title">Your documents</h1>
    <p className="documents-intro">
      Upload plain-text documents to ground answers in your own material. Indexing runs in the background;
      status updates automatically until each document is ready.
    </p>

    <div className="upload-card">
      <div className="upload-icon"><UploadCloud size={25} /></div>
      <div>
        <h2>Upload a text document</h2>
        <p>Only <code>.txt</code> files are accepted. The server validates encoding and limits file size.</p>
      </div>
      <label className={`upload-button ${uploading ? "is-busy" : ""}`}>
        <input ref={fileInput} type="file" accept=".txt,text/plain" disabled={uploading} onChange={upload} />
        {uploading ? <><LoaderCircle size={15} className="spin" /> Uploading…</> : <><UploadCloud size={15} /> Choose file</>}
      </label>
    </div>

    {error && <div className="documents-alert error" role="alert">{error}</div>}
    {notice && <div className="documents-alert notice" role="status">{notice}</div>}

    <div className="documents-heading">
      <h2><Database size={15} /> Indexed documents <span>{documents.length}</span></h2>
      <button className="text-button" onClick={load} disabled={loading}>
        <RefreshCw size={13} /> Refresh
      </button>
    </div>

    {loading ? <div className="documents-empty"><LoaderCircle size={18} className="spin" /> Loading your documents…</div>
      : documents.length === 0 ? <div className="documents-empty">
          <FileText size={22} />
          <strong>No documents yet</strong>
          <p>Upload a text file above, wait for “Ready to search”, then ask a question in the chat workspace.</p>
        </div>
      : <div className="document-list">
          {documents.map((document) => <article className="document-row" key={document.document_id}>
            <span className={`document-status ${document.status}`} title={statusCopy[document.status] || document.status}>
              <StatusIcon status={document.status} />
            </span>
            <div className="document-main">
              <strong>{document.title || document.original_filename}</strong>
              <small>{document.original_filename} · {formatBytes(document.size_bytes)}</small>
            </div>
            <div className="document-meta">
              <span className={`status-pill ${document.status}`}>{statusCopy[document.status] || document.status}</span>
              <small>{document.status === "ready" ? `${document.chunk_count} chunks` : " "}</small>
            </div>
          </article>)}
        </div>}
  </section>;
}
