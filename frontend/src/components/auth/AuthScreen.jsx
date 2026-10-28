import { useState } from "react";
import { ArrowRight, Eye, EyeOff, Layers3, LockKeyhole, ShieldCheck, UserRound } from "lucide-react";

export default function AuthScreen({ onLogin, onRegister, busy, error, clearError }) {
  const [mode, setMode] = useState("login");
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [notice, setNotice] = useState("");
  const [localError, setLocalError] = useState("");
  const registering = mode === "register";

  function switchMode(next) {
    setMode(next);
    setPassword("");
    setLocalError("");
    setNotice("");
    clearError();
  }

  async function submit(event) {
    event.preventDefault();
    setLocalError("");
    setNotice("");
    const name = username.trim();
    if (!name) {
      setLocalError("Enter your username.");
      return;
    }
    if (registering && password.length < 8) {
      setLocalError("Choose a password with at least 8 characters.");
      return;
    }
    if (!password) {
      setLocalError("Enter your password.");
      return;
    }
    try {
      if (registering) {
        await onRegister(name, password);
        setMode("login");
        setPassword("");
        setNotice("Account created. Sign in to continue.");
      } else {
        await onLogin(name, password);
      }
    } catch {
      // The hook exposes the server-safe message through `error`.
    }
  }

  const message = localError || error;

  return <main className="auth-screen">
    <section className="auth-card" aria-labelledby="auth-title">
      <div className="auth-brand">
        <span className="brand-mark"><Layers3 size={22} /></span>
        <span>RAG<span className="brand-light"> Learning</span><small>KNOWLEDGE WORKSPACE</small></span>
      </div>
      <span className="eyebrow">{registering ? "CREATE YOUR WORKSPACE" : "WELCOME BACK"}</span>
      <h1 id="auth-title">{registering ? "Start with your documents." : "Sign in to your knowledge."}</h1>
      <p className="auth-intro">
        {registering
          ? "Create an account, upload a text document, and ask questions grounded in your own sources."
          : "Your chats, uploads, and sources stay scoped to your account. Access tokens live in memory only."}
      </p>

      <div className="auth-tabs" role="tablist" aria-label="Account access">
        <button type="button" role="tab" aria-selected={!registering} className={!registering ? "selected" : ""} onClick={() => switchMode("login")}>Sign in</button>
        <button type="button" role="tab" aria-selected={registering} className={registering ? "selected" : ""} onClick={() => switchMode("register")}>Create account</button>
      </div>

      <form className="auth-form" onSubmit={submit} noValidate>
        <label htmlFor="auth-username">Username</label>
        <div className="auth-input">
          <UserRound size={16} />
          <input id="auth-username" name="username" autoComplete="username" value={username} maxLength={256}
            disabled={busy} onChange={(event) => setUsername(event.target.value)} />
        </div>
        <label htmlFor="auth-password">Password</label>
        <div className="auth-input">
          <LockKeyhole size={16} />
          <input id="auth-password" name="password" type={showPassword ? "text" : "password"}
            autoComplete={registering ? "new-password" : "current-password"} value={password} maxLength={256}
            disabled={busy} onChange={(event) => setPassword(event.target.value)} />
          <button type="button" className="icon-button auth-reveal" aria-label={showPassword ? "Hide password" : "Show password"}
            onClick={() => setShowPassword(!showPassword)}>{showPassword ? <EyeOff size={16} /> : <Eye size={16} />}</button>
        </div>
        {registering && <p className="helper">At least 8 characters. Usernames are case-insensitive.</p>}
        {message && <div className="auth-error" role="alert">{message}</div>}
        {notice && <div className="auth-notice" role="status">{notice}</div>}
        <button className="auth-submit" type="submit" disabled={busy}>
          {busy ? "Please wait…" : registering ? "Create account" : "Sign in"}
          {!busy && <ArrowRight size={16} />}
        </button>
      </form>

      <div className="auth-trust">
        <ShieldCheck size={16} />
        <p><strong>Tenant-scoped by design.</strong> Retrieval always filters by your account. Your refresh token is an HttpOnly cookie; the short-lived access token is never written to browser storage.</p>
      </div>
    </section>
    <footer className="auth-footer">Grounded answers, with sources you can inspect.</footer>
  </main>;
}
