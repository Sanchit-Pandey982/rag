import { useEffect, useState } from "react";
import { RefreshCw } from "lucide-react";

export default function ConnectionStatus() {
  const [connection, setConnection] = useState({ state: "checking", label: "Checking API" });
  const [attempt, setAttempt] = useState(0);

  useEffect(() => {
    const controller = new AbortController();
    async function check() {
      try {
        const signal = AbortSignal.any([controller.signal, AbortSignal.timeout(5000)]);
        const [health, ready] = await Promise.all([
          fetch("/health", { signal, cache: "no-store" }),
          fetch("/ready", { signal, cache: "no-store" }),
        ]);
        const healthBody = await health.json();
        if (!health.ok || healthBody.status !== "ok") throw new Error("Health check failed");
        const readyBody = await ready.json();
        if (!controller.signal.aborted) setConnection(
          ready.ok && readyBody.status === "ready"
            ? { state: "ready", label: "API ready" }
            : { state: "warning", label: "API not ready" },
        );
      } catch {
        if (!controller.signal.aborted) setConnection({ state: "offline", label: "API unavailable" });
      }
    }
    check();
    const interval = setInterval(check, 30000);
    return () => { controller.abort(); clearInterval(interval); };
  }, [attempt]);

  return <div className="connection">
    <span className={`status-dot ${connection.state}`} />
    <span role="status">{connection.label}</span>
    <button className="icon-button" onClick={() => setAttempt((value) => value + 1)} aria-label="Refresh API status" title="Refresh API status"><RefreshCw size={13} /></button>
  </div>;
}
