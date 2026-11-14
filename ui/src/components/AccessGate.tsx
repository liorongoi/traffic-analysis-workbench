import { FormEvent, ReactNode, useEffect, useState } from "react";

// On the network (server started with --lan) the API needs the access key. The browser asks for it once and then
// keeps a cookie; on this computer alone (127.0.0.1) nothing is asked.
export default function AccessGate({ children }: { children: ReactNode }) {
  const [state, setState] = useState<"checking" | "open" | "ask">("checking");
  const [key, setKey] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    fetch("/api/access", { credentials: "same-origin" })
      .then((r) => r.json())
      .then((d) => setState(d.required && !d.ok ? "ask" : "open"))
      .catch(() => setState("open"));   // an older server without access keys
  }, []);

  async function submit(e: FormEvent) {
    e.preventDefault();
    setBusy(true);
    setError("");
    try {
      const r = await fetch("/api/access", {
        method: "POST",
        credentials: "same-origin",
        headers: { "X-TWB": "1", "Content-Type": "application/json" },
        body: JSON.stringify({ key }),
      });
      const d = await r.json().catch(() => ({}));
      if (r.ok) setState("open");
      else setError(d.message ?? "That key was not accepted.");
    } catch {
      setError("The server did not answer.");
    } finally {
      setBusy(false);
    }
  }

  if (state === "checking") return null;
  if (state === "open") return <>{children}</>;
  return (
    <div style={{ minHeight: "100vh", display: "grid", placeItems: "center", padding: 16 }}>
      <form className="card" onSubmit={submit} style={{ width: "min(420px, 100%)" }}>
        <h2>LanePulse access key</h2>
        <p>This server is shared on the network. Enter the access key you were given. The server's administrator
          can show it with <code>sudo lanepulse key</code>.</p>
        <input autoFocus value={key} onChange={(e) => setKey(e.target.value)} aria-label="Access key"
          autoComplete="current-password" type="password" style={{ width: "100%", marginBottom: 12 }} />
        {error && <p role="alert" style={{ color: "var(--bad, #ff6b6b)" }}>{error}</p>}
        <button className="primary" type="submit" disabled={busy || !key.trim()}>Open LanePulse</button>
      </form>
    </div>
  );
}
