import { FormEvent, useEffect, useState } from "react";
import { createPortal } from "react-dom";
import { post } from "../api";

// On a server shared on the network: change the password (knowing the current one). Until someone chooses their own,
// the button asks for it, since the one made at install was printed on the server's screen.
export default function PasswordButton() {
  const [state, setState] = useState<{ required: boolean; generated: boolean } | null>(null);
  const [open, setOpen] = useState(false);
  const [cur, setCur] = useState("");
  const [next, setNext] = useState("");
  const [again, setAgain] = useState("");
  const [msg, setMsg] = useState<{ ok: boolean; text: string } | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    fetch("/api/access", { credentials: "same-origin" }).then((r) => r.json())
      .then((d) => setState({ required: !!d.required, generated: !!d.generated })).catch(() => setState(null));
  }, []);
  if (!state?.required) return null;

  async function submit(e: FormEvent) {
    e.preventDefault();
    if (next !== again) { setMsg({ ok: false, text: "The two new passwords are not the same." }); return; }
    setBusy(true); setMsg(null);
    try {
      await post("/access/password", { current: cur, new: next });
      setMsg({ ok: true, text: "Password changed. Other browsers must enter the new one." });
      setState({ required: true, generated: false });
      setCur(""); setNext(""); setAgain("");
    } catch (err: any) {
      setMsg({ ok: false, text: err?.message ?? "The password was not changed." });
    } finally { setBusy(false); }
  }

  return (
    <>
      <button type="button" className={`password-button${state.generated ? " nudge" : ""}`} onClick={() => setOpen(true)}
        title={state.generated ? "Choose your own password" : "Change the LanePulse password"}
        aria-label={state.generated ? "Choose your own password" : "Change the LanePulse password"}>
        <span aria-hidden="true">🔑</span>
      </button>
      {open && createPortal(
        <div className="modal-backdrop" role="dialog" aria-modal="true" aria-label="Change the password"
          onClick={(e) => { if (e.target === e.currentTarget) setOpen(false); }}
          style={{ position: "fixed", inset: 0, background: "rgba(0,0,0,.55)", display: "grid", placeItems: "center", padding: 16, zIndex: 50 }}>
          <form className="card" onSubmit={submit} style={{ width: "min(420px, 100%)" }}>
            <h2>Change the password</h2>
            {state.generated && <p>The password made at install is still in use. Choose your own: at least 8 characters.</p>}
            <label>Current password
              <input type="password" autoComplete="current-password" value={cur} onChange={(e) => setCur(e.target.value)} style={{ width: "100%" }} />
            </label>
            <label>New password (at least 8 characters)
              <input type="password" autoComplete="new-password" value={next} onChange={(e) => setNext(e.target.value)} style={{ width: "100%" }} />
            </label>
            <label>New password again
              <input type="password" autoComplete="new-password" value={again} onChange={(e) => setAgain(e.target.value)} style={{ width: "100%" }} />
            </label>
            {msg && <p role="status" style={{ color: msg.ok ? "var(--good, #4cd7a0)" : "var(--bad, #ff6b6b)" }}>{msg.text}</p>}
            <div style={{ display: "flex", gap: 8, marginTop: 12 }}>
              <button className="primary" type="submit" disabled={busy || !cur || next.length < 8}>Change password</button>
              <button type="button" onClick={() => { setOpen(false); setMsg(null); }}>Close</button>
            </div>
            <p className="muted" style={{ marginTop: 14, fontSize: 13 }}>Forgot the current one? On the server:
              <code> sudo lanepulse password</code> sets a new one.</p>
          </form>
        </div>, document.body)}
    </>
  );
}
