import { FormEvent, useState } from "react";
import { ApiError, post } from "../api";

export default function Login({ needsSetup, notice, onDone }: { needsSetup: boolean; notice: string | null; onDone: () => void }) {
  const [username, setU] = useState("");
  const [password, setP] = useState("");
  const [name, setName] = useState("");
  const [err, setErr] = useState<ApiError | null>(null);
  const [busy, setBusy] = useState(false);

  const submit = async (e: FormEvent, takeOver = false) => {
    e.preventDefault();
    setBusy(true);
    setErr(null);
    try {
      if (needsSetup) await post("/setup", { username, display_name: name, password });
      await post("/login", { username, password, take_over: takeOver });
      onDone();
    } catch (x) {
      setErr(x as ApiError);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="page" style={{ maxWidth: 420, marginTop: "10vh" }}>
      <div className="card">
        <h1>{needsSetup ? "Create the administrator account" : "Sign in"}</h1>
        {needsSetup && <p className="muted">No accounts exist yet. This first account can add the operators' accounts. It can only be created on the workstation itself.</p>}
        {notice && <div className="msg warn">{notice}</div>}
        <form onSubmit={(e) => submit(e)}>
          <label className="field"><span>User name</span>
            <input autoFocus value={username} onChange={(e) => setU(e.target.value)} autoComplete="username" required />
          </label>
          {needsSetup && (
            <label className="field"><span>Full name</span>
              <input value={name} onChange={(e) => setName(e.target.value)} />
            </label>
          )}
          <label className="field"><span>Password{needsSetup ? " (at least 8 characters)" : ""}</span>
            <input type="password" value={password} onChange={(e) => setP(e.target.value)} autoComplete={needsSetup ? "new-password" : "current-password"} required />
          </label>
          {err && <div className="msg err">{err.message}</div>}
          <div className="row">
            <button className="primary" disabled={busy}>{needsSetup ? "Create and sign in" : "Sign in"}</button>
            {err?.code === "operator_active" && (
              <button type="button" className="danger" onClick={(e) => submit(e as unknown as FormEvent, true)}>Take over</button>
            )}
          </div>
        </form>
      </div>
    </div>
  );
}
