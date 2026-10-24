import { useEffect, useState } from "react";
import { api, ApiError, post } from "../api";

type M = { id: string; name: string; input_type: string; mode: string; processing: string; monitoring: boolean; created_at: string;
  status: Record<string, number>; job: { state: string; done: number; total: number } | null };

export default function Missions() {
  const [list, setList] = useState<M[] | null>(null);
  const [sys, setSys] = useState<any>(null);
  const [msg, setMsg] = useState<{ kind: string; text: string } | null>(null);
  const loadSys = () => api("/system").then(setSys).catch(() => {});
  useEffect(() => { api("/missions").then((r) => setList(r.missions)); loadSys(); }, []);
  return (
    <div className="page">
      <div className="row" style={{ justifyContent: "space-between" }}>
        <h1>Missions</h1>
        <a className="btn primary" href="#/new"><button className="primary">New mission</button></a>
      </div>
      {list === null && <p className="muted">Loading…</p>}
      {list?.length === 0 && <div className="card"><p>No missions yet.</p><p className="muted">A mission reads one source folder of images or videos, analyses them in one mode and exports to one output folder.</p></div>}
      {list && list.length > 0 && (
        <div className="tablewrap">
          <table className="t">
            <thead><tr><th>Name</th><th>Input</th><th>Mode</th><th>Processing</th><th className="num">Files</th><th className="num">Queued</th><th>Activity</th><th>Created (UTC)</th></tr></thead>
            <tbody>
              {list.map((m) => (
                <tr key={m.id} className="click" onClick={() => (window.location.hash = `#/m/${m.id}`)}>
                  <td><b className="bidi">{m.name}</b></td>
                  <td>{m.input_type}</td>
                  <td>{m.mode.replace(/_/g, " ")}</td>
                  <td>{m.processing === "continuous" ? `continuous, monitoring ${m.monitoring ? "on" : "off"}` : "one-time"}</td>
                  <td className="num">{m.status.source_versions}</td>
                  <td className="num">{m.status.queued_items}</td>
                  <td>{m.job && ["running", "paused"].includes(m.job.state) ? `${m.job.state} ${m.job.done}/${m.job.total}` : "idle"}</td>
                  <td className="mono">{m.created_at}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      {sys && <System sys={sys} msg={msg} backupNow={async () => {
        setMsg(null);
        try { const r = await post("/admin/backup"); setMsg({ kind: "ok", text: `Backed up to ${r.folder}` }); }
        catch (e) { setMsg({ kind: "err", text: (e as ApiError).message }); }
        loadSys();
      }} />}
    </div>
  );
}

const ACCEL: Record<string, string> = { cuda: "NVIDIA GPU (CUDA)", directml: "GPU (DirectML)", cpu: "CPU" };

function System({ sys, msg, backupNow }: { sys: any; msg: any; backupNow: () => void }) {
  const b = sys.backup;
  return (
    <div className="card" style={{ marginTop: 14 }}>
      <h2 style={{ marginTop: 0 }}>System</h2>
      <p style={{ margin: "4px 0" }}>Version {sys.version}. Models run on: {sys.runtime ? <>{ACCEL[sys.runtime.accelerator]}{sys.runtime.notes.length > 0 && <span className="muted"> ({sys.runtime.notes.join("; ")})</span>}</> : <span className="muted">decided when the first job loads the detector</span>}.</p>
      <p style={{ margin: "4px 0" }}>
        {b.folder ? <>Nightly database backup at {b.at} to <span className="mono">{b.folder}</span>. </> : <>No backup folder is set: start the server with <span className="mono">--backup-dir</span>. </>}
        {b.last ? <>Last: {b.last.at} ({b.last.detail.folder ?? b.last.detail.error}).</> : "No backup yet."}
        {b.last_error && <span className="msg err">Last scheduled backup failed: {b.last_error}</span>}
      </p>
      <div className="row"><button disabled={!b.folder} onClick={backupNow}>Back up now</button>{msg && <span className={`msg ${msg.kind}`}>{msg.text}</span>}</div>
      {sys.audit && sys.audit.length > 0 && <details style={{ marginTop: 8 }}><summary>Audit log</summary>
        <ul className="muted">{sys.audit.map((e: any) => <li key={e.id}><span className="mono">{e.at}</span> {e.author}: {e.action.replace(/_/g, " ")}{e.detail.name ? ` “${e.detail.name}”` : ""}{e.detail.folder ? ` ${e.detail.folder}` : ""}</li>)}</ul></details>}
    </div>
  );
}
