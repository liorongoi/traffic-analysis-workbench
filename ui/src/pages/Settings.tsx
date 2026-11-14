import { useEffect, useState } from "react";
import { api, ApiError, post } from "../api";
import NameFormats from "./NameFormats";
import Workers from "./Workers";

// Settings: what is set once, or seldom, for the whole workstation. Kept off the working pages so they show only what
// the operator needs day to day. Each mission's own one-time choices are on that mission's Settings tab.
const SECTIONS: [string, string, string][] = [
  ["computer", "This computer", "Version, where detection runs, detector sizes, backups and the audit log"],
  ["formats", "Filename formats", "How your cameras name their files, so capture times are read right"],
  ["workers", "Other computers", "Let other computers on the network help analyse long videos"],
];

export default function Settings({ section }: { section?: string }) {
  const on = SECTIONS.some(([k]) => k === section) ? section! : "computer";
  return (
    <div className="page settings">
      <div className="page-head">
        <div>
          <h1>Settings</h1>
          <p>Things you set up once for this workstation. Each mission's own setup is on its Settings tab.</p>
        </div>
      </div>
      <nav className="tabs" aria-label="Settings sections">
        {SECTIONS.map(([k, l, d]) => <a key={k} href={`#/settings/${k}`} className={on === k ? "on" : ""} title={d} aria-current={on === k ? "page" : undefined}>{l}</a>)}
      </nav>
      {on === "computer" && <ThisComputer />}
      {on === "formats" && <NameFormats embedded />}
      {on === "workers" && <Workers embedded />}
    </div>
  );
}

function ThisComputer() {
  const [sys, setSys] = useState<any>(null);
  const [err, setErr] = useState("");
  const [msg, setMsg] = useState<{ kind: string; text: string } | null>(null);
  const load = () => api("/system").then(setSys).catch((e) => setErr(e instanceof Error ? e.message : "Could not load system status."));
  useEffect(() => { load(); }, []);
  const backupNow = async () => {
    setMsg(null);
    try { const r = await post("/admin/backup"); setMsg({ kind: "ok", text: `Backed up to ${r.folder}` }); }
    catch (e) { setMsg({ kind: "err", text: (e as ApiError).message }); }
    load();
  };
  return (
    <div className="grid" style={{ gap: 14 }}>
      {err && <div className="card msg err" role="alert"><b>Could not load system status.</b><p>{err}</p><button onClick={() => { setErr(""); load(); }}>Try again</button></div>}
      {sys && <System sys={sys} msg={msg} backupNow={backupNow} />}
      <SizesCard />
    </div>
  );
}

/** Which size of each fast detector runs here: chosen from this computer's hardware, or by the operator. */
function SizesCard() {
  const [h, setH] = useState<any>(null);
  const [err, setErr] = useState<string | null>(null);
  useEffect(() => { api("/hardware").then(setH).catch((e) => setErr(String(e?.message ?? e))); }, []);
  const choose = (family: string, size: string) =>
    api("/hardware/sizes", { method: "PUT", body: { family, size: size || null } }).then((r) => { setH(r); setErr(null); })
      .catch((e) => setErr(e instanceof ApiError ? e.message : String(e)));
  if (!h) return err ? <div className="card"><div className="msg err">{err}</div></div> : null;
  const hw = h.hardware;
  const gpus = (hw.nvidia ?? []).map((g: any) => `${g.name} (${g.memory_gb} GB)`).join(", ");
  return (
    <div className="card">
      <h2 style={{ marginTop: 0 }}>Detector sizes on this computer</h2>
      <p style={{ marginTop: 0 }}>This computer: {hw.cpu_threads} processor threads, {hw.memory_gb ?? "?"} GB memory
        {gpus ? `, ${gpus}` : hw.apple_silicon ? ", Apple GPU" : ", no NVIDIA GPU"}. Detection runs on <b>{h.run_on}</b>;
        training on <b>{h.train_on}</b>. Larger sizes find more but are slower.</p>
      <table className="t">
        <thead><tr><th>Detector</th><th>Size for detection</th><th>Recommended for training</th><th>Installed</th></tr></thead>
        <tbody>{Object.entries(h.families).map(([f, d]: [string, any]) => (
          <tr key={f}><td>{d.title}</td>
            <td>{d.installed.length === 0 ? <span className="muted">not installed</span> :
              <select value={d.chosen && d.installed.includes(d.chosen) ? d.chosen : ""} onChange={(ev) => choose(f, ev.target.value)}>
                <option value="">Automatic ({d.auto} for this computer)</option>
                {d.installed.map((sz: string) => <option key={sz} value={sz}>{sz}</option>)}
              </select>}</td>
            <td>{d.train_auto ?? <span className="muted">–</span>}</td>
            <td className="muted">{d.installed.join(", ") || "–"}</td></tr>))}</tbody>
      </table>
      <p className="muted" style={{ marginBottom: 0 }}>YOLOX runs in missions with the standard detector, and RF-DETR in missions that choose
        it, unless a model is promoted. A promoted or saved model always runs as it is.</p>
      {err && <div className="msg err">{err}</div>}
    </div>
  );
}


const ACCEL: Record<string, string> = { cuda: "NVIDIA GPU (CUDA)", directml: "GPU (DirectML)", cpu: "CPU" };

function System({ sys, msg, backupNow }: { sys: any; msg: any; backupNow: () => void }) {
  const b = sys.backup;
  return (
    <div className="card">
      <h2>This workstation and backups</h2>
      <p style={{ margin: "4px 0" }}>Version {sys.version}. Models run on: {sys.runtime ? <>{ACCEL[sys.runtime.accelerator]}{sys.runtime.notes.length > 0 && <span className="muted"> ({sys.runtime.notes.join("; ")})</span>}</> : <span className="muted">decided when the first job loads the detector</span>}.</p>
      <p style={{ margin: "4px 0" }}>
        {b.folder ? <>Nightly database backup at {b.at} to <span className="mono">{b.folder}</span>. </> : <>No backup folder is set: start the server with <span className="mono">--backup-dir</span>. </>}
        {b.last ? <>Last: {b.last.at} ({b.last.detail.folder ?? b.last.detail.error}).</> : "No backup yet."}
        {b.last_error && <span className="msg err">Last scheduled backup failed: {b.last_error}</span>}
      </p>
      <div className="row"><button disabled={!b.folder} title={!b.folder ? "Set --backup-dir when starting the server to enable backups." : undefined} onClick={backupNow}>Back up now</button>{msg && <span className={`msg ${msg.kind}`} role={msg.kind === "err" ? "alert" : "status"} aria-live="polite">{msg.text}</span>}</div>
      {sys.audit && sys.audit.length > 0 && <details style={{ marginTop: 8 }}><summary>Audit log</summary>
        <ul className="muted">{sys.audit.map((e: any) => <li key={e.id}><span className="mono">{e.at}</span> {e.author}: {e.action.replace(/_/g, " ")}{e.detail.name ? ` “${e.detail.name}”` : ""}{e.detail.folder ? ` ${e.detail.folder}` : ""}</li>)}</ul></details>}
    </div>
  );
}
