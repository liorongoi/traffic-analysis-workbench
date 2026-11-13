import { useCallback, useEffect, useState } from "react";
import { api, ApiError, post } from "../api";

// Workers on other computers: the operator lets other computers help, makes a pairing code, and runs one command on
// each helping computer. Nothing to type here: the code and the command are made for you, ready to copy.

type Status = {
  enabled: boolean; lan: boolean; port: number; addresses: string[];
  helpers: { name: string; connected: boolean; seen_s_ago: number; segments: number; seconds: number; lost: number;
    info: { cpus?: number; system?: string }; working_on: { mission_id: string; segment: number } | null }[];
  queue: { mission_id: string; version_id: number; segments: number; done: number; waiting: number; open_to_others: boolean }[];
  codes: { id: number; label: string; author: string; created_at: string; revoked_at: string | null }[];
};

export default function Workers() {
  const [st, setSt] = useState<Status | null>(null);
  const [fresh, setFresh] = useState<{ code: string; label: string } | null>(null);
  const [msg, setMsg] = useState<{ kind: string; text: string } | null>(null);
  const load = useCallback(() => api("/workers").then(setSt).catch((e) => setMsg({ kind: "err", text: (e as ApiError).message })), []);
  useEffect(() => { load(); const t = setInterval(load, 3000); return () => clearInterval(t); }, [load]);
  if (!st) return <div className="page muted">Loading…</div>;

  const act = async (fn: () => Promise<unknown>) => { setMsg(null); try { await fn(); await load(); } catch (e) { setMsg({ kind: "err", text: (e as ApiError).message }); } };
  const host = st.addresses[0] ?? "<this computer's address>";
  const command = fresh ? `trafficwb-worker --server http://${host}:${st.port} --code ${fresh.code}` : "";
  const live = st.codes.filter((c) => !c.revoked_at);
  const addComputer = () => act(async () => {
    const r = await post("/workers/codes", { label: `computer ${st.codes.length + 1}` });
    setFresh({ code: r.code, label: r.label });
  });

  return (
    <div className="page grid workers" style={{ gap: 12 }}>
      <div className="page-head"><h1>Workers</h1>
        <p>Other computers on your network can help analyse long videos. Each one takes a few minutes of a recording at a
          time, runs the mission's exact model on it and sends back what it found, so an hour of video is done sooner.</p></div>
      {msg && <div className={`msg ${msg.kind}`}>{msg.text}</div>}

      <div className="card grid" style={{ gap: 10 }}>
        <div className="row">
          <label className="row">Other computers may help
            <select aria-label="Other computers may help" value={st.enabled ? "on" : "off"}
              onChange={(e) => act(() => api("/workers/switch", { method: "PUT", body: { enabled: e.target.value === "on" } }))}>
              <option value="off">Off: this computer works alone</option>
              <option value="on">On: computers with a pairing code may help</option>
            </select></label>
          <span style={{ flex: 1 }} />
          <button className="primary" disabled={!st.enabled} onClick={addComputer}>Add a computer</button>
        </div>
        {!st.lan && <div className="msg warn">This server accepts connections from this computer only. Start it with network
          access (<span className="mono">trafficwb-web serve --lan</span>) so other computers can reach it.</div>}
        {fresh && <div className="msg ok grid" style={{ gap: 6 }}>
          <div>Pairing code for <strong>{fresh.label}</strong>: <span className="mono big-code">{fresh.code}</span>
            <span className="muted"> (shown once)</span></div>
          <div>On that computer, with LanePulse installed, run:</div>
          <div className="row"><code className="mono cmd">{command}</code>
            <button onClick={() => navigator.clipboard?.writeText(command)}>Copy</button></div>
        </div>}
        <p className="muted" style={{ margin: 0 }}>Only segments of video missions are shared. A helping computer reads just the part
          of the recording it works on and keeps nothing afterwards. If it stops, its part goes back to the queue after two
          minutes. A computer needs SAM 3.1 installed to help missions that use it.</p>
      </div>

      <div className="card">
        <h2>Computers</h2>
        {st.helpers.length === 0 ? <p className="muted">No computer has connected yet.</p> : (
          <table className="t"><thead><tr><th>Computer</th><th>Status</th><th>Now</th><th className="num">Segments done</th>
            <th className="num">Time</th><th className="num">Lost</th><th>System</th></tr></thead>
            <tbody>{st.helpers.map((h) => (
              <tr key={h.name}>
                <td className="mono">{h.name}</td>
                <td><span className={`badge ${h.connected ? "approved" : ""}`}>{h.connected ? "connected" : `last seen ${Math.round(h.seen_s_ago)} s ago`}</span></td>
                <td>{h.working_on ? `segment ${h.working_on.segment + 1}` : <span className="muted">waiting</span>}</td>
                <td className="num">{h.segments}</td>
                <td className="num mono">{Math.round(h.seconds)} s</td>
                <td className="num">{h.lost || ""}</td>
                <td className="muted small">{h.info.cpus ? `${h.info.cpus} cores · ` : ""}{h.info.system ?? ""}</td>
              </tr>))}</tbody></table>)}
      </div>

      {st.queue.length > 0 && <div className="card">
        <h2>Recordings being analysed in segments</h2>
        <table className="t"><thead><tr><th>Recording</th><th className="num">Segments</th><th className="num">Done</th><th className="num">Waiting</th><th>Shared</th></tr></thead>
          <tbody>{st.queue.map((q) => (
            <tr key={`${q.mission_id}-${q.version_id}`}>
              <td><a href={`#/m/${q.mission_id}/events/rec-${q.version_id}`}>recording {q.version_id}</a></td>
              <td className="num">{q.segments}</td><td className="num">{q.done}</td><td className="num">{q.waiting}</td>
              <td>{q.open_to_others ? "with other computers" : "this computer only"}</td>
            </tr>))}</tbody></table>
      </div>}

      {st.codes.length > 0 && <div className="card">
        <h2>Pairing codes</h2>
        <table className="t"><thead><tr><th>For</th><th>Made</th><th>State</th><th></th></tr></thead>
          <tbody>{st.codes.map((c) => (
            <tr key={c.id}>
              <td>{c.label}</td><td className="mono muted">{c.created_at.replace("T", " ").replace("Z", "")}</td>
              <td>{c.revoked_at ? <span className="muted">stopped</span> : <span className="badge approved">valid</span>}</td>
              <td>{!c.revoked_at && <button className="danger" onClick={() => act(() => api(`/workers/codes/${c.id}`, { method: "DELETE" }))}>Stop this code</button>}</td>
            </tr>))}</tbody></table>
        {live.length > 0 && <p className="muted" style={{ marginBottom: 0 }}>Stopping a code ends its computers' work at once; their segments go back to the queue.</p>}
      </div>}
    </div>
  );
}
