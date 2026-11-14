import { useCallback, useEffect, useState } from "react";
import { api, ApiError, post } from "../api";

// Workers on other computers: the operator lets other computers help. A helping computer opens its "help another
// computer" launcher, finds this server on the network by itself and asks; the operator allows it from a dropdown.
// For a computer on another network, a pairing code and a ready-made command take the place of finding.

type Status = {
  enabled: boolean; lan: boolean; port: number; addresses: string[];
  helpers: { name: string; connected: boolean; seen_s_ago: number; segments: number; seconds: number; lost: number;
    info: { cpus?: number; system?: string }; working_on: { mission_id: string; segment: number } | null }[];
  queue: { mission_id: string; version_id: number; segments: number; done: number; waiting: number; open_to_others: boolean }[];
  codes: { id: number; label: string; author: string; created_at: string; revoked_at: string | null }[];
  asking: { id: number; name: string; address: string | null; created_at: string; info: { cpus?: number; system?: string } }[];
  discovery: string | null;
};

// The helping computer runs the "help another computer" launcher its LanePulse install already has.
type Os = "windows" | "mac" | "server" | "python";
const LAUNCHERS: Record<Os, { label: string; run: string; open: string }> = {
  windows: { label: "Windows", run: `"%LOCALAPPDATA%\\TrafficWorkbench\\help-another-computer.cmd"`,
    open: "open LanePulse - help another computer from the Start menu" },
  mac: { label: "Mac", run: `~/LanePulse/"Help another computer.command"`,
    open: "double-click Help another computer in the LanePulse folder of your home folder (after opening the LanePulse app once)" },
  server: { label: "Linux server", run: "lanepulse worker", open: "run lanepulse worker" },
  python: { label: "Python install (pip)", run: "trafficwb-worker", open: "run trafficwb-worker" },
};
const guessOs = (): Os => /Mac/i.test(navigator.platform) ? "mac" : /Win/i.test(navigator.platform) ? "windows" : "server";

/** Embedded on the Settings page, it drops its own page frame and title. */
export default function Workers({ embedded = false }: { embedded?: boolean }) {
  const [st, setSt] = useState<Status | null>(null);
  const [fresh, setFresh] = useState<{ code: string; label: string } | null>(null);
  const [msg, setMsg] = useState<{ kind: string; text: string } | null>(null);
  const [os, setOs] = useState<Os>(() => { try { return (localStorage.getItem("twb.worker-os") as Os) || guessOs(); } catch { return guessOs(); } });
  const load = useCallback(() => api("/workers").then(setSt).catch((e) => setMsg({ kind: "err", text: (e as ApiError).message })), []);
  useEffect(() => { load(); const t = setInterval(load, 3000); return () => clearInterval(t); }, [load]);
  if (!st) return <div className={embedded ? "muted" : "page muted"}>Loading…</div>;

  const act = async (fn: () => Promise<unknown>) => { setMsg(null); try { await fn(); await load(); } catch (e) { setMsg({ kind: "err", text: (e as ApiError).message }); } };
  const host = st.addresses[0] ?? "<this computer's address>";
  const address = `http://${host}:${st.port}`;
  const command = fresh ? `${LAUNCHERS[os].run} --server ${address} --code ${fresh.code}` : "";
  const pickOs = (v: Os) => { setOs(v); try { localStorage.setItem("twb.worker-os", v); } catch { /* kept for this visit only */ } };
  const live = st.codes.filter((c) => !c.revoked_at);
  const addComputer = () => act(async () => {
    const r = await post("/workers/codes", { label: `computer ${st.codes.length + 1}` });
    setFresh({ code: r.code, label: r.label });
  });

  return (
    <div className={`${embedded ? "" : "page "}grid workers`} style={{ gap: 12 }}>
      <div className="page-head"><div>{embedded ? <h2 style={{ margin: 0 }}>Other computers that help</h2> : <h1>Workers</h1>}
        <p>Other computers on your network can help analyse long videos. Each one takes a few minutes of a recording at a
          time, runs the mission's exact model on it and sends back what it found, so an hour of video is done sooner.</p></div></div>
      {msg && <div className={`msg ${msg.kind}`}>{msg.text}</div>}

      <div className="card grid" style={{ gap: 10 }}>
        <div className="row">
          <label className="row">Other computers may help
            <select aria-label="Other computers may help" value={st.enabled ? "on" : "off"}
              onChange={(e) => act(() => api("/workers/switch", { method: "PUT", body: { enabled: e.target.value === "on" } }))}>
              <option value="off">Off: this computer works alone</option>
              <option value="on">On: computers you allow may help</option>
            </select></label>
          <span style={{ flex: 1 }} />
          <button disabled={!st.enabled} onClick={addComputer}
            title="For a computer on another network or a VPN, which cannot find this server by itself">Make a pairing code</button>
        </div>
        {st.enabled && <div className="row join-how">
          <span>To add a computer running</span>
          <select aria-label="The other computer runs" value={os} onChange={(e) => pickOs(e.target.value as Os)}>
            {(Object.keys(LAUNCHERS) as Os[]).map((k) => <option key={k} value={k}>{LAUNCHERS[k].label}</option>)}
          </select>
          <span>with LanePulse installed: on it, <strong>{LAUNCHERS[os].open}</strong>. It finds this server and asks to help here.</span>
        </div>}
        {st.lan && st.discovery && st.discovery !== "on" && <div className="msg warn">Other computers cannot find this
          server by themselves ({st.discovery}). Use a pairing code instead.</div>}
        {!st.lan && <div className="msg warn">This server accepts connections from this computer only. Start it with network
          access (<span className="mono">trafficwb-web serve --lan</span>) so other computers can reach it.</div>}
        {fresh && <div className="msg ok grid" style={{ gap: 6 }}>
          <div>Pairing code for <strong>{fresh.label}</strong>: <span className="mono big-code">{fresh.code}</span>
            <span className="muted"> (shown once)</span></div>
          <div>On that computer ({LAUNCHERS[os].label}), run:</div>
          <div className="row"><code className="mono cmd">{command}</code>
            <button onClick={() => navigator.clipboard?.writeText(command)}>Copy</button></div>
          <div className="muted small">This server's address is <span className="mono">{address}</span>{" "}
            <button onClick={() => navigator.clipboard?.writeText(address)}>Copy address</button></div>
        </div>}
        <p className="muted" style={{ margin: 0 }}>Only segments of video missions are shared. A helping computer reads just the part
          of the recording it works on and keeps nothing afterwards. If it stops, its part goes back to the queue after two
          minutes. A computer needs SAM 3.1 installed to help missions that use it.</p>
      </div>

      {st.asking.length > 0 && <div className="card asking">
        <h2>Asking to help</h2>
        <table className="t"><thead><tr><th>Computer</th><th>From</th><th>System</th><th>Asked</th><th>Answer</th></tr></thead>
          <tbody>{st.asking.map((a) => (
            <tr key={a.id}>
              <td className="mono">{a.name}</td><td className="mono muted">{a.address ?? ""}</td>
              <td className="muted small">{a.info.cpus ? `${a.info.cpus} cores · ` : ""}{a.info.system ?? ""}</td>
              <td className="mono muted">{a.created_at.replace("T", " ").replace("Z", "").slice(0, 16)}</td>
              <td><select aria-label={`Answer ${a.name}`} value=""
                onChange={(e) => e.target.value && act(() => api(`/workers/asking/${a.id}`, { method: "PUT", body: { allow: e.target.value === "allow" } }))}>
                <option value="">Waiting for your answer</option>
                <option value="allow">Allow: it may help</option>
                <option value="refuse">Refuse</option>
              </select></td>
            </tr>))}</tbody></table>
        {!st.enabled && <p className="muted" style={{ marginBottom: 0 }}>A computer you allow starts helping when other computers may help (above).</p>}
      </div>}

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
        <h2>Allowed computers and pairing codes</h2>
        <table className="t"><thead><tr><th>For</th><th>Made</th><th>State</th><th></th></tr></thead>
          <tbody>{st.codes.map((c) => (
            <tr key={c.id}>
              <td>{c.label}</td><td className="mono muted">{c.created_at.replace("T", " ").replace("Z", "")}</td>
              <td>{c.revoked_at ? <span className="muted">stopped</span> : <span className="badge approved">allowed</span>}</td>
              <td>{!c.revoked_at && <button className="danger" onClick={() => act(() => api(`/workers/codes/${c.id}`, { method: "DELETE" }))}>Stop</button>}</td>
            </tr>))}</tbody></table>
        {live.length > 0 && <p className="muted" style={{ marginBottom: 0 }}>Stopping ends that computer's work at once; its segments go back to the queue. A computer that found this server asks again next time it starts.</p>}
      </div>}
    </div>
  );
}
