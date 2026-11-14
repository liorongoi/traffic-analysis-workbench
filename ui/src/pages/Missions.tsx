import { useEffect, useState } from "react";
import { api, ApiError, post } from "../api";
import { DeleteMissionDialog } from "./Mission";

type Review = { unreviewed: number; first_unreviewed: number | null; approved: number; processed: number; latest: number | null };
type M = { id: string; name: string; input_type: string; mode: string; processing: string; monitoring: boolean; created_at: string;
  status: Record<string, number>; job: { state: string; done: number; total: number } | null; review: Review | null };

export default function Missions() {
  const [list, setList] = useState<M[] | null>(null);
  const [loadError, setLoadError] = useState("");
  // Setup lives in Settings; the home page only flags what needs the operator from there.
  const [sys, setSys] = useState<any>(null);
  const [asking, setAsking] = useState(0);
  const loadSys = () => {
    api("/system").then(setSys).catch(() => setSys(null));
    api("/workers").then((w) => setAsking(w.asking?.length ?? 0)).catch(() => setAsking(0));
  };
  const refresh = () => api("/missions").then((r) => setList(r.missions)).catch((e) => {
    setList((l) => l ?? []);
    setLoadError(e instanceof Error ? e.message : "Could not load missions.");
  });
  const loadMissions = () => {
    setList(null);
    setLoadError("");
    refresh();
    loadSys();
  };
  useEffect(() => { loadMissions(); }, []);
  // Keep progress bars moving while something is processing.
  const busy = list?.some((m) => m.job && ["running", "paused"].includes(m.job.state));
  useEffect(() => {
    if (!busy) return;
    const t = window.setInterval(refresh, 4000);
    return () => window.clearInterval(t);
  }, [busy]);

  const sum = (f: (m: M) => number) => (list ?? []).reduce((a, m) => a + f(m), 0);
  const waiting = sum((m) => m.review?.unreviewed ?? 0);
  const running = (list ?? []).filter((m) => m.job?.state === "running").length;
  const damaged = sum((m) => (m.status.malformed ?? 0) + (m.status.corrupt ?? 0));
  const notes = (list ?? []).filter((m) => m.status.unacknowledged_notifications > 0);
  return (
    <div className="page">
      <div className="page-head">
        <div>
          <h1>Missions</h1>
          <p>{list && list.length > 0
            ? `${list.length} mission${list.length > 1 ? "s" : ""}. ${running ? `${running} processing, ` : ""}${waiting ? `${waiting} image${waiting > 1 ? "s" : ""} waiting for your review.` : "Nothing waiting for review."}`
            : "Each mission watches one source folder and keeps its own results, reviews and exports."}</p>
        </div>
        <a className="btn primary" href="#/new">New mission</a>
      </div>
      {list === null && <p className="muted" role="status" aria-live="polite">Loading missions…</p>}
      {loadError && <div className="card msg err" role="alert"><b>Could not load missions.</b><p>{loadError}</p><button onClick={loadMissions}>Try again</button></div>}
      {list?.length === 0 && !loadError && <div className="card drop"><h2>No missions yet</h2><p className="muted">Choose the folder with a camera's images or recordings. LanePulse reads it and fills in the rest. Your original files stay untouched.</p><a className="btn primary big" href="#/new">Create your first mission</a></div>}
      {list && list.length > 0 && (
        <>
          <div className="hud-figs">
            <div><div className="n">{sum((m) => m.review?.processed ?? 0).toLocaleString()}</div><div className="l">Images analysed</div></div>
            <div><div className={`n${waiting ? " attn" : " zero"}`}>{waiting.toLocaleString()}</div><div className="l">Waiting for your review</div></div>
            <div><div className={`n${running ? "" : " zero"}`}>{running}</div><div className="l">Missions processing now</div></div>
            <div><div className={`n${damaged ? " attn" : " zero"}`}>{damaged}</div><div className="l">Unreadable or damaged files</div></div>
          </div>
          <div className={`home-grid${notes.length > 0 || asking > 0 || sys?.backup?.last_error ? "" : " solo"}`}>
            <div className="mlist">{list.map((m) => <MissionCard key={m.id} m={m} reload={refresh} />)}</div>
            <div className="home-side">
              {(notes.length > 0 || asking > 0 || sys?.backup?.last_error) && (
                <div className="card">
                  <h2>Needs your attention</h2>
                  <div className="attention">
                    {notes.map((m) => (
                      <div className="item" key={m.id}>
                        <p><b className="bidi">{m.name}</b> has {m.status.unacknowledged_notifications} new notification{m.status.unacknowledged_notifications > 1 ? "s" : ""}.</p>
                        <a className="btn" href={`#/m/${m.id}`}>Read them</a>
                      </div>
                    ))}
                    {asking > 0 && <div className="item">
                      <p>{asking} other computer{asking > 1 ? "s are" : " is"} asking to help with video.</p>
                      <a className="btn" href="#/settings/workers">Answer</a>
                    </div>}
                    {sys?.backup?.last_error && <div className="item">
                      <p>The last nightly backup failed.</p>
                      <a className="btn" href="#/settings/computer">See why</a>
                    </div>}
                  </div>
                </div>
              )}
            </div>
          </div>
        </>
      )}
    </div>
  );
}

/** One mission: its latest frame, how far processing got, and the next thing to do on it, one click away. */
function MissionCard({ m, reload }: { m: M; reload: () => void }) {
  const [msg, setMsg] = useState<{ kind: string; text: string } | null>(null);
  const [busy, setBusy] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const video = m.input_type === "video";
  const r = m.review;
  const job = m.job && ["running", "paused"].includes(m.job.state) ? m.job : null;
  const total = m.status.source_versions;
  const done = job ? job.done : r?.processed ?? total;
  const of = job ? job.total : video ? total : Math.max(total, r?.processed ?? 0);
  const pct = of ? Math.round((100 * done) / of) : 0;
  const act = async (fn: () => Promise<string>) => {
    setBusy(true); setMsg(null);
    try { setMsg({ kind: "ok", text: await fn() }); } catch (e) { setMsg({ kind: "err", text: (e as ApiError).message }); }
    setBusy(false); reload();
  };
  return (
    <div className="card mcard">
      <a className="mthumb" href={`#/m/${m.id}`} tabIndex={-1} aria-hidden="true">
        {r?.latest ? <img src={`/api/missions/${m.id}/thumbs/${r.latest}?w=352`} alt="" loading="lazy" /> : <span className="ph">{video ? "Video" : "No images yet"}</span>}
      </a>
      <div className="mbody">
        <div className="mtitle">
          <h2><a href={`#/m/${m.id}`} className="bidi">{m.name}</a></h2>
          <span className="tag">{video ? "Video" : "Still images"}</span>
          {job ? <span className="tag live"><i aria-hidden="true" />{job.state === "paused" ? "Paused" : "Processing"}</span>
            : m.monitoring ? <span className="tag watch"><i aria-hidden="true" />Watching the folder</span> : null}
        </div>
        <div className="mbar"><div className="progress" role="progressbar" aria-valuenow={pct} aria-valuemin={0} aria-valuemax={100} aria-label="Processed"><div style={{ width: `${pct}%` }} /></div>
          <span className="mono">{done.toLocaleString()} / {of.toLocaleString()} {video ? "recordings" : "analysed"}</span></div>
        <div className="mline">
          {r ? <>{r.approved} approved{r.unreviewed ? <> · <span className="attn">{r.unreviewed} waiting for review</span></> : r.processed ? " · all reviewed" : ""}</>
            : <>{m.mode.replace(/_/g, " ")}</>}
          {m.status.queued_items > 0 && !job && <> · {m.status.queued_items} queued</>}
          {" · "}created <span className="mono">{m.created_at.slice(0, 10)}</span>
        </div>
        {msg && <span className={`msg ${msg.kind}`} role={msg.kind === "err" ? "alert" : "status"}>{msg.text}</span>}
      </div>
      <div className="macts">
        {r && r.unreviewed > 0 && r.first_unreviewed && <a className="btn attn" href={`#/m/${m.id}/i/${r.first_unreviewed}`}>Review {r.unreviewed}</a>}
        {video && <a className="btn" href={`#/m/${m.id}/events`}>Events</a>}
        {!video && !(r && r.unreviewed > 0) && <a className="btn" href={`#/m/${m.id}/results`}>Results</a>}
        {!job && m.status.queued_items > 0 && <button disabled={busy} onClick={() => act(async () => { await post(`/missions/${m.id}/job`); return "Processing started."; })}>Process</button>}
        {!video && (r?.processed ?? 0) > 0 && <button disabled={busy} onClick={() => act(async () => {
          const x = await post(`/missions/${m.id}/export`, { filter: {} });
          return `Exported ${x.files.length} file${x.files.length > 1 ? "s" : ""}.`;
        })}>Export</button>}
        <button className="mdel" disabled={!!job} title={job ? "Cancel processing before deleting this mission." : `Delete ${m.name}`}
          aria-label={`Delete ${m.name}`} onClick={() => setDeleting(true)}>Delete…</button>
      </div>
      {deleting && <DeleteMissionDialog id={m.id} name={m.name} onClose={() => setDeleting(false)}
        onDeleted={() => { setDeleting(false); reload(); }} />}
    </div>
  );
}

