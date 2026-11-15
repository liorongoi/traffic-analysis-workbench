import { useCallback, useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { api, ApiError, Filter, fmtDuration, Job, plausibleFps, post, REVIEW_LABEL, ResultRow } from "../api";
import { go } from "../router";
import { useTopSlot } from "../components/TopSlot";
import DateRange from "../components/DateRange";
import { ScoreSlider } from "../components/ScoreSlider";
import SearchTab from "./Search";
import EventsTab from "./Events";
import PlatesTab from "./Plates";
import LearningTab, { MissionRunMode } from "./Learning";
import NightReviewTab from "./NightReview";
import SortingTab from "./Sorting";
import { CameraChoice, DETECTOR_LABEL, DetectorChoice, ModelChoice, ParallelChoice, SamInterval, TrackFrames, partsOf, usesSam } from "./Wizard";
import FolderPicker from "../components/FolderPicker";
import { TargetsEditor } from "../components/Targets";
import { Choice, NamePick } from "../components/Pick";

const MIN_VEHICLES = [0, 1, 2, 3, 5, 10, 20, 50];
/** Names to give a camera view. */
const VIEW_NAMES = ["North approach", "South approach", "East approach", "West approach", "Junction", "Roundabout", "Main road",
  "Side road", "Gate", "Entrance", "Exit", "Parking", "Bridge", "Tunnel", "Toll plaza", "Day view", "Night view"];

export type SavedQuery = { filter: Filter; sort: string; desc: boolean };
export const queryKey = (mid: string) => `twb-query-${mid}`;
export function loadQuery(mid: string): SavedQuery {
  try {
    return JSON.parse(sessionStorage.getItem(queryKey(mid)) ?? "") as SavedQuery;
  } catch {
    return { filter: {}, sort: "capture_utc", desc: false };
  }
}

export default function MissionPage({ missionId, tab, extra }: { missionId: string; tab: string; extra?: string }) {
  const [m, setM] = useState<any>(null);
  const [err, setErr] = useState<string | null>(null);
  const [review, setReview] = useState<any>(null);
  const slot = useTopSlot();
  const loadMission = useCallback(() => api(`/missions/${missionId}`).then(setM).catch((e: ApiError) => setErr(e.message)), [missionId]);
  // The mission list carries what waits for review, for the one button that is always the next step. It covers every
  // mission, so it is read when the page opens, after a change and when processing starts or ends, not on every tick.
  const loadReview = useCallback(() => api(`/missions?id=${encodeURIComponent(missionId)}`).then((r) => setReview((r.missions as any[]).find((x) => x.id === missionId)?.review ?? null))
    .catch(() => undefined), [missionId]);
  const load = useCallback(() => { loadReview(); return loadMission(); }, [loadReview, loadMission]);
  useEffect(() => { load(); }, [load]);
  const active = m?.job && ["running", "paused"].includes(m.job.state);
  const jobState = m?.job?.state;
  useEffect(() => { if (jobState) loadReview(); }, [jobState, loadReview]);
  useEffect(() => {
    // Nothing is asked for while the tab is hidden; coming back to it refreshes at once.
    const tick = () => { if (!document.hidden) loadMission(); };
    const t = setInterval(tick, active ? 1500 : 8000);
    document.addEventListener("visibilitychange", tick);
    return () => { clearInterval(t); document.removeEventListener("visibilitychange", tick); };
  }, [loadMission, active]);

  if (err) return <div className="page"><div className="msg err">{err}</div></div>;
  if (!m) return <div className="page muted">Loading…</div>;
  // Sorting indoor from outdoor finds no vehicles: its own tab replaces the vehicle tabs.
  const sorting = m.config.mode === "indoor_outdoor";
  const isVideo = m.config.input_type === "video";
  // Video: Videos (each one with its own progress), Results (the vehicles, a Find box and Export), Learning.
  const tabs = isVideo
    ? [["overview", "Videos"], ["events", "Results"], ...(m.config.mode === "plate_recognition" ? [["plates", "Plates"]] : []), ["learning", "Learning"]]
    : [["overview", "Overview"], ...(m.config.mode === "object_search" ? [["search", "Search"]] : []),
      ...(m.config.mode === "plate_recognition" ? [["plates", "Plates"]] : []),
      ["results", "Results"], ["night", "Night review"], ["views", "Camera views"], ["learning", "Learning"]];
  const job = m.job && ["running", "paused"].includes(m.job.state) ? m.job : null;
  const pct = job && job.total ? Math.round(100 * (job.progress ?? job.done / job.total)) : null;
  const facts = (sorting ? ["Still images", "Indoor and outdoor", m.config.iana_zone]
    : [isVideo ? "Video" : "Still images", cap(m.config.mode.replace(/_/g, " ")), m.config.iana_zone,
      DETECTOR_LABEL[m.config.detector ?? "standard"] ?? "Standard detector"]).join(" · ");
  const strip = (
  <div className="mstrip">
    <a className="mstrip-back" href="#/" aria-label="All missions" title="All missions">‹</a>
    <h1 className="bidi" title={facts}>{m.config.name}</h1>
    {job ? <span className="tag live"><i aria-hidden="true" />{job.state === "paused" ? "Paused" : `Processing${pct != null ? ` ${pct}%` : ""}`}</span>
      : m.config.monitoring_enabled || (isVideo && (m.config.analyse_new || m.config.run_mode === "auto"))
        ? <span className="tag watch"><i aria-hidden="true" />{isVideo ? "New videos start by themselves" : "Watching the folder"}</span> : null}
    <nav className="tabs mstrip-tabs" aria-label="Mission">
      {(sorting ? [tabs[0], ["sort", "Indoor and outdoor"]] : tabs).map(([k, l]) => <a key={k} href={`#/m/${missionId}/${k}`} className={tab === k ? "on" : ""} aria-current={tab === k ? "page" : undefined}>{l}</a>)}
    </nav>
    <span className="grow" />
    {!isVideo && !sorting && review?.unreviewed > 0 && review.first_unreviewed &&
      <a className="btn attn" href={`#/m/${missionId}/i/${review.first_unreviewed}`}>Review {review.unreviewed}</a>}
    <a className={`gear${tab === "setup" ? " on" : ""}`} href={`#/m/${missionId}/setup`} title={`This mission's setup: ${facts}`} aria-label="Mission setup"><GearIcon /></a>
  </div>
  );
  return (
    <div className="page">
      {/* One strip instead of a breadcrumb, a title, a row of facts and a row of tabs, and it sits in the application bar
          itself, so mission pages have one bar and the work below gets the room. */}
      {slot ? createPortal(strip, slot) : strip}
      {tab === "overview" && (isVideo ? <VideosTab m={m} reload={load} /> : <Overview m={m} reload={load} />)}
      {tab === "sort" && sorting && <SortingTab m={m} reload={load} />}
      {tab === "results" && !isVideo && <Results m={m} reload={load} />}
      {tab === "night" && !isVideo && <NightReviewTab missionId={missionId} />}
      {tab === "views" && !isVideo && <Views missionId={missionId} />}
      {(tab === "setup" || tab === "settings") && <MissionSetup m={m} reload={load} />}
      {tab === "learning" && <LearningTab missionId={missionId} job={m.job} reload={load} />}
      {tab === "plates" && m.config.mode === "plate_recognition" && <PlatesTab missionId={missionId} />}
      {tab === "events" && isVideo && <EventsTab missionId={missionId} extra={extra ?? null}
        find={m.config.mode !== "plate_recognition" ? <SearchTab missionId={missionId} runId={null} embedded /> : null} />}
      {tab === "search" && (isVideo ? <EventsTab missionId={missionId} extra={null} find={<SearchTab missionId={missionId} runId={extra ? Number(extra) : null} embedded />} />
        : <SearchTab missionId={missionId} runId={extra ? Number(extra) : null} />)}
    </div>
  );
}

const cap = (s: string) => s.charAt(0).toUpperCase() + s.slice(1);

function JobCard({ job, mid, reload, unit }: { job: Job | null; mid: string; reload: () => void; unit: string }) {
  const [err, setErr] = useState<string | null>(null);
  const act = async (path: string) => {
    setErr(null);
    try { await post(`/missions/${mid}/job${path}`); reload(); } catch (e) { setErr((e as ApiError).message); }
  };
  const active = job && ["running", "paused"].includes(job.state);
  const pct = job && job.total ? Math.round(100 * (job.progress ?? job.done / job.total)) : 0;
  return (
    <div className="card">
      <h2>{job?.kind === "search" ? "Search" : job?.kind === "teach" ? "SAM 3.1 labelling for training" : "Processing"}</h2>
      {!job && <p className="muted">Nothing has run since the server started.</p>}
      {job && (
        <>
          <div className="row" style={{ justifyContent: "space-between" }}>
            <b>{job.state === "paused" ? "Paused" : job.state === "running" ? job.stage : job.state}</b>
            <span className="muted">{job.done} of {job.total} {unit}</span>
          </div>
          <div className="progress" style={{ margin: "8px 0" }}><div style={{ width: `${pct}%` }} /></div>
          <div className="muted">Elapsed {fmtDuration(job.elapsed_s)} · remaining {job.eta_s != null ? `about ${fmtDuration(job.eta_s)}` : "–"} ({job.eta_basis})</div>
          {job.resource_usage && <div className="muted">App usage: {job.resource_usage.process_cpu_percent.toFixed(1)}% CPU · {fmtBytes(job.resource_usage.process_rss_bytes)} RAM · system available {fmtBytes(job.resource_usage.system_memory_available_bytes)} / {fmtBytes(job.resource_usage.system_memory_total_bytes)}</div>}
          {job.error && <div className="msg err">{job.error}</div>}
          {job.result?.scan && job.result.scan.notes.length > 0 && (
            <div className={job.result.scan.queued ? "muted" : "msg err"}>
              <b>Scan: {job.result.scan.queued} file(s) queued.</b>
              <ul style={{ margin: "4px 0 0" }}>{job.result.scan.notes.map((n, i) => <li key={i}>{n}</li>)}</ul>
              {!!job.result.scan.no_capture_time && <a className="btn" href="#/settings/formats">Show the app how your files are named</a>}
            </div>
          )}
        </>
      )}
      <div className="row" style={{ marginTop: 10 }}>
        {!active && <button className="primary" onClick={() => act("")}>Scan and process</button>}
        {job?.state === "running" && <button onClick={() => act("/pause")}>Pause</button>}
        {job?.state === "paused" && <button className="primary" onClick={() => act("/resume")}>Resume</button>}
        {active && <button className="danger" onClick={() => act("/cancel")}>Cancel run</button>}
      </div>
      <p className="muted" style={{ marginBottom: 0 }}>Work continues on the workstation if you close the browser.</p>
      {err && <div className="msg err">{err}</div>}
    </div>
  );
}

/** A video mission's main screen: one list of its videos, each with its own progress bar, position and time left,
 *  and a Review button once done. New videos, and videos still being copied, start by themselves. The image-only
 *  counters, runs and notifications are under "Details". */
function VideosTab({ m, reload }: { m: any; reload: () => void }) {
  const [data, setData] = useState<any>(null);
  const [err, setErr] = useState<string | null>(null);
  const [details, setDetails] = useState(false);
  const job: Job | null = m.job;
  const busy = !!job && ["running", "paused"].includes(job.state);
  const load = useCallback(async (refresh: boolean) => {
    try { setData(await api(`/missions/${m.id}/catalog?refresh=${refresh}`)); }
    catch (e) { setErr((e as ApiError).message); }
  }, [m.id]);
  useEffect(() => { load(true); }, [load]);
  const waitingN = data?.waiting?.length ?? 0;
  useEffect(() => {
    const t = setInterval(() => load(!busy), busy ? 1500 : waitingN ? 5000 : 10000);
    return () => clearInterval(t);
  }, [busy, waitingN, load]);
  const jobState = m.job?.state;
  useEffect(() => { if (jobState && !["running", "paused"].includes(jobState)) load(true); }, [jobState, load]);
  const act = async (fn: () => Promise<unknown>) => { setErr(null); try { await fn(); reload(); load(false); } catch (e) { setErr((e as ApiError).message); } };
  if (!data) return <div className="card muted">{err ?? "Looking for videos…"}</div>;
  const vids: any[] = data.videos;
  const groups: any[] = data.groups ?? [];
  const name = (id: number) => vids.find((v) => v.id === id)?.rel_path.split(/[\\/]/).pop() ?? id;
  const todo = vids.filter((v) => !v.error && !v.analysed);
  const notStarted = todo.filter((v) => ["queued", "new", "cancelled", "failed"].includes(v.state));
  // Overall progress: seconds of video analysed out of the seconds to analyse (the analysed ones and the rest).
  const inRun = vids.filter((v) => !v.error && (v.analysed || v.progress != null));
  const total = inRun.reduce((a, v) => a + (v.duration_s ?? 0), 0);
  const doneS = inRun.reduce((a, v) => a + (v.duration_s ?? 0) * (v.progress ?? 0), 0);
  const overall = total ? doneS / total : 0;
  const running = vids.find((v) => v.state === "running");
  const leftS = running?.eta_s != null ? running.eta_s + todo.filter((v) => v !== running).reduce((a, v) => a + (v.duration_s ?? 0) * (1 - (v.progress ?? 0)), 0)
    * (running.eta_s / Math.max(1, (running.duration_s ?? 0) * (1 - (running.progress ?? 0)))) : null;
  const errors = vids.filter((v) => v.error);
  const events = vids.reduce((a, v) => a + (v.events ?? 0), 0);
  const rows: any[] = [];
  const seen = new Set<number>();
  for (const v of vids) {
    if (seen.has(v.id)) continue;
    const g = v.group_id ? groups.find((x) => x.id === v.group_id) : null;
    if (g) {
      rows.push({ group: g });
      for (const id of g.members) { const p = vids.find((x) => x.id === id); if (p) { rows.push(p); seen.add(id); } }
    } else rows.push(v);
  }
  return (
    <div className="grid" style={{ gap: 12 }}>
      <div className="card vv-sum">
        <div className="vv-sum-n"><b>{busy ? `${Math.round(overall * 100)}%` : todo.length ? `${vids.length - todo.length - errors.length} of ${vids.length - errors.length}` : vids.length}</b>
          <span>{busy ? (job?.state === "paused" ? "paused" : `analysing ${running ? name(running.id) : "…"}`) : todo.length ? "videos analysed" : `video${vids.length === 1 ? "" : "s"}, ${events} vehicle${events === 1 ? "" : "s"}`}</span></div>
        <div className="vv-sum-bar"><div className={`vv-bar${!busy && !todo.length && vids.length ? " done" : ""}`}><i style={{ width: `${Math.round((busy ? overall : vids.length ? (vids.length - todo.length) / vids.length : 0) * 100)}%` }} /></div>
          <span className="muted small">{busy ? (leftS != null ? `about ${fmtDuration(Math.round(leftS))} left` : "working out the time left…")
            : waitingN ? `${waitingN} video${waitingN === 1 ? " is" : "s are"} still being copied; analysis starts when the copy is done`
            : notStarted.length ? `${notStarted.length} video${notStarted.length === 1 ? "" : "s"} not analysed yet` : "Everything here is analysed. New videos in the folder start by themselves."}</span></div>
        <div className="row">
          {job?.state === "running" && <button onClick={() => act(() => post(`/missions/${m.id}/job/pause`))}>Pause</button>}
          {job?.state === "paused" && <button className="primary" onClick={() => act(() => post(`/missions/${m.id}/job/resume`))}>Resume</button>}
          {busy && <button className="danger" onClick={() => act(() => post(`/missions/${m.id}/job/cancel`))}>Stop</button>}
          {!busy && notStarted.length > 0 && <button className="primary" onClick={() => act(() => post(`/missions/${m.id}/catalog/analyse`, { version_ids: notStarted.map((v) => v.id) }))}>
            Analyse {notStarted.length} video{notStarted.length === 1 ? "" : "s"}</button>}
        </div>
      </div>
      {err && <div className="msg err" role="alert">{err}</div>}
      {errors.length > 0 && <div className="msg warn">{errors.length} video{errors.length === 1 ? "" : "s"} cannot be opened: {errors.map((v) => v.rel_path).join(", ")}.
        {" "}LanePulse reads AVI, MP4, MKV, MOV, WMV, TS, WebM and H.264/H.265 files; a camera's own format (for example .dav) may need to be exported from the camera software first.</div>}
      {(data.suggestions ?? []).map((sg: any) => (
        <div key={sg.members.join("-")} className="msg suggest row">
          <span><strong>Parts of one recording?</strong> {sg.members.length} videos look like parts of one ({sg.reason}).</span>
          <span style={{ flex: 1 }} />
          <button disabled={busy} onClick={() => act(() => post(`/missions/${m.id}/catalog/merge`, { version_ids: sg.members }))}>Join them into one</button>
        </div>))}
      <div className="vv-list">
        {rows.map((v) => v.group ? (
          <div key={`g${v.group.id}`} className="vv-group"><strong>{v.group.name}</strong><span className="muted"> · parts of one recording, analysed as one</span>
            {v.group.members.every((id: number) => vids.find((x) => x.id === id)?.analysed) && <a href={`#/m/${m.id}/events/rec-g${v.group.id}`}>Results</a>}
            <button className="linkish" disabled={busy} onClick={() => act(() => api(`/missions/${m.id}/catalog/groups/${v.group.id}`, { method: "DELETE" }))}>Separate</button></div>
        ) : (
          <VideoRow key={v.id} v={v} mid={m.id} busy={busy} paused={job?.state === "paused"}
            again={() => act(() => post(`/missions/${m.id}/catalog/analyse`, { version_ids: [v.id] }))} />))}
        {(data.waiting ?? []).map((w: string) => (
          <div key={`w${w}`} className="vv-row wait"><div className="vv-thumb ph" /><div className="vv-main"><div className="vv-name mono bidi">{w}</div>
            <div className="vv-bar"><i style={{ width: 0 }} /></div><div className="muted small">Still being copied into the folder. It starts by itself when the copy is done.</div></div>
            <div /><div className="vv-state"><span className="badge">copying</span></div></div>))}
        {vids.length === 0 && waitingN === 0 && <div className="card muted">No videos in the input folder yet. Videos added there start by themselves.</div>}
      </div>
      <button className="linkish" style={{ justifySelf: "start" }} onClick={() => setDetails(!details)} aria-expanded={details}>
        {details ? "Hide details" : "Details: input folders, runs and messages"}</button>
      {details && <div className="grid cols-2">
        <SourcesCard m={m} reload={reload} />
        <RunsCard m={m} reload={reload} />
        <div className="card">
          <div className="row" style={{ justifyContent: "space-between" }}><h2>Messages</h2>
            <button onClick={async () => { await post(`/missions/${m.id}/notifications/ack`); reload(); }}>Mark all read</button></div>
          <div style={{ maxHeight: 260, overflowY: "auto" }}>
            {m.notifications.length === 0 && <p className="muted">None.</p>}
            {m.notifications.map((n: any) => (
              <div key={n.id} style={{ padding: "4px 0", borderBottom: "1px solid var(--line)", opacity: n.acknowledged ? 0.6 : 1 }}>
                <span className={`badge ${n.severity === "error" ? "failed" : ""}`}>{n.severity}</span> <span className="bidi">{n.message}</span>
                <div className="muted mono">{n.created_at}</div>
              </div>))}
          </div>
        </div>
      </div>}
    </div>
  );
}

function VideoRow({ v, mid, busy, paused, again }: { v: any; mid: string; busy: boolean; paused: boolean; again: () => void }) {
  const now = v.state === "running";
  const [noThumb, setNoThumb] = useState(false);
  const pct = Math.round((v.progress ?? 0) * 100);
  const fps = plausibleFps(v.nominal_fps);
  return (
    <div className={`vv-row${now ? " now" : ""}${v.error ? " bad" : ""}`}>
      {v.error || noThumb ? <div className="vv-thumb ph" /> : <img loading="lazy" src={`/api/missions/${mid}/recordings/${v.id}/thumb?h=96`} alt="" className="vv-thumb" onError={() => setNoThumb(true)} />}
      <div className="vv-main">
        <div className="vv-name mono bidi" title={v.rel_path}>{v.part ? <span className="badge part">part {v.part}</span> : null} {v.rel_path}</div>
        <div className={`vv-bar${v.analysed ? " done" : ""}`}><i style={{ width: `${v.analysed ? 100 : pct}%` }} /></div>
        <div className="muted small">
          {v.error ? <span title={v.error}>Cannot be opened: {v.error}</span>
            : now ? <>{fmtDuration(v.position_s ?? 0)} of {fmtDuration(v.duration_s)}{v.eta_s != null ? ` · about ${fmtDuration(v.eta_s)} left` : ""}{paused ? " · paused" : ""}</>
            : <>{fmtDuration(v.duration_s)}{v.width ? ` · ${v.width}×${v.height}` : ""}{fps ? ` · ${Math.round(fps)} fps` : ""}{v.codec ? ` · ${v.codec}` : ""}</>}
        </div>
      </div>
      <div className="vv-num">{v.analysed ? <><b>{v.events}</b><span>vehicle{v.events === 1 ? "" : "s"}</span></> : now ? <b className="live">{pct}%</b> : null}</div>
      <div className="vv-state">
        {v.analysed ? <>
            <a className="btn primary" href={`#/m/${mid}/events/play-${v.id}`}>Review</a>
            <button className="linkish small" disabled={busy} onClick={again} title="Analyse this video again; its newest analysis is the one that counts">Analyse again</button></>
          : v.error ? <span className="badge failed">cannot open</span>
          : now ? <span className="badge live">{paused ? "paused" : "analysing"}</span>
          : v.state === "failed" ? <span className="badge failed" title="Its analysis stopped with an error; see Details">failed</span>
          : v.state === "cancelled" ? <span className="badge">stopped</span>
          : <span className="badge">{busy ? "queued" : "not analysed"}</span>}
      </div>
    </div>
  );
}

function fmtBytes(n: number): string {
  if (n >= 1024 ** 3) return `${(n / 1024 ** 3).toFixed(1)} GB`;
  return `${(n / 1024 ** 2).toFixed(0)} MB`;
}

function Overview({ m, reload }: { m: any; reload: () => void }) {
  const s = m.status;
  const pc = m.processing_counts as Record<string, number>;
  const toggle = async () => {
    await api(`/missions/${m.id}/monitoring`, { method: "PUT", body: { enabled: !m.config.monitoring_enabled } });
    reload();
  };
  // The third field marks figures that show red when they are not zero.
  const stats: [string, number, boolean?][] = [["Files", s.source_files], ["Processed", pc.done ?? 0], ["Queued", (pc.queued ?? 0) + (pc.running ?? 0)],
    ["Failed", pc.failed ?? 0, true], ["Waiting to finish copying", s.pending_stability], ["Missing from source", s.missing], ["Names not readable", s.malformed], ["Damaged files", s.corrupt, true]];
  return (
    <div className="grid" style={{ gap: 14 }}>
      <div className="grid cols-4">
        {stats.map(([l, n, alert]) => <div key={l} className={`card stat${!n ? " zero" : alert ? " alert" : ""}`}><div className="n">{n}</div><div className="l">{l}</div></div>)}
      </div>
      {m.config.input_type === "video" && <CatalogCard m={m} reload={reload} />}
      <div className="grid cols-2">
        <JobCard job={m.job} mid={m.id} reload={reload} unit={m.config.input_type === "video" ? "recordings" : "images"} />
        <SourcesCard m={m} reload={reload} />
        <div className="card">
          <h2>Monitoring</h2>
          {m.config.processing === "continuous" ? (
            <>
              <p>New files in {(m.config.source_dirs ?? [m.config.source_dir]).length > 1 ? "the mission's source folders" : <span className="mono bidi">{m.config.source_dir}</span>} are picked up {m.config.monitoring_enabled ? "automatically" : "only when you start a scan"}.</p>
              <button className={m.config.monitoring_enabled ? "danger" : "primary"} onClick={toggle}>
                {m.config.monitoring_enabled ? "Stop monitoring" : "Start monitoring"}
              </button>
              <p className="muted">Monitoring survives a restart of the application or the workstation.</p>
            </>
          ) : <p className="muted">One-time mission: it processes the files present at its first scan.</p>}
        </div>
      </div>
      <div className="grid cols-2">
        <RunsCard m={m} reload={reload} />
        <div className="card">
          <div className="row" style={{ justifyContent: "space-between" }}><h2>Notifications</h2>
            <button onClick={async () => { await post(`/missions/${m.id}/notifications/ack`); reload(); }}>Mark all read</button></div>
          <div style={{ maxHeight: 260, overflowY: "auto" }}>
            {m.notifications.map((n: any) => (
              <div key={n.id} style={{ padding: "4px 0", borderBottom: "1px solid var(--line)", opacity: n.acknowledged ? 0.6 : 1 }}>
                <span className={`badge ${n.severity === "error" ? "failed" : ""}`}>{n.severity}</span> <span className="bidi">{n.message}</span>
                <div className="muted mono">{n.created_at}</div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

/** What this mission decided before it ran: the detector, how its model learns, where results go, and deleting it.
 *  New missions take these from Settings; this is where one mission is changed. Its input folders stay on the
 *  Overview, because adding a folder is part of the work. */
function MissionSetup({ m, reload }: { m: any; reload: () => void }) {
  const c = m.config;
  const facts = (
      <div className="setup-facts card">
        <h2 style={{ margin: 0 }}>This mission's setup</h2>
        <dl>
          <dt>Kind</dt><dd>{c.input_type === "video" ? "Video" : "Still images"} · {c.input_type === "video" && c.mode === "general_traffic" ? "Count and track vehicles" : cap(c.mode.replace(/_/g, " "))}</dd>
          {c.input_type !== "video" && <><dt>Time zone</dt><dd>{c.iana_zone} ({c.country})</dd></>}
          <dt>Results</dt><dd className="mono bidi">{c.output_dir} ({c.export_formats.join(", ")})</dd>
          <dt>New files</dt><dd>{c.input_type === "video" ? (c.analyse_new || c.run_mode === "auto" ? "New videos start by themselves" : "New videos wait until you press Analyse")
            : c.processing === "continuous" ? "Keeps watching the input folder" : "Analysed once"}</dd>
        </dl>
        <p className="muted" style={{ margin: 0 }}>New missions take their setup from <a href={`#/settings/${c.input_type === "video" ? "video" : "images"}`}>Settings</a>. Changes here apply to this mission only.</p>
      </div>);
  // The facts and the run mode share the top row; the detector gets the full width, its choices side by side.
  return (
    <div className="grid" style={{ gap: 14 }}>
      {c.input_type === "video" ? <>{facts}
        <MissionTargets m={m} reload={reload} />
        {/* Video: the choices came from Settings when the mission was made; they are not asked again, only shown,
            and opened here when this one mission should differ. */}
        <details className="card fold">
          <summary><h2>Detector and camera: {DETECTOR_LABEL[c.detector ?? "standard"] ?? c.detector}, {c.camera === "moving" ? "moving camera" : "fixed camera"}. Change for this mission…</h2></summary>
          <div className="grid" style={{ gap: 14 }}>
            <DetectorCard m={m} reload={reload} />
            <MissionRunMode missionId={m.id} />
          </div>
        </details>
      </> : <>
        <div className="grid cols-2" style={{ alignItems: "stretch" }}>{facts}<MissionRunMode missionId={m.id} /></div>
        <MissionTargets m={m} reload={reload} />
        <DetectorCard m={m} reload={reload} />
      </>}
      <DeleteMission m={m} />
    </div>
  );
}

function GearIcon() {
  return (
    <svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <circle cx="12" cy="12" r="3.2" />
      <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 1 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 1 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 1 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 1 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z" />
    </svg>
  );
}

function SourcesCard({ m, reload }: { m: any; reload: () => void }) {
  const have: string[] = m.config.source_dirs ?? [m.config.source_dir];
  const [add, setAdd] = useState<string[]>([]);
  const [err, setErr] = useState<string | null>(null);
  const [adding, setAdding] = useState(false);
  const save = async () => {
    setErr(null);
    try { await post(`/missions/${m.id}/sources`, { add }); setAdd([]); setAdding(false); reload(); }
    catch (e) { setErr((e as ApiError).message); }
  };
  return (
    <div className="card">
      <h2>Source folders</h2>
      <ul className="folderlist">{have.map((f) => <li key={f}><span className="mono bidi">{f}</span></li>)}</ul>
      <p className="muted">{m.config.include_subdirs ? "Subfolders are included. " : ""}Files are only read, never changed. A folder added here is read at the next scan.</p>
      {!adding ? <button onClick={() => setAdding(true)}>Add folders…</button> : <>
        <FolderPicker multiple label="Folders to add" value={add} onChange={setAdd} placeholder="Type a path or press Browse" />
        <div className="row"><button className="primary" disabled={!add.length} onClick={save}>Add {add.length || ""} folder{add.length === 1 ? "" : "s"}</button>
          <button onClick={() => { setAdding(false); setAdd([]); setErr(null); }}>Cancel</button></div>
      </>}
      {err && <div className="msg err" role="alert">{err}</div>}
    </div>
  );
}

function DetectorCard({ m, reload }: { m: any; reload: () => void }) {
  const [sam, setSam] = useState<any>(null);
  const [value, setValue] = useState<string>(m.config.detector ?? "standard");
  const [err, setErr] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState<any[]>([]);
  useEffect(() => { api("/sam3").then(setSam).catch(() => setSam({ installed: false, reason: "Could not check." })); }, []);
  useEffect(() => { api("/models/saved").then((r) => setSaved(r.saved ?? [])).catch(() => setSaved([])); }, []);
  const setModel = async (v: string | null) => {
    setErr(null);
    try { await api(`/missions/${m.id}/vehicle-model`, { method: "PUT", body: { saved: v } }); reload(); }
    catch (e) { setErr((e as ApiError).message); }
  };
  useEffect(() => { setValue(m.config.detector ?? "standard"); }, [m.config.detector]);
  const busy = m.job && ["running", "paused"].includes(m.job.state);
  const video = m.config.input_type === "video";
  const [every, setEvery] = useState<number>(m.config.sam3_interval_s ?? 1);
  useEffect(() => { setEvery(m.config.sam3_interval_s ?? 1); }, [m.config.sam3_interval_s]);
  const [camera, setCamera] = useState<string>(m.config.camera ?? "fixed");
  useEffect(() => { setCamera(m.config.camera ?? "fixed"); }, [m.config.camera]);
  const [workers, setWorkers] = useState<number>(m.config.video_workers ?? 1);
  useEffect(() => { setWorkers(m.config.video_workers ?? 1); }, [m.config.video_workers]);
  const [minFrames, setMinFrames] = useState<number>(m.config.track_min_frames ?? 0);
  useEffect(() => { setMinFrames(m.config.track_min_frames ?? 0); }, [m.config.track_min_frames]);
  const changed = value !== (m.config.detector ?? "standard") || (video && (every !== (m.config.sam3_interval_s ?? 1) || camera !== (m.config.camera ?? "fixed") || workers !== (m.config.video_workers ?? 1) || minFrames !== (m.config.track_min_frames ?? 0)));
  const saveIt = async () => {
    setErr(null);
    setSaving(true);
    try { await api(`/missions/${m.id}/detector`, { method: "PUT", body: { detector: value, ...(video ? { sam3_interval_s: every, camera, video_workers: workers, track_min_frames: minFrames } : {}) } }); reload(); }
    catch (e) { setErr((e as ApiError).message); }
    finally { setSaving(false); }
  };
  if (m.config.mode === "indoor_outdoor") return <div className="card"><h2>Model</h2><p className="muted">Indoor or outdoor is decided by CLIP ViT-B/32 (OpenAI, MIT licence), installed with LanePulse. It finds no vehicles, so there is no detector to choose. See the <a href={`#/m/${m.id}/sort`}>Indoor and outdoor</a> tab.</p></div>;
  return (
    <div className="card set-wide">
      <h2>Detector</h2>
      <DetectorChoice value={value} onChange={setValue} sam={sam ?? undefined} disabled={busy} legend={false} video={video} />
      {video && usesSam(value) && <SamInterval value={every} onChange={setEvery} both={partsOf(value).fast !== null} disabled={busy} />}
      {video && <CameraChoice value={camera} onChange={setCamera} disabled={busy} />}
      {video && <ParallelChoice value={workers} onChange={setWorkers} disabled={busy} />}
      {video && <TrackFrames value={minFrames || null} onChange={setMinFrames} disabled={busy}
        step={value === "sam3" ? every : 0.2} auto={value === "sam3" ? 3 : 5} />}
      {value !== "sam3" && <ModelChoice saved={saved} value={m.config.vehicle_model ?? null} onChange={setModel} disabled={busy} rfdetr={partsOf(value).fast === "rfdetr"} />}
      {changed && <p className="muted">{video ? "Recordings already processed keep their detections; the change applies to the next run." : "Images already processed keep their detections until you run them again (the button appears once the detector is saved)."} Each detection records the model that made it.</p>}
      {busy && <p className="muted">The detector can be changed when processing has finished.</p>}
      {!video && !changed && <RerunButton m={m} reload={reload} />}
      {changed && <div className="row"><button className="primary" disabled={busy || saving} onClick={saveIt}>{saving ? "Saving…" : value !== (m.config.detector ?? "standard") ? "Use this detector" : "Save"}</button>
        <button onClick={() => { setValue(m.config.detector ?? "standard"); setEvery(m.config.sam3_interval_s ?? 1); setCamera(m.config.camera ?? "fixed"); setWorkers(m.config.video_workers ?? 1); setMinFrames(m.config.track_min_frames ?? 0); }}>Cancel</button></div>}
      {err && <div className="msg err" role="alert">{err}</div>}
    </div>
  );
}

/** What this mission looks for, in a panel of its own: the vehicle classes and any of its own (a pedestrian, a
 *  tractor). New missions take this from Settings; here it is changed for this one mission. */
function MissionTargets({ m, reload }: { m: any; reload: () => void }) {
  const [sam, setSam] = useState<any>(null);
  useEffect(() => { api("/sam3").then(setSam).catch(() => setSam(null)); }, []);
  if (m.config.mode === "indoor_outdoor" || !sam?.vehicle_prompts) return null;
  const busy = !!m.job && ["running", "paused"].includes(m.job.state);
  const det = m.config.detector ?? "standard";
  return (
    <div className="card" id="targets">
      <TargetsEditor words={m.config.sam3_words ?? null} uncounted={m.config.sam3_uncounted ?? null} defaults={sam.vehicle_prompts}
        busy={busy} samOnly={det === "sam3"} standardOnly={!usesSam(det)} standardKnows={sam.standard_knows ?? []}
        note={`Changes apply to the next analysis of this mission. New missions look for what is set in Settings.`}
        onSave={async (words, uncounted) => { await api(`/missions/${m.id}/sam3-words`, { method: "PUT", body: { words, uncounted } }); reload(); }} />
    </div>
  );
}

const COLS: [string, string, string][] = [["filename", "File", ""], ["capture_utc", "Local time", ""], ["view", "View", ""],
  ["vehicle_count", "Vehicles", "num"], ["", "Unresolved at night", "num"], ["", "Found by", ""], ["processing_state", "Processing", ""], ["review_status", "Review", ""]];

function Results({ m, reload }: { m: any; reload: () => void }) {
  const missionId: string = m.id;
  const [q, setQ] = useState<SavedQuery>(() => loadQuery(missionId));
  const [page, setPage] = useState(0);
  const [data, setData] = useState<{ total: number; rows: ResultRow[] } | null>(null);
  const [views, setViews] = useState<any[]>([]);
  const [bulk, setBulk] = useState<any>(null);
  const [msg, setMsg] = useState<{ kind: string; text: string } | null>(null);
  const per = 50;

  const seq = useRef(0);
  const load = useCallback(async () => {
    sessionStorage.setItem(queryKey(missionId), JSON.stringify(q));
    const mine = ++seq.current;  // a slower, older response must not overwrite a newer one
    const r = await post(`/missions/${missionId}/results?sort=${q.sort}&desc=${q.desc}&offset=${page * per}&limit=${per}`, q.filter);
    if (mine === seq.current) setData(r);
  }, [missionId, q, page]);
  useEffect(() => { load(); }, [load]);
  useEffect(() => { api(`/missions/${missionId}/views`).then((r) => setViews(r.views)); }, [missionId]);

  const setF = (k: keyof Filter, v: string) => { setPage(0); setQ({ ...q, filter: { ...q.filter, [k]: v === "" ? undefined : k === "view_id" || k === "min_vehicles" ? Number(v) : v } }); };
  const sortBy = (k: string) => k && setQ((prev) => ({ ...prev, sort: k, desc: prev.sort === k ? !prev.desc : false }));
  const doExport = async () => {
    setMsg(null);
    try {
      const r = await post(`/missions/${missionId}/export`, { filter: q.filter });
      setMsg({ kind: "ok", text: `Exported: ${r.files.join(", ")}` });
    } catch (e) { setMsg({ kind: "err", text: (e as ApiError).message }); }
  };

  return (
    <div className="grid" style={{ gap: 10 }}>
      <FoundBy m={m} reload={reload} />
      <div className="card row">
        <input placeholder="Search file name" value={q.filter.search ?? ""} onChange={(e) => setF("search", e.target.value)} />
        <select value={q.filter.view_id ?? ""} onChange={(e) => setF("view_id", e.target.value)}>
          <option value="">All views</option>{views.map((v) => <option key={v.id} value={v.id}>{v.name}</option>)}
        </select>
        <select value={q.filter.review_status ?? ""} onChange={(e) => setF("review_status", e.target.value)}>
          <option value="">Any review status</option>{Object.entries(REVIEW_LABEL).map(([k, v]) => <option key={k} value={k}>{v}</option>)}
        </select>
        <select value={q.filter.processing_state ?? ""} onChange={(e) => setF("processing_state", e.target.value)}>
          <option value="">Any processing state</option>{["done", "queued", "running", "failed", "cancelled"].map((s) => <option key={s}>{s}</option>)}
        </select>
        <DateRange from={q.filter.local_date_from ?? ""} to={q.filter.local_date_to ?? ""} country={m.config.country}
          near={data?.rows?.[0]?.capture_local?.slice(0, 10) ?? null}
          onChange={(a, b) => { setPage(0); setQ({ ...q, filter: { ...q.filter, local_date_from: a || undefined, local_date_to: b || undefined } }); }} />
        <select aria-label="Vehicles at least" value={q.filter.min_vehicles ?? ""} onChange={(e) => setF("min_vehicles", e.target.value)}>
          <option value="">Any number of vehicles</option>
          {!["", undefined, null, ...MIN_VEHICLES.map(String)].includes(q.filter.min_vehicles as any) && <option value={q.filter.min_vehicles as any}>{q.filter.min_vehicles} or more vehicles</option>}
          {MIN_VEHICLES.map((n) => <option key={n} value={String(n)}>{n} or more vehicle{n === 1 ? "" : "s"}</option>)}
        </select>
        <span style={{ flex: 1 }} />
        <ScoreSlider what="boxes" value={q.filter.min_score ?? 0}
          onChange={(v) => setQ({ ...q, filter: { ...q.filter, min_score: v > 0 ? v : undefined } })} />
        <button onClick={async () => setBulk(await post(`/missions/${missionId}/approve-all/preview`, q.filter))}>Approve all and export…</button>
        <button onClick={doExport}>Export</button>
      </div>
      {msg && <div className={`msg ${msg.kind}`}>{msg.text}</div>}
      {data && data.total === 0 && <div className="card muted">No images match this filter. Clear the filters to see all processed images.</div>}
      {data && data.total > 0 && (
        <>
          <div className="tablewrap results-wrap">
            <table className="t">
              <thead><tr><th style={{ width: 100 }}></th>{COLS.map(([k, l, c]) => <th key={l} className={c} onClick={() => sortBy(k)}>{l}{q.sort === k ? (q.desc ? " ▼" : " ▲") : ""}</th>)}<th>Notes</th></tr></thead>
              <tbody>
                {data.rows.map((r) => (
                  <tr key={r.source_version_id} className="click" onClick={() => go(`/m/${missionId}/i/${r.source_version_id}`)}>
                    <td><img loading="lazy" src={`/api/missions/${missionId}/thumbs/${r.source_version_id}?w=96`} alt="" style={{ height: 40, display: "block", borderRadius: 3 }} /></td>
                    <td className="mono bidi">{r.filename}</td>
                    <td className="mono" title={`UTC ${r.capture_utc}`}>{r.capture_local?.replace("T", " ") ?? ""}</td>
                    <td>{r.view ?? <span className="muted">no view</span>}</td>
                    <td className="num">{r.vehicle_count ?? "–"}</td>
                    <td className="num">{r.candidate_count_max ? `${r.candidate_count}–${r.candidate_count_max}` : r.condition && r.condition !== "day" ? "0" : ""}</td>
                    <td>{r.detector ?? ""}</td>
                    <td>{r.processing_state === "failed" ? <span className="badge failed">failed</span> : r.processing_state}</td>
                    <td><span className={`badge ${r.review_status}`}>{REVIEW_LABEL[r.review_status]}</span></td>
                    <td>{r.condition && r.condition !== "day" && <span className="badge flag">{r.condition}</span>} {r.flags.map((f) => <span key={f} className="badge flag">{f.replace(/_/g, " ")}</span>)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="row" style={{ justifyContent: "space-between" }}>
            <span className="muted">{page * per + 1}–{Math.min(data.total, (page + 1) * per)} of {data.total} images. Counts are per image, not unique vehicles.</span>
            <span className="row"><button disabled={page === 0} onClick={() => setPage(page - 1)}>Previous</button>
              <button disabled={(page + 1) * per >= data.total} onClick={() => setPage(page + 1)}>Next</button></span>
          </div>
        </>
      )}
      {bulk && (
        <div className="dialog-bg" onClick={() => setBulk(null)}>
          <div className="dialog" onClick={(e) => e.stopPropagation()}>
            <h2>Approve all and export</h2>
            <p>This marks <b>{bulk.count}</b> processed, unreviewed image(s) in the current filter as <b>bulk approved</b>, then exports.</p>
            <ul className="muted">
              <li>Scope: {Object.keys(bulk.scope).length ? JSON.stringify(bulk.scope) : "all images"}.</li>
              <li>Bulk approval is recorded as <i>not reviewed by a person</i> and is never used as ground truth.</li>
              <li>Images that arrive or finish processing after this moment are not included.</li>
              {bulk.excluded_not_processed > 0 && <li>{bulk.excluded_not_processed} image(s) are not processed yet and are left out.</li>}
            </ul>
            <div className="row" style={{ justifyContent: "flex-end" }}>
              <button onClick={() => setBulk(null)}>Cancel</button>
              <button className="primary" disabled={bulk.count === 0} onClick={async () => {
                try {
                  await post(`/missions/${missionId}/approve-all/commit`, { token: bulk.token });
                  setBulk(null);
                  await doExport();
                  load();
                } catch (e) { setBulk(null); setMsg({ kind: "err", text: (e as ApiError).message }); }
              }}>Approve {bulk.count} and export</button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

function Views({ missionId }: { missionId: string }) {
  const [data, setData] = useState<{ views: any[]; unassigned: number[] } | null>(null);
  const [names, setNames] = useState<Record<number, string>>({});
  const load = () => api(`/missions/${missionId}/views`).then(setData);
  useEffect(() => { load(); }, [missionId]);
  const act = async (id: number, action: string, body: object = {}) => { await post(`/missions/${missionId}/views/${id}/actions/${action}`, body); load(); };
  if (!data) return <p className="muted">Loading…</p>;
  return (
    <div className="grid" style={{ gap: 12 }}>
      {data.views.length === 0 && <div className="card muted">No camera views yet. Views are proposed when images are processed.</div>}
      {data.views.some((v) => v.night_bands) && <ApplyNightDecisions missionId={missionId} />}
      {data.views.map((v) => (
        <div key={v.id} className="card">
          <div className="row" style={{ justifyContent: "space-between" }}>
            <div className="row">
              <NamePick label={`Name of view ${v.id}`} value={names[v.id] ?? v.name} onChange={(n) => setNames({ ...names, [v.id]: n })}
                suggestions={[v.name, ...VIEW_NAMES, `Camera ${v.id}`]} />
              {names[v.id] && names[v.id] !== v.name && <button onClick={() => act(v.id, "rename", { name: names[v.id] })}>Rename</button>}
              <span className="badge">{v.state.replace("_", " ")}</span>
            </div>
            <a href={`#/m/${missionId}/regions/${v.id}`}><button>Draw regions</button></a>
          </div>
          <p className="muted">{v.size} images · {Object.entries(v.conditions).map(([c, n]) => `${n} ${c}`).join(", ")} · {v.ref_width}×{v.ref_height}</p>
          <NightBandsRow missionId={missionId} view={v} reload={load} />
          {v.state === "needs_review" && <div className="msg warn">The camera seems to have moved. Regions are not applied to new images until you check them.</div>}
          {!v.link_confirmed && (
            <div className="msg warn row" style={{ justifyContent: "space-between" }}>
              <span>Day and night images were joined as one view ({v.link_note}). Check the examples below.</span>
              <button className="primary" onClick={() => act(v.id, "confirm")}>Confirm same view</button>
            </div>
          )}
          {Object.entries(v.representatives as Record<string, number[]>).map(([cond, ids]) => (
            <div key={cond} style={{ marginBottom: 6 }}>
              <div className="muted">{cond}</div>
              <div className="thumbs">{ids.map((id) => <a key={id} href={`#/m/${missionId}/i/${id}`}><img src={`/api/missions/${missionId}/thumbs/${id}?w=200`} alt="" /></a>)}</div>
            </div>
          ))}
          <ViewMembership missionId={missionId} view={v} views={data.views} reload={load} />
          {data.views.length > 1 && (
            <div className="row"><span className="muted">Merge another view into this one:</span>
              <select defaultValue="" onChange={(e) => e.target.value && act(v.id, "merge", { absorb: Number(e.target.value) })}>
                <option value="">choose…</option>{data.views.filter((o) => o.id !== v.id).map((o) => <option key={o.id} value={o.id}>{o.name}</option>)}
              </select></div>
          )}
        </div>
      ))}
      {data.unassigned.length > 0 && (
        <div className="card">
          <h2>Not assigned to a view ({data.unassigned.length})</h2>
          <p className="muted">These images did not match any view closely enough (for example rain on the lens). They are analysed without regions.</p>
          <div className="thumbs">{data.unassigned.slice(0, 30).map((id) => <a key={id} href={`#/m/${missionId}/i/${id}`}><img src={`/api/missions/${missionId}/thumbs/${id}?w=200`} alt="" /></a>)}</div>
          {data.views.length > 0 && <UnassignedMembership missionId={missionId} views={data.views} reload={load} />}
        </div>
      )}
    </div>
  );
}

function UnassignedMembership({ missionId, views, reload }: { missionId: string; views: any[]; reload: () => void }) {
  const pageSize = 60;
  const [offset, setOffset] = useState(0);
  const [ids, setIds] = useState<number[]>([]);
  const [total, setTotal] = useState(0);
  const [selected, setSelected] = useState<number[]>([]);
  const [target, setTarget] = useState("");
  const [message, setMessage] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const load = async () => {
    const r = await api(`/missions/${missionId}/views/unassigned/members?offset=${offset}&limit=${pageSize}`);
    setIds(r.ids); setTotal(r.total);
  };
  useEffect(() => { setSelected([]); load(); }, [missionId, offset]);
  const toggle = (id: number) => setSelected((xs) => xs.includes(id) ? xs.filter((x) => x !== id) : [...xs, id]);
  const assign = async () => {
    if (!target || !selected.length) return;
    setBusy(true); setMessage(null);
    try {
      await post(`/missions/${missionId}/views/${target}/actions/reassign`, { version_ids: selected, target_view: Number(target) });
      setSelected([]); setMessage("Selected images assigned to the chosen view."); await load(); reload();
    } catch (e) { setMessage((e as ApiError).message); }
    finally { setBusy(false); }
  };
  return <details style={{ marginTop: 8 }}>
    <summary>Assign ungrouped images</summary>
    <div className="thumbs" style={{ marginTop: 8 }}>
      {ids.map((id) => <label key={id} style={{ display: "inline-grid", gap: 3, cursor: "pointer" }}>
        <img src={`/api/missions/${missionId}/thumbs/${id}?w=200`} alt={`Image ${id}`} />
        <span><input type="checkbox" checked={selected.includes(id)} onChange={() => toggle(id)} /> Image {id}</span>
      </label>)}
    </div>
    <div className="row" style={{ marginTop: 8 }}>
      <button disabled={!offset} onClick={() => { setOffset(Math.max(0, offset - pageSize)); setSelected([]); }}>Previous images</button>
      <span className="muted">{total ? offset + 1 : 0}–{Math.min(offset + pageSize, total)} of {total}</span>
      <button disabled={offset + pageSize >= total} onClick={() => { setOffset(offset + pageSize); setSelected([]); }}>Next images</button>
      <select value={target} onChange={(e) => setTarget(e.target.value)}><option value="">Choose destination view…</option>
        {views.map((v) => <option key={v.id} value={String(v.id)}>{v.name}</option>)}
      </select>
      <button disabled={!selected.length || !target || busy} onClick={assign}>Assign selected</button>
    </div>
    {message && <p className="muted" role="status">{message}</p>}
  </details>;
}

function ViewMembership({ missionId, view, views, reload }: { missionId: string; view: any; views: any[]; reload: () => void }) {
  const pageSize = 60;
  const [offset, setOffset] = useState(0);
  const [ids, setIds] = useState<number[]>([]);
  const [total, setTotal] = useState(0);
  const [selected, setSelected] = useState<number[]>([]);
  const [target, setTarget] = useState<string>("");
  const [name, setName] = useState("");
  const [message, setMessage] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const load = async () => {
    const r = await api(`/missions/${missionId}/views/${view.id}/members?offset=${offset}&limit=${pageSize}`);
    setIds(r.ids); setTotal(r.total);
  };
  useEffect(() => { setSelected([]); load(); }, [missionId, view.id, offset]);
  const changePage = (next: number) => { setOffset(next); setSelected([]); };
  const toggle = (id: number) => setSelected((xs) => xs.includes(id) ? xs.filter((x) => x !== id) : [...xs, id]);
  const act = async (action: "split" | "reassign") => {
    if (!selected.length) return;
    setBusy(true); setMessage(null);
    try {
      const body: any = { version_ids: selected };
      if (action === "split") body.name = name.trim() || null;
      else body.target_view = target === "unassigned" ? null : Number(target);
      await post(`/missions/${missionId}/views/${view.id}/actions/${action}`, body);
      setSelected([]); setName(""); setMessage(action === "split" ? "Selected images moved into a new view." : "Selected images reassigned.");
      await load(); reload();
    } catch (e) { setMessage((e as ApiError).message); }
    finally { setBusy(false); }
  };
  return (
    <details style={{ marginTop: 8 }}>
      <summary>Manage image assignments ({view.size})</summary>
      <p className="muted">Select images from this view to move them to another view or split them into a new view. Assignment changes remain in history.</p>
      <div className="thumbs">
        {ids.map((id) => <label key={id} style={{ display: "inline-grid", gap: 3, cursor: "pointer" }}>
          <img src={`/api/missions/${missionId}/thumbs/${id}?w=200`} alt={`Image ${id}`} />
          <span><input type="checkbox" checked={selected.includes(id)} onChange={() => toggle(id)} /> Image {id}</span>
        </label>)}
      </div>
      <div className="row" style={{ marginTop: 8 }}>
        <button disabled={!offset} onClick={() => changePage(Math.max(0, offset - pageSize))}>Previous images</button>
        <span className="muted">{total ? offset + 1 : 0}–{Math.min(offset + pageSize, total)} of {total}</span>
        <button disabled={offset + pageSize >= total} onClick={() => changePage(offset + pageSize)}>Next images</button>
      </div>
      <div className="row" style={{ marginTop: 8 }}>
        <NamePick optional label="New view name" value={name} onChange={setName} suggestions={VIEW_NAMES} />
        <button disabled={!selected.length || busy} onClick={() => act("split")}>Split selected into new view</button>
        <select value={target} onChange={(e) => setTarget(e.target.value)}>
          <option value="">Choose destination…</option>
          <option value="unassigned">Unassigned</option>
          {views.filter((v) => v.id !== view.id).map((v) => <option key={v.id} value={String(v.id)}>{v.name}</option>)}
        </select>
        <button disabled={!selected.length || !target || busy} onClick={() => act("reassign")}>Reassign selected</button>
      </div>
      {message && <p className="muted" role="status">{message}</p>}
    </details>
  );
}

function ApplyNightDecisions({ missionId }: { missionId: string }) {
  const [msg, setMsg] = useState<{ ok: boolean; text: string } | null>(null);
  const [busy, setBusy] = useState(false);
  const run = async () => {
    setBusy(true); setMsg(null);
    try {
      const r: any = await post(`/missions/${missionId}/night/apply-decisions`);
      setMsg({ ok: true, text: r.reanalysed ? `${r.reanalysed} night image(s) re-checked.` : "Nothing new to apply." });
    } catch (e) { setMsg({ ok: false, text: (e as ApiError).message }); }
    setBusy(false);
  };
  return (
    <div className="card">
      <div className="row" style={{ justifyContent: "space-between" }}>
        <span className="muted">Night candidates learn from your decisions: a place where you said “Not a vehicle” several times is scored down in the view's other images, and a place where you confirmed vehicles is kept. This is applied at every processing run, or now:</span>
        <button onClick={run} disabled={busy}>{busy ? "Re-checking…" : "Apply my night decisions now"}</button>
      </div>
      {msg && <div className={`msg ${msg.ok ? "ok" : "err"}`}>{msg.text}</div>}
    </div>
  );
}

const FAR_PX = [8, 12, 16, 20, 24, 28, 32, 40, 48];
const CLOSE_PX = [40, 50, 60, 70, 80, 100, 120, 150];

function NightBandsRow({ missionId, view, reload }: { missionId: string; view: any; reload: () => void }) {
  const b = view.night_bands;
  const [far, setFar] = useState<string>(String(b?.far_below ?? 24));
  const [close, setClose] = useState<string>(String(b?.close_from ?? 60));
  const [msg, setMsg] = useState<{ ok: boolean; text: string } | null>(null);
  if (!b) return null;
  const save = async () => {
    setMsg(null);
    try {
      const r: any = await api(`/missions/${missionId}/views/${view.id}/bands`, { method: "PUT", body: { far_below: Number(far), close_from: Number(close) } });
      setMsg({ ok: true, text: `Saved. ${r.reanalysed} night image(s) re-analysed.` });
      reload();
    } catch (e) { setMsg({ ok: false, text: (e as ApiError).message }); }
  };
  const changed = Number(far) !== b.far_below || Number(close) !== b.close_from;
  return (
    <div style={{ margin: "6px 0" }}>
      <div className="row">
        <span className="muted">Night distance bands: far while a vehicle would be under</span>
        <Choice label="Far below, pixels wide" value={Number(far)} onChange={(n) => setFar(String(n))} options={FAR_PX} />
        <span className="muted">px wide, close from</span>
        <Choice label="Close from, pixels wide" value={Number(close)} onChange={(n) => setClose(String(n))} options={CLOSE_PX} />
        <span className="muted">px{b.operator_set ? "" : " (defaults)"}</span>
        {changed && <button onClick={save}>Save and re-analyse</button>}
      </div>
      <div className="muted">
        {b.width_map
          ? <>Expected vehicle width = {b.width_map.a.toFixed(3)} × row + {b.width_map.b.toFixed(1)} px, from {b.width_map.samples} daytime cars.
            {b.rows && <> Far above row {Math.round(b.rows.far_until_row)}, close below row {Math.round(b.rows.close_from_row)}.</>}</>
          : "No width map yet: it needs about 20 daytime cars in this view. Until then night lights are not judged by distance."}
      </div>
      {msg && <div className={`msg ${msg.ok ? "ok" : "err"}`}>{msg.text}</div>}
    </div>
  );
}

type RunSummary = {
  id: number; kind: string; kind_label: string; state: string; created_at: string; started_at: string | null; finished_at: string | null;
  detector: { key: string; label: string; saved_model: string | null }; model: string | null;
  images: number; done: number; failed: number; detections: number; by_class: Record<string, number>; by_model: Record<string, number>; shown: number;
};

const plural = (n: number, w: string) => `${n} ${w}${n === 1 ? "" : "s"}`;

/** Every run keeps its own detections, so runs with different detectors can be compared here. */
function RunsCard({ m, reload }: { m: any; reload: () => void }) {
  const runs: RunSummary[] = m.runs ?? [];
  const stills = m.config.input_type !== "video" && m.config.mode !== "indoor_outdoor";
  return (
    <div className="card">
      <div className="row" style={{ justifyContent: "space-between" }}>
        <h2>Runs</h2>
        {stills && <RerunButton m={m} reload={reload} small />}
      </div>
      {runs.length === 0 ? <p className="muted">Nothing has run yet.</p> : (
        <div className="tablewrap">
          <table className="t"><thead><tr><th>Run</th><th>Detector</th><th className="num">Images</th><th>Found</th><th>State</th><th>Finished (UTC)</th></tr></thead>
            <tbody>{runs.map((r) => (
              <tr key={r.id}>
                <td>{r.id}<div className="muted">{r.kind_label}</div></td>
                <td title={r.model ? `Model loaded: ${r.model}` : undefined}>{m.config.mode === "indoor_outdoor" ? "CLIP (indoor or outdoor)" : r.detector.label}
                  {r.shown > 0 && stills && <div><span className="badge approved" title="The Results tab shows what this run found on these images">shown on {plural(r.shown, "image")}</span></div>}</td>
                <td className="num">{r.done}{r.images !== r.done ? <span className="muted"> of {r.images}</span> : null}</td>
                <td>{r.detections ? <>{r.detections} box{r.detections === 1 ? "" : "es"}<div className="muted">{Object.entries(r.by_class).map(([c, n]) => `${n} ${c}`).join(", ")}</div>
                  {Object.keys(r.by_model).length > 1 && <div className="muted">{Object.entries(r.by_model).map(([k, n]) => `${k}: ${n}`).join(" · ")}</div>}</>
                  : <span className="muted">–</span>}</td>
                <td>{r.state.replace(/_/g, " ")}{r.failed > 0 && <div className="muted">{r.failed} failed</div>}</td>
                <td className="mono">{r.finished_at?.replace("T", " ").slice(0, 16) ?? ""}</td>
              </tr>))}</tbody></table>
        </div>
      )}
      {stills && runs.length > 1 && <p className="muted" style={{ marginBottom: 0 }}>Each run keeps what it found. The Results tab shows every image's latest finished run; the export names the run and the model of every box.</p>}
    </div>
  );
}

/** Above the results: which detector found the vehicles shown, and running the images again with another one. */
function FoundBy({ m, reload }: { m: any; reload: () => void }) {
  if (m.config.mode === "indoor_outdoor") return null;
  const runs: RunSummary[] = m.runs ?? [];
  const shown = runs.filter((r) => r.shown > 0);
  const earlier = runs.filter((r) => r.shown === 0 && r.detections > 0).length;
  const job = m.job && ["running", "paused"].includes(m.job.state);
  return (
    <div className="card row" style={{ justifyContent: "space-between" }}>
      <span>
        {shown.length === 0 ? <span className="muted">No image has finished processing yet.</span> : <>Vehicles shown were found by{" "}
          {shown.map((r, i) => <span key={r.id}>{i > 0 && (i === shown.length - 1 ? " and " : ", ")}<b>{r.detector.label}</b> (run {r.id}{shown.length > 1 ? `, ${plural(r.shown, "image")}` : ""})</span>)}.</>}
        {earlier > 0 && <span className="muted"> {plural(earlier, "earlier run")} {earlier === 1 ? "is" : "are"} kept: compare them under Runs on the <a href={`#/m/${m.id}/overview`}>Overview</a>.</span>}
        {job && <span className="muted"> Processing now: results switch to the new run image by image as it finishes them.</span>}
      </span>
      <RerunButton m={m} reload={reload} />
    </div>
  );
}

/** Run every processed image again, with the mission's detector or another one. Nothing is overwritten. */
function RerunButton({ m, reload }: { m: any; reload: () => void; small?: boolean }) {
  const [open, setOpen] = useState(false);
  const [value, setValue] = useState<string>(m.config.detector ?? "standard");
  const [sam, setSam] = useState<any>(null);
  const [err, setErr] = useState<string | null>(null);
  const [going, setGoing] = useState(false);
  const pc = (m.processing_counts ?? {}) as Record<string, number>;
  const n = (pc.done ?? 0) + (pc.failed ?? 0) + (pc.cancelled ?? 0);
  const busy = !!(m.job && ["running", "paused"].includes(m.job.state));
  useEffect(() => { if (open && !sam) api("/sam3").then(setSam).catch(() => setSam({ installed: false, reason: "Could not check." })); }, [open, sam]);
  if (m.config.input_type === "video" || m.config.mode === "indoor_outdoor") return null;
  const current = m.config.detector ?? "standard";
  const start = async () => {
    setErr(null);
    setGoing(true);
    try { await post(`/missions/${m.id}/rerun`, { detector: value }); setOpen(false); reload(); }
    catch (e) { setErr((e as ApiError).message); }
    finally { setGoing(false); }
  };
  return (
    <>
      <button disabled={busy || n === 0}
        title={busy ? "Processing is running. Run again when it has finished." : n === 0 ? "Nothing has been processed yet." : undefined}
        onClick={() => { setValue(current); setErr(null); setOpen(true); }}>Run again with another detector…</button>
      {open && createPortal(
        <div className="dialog-bg" onClick={() => setOpen(false)}>
          <div className="dialog" onClick={(e) => e.stopPropagation()} style={{ maxWidth: 720, maxHeight: "90vh", overflowY: "auto" }}>
            <h2>Run the images again</h2>
            <p>All {plural(n, "processed image")} of <b className="bidi">{m.config.name}</b> are analysed again with the detector you tick. It also becomes this mission's detector for new images.</p>
            <DetectorChoice value={value} onChange={setValue} sam={sam ?? undefined} legend={false} />
            <ul className="muted">
              <li>Nothing is overwritten. The earlier run keeps its boxes, and the Runs list on the Overview compares the runs side by side.</li>
              <li>The Results tab switches to the new run image by image as it finishes. Until then each image shows what the earlier run found.</li>
              <li>Vehicles you drew by hand carry over. Corrections to the earlier detector's boxes stay with the earlier run.</li>
            </ul>
            {err && <div className="msg err" role="alert">{err}</div>}
            <div className="row" style={{ justifyContent: "flex-end" }}>
              <button onClick={() => setOpen(false)}>Cancel</button>
              <button className="primary" disabled={going} onClick={start}>{going ? "Starting…" : `Run ${plural(n, "image")} with ${DETECTOR_LABEL[value]?.split(" (")[0] ?? value}`}</button>
            </div>
          </div>
        </div>
      , document.body)}
    </>
  );
}

function DeleteMission({ m }: { m: any }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="card">
      <div className="row" style={{ justifyContent: "space-between" }}>
        <span className="muted">Deleting a mission removes its database, crops and clips. Source images and exported files stay where they are. Administrators only.</span>
        <button className="danger" onClick={() => setOpen(true)}>Delete mission…</button>
      </div>
      {open && <DeleteMissionDialog id={m.id} name={m.config.name} source={m.config.source_dir} output={m.config.output_dir}
        onClose={() => setOpen(false)} onDeleted={() => go("/")} />}
    </div>
  );
}

/** The one place a mission is deleted from: its setup page and the missions list both open it. */
export function DeleteMissionDialog({ id, name, source, output, onClose, onDeleted }:
  { id: string; name: string; source?: string; output?: string; onClose: () => void; onDeleted: () => void }) {
  const [sure, setSure] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [going, setGoing] = useState(false);
  return createPortal(
    <div className="dialog-bg" onClick={onClose}>
      <div className="dialog" onClick={(e) => e.stopPropagation()}>
        <h2>Delete mission</h2>
        <p>This permanently removes the analysis results, corrections, reviews, datasets and evidence of <b className="bidi">{name}</b>. It cannot be undone except from a backup. The deletion is recorded in the audit log.</p>
        <p className="muted">Not touched: the source images{source ? <> in <span className="mono bidi">{source}</span></> : null} and exported files{output ? <> in <span className="mono bidi">{output}</span></> : null}.</p>
        <label className="row"><input type="checkbox" checked={sure} onChange={(e) => setSure(e.target.checked)} />
          <span>Yes, permanently delete <b className="bidi">{name}</b></span></label>
        {err && <div className="msg err" role="alert">{err}</div>}
        <div className="row" style={{ justifyContent: "flex-end" }}>
          <button onClick={onClose}>Cancel</button>
          <button className="danger" disabled={!sure || going} onClick={async () => {
            setGoing(true);
            try { await post(`/missions/${id}/delete`, { confirm_name: name }); onDeleted(); }
            catch (e) { setErr((e as ApiError).message); }
            finally { setGoing(false); }
          }}>Delete permanently</button>
        </div>
      </div>
    </div>, document.body);
}

const STATE_TEXT: Record<string, string> = {
  new: "New", queued: "Not analysed", running: "Analysing", done: "Analysed", failed: "Failed", cancelled: "Stopped",
};

/** Every video in the source folders: tick the ones to analyse (the new ones are ticked already) and run them in one job. */
function CatalogCard({ m, reload }: { m: any; reload: () => void }) {
  const [vids, setVids] = useState<any[] | null>(null);
  const [groups, setGroups] = useState<any[]>([]);
  const [suggestions, setSuggestions] = useState<{ members: number[]; reason: string }[]>([]);
  const [sel, setSel] = useState<Set<number>>(new Set());
  const [err, setErr] = useState<string | null>(null);
  const busy = m.job && ["running", "paused"].includes(m.job.state);
  const load = useCallback(async (refresh: boolean) => {
    try {
      const r = await api(`/missions/${m.id}/catalog?refresh=${refresh}`);
      setVids(r.videos); setGroups(r.groups ?? []); setSuggestions(r.suggestions ?? []);
      setSel((old) => (old.size && refresh === false ? old : new Set(r.videos.filter((v: any) => v.selected).map((v: any) => v.id))));
    } catch (e) { setErr((e as ApiError).message); }
  }, [m.id]);
  useEffect(() => { load(true); }, [load]);
  const jobState = m.job?.state;
  useEffect(() => { if (jobState && !["running", "paused"].includes(jobState)) load(true); }, [jobState, load]);
  useEffect(() => { if (!busy) return; const t = setInterval(() => load(false), 4000); return () => clearInterval(t); }, [busy, load]);
  if (!vids) return <div className="card muted">Looking for videos…</div>;
  const toggle = (id: number) => setSel((s) => { const n = new Set(s); n.has(id) ? n.delete(id) : n.add(id); return n; });
  const all = vids.filter((v) => !v.error);
  const allOn = all.length > 0 && all.every((v) => sel.has(v.id));
  const total = vids.filter((v) => sel.has(v.id)).reduce((a, v) => a + (v.duration_s ?? 0), 0);
  const analyse = async () => {
    setErr(null);
    try { await post(`/missions/${m.id}/catalog/analyse`, { version_ids: [...sel] }); reload(); load(false); }
    catch (e) { setErr((e as ApiError).message); }
  };
  const merge = async (ids: number[]) => {
    setErr(null);
    try { await post(`/missions/${m.id}/catalog/merge`, { version_ids: ids }); load(false); }
    catch (e) { setErr((e as ApiError).message); }
  };
  const separate = async (gid: number) => {
    setErr(null);
    try { await api(`/missions/${m.id}/catalog/groups/${gid}`, { method: "DELETE" }); load(false); }
    catch (e) { setErr((e as ApiError).message); }
  };
  const name = (id: number) => vids.find((v) => v.id === id)?.rel_path.split("/").pop() ?? id;
  const tickedFree = [...sel].filter((id) => !vids.find((v) => v.id === id)?.group_id);
  // Parts of one merged recording are listed together, under a line that names it.
  const rows: any[] = [];
  const seen = new Set<number>();
  for (const v of vids) {
    if (seen.has(v.id)) continue;
    const g = v.group_id ? groups.find((x) => x.id === v.group_id) : null;
    if (g) {
      rows.push({ group: g });
      for (const id of g.members) { const p = vids.find((x) => x.id === id); if (p) { rows.push(p); seen.add(id); } }
    } else rows.push(v);
  }
  return (
    <div className="card catalog">
      <div className="row" style={{ justifyContent: "space-between" }}>
        <h2 style={{ margin: 0 }}>Videos in the input folder</h2>
        <div className="row">
          <span className="muted">{sel.size} of {vids.length} ticked{total ? ` · ${fmtDuration(total)} of video` : ""}</span>
          <button className="primary" disabled={busy || sel.size === 0} onClick={analyse}>
            {busy ? "Analysing…" : `Analyse ${sel.size} video${sel.size === 1 ? "" : "s"}`}</button>
        </div>
      </div>
      {err && <div className="msg err" role="alert">{err}</div>}
      {suggestions.map((sg) => (
        <div key={sg.members.join("-")} className="msg suggest row">
          <span><strong>Parts of one recording?</strong> {sg.members.map(name).join(", ")} <span className="muted">({sg.reason})</span></span>
          <span style={{ flex: 1 }} />
          <button className="primary" disabled={busy} onClick={() => merge(sg.members)}>Merge into one</button>
        </div>))}
      {tickedFree.length >= 2 && (
        <div className="row"><button disabled={busy} onClick={() => merge(tickedFree)}
          title="The ticked videos become parts of one recording: one job, a target crossing from one part into the next counts once, one results tab">
          Merge the {tickedFree.length} ticked videos into one recording</button></div>)}
      {vids.length === 0 ? <p className="muted">No videos in the source folder yet.</p> : (
        <div className="catalog-scroll"><table className="t">
          <thead><tr>
            <th><input type="checkbox" aria-label="Tick all" checked={allOn} disabled={busy}
              onChange={() => setSel(allOn ? new Set() : new Set(all.map((v) => v.id)))} /></th>
            <th></th><th>Video</th><th className="num">Length</th><th>Picture</th><th>Start (UTC)</th><th>Status</th><th className="num">Events</th><th></th>
          </tr></thead>
          <tbody>{rows.map((v) => v.group ? (
            <tr key={`g${v.group.id}`} className="grouprow">
              <td></td><td colSpan={2}><strong>{v.group.name}</strong> <span className="muted">· merged recording, analysed as one</span></td>
              <td className="num mono">{fmtDuration(v.group.members.reduce((a: number, id: number) => a + (vids.find((x) => x.id === id)?.duration_s ?? 0), 0))}</td>
              <td colSpan={3}></td>
              <td className="num">{v.group.members.every((id: number) => vids.find((x) => x.id === id)?.analysed)
                ? <a href={`#/m/${m.id}/events/rec-g${v.group.id}`}>results</a> : ""}</td>
              <td><button className="linkish" disabled={busy} onClick={() => separate(v.group.id)}>Separate</button></td>
            </tr>
          ) : (
            <tr key={v.id} className={sel.has(v.id) ? "on" : ""} onClick={() => !busy && !v.error && toggle(v.id)} style={{ cursor: busy || v.error ? "default" : "pointer" }}>
              <td><input type="checkbox" aria-label={`Tick ${v.rel_path}`} checked={sel.has(v.id)} disabled={busy || !!v.error}
                onClick={(e) => e.stopPropagation()} onChange={() => toggle(v.id)} /></td>
              <td>{!v.error && <img loading="lazy" src={`/api/missions/${m.id}/recordings/${v.id}/thumb?h=96`} alt="" className="thumb" />}</td>
              <td className="mono bidi">{v.part ? <span className="badge part">part {v.part}</span> : null} {v.rel_path}</td>
              <td className="num mono">{fmtDuration(v.duration_s)}</td>
              <td className="muted">{v.width ? `${v.width}×${v.height}` : ""}{plausibleFps(v.nominal_fps) ? ` · ${Math.round(plausibleFps(v.nominal_fps)!)} fps` : ""}</td>
              <td className="mono muted">{v.start_utc ?? "unknown"}</td>
              <td>{v.error ? <span className="badge failed" title={v.error}>cannot decode</span> : <span className={`badge ${v.state}`}>{STATE_TEXT[v.state] ?? v.state}</span>}</td>
              <td className="num">{v.analysed ? <a href={`#/m/${m.id}/events/rec-${v.id}`} onClick={(e) => e.stopPropagation()}>{v.events}</a> : ""}</td>
              <td>{v.analysed && <a className="play-link" href={`#/m/${m.id}/events/play-${v.id}`} onClick={(e) => e.stopPropagation()}
                title="Play with the detections drawn, review and edit them, and approve for learning">▶ Review</a>}</td>
            </tr>))}</tbody>
        </table></div>)}
      <p className="muted" style={{ marginBottom: 0 }}>New videos are ticked for you. Tick an analysed video to analyse it again; its newest analysis is the one that counts.
        Ticking one part of a merged recording analyses all of its parts.</p>
    </div>
  );
}
