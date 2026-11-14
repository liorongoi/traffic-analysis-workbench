import { useCallback, useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { api, ApiError, Filter, fmtDuration, Job, post, REVIEW_LABEL, ResultRow } from "../api";
import { go } from "../router";
import { ScoreSlider } from "../components/ScoreSlider";
import SearchTab from "./Search";
import EventsTab from "./Events";
import PlatesTab from "./Plates";
import LearningTab, { MissionRunMode } from "./Learning";
import NightReviewTab from "./NightReview";
import SortingTab from "./Sorting";
import { CameraChoice, DETECTOR_LABEL, DetectorChoice, ModelChoice, ParallelChoice, SamInterval, TrackFrames, partsOf, usesSam } from "./Wizard";
import FolderPicker from "../components/FolderPicker";
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
  const load = useCallback(() => {
    // The mission list carries what waits for review, for the one button that is always the next step.
    api("/missions").then((r) => setReview((r.missions as any[]).find((x) => x.id === missionId)?.review ?? null)).catch(() => undefined);
    return api(`/missions/${missionId}`).then(setM).catch((e: ApiError) => setErr(e.message));
  }, [missionId]);
  useEffect(() => { load(); }, [load]);
  const active = m?.job && ["running", "paused"].includes(m.job.state);
  useEffect(() => {
    const t = setInterval(load, active ? 1500 : 8000);
    return () => clearInterval(t);
  }, [load, active]);

  if (err) return <div className="page"><div className="msg err">{err}</div></div>;
  if (!m) return <div className="page muted">Loading…</div>;
  // Sorting indoor from outdoor finds no vehicles: its own tab replaces the vehicle tabs.
  const sorting = m.config.mode === "indoor_outdoor";
  const isVideo = m.config.input_type === "video";
  const tabs = [["overview", "Overview"], ...(m.config.mode === "object_search" ? [["search", "Search"]] : []),
    ...(m.config.mode === "plate_recognition" ? [["plates", "Plates"]] : []),
    ...(isVideo ? [["events", "Events"]] : [["results", "Results"], ["night", "Night review"], ["views", "Camera views"]]), ["learning", "Learning"]];
  const job = m.job && ["running", "paused"].includes(m.job.state) ? m.job : null;
  const pct = job && job.total ? Math.round((100 * job.done) / job.total) : null;
  const facts = (sorting ? ["Still images", "Indoor and outdoor", m.config.iana_zone]
    : [isVideo ? "Video" : "Still images", cap(m.config.mode.replace(/_/g, " ")), m.config.iana_zone,
      DETECTOR_LABEL[m.config.detector ?? "standard"] ?? "Standard detector"]).join(" · ");
  return (
    <div className="page">
      {/* One strip instead of a breadcrumb, a title, a row of facts and a row of tabs: the work below gets the room. */}
      <div className="mstrip">
        <a className="mstrip-back" href="#/" aria-label="All missions" title="All missions">‹</a>
        <h1 className="bidi" title={facts}>{m.config.name}</h1>
        {job ? <span className="tag live"><i aria-hidden="true" />{job.state === "paused" ? "Paused" : `Processing${pct != null ? ` ${pct}%` : ""}`}</span>
          : m.config.monitoring_enabled ? <span className="tag watch"><i aria-hidden="true" />Watching the folder</span> : null}
        <nav className="tabs mstrip-tabs" aria-label="Mission">
          {(sorting ? [tabs[0], ["sort", "Indoor and outdoor"]] : tabs).map(([k, l]) => <a key={k} href={`#/m/${missionId}/${k}`} className={tab === k ? "on" : ""} aria-current={tab === k ? "page" : undefined}>{l}</a>)}
        </nav>
        <span className="grow" />
        {!isVideo && !sorting && review?.unreviewed > 0 && review.first_unreviewed &&
          <a className="btn attn" href={`#/m/${missionId}/i/${review.first_unreviewed}`}>Review {review.unreviewed}</a>}
        <a className={`gear${tab === "setup" ? " on" : ""}`} href={`#/m/${missionId}/setup`} title={`This mission's setup: ${facts}`} aria-label="Mission setup"><GearIcon /></a>
      </div>
      {tab === "overview" && <Overview m={m} reload={load} />}
      {tab === "sort" && sorting && <SortingTab m={m} reload={load} />}
      {tab === "results" && !isVideo && <Results m={m} reload={load} />}
      {tab === "night" && !isVideo && <NightReviewTab missionId={missionId} />}
      {tab === "views" && !isVideo && <Views missionId={missionId} />}
      {(tab === "setup" || tab === "settings") && <MissionSetup m={m} reload={load} />}
      {tab === "learning" && <LearningTab missionId={missionId} job={m.job} reload={load} />}
      {tab === "plates" && m.config.mode === "plate_recognition" && <PlatesTab missionId={missionId} />}
      {tab === "events" && isVideo && <EventsTab missionId={missionId} extra={extra ?? null} />}
      {tab === "search" && <SearchTab missionId={missionId} runId={extra ? Number(extra) : null} />}
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
  const pct = job && job.total ? Math.round((100 * job.done) / job.total) : 0;
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
  return (
    <div className="grid" style={{ gap: 14 }}>
      <div className="setup-facts card">
        <h2 style={{ margin: 0 }}>This mission's setup</h2>
        <dl>
          <dt>Kind</dt><dd>{c.input_type === "video" ? "Video" : "Still images"} · {cap(c.mode.replace(/_/g, " "))}</dd>
          <dt>Time zone</dt><dd>{c.iana_zone} ({c.country})</dd>
          <dt>Results</dt><dd className="mono bidi">{c.output_dir} ({c.export_formats.join(", ")})</dd>
          <dt>New files</dt><dd>{c.processing === "continuous" ? "Keeps watching the input folder" : "Analysed once"}</dd>
        </dl>
        <p className="muted" style={{ margin: 0 }}>New missions take their setup from <a href={`#/settings/${c.input_type === "video" ? "video" : "images"}`}>Settings</a>. Changes here apply to this mission only.</p>
      </div>
      <div className="grid cols-2" style={{ alignItems: "start" }}>
        <DetectorCard m={m} reload={reload} />
        <MissionRunMode missionId={m.id} />
      </div>
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
    <div className="card">
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
      {!changed && sam?.vehicle_prompts && (
        <SamWords m={m} defaults={sam.vehicle_prompts} busy={busy} reload={reload} samOnly={m.config.detector === "sam3"} standardOnly={!usesSam(m.config.detector ?? "standard")} standardKnows={sam.standard_knows ?? []} />
      )}
    </div>
  );
}

const CLASS_LABEL: Record<string, string> = { car: "Car", truck: "Truck", bus: "Bus", motorcycle: "Motorcycle", bicycle: "Bicycle" };

/** Road words offered in the drop-down beside each "add a word" box, grouped under the class they belong to. SAM 3.1
 *  has no fixed class list (it looks for whatever phrase it is given), so these are only suggestions. */
type Groups = [string, string[]][];
const SUGGESTED_WORDS: Record<string, Groups> = {
  car: [["Passenger cars", ["car", "taxi", "police car", "station wagon", "suv"]],
    ["Vans and pickups", ["van", "minivan", "pickup truck", "camper van", "motorhome"]],
    ["Cars with trailers", ["car with trailer", "car towing a caravan"]],
    ["Emergency", ["ambulance"]]],
  truck: [["Semi-trailers and full trailers", ["semi-trailer truck", "tractor-trailer", "articulated lorry", "truck with trailer", "car transporter"]],
    ["Rigid trucks", ["truck", "lorry", "box truck", "delivery truck"]],
    ["Special trucks", ["dump truck", "tanker truck", "fire truck", "garbage truck", "tow truck", "snow plough"]]],
  bus: [["Buses", ["bus", "minibus", "coach", "double-decker bus", "articulated bus", "school bus"]]],
  motorcycle: [["Motorcycles", ["motorcycle", "motorbike", "scooter", "moped"]]],
  bicycle: [["Bicycles", ["bicycle", "cyclist", "cargo bike", "electric scooter"]]],
};
/** Extra search words, so typing "hgv", "semi" or "camper" also finds a suggestion. */
const KEYWORDS: Record<string, string[]> = {
  "truck": ["hgv", "heavy"], "lorry": ["hgv", "heavy"], "semi-trailer truck": ["semi", "hgv", "articulated", "heavy"],
  "tractor-trailer": ["semi", "18-wheeler", "hgv", "heavy"], "articulated lorry": ["semi", "hgv", "heavy"],
  "truck with trailer": ["full trailer", "drawbar", "hgv", "heavy"], "car transporter": ["trailer", "hgv"],
  "tanker truck": ["hgv"], "car with trailer": ["trailer", "towing"], "car towing a caravan": ["trailer", "towing", "camper"],
  "camper van": ["camper", "rv"], "motorhome": ["camper", "rv"], "pickup truck": ["pickup"],
  "ambulance": ["emergency"], "police car": ["emergency"], "fire truck": ["emergency"],
  "coach": ["bus"], "minibus": ["bus"], "moped": ["two-wheeler"], "scooter": ["two-wheeler"], "motorcycle": ["two-wheeler", "mc"],
  "bicycle": ["bike", "two-wheeler"], "cyclist": ["bike"], "suv": ["jeep"], "taxi": ["cab"],
};
const matches = (w: string, q: string) => w.includes(q) || (KEYWORDS[w] ?? []).some((k) => k.includes(q));

/** Classes a mission may add, grouped, each with the words to start from. */
const SUGGESTED_CLASSES: [string, [string, string[]][]][] = [
  ["Other road users", [["pedestrian", ["pedestrian", "person"]]]],
  ["Farm and work vehicles", [["tractor", ["tractor"]], ["farm machinery", ["combine harvester", "tractor"]],
    ["construction vehicle", ["excavator", "wheel loader", "road roller"]], ["snow plough", ["snow plough", "snowplow"]]]],
  ["Emergency", [["emergency vehicle", ["ambulance", "fire truck", "police car"]]]],
  ["Trailers on their own", [["trailer", ["trailer", "caravan"]]]],
  ["Rail", [["tram", ["tram"]]]],
  ["Animals", [["animal", ["deer", "moose", "dog", "horse"]]]],
];
const CLASS_WORDS: Record<string, string[]> = Object.fromEntries(SUGGESTED_CLASSES.flatMap(([, cs]) => cs));

const classTitle = (c: string) => CLASS_LABEL[c] ?? c[0].toUpperCase() + c.slice(1);

/** A type-in box: type a leading word and every related suggestion is listed, grouped by kind (a group whose name
 *  matches is shown whole, so "trailer" lists all trailer combinations). Tick one or several and add them together,
 *  or press Enter to add the typed word itself. */
function AddPick({ groups, have, onAdd, disabled, label, prompt, maxLength, wide, targets }:
  { wide?: boolean; targets?: [string, string][]; groups: Groups; have: string[]; onAdd: (ws: string[], target?: string) => void; disabled: boolean; label: string; prompt: string; maxLength: number }) {
  const [text, setText] = useState("");
  const [open, setOpen] = useState(false);
  const [picked, setPicked] = useState<string[]>([]);
  const q = text.trim().toLowerCase();
  const shown: Groups = groups.map(([g, ws]): [string, string[]] => {
    const left = ws.filter((w) => !have.includes(w));
    return [g, !q || g.toLowerCase().includes(q) ? left : left.filter((w) => matches(w, q))];
  }).filter(([, ws]) => ws.length);
  const flat = shown.flatMap(([, ws]) => ws);
  const toggle = (w: string) => setPicked(picked.includes(w) ? picked.filter((x) => x !== w) : [...picked, w]);
  const toggleGroup = (ws: string[]) => {
    const all = ws.every((w) => picked.includes(w));
    setPicked(all ? picked.filter((w) => !ws.includes(w)) : [...picked, ...ws.filter((w) => !picked.includes(w))]);
  };
  const commit = (ws: string[], target?: string) => {
    const clean = ws.map((w) => w.trim()).filter(Boolean);
    if (clean.length) onAdd(clean, target);
    setText(""); setPicked([]); setOpen(false);
  };
  return (
    <div className={wide ? "addpick wide" : "addpick"} onBlur={(e) => { if (!e.currentTarget.contains(e.relatedTarget as Node)) setOpen(false); }}>
      <input aria-label={label} placeholder={prompt} value={text} maxLength={maxLength} disabled={disabled}
        role="combobox" aria-expanded={open} autoComplete="off"
        onFocus={() => setOpen(true)} onChange={(e) => { setText(e.target.value); setOpen(true); }}
        onKeyDown={(e) => {
          if (e.key === "Enter") { e.preventDefault(); if (picked.length) commit(picked); else if (!targets) commit([text]); }
          else if (e.key === "Escape") { setText(""); setPicked([]); setOpen(false); }
        }} />
      {open && !disabled && (shown.length > 0 || q) && (
        <div className="addpick-menu" role="listbox" aria-multiselectable="true" tabIndex={-1}>
          {q && !flat.includes(q) && !picked.length && !targets && (
            <button type="button" className="addpick-own" onMouseDown={(e) => e.preventDefault()} onClick={() => commit([text])}>Add “{text.trim()}”</button>
          )}
          {q && !flat.includes(q) && !picked.length && targets && (
            <div className="addpick-targets">Add “{text.trim()}” to
              {targets.map(([t, l]) => <button type="button" key={t} onMouseDown={(e) => e.preventDefault()} onClick={() => commit([text], t)}>{l}</button>)}
            </div>
          )}
          <div className="addpick-list">
            {shown.map(([g, ws]) => (
              <div key={g}>
                <label className="addpick-group" onMouseDown={(e) => e.preventDefault()}>
                  <input type="checkbox" checked={ws.every((w) => picked.includes(w))} onChange={() => toggleGroup(ws)} /> {g}
                </label>
                {ws.map((w) => (
                  <label key={w} className="addpick-opt" role="option" aria-selected={picked.includes(w)} onMouseDown={(e) => e.preventDefault()}>
                    <input type="checkbox" checked={picked.includes(w)} onChange={() => toggle(w)} /> {w}
                  </label>
                ))}
              </div>
            ))}
          </div>
          {picked.length > 0 && (
            <div className="addpick-foot">
              <button type="button" className="linkish" onMouseDown={(e) => e.preventDefault()} onClick={() => setPicked([])}>Clear</button>
              <button type="button" className="primary" onMouseDown={(e) => e.preventDefault()} onClick={() => commit(picked)}>Add {picked.length}</button>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

/** The words SAM 3.1 is asked for, per class: each word found counts as that class. Classes of the mission's own
 *  (e.g. pedestrian, tractor) can be added, and left out of the vehicle count. Folded to one summary line until opened. */
/** Which detector of the mission finds a class: the standard one knows the vehicle classes and some others (a
 *  person); anything else only SAM 3.1 can find. */
function foundByText(c: string, mine: boolean, knows: string[], samOnly?: boolean, standardOnly?: boolean): string {
  const std = !mine || knows.includes(c);
  if (samOnly) return "SAM 3.1";
  if (standardOnly) return std ? "standard" : "needs SAM 3.1 or Both";
  return std ? "standard + SAM 3.1" : "SAM 3.1 only";
}

function SamWords({ m, defaults, busy, reload, samOnly, standardOnly, standardKnows = [] }: { m: any; defaults: Record<string, string[]>; busy: boolean; reload: () => void; samOnly?: boolean; standardOnly?: boolean; standardKnows?: string[] }) {
  // A vehicle class missing from the saved words keeps its default words, as on the server.
  const saved: Record<string, string[]> = m.config.sam3_words ? { ...defaults, ...m.config.sam3_words } : defaults;
  const savedOut: string[] = m.config.sam3_uncounted ?? [];
  const [open, setOpen] = useState(false);
  const [words, setWords] = useState<Record<string, string[]>>(saved);
  const [out, setOut] = useState<string[]>(savedOut);
  const [err, setErr] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  useEffect(() => { setWords(m.config.sam3_words ? { ...defaults, ...m.config.sam3_words } : defaults); setOut(m.config.sam3_uncounted ?? []); }, [m.config.sam3_words, m.config.sam3_uncounted]);
  const builtIn = Object.keys(defaults);
  const own = Object.keys(words).filter((c) => !builtIn.includes(c));
  const classes = [...builtIn, ...own];
  const used = Object.values(words).flat();
  const key = (w: Record<string, string[]>, o: string[]) => JSON.stringify([Object.keys(w).sort().map((c) => [c, w[c]]), [...o].sort()]);
  const changed = key(words, out) !== key(saved, savedOut);
  const isDefault = key(words, out) === key(defaults, []);
  const norm = (t: string) => t.trim().toLowerCase().replace(/\s+/g, " ");
  const remove = (c: string, w: string) => setWords({ ...words, [c]: (words[c] ?? []).filter((x) => x !== w) });
  const addClass = (ts: string[]) => {
    setErr(null);
    const names = ts.map(norm).filter((n, i, a) => n && a.indexOf(n) === i);
    const dup = names.filter((n) => classes.includes(n));
    if (dup.length) { setErr(`There is already a class called ${dup.join(", ")}.`); return; }
    const next = { ...words };
    for (const n of names) next[n] = CLASS_WORDS[n] ?? [n];
    setWords(next);
    setOut([...out, ...names]);  // a new class is not a vehicle until said so
  };
  const dropClass = (c: string) => {
    const { [c]: _, ...rest } = words;
    setWords(rest);
    setOut(out.filter((x) => x !== c));
  };
  const save = async (body: Record<string, string[]> | null, uncounted: string[]) => {
    setErr(null);
    setSaving(true);
    try { await api(`/missions/${m.id}/sam3-words`, { method: "PUT", body: { words: body, uncounted } }); reload(); setOpen(false); }
    catch (e) { setErr((e as ApiError).message); }
    finally { setSaving(false); }
  };
  // One search for all classes: each suggestion is listed under "Class · group" and lands in that class.
  const searchGroups: Groups = classes.flatMap((c): Groups => (SUGGESTED_WORDS[c] ?? (CLASS_WORDS[c] ? [["Suggested", CLASS_WORDS[c]]] : []))
    .map(([g, ws]) => [`${classTitle(c)} · ${g}`, ws]));
  const wordClass: Record<string, string> = Object.fromEntries(searchGroups.flatMap(([g, ws]) =>
    ws.map((w) => [w, classes.find((c) => g.startsWith(classTitle(c) + " · "))!])));
  const addFound = (ws: string[], target?: string) => {
    const next = { ...words };
    for (const t of ws) {
      const w = norm(t), c = target ?? wordClass[w];
      if (!w || !c || (next[c] ?? []).includes(w)) continue;
      next[c] = [...(next[c] ?? []), w];
    }
    setWords(next);
  };
  const classGroups: Groups = SUGGESTED_CLASSES.map(([g, cs]) => [g, cs.map(([c]) => c)]);
  return (
    <div className="samwords">
      <div className="row samwords-head">
        <h3>{standardOnly ? "What this mission looks for" : "What this mission looks for, and the words SAM 3.1 uses"}</h3>
        <button className="linkish" onClick={() => { if (open && changed) { setWords(saved); setOut(savedOut); } setOpen(!open); }}>
          {open ? (changed ? "Cancel" : "Close") : "Edit"}</button>
      </div>
      {!open && (
        <p className="samwords-sum">
          {classes.filter((c) => (saved[c] ?? []).length).map((c, i) => (
            <span key={c}>{i > 0 && " · "}<b>{classTitle(c)}</b>{out.includes(c) ? " (not counted)" : ""}: {(saved[c] ?? []).join(", ")}</span>
          ))}
        </p>
      )}
      {open && <>
        <div className="samwords-search">
          <AddPick wide groups={searchGroups} have={used} disabled={busy} label="Find words to add"
            prompt="Choose words to add ▾ (typing narrows the list)" maxLength={40}
            targets={classes.map((c) => [c, classTitle(c)])} onAdd={addFound} />
        </div>
        <table className="samwords-table">
          <thead><tr><th>Class</th><th>Search words</th><th>Found by</th><th title="Counted as a vehicle">Counted</th><th /></tr></thead>
          <tbody>
            {classes.map((c) => {
              const mine = own.includes(c);
              return (
                <tr key={c}>
                  <td className="cls">{classTitle(c)}</td>
                  <td>
                    <div className="chips">
                      {(words[c] ?? []).map((w) => (
                        <span key={w} className="wordchip">{w}
                          <button aria-label={`Remove ${w}`} disabled={busy} onClick={() => remove(c, w)}>×</button>
                        </span>
                      ))}
                      {(words[c] ?? []).length === 0 && <span className="muted">not looked for</span>}
                    </div>
                  </td>
                  <td className="small">{foundByText(c, mine, standardKnows, samOnly, standardOnly)}</td>
                  <td className="ctr">
                    <input type="checkbox" aria-label={`Count ${c} as a vehicle`} checked={!out.includes(c)} disabled={busy || !mine}
                      title={mine ? "Count what this class finds as vehicles" : "Vehicle classes are always counted"}
                      onChange={(e) => setOut(e.target.checked ? out.filter((x) => x !== c) : [...out, c])} />
                  </td>
                  <td className="ctr">
                    {mine && <button className="iconbtn" aria-label={`Remove class ${c}`} title="Remove class" disabled={busy} onClick={() => dropClass(c)}>×</button>}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
        <div className="row samwords-foot">
          <AddPick wide groups={classGroups.map(([g, cs]) => [g, cs.filter((c) => !classes.includes(c))])} have={[]} disabled={busy}
            label="Add a class" prompt="+ Choose a class ▾" maxLength={24} onAdd={addClass} />
          <span className="spacer" />
          {!isDefault && <button className="linkish" disabled={busy || saving} onClick={() => save(null, [])}>Reset to defaults</button>}
          {changed && <button className="primary" disabled={busy || saving} onClick={() => save(words, out)}>{saving ? "Saving…" : "Save"}</button>}
        </div>
        <p className="muted small">SAM 3.1 has no fixed class list: choose suggested words from the list (typing only narrows it, or adds a word of your own). Each word adds about half a second per image. Classes you add are reviewed and exported like the others, and are not used to train the standard detector.</p>
      </>}
      {err && <div className="msg err" role="alert">{err}</div>}
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
        <label className="row muted">From <input type="date" value={q.filter.local_date_from ?? ""} onChange={(e) => setF("local_date_from", e.target.value)} /></label>
        <label className="row muted">to <input type="date" value={q.filter.local_date_to ?? ""} onChange={(e) => setF("local_date_to", e.target.value)} /></label>
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
          <div className="tablewrap">
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
function RerunButton({ m, reload, small }: { m: any; reload: () => void; small?: boolean }) {
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
      <button className={small ? undefined : "primary"} disabled={busy || n === 0}
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
        <table className="t">
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
              <td className="muted">{v.width ? `${v.width}×${v.height}` : ""}{v.nominal_fps ? ` · ${Number(v.nominal_fps).toFixed(0)} fps` : ""}</td>
              <td className="mono muted">{v.start_utc ?? "unknown"}</td>
              <td>{v.error ? <span className="badge failed" title={v.error}>cannot decode</span> : <span className={`badge ${v.state}`}>{STATE_TEXT[v.state] ?? v.state}</span>}</td>
              <td className="num">{v.analysed ? <a href={`#/m/${m.id}/events/rec-${v.id}`} onClick={(e) => e.stopPropagation()}>{v.events}</a> : ""}</td>
              <td>{v.analysed && <a className="play-link" href={`#/m/${m.id}/events/play-${v.id}`} onClick={(e) => e.stopPropagation()}
                title="Play with the detections drawn, review and edit them, and approve for learning">▶ Review</a>}</td>
            </tr>))}</tbody>
        </table>)}
      <p className="muted" style={{ marginBottom: 0 }}>New videos are ticked for you. Tick an analysed video to analyse it again; its newest analysis is the one that counts.
        Ticking one part of a merged recording analyses all of its parts.</p>
    </div>
  );
}
