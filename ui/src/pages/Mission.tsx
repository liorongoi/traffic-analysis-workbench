import { useCallback, useEffect, useRef, useState } from "react";
import { api, ApiError, Filter, fmtDuration, Job, post, REVIEW_LABEL, ResultRow } from "../api";
import { go } from "../router";
import SearchTab from "./Search";
import EventsTab from "./Events";
import PlatesTab from "./Plates";
import LearningTab from "./Learning";
import NightReviewTab from "./NightReview";
import { DetectorChoice } from "./Wizard";

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
  const load = useCallback(() => api(`/missions/${missionId}`).then(setM).catch((e: ApiError) => setErr(e.message)), [missionId]);
  useEffect(() => { load(); }, [load]);
  const active = m?.job && ["running", "paused"].includes(m.job.state);
  useEffect(() => {
    const t = setInterval(load, active ? 1500 : 8000);
    return () => clearInterval(t);
  }, [load, active]);

  if (err) return <div className="page"><div className="msg err">{err}</div></div>;
  if (!m) return <div className="page muted">Loading…</div>;
  const isVideo = m.config.input_type === "video";
  const tabs = [["overview", "Overview"], ...(m.config.mode === "object_search" ? [["search", "Search"]] : []),
    ...(m.config.mode === "plate_recognition" ? [["plates", "Plates"]] : []),
    ...(isVideo ? [["events", "Events"]] : [["results", "Results"], ["night", "Night review"], ["views", "Camera views"], ["learning", "Learning"]])];
  return (
    <div className="page">
      <div className="page-head">
        <div>
          <nav className="crumbs" aria-label="Breadcrumb"><a href="#/">Missions</a><span aria-hidden="true">/</span><span className="bidi">{m.config.name}</span></nav>
          <h1 className="bidi">{m.config.name}</h1>
          <div className="meta">
            <span>{isVideo ? "Video" : "Still images"}</span>
            <span>{cap(m.config.mode.replace(/_/g, " "))}</span>
            <span>{m.config.iana_zone}</span>
            <span>{m.config.processing === "continuous" ? `Continuous, monitoring ${m.config.monitoring_enabled ? "on" : "off"}` : "One-time"}</span>
            {!isVideo && <span>{m.config.detector === "sam3" ? "SAM 3.1 detector" : m.config.detector === "both" ? "Standard + SAM 3.1 detectors" : "Standard detector"}</span>}
          </div>
        </div>
      </div>
      <nav className="tabs">
        {tabs.map(([k, l]) => <a key={k} href={`#/m/${missionId}/${k}`} className={tab === k ? "on" : ""}>{l}</a>)}
      </nav>
      {tab === "overview" && <Overview m={m} reload={load} />}
      {tab === "results" && !isVideo && <Results missionId={missionId} />}
      {tab === "night" && !isVideo && <NightReviewTab missionId={missionId} />}
      {tab === "views" && !isVideo && <Views missionId={missionId} />}
      {tab === "learning" && !isVideo && <LearningTab missionId={missionId} job={m.job} reload={load} />}
      {tab === "plates" && m.config.mode === "plate_recognition" && <PlatesTab missionId={missionId} />}
      {tab === "events" && isVideo && <EventsTab missionId={missionId} eventId={extra ? Number(extra) : null} />}
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
      <h2>{job?.kind === "search" ? "Search" : "Processing"}</h2>
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
      <div className="grid cols-2">
        <JobCard job={m.job} mid={m.id} reload={reload} unit={m.config.input_type === "video" ? "recordings" : "images"} />
        <div className="card">
          <h2>Monitoring</h2>
          {m.config.processing === "continuous" ? (
            <>
              <p>New files in <span className="mono bidi">{m.config.source_dir}</span> are picked up {m.config.monitoring_enabled ? "automatically" : "only when you start a scan"}.</p>
              <button className={m.config.monitoring_enabled ? "danger" : "primary"} onClick={toggle}>
                {m.config.monitoring_enabled ? "Stop monitoring" : "Start monitoring"}
              </button>
              <p className="muted">Monitoring survives a restart of the application or the workstation.</p>
            </>
          ) : <p className="muted">One-time mission: it processes the files present at its first scan.</p>}
        </div>
        {m.config.input_type === "images" && <DetectorCard m={m} reload={reload} />}
      </div>
      <div className="grid cols-2">
        <div className="card">
          <h2>Runs</h2>
          <table className="t"><thead><tr><th>Run</th><th>Kind</th><th>State</th><th>Started (UTC)</th><th>Finished (UTC)</th></tr></thead>
            <tbody>{m.runs.map((r: any) => <tr key={r.id}><td>{r.id}</td><td>{r.kind}</td><td>{r.state}</td><td className="mono">{r.started_at ?? ""}</td><td className="mono">{r.finished_at ?? ""}</td></tr>)}</tbody></table>
        </div>
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
      <DeleteMission m={m} />
    </div>
  );
}

function DetectorCard({ m, reload }: { m: any; reload: () => void }) {
  const [sam, setSam] = useState<any>(null);
  const [value, setValue] = useState<string>(m.config.detector ?? "standard");
  const [err, setErr] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  useEffect(() => { api("/sam3").then(setSam).catch(() => setSam({ installed: false, reason: "Could not check." })); }, []);
  useEffect(() => { setValue(m.config.detector ?? "standard"); }, [m.config.detector]);
  const busy = m.job && ["running", "paused"].includes(m.job.state);
  const changed = value !== (m.config.detector ?? "standard");
  const saveIt = async () => {
    setErr(null);
    setSaving(true);
    try { await api(`/missions/${m.id}/detector`, { method: "PUT", body: { detector: value } }); reload(); }
    catch (e) { setErr((e as ApiError).message); }
    finally { setSaving(false); }
  };
  return (
    <div className="card">
      <h2>Vehicle detector</h2>
      <DetectorChoice value={value} onChange={setValue} sam={sam ?? undefined} disabled={busy} legend={false} />
      {changed && <p className="muted">Images already processed keep their detections; the change applies to the next run. Each detection records the model that made it.</p>}
      {busy && <p className="muted">The detector can be changed when processing has finished.</p>}
      {changed && <div className="row"><button className="primary" disabled={busy || saving} onClick={saveIt}>{saving ? "Saving…" : "Use this detector"}</button>
        <button onClick={() => setValue(m.config.detector ?? "standard")}>Cancel</button></div>}
      {err && <div className="msg err" role="alert">{err}</div>}
      {!changed && ["sam3", "both"].includes(m.config.detector) && sam?.vehicle_prompts && (
        <SamWords m={m} defaults={sam.vehicle_prompts} busy={busy} reload={reload} />
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
    ["Emergency", ["ambulance"]]],
  truck: [["Semi-trailers and full trailers", ["semi-trailer truck", "tractor-trailer", "articulated lorry", "truck with trailer", "car transporter"]],
    ["Rigid trucks", ["truck", "lorry", "box truck", "delivery truck"]],
    ["Special trucks", ["dump truck", "tanker truck", "fire truck", "garbage truck", "tow truck", "snow plough"]]],
  bus: [["Buses", ["bus", "minibus", "coach", "double-decker bus", "articulated bus", "school bus"]]],
  motorcycle: [["Motorcycles", ["motorcycle", "motorbike", "scooter", "moped"]]],
  bicycle: [["Bicycles", ["bicycle", "cyclist", "cargo bike", "electric scooter"]]],
};
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
/** The groups offered for a class: its own first, then the rest under "Other". */
function groupsFor(c: string): Groups {
  const own: Groups = SUGGESTED_WORDS[c] ?? (CLASS_WORDS[c] ? [["Suggested", CLASS_WORDS[c]]] : []);
  const ownWords = new Set(own.flatMap(([, ws]) => ws));
  const others: Groups = Object.entries(SUGGESTED_WORDS).filter(([k]) => k !== c)
    .map(([k, gs]) => [`Other: ${CLASS_LABEL[k] ?? k}`, gs.flatMap(([, ws]) => ws).filter((w) => !ownWords.has(w))]);
  return [...own, ...others];
}

const classTitle = (c: string) => CLASS_LABEL[c] ?? c[0].toUpperCase() + c.slice(1);

/** A type-in box: type any word and press Enter, or pick one of the suggestions, which are grouped by kind and
 *  narrowed to what has been typed. */
function AddPick({ groups, have, onAdd, disabled, label, prompt, maxLength, wide }:
  { wide?: boolean; groups: Groups; have: string[]; onAdd: (w: string) => void; disabled: boolean; label: string; prompt: string; maxLength: number }) {
  const [text, setText] = useState("");
  const [open, setOpen] = useState(false);
  const [hi, setHi] = useState(-1);
  const q = text.trim().toLowerCase();
  const shown: Groups = groups.map(([g, ws]): [string, string[]] => [g, ws.filter((w) => !have.includes(w) && (!q || w.includes(q)))])
    .filter(([, ws]) => ws.length);
  const flat = shown.flatMap(([, ws]) => ws);
  const pick = (w: string) => { if (w.trim()) onAdd(w); setText(""); setHi(-1); };
  return (
    <div className={wide ? "addpick wide" : "addpick"} onBlur={(e) => { if (!e.currentTarget.contains(e.relatedTarget as Node)) setOpen(false); }}>
      <input aria-label={label} placeholder={prompt} value={text} maxLength={maxLength} disabled={disabled}
        role="combobox" aria-expanded={open} autoComplete="off"
        onFocus={() => setOpen(true)} onChange={(e) => { setText(e.target.value); setOpen(true); setHi(-1); }}
        onKeyDown={(e) => {
          if (e.key === "ArrowDown") { e.preventDefault(); setOpen(true); setHi(Math.min(hi + 1, flat.length - 1)); }
          else if (e.key === "ArrowUp") { e.preventDefault(); setHi(Math.max(hi - 1, -1)); }
          else if (e.key === "Enter") { e.preventDefault(); pick(hi >= 0 ? flat[hi] : text); }
          else if (e.key === "Escape") { setText(""); setOpen(false); }
        }} />
      {open && !disabled && (shown.length > 0 || q) && (
        <div className="addpick-menu" role="listbox">
          {q && !flat.includes(q) && <button type="button" className="addpick-own" onMouseDown={(e) => e.preventDefault()} onClick={() => pick(text)}>Add “{text.trim()}”</button>}
          {shown.map(([g, ws]) => (
            <div key={g}>
              <div className="addpick-group">{g}</div>
              {ws.map((w) => (
                <button type="button" key={w} role="option" aria-selected={flat[hi] === w} className={flat[hi] === w ? "on" : ""}
                  onMouseDown={(e) => e.preventDefault()} onClick={() => pick(w)}>{w}</button>
              ))}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

/** The words SAM 3.1 is asked for, per class: each word found counts as that class. Classes of the mission's own
 *  (e.g. pedestrian, tractor) can be added, and left out of the vehicle count. Folded to one summary line until opened. */
function SamWords({ m, defaults, busy, reload }: { m: any; defaults: Record<string, string[]>; busy: boolean; reload: () => void }) {
  const saved: Record<string, string[]> = m.config.sam3_words ?? defaults;
  const savedOut: string[] = m.config.sam3_uncounted ?? [];
  const [open, setOpen] = useState(false);
  const [words, setWords] = useState<Record<string, string[]>>(saved);
  const [out, setOut] = useState<string[]>(savedOut);
  const [err, setErr] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  useEffect(() => { setWords(m.config.sam3_words ?? defaults); setOut(m.config.sam3_uncounted ?? []); }, [m.config.sam3_words, m.config.sam3_uncounted]);
  const builtIn = Object.keys(defaults);
  const own = Object.keys(words).filter((c) => !builtIn.includes(c));
  const classes = [...builtIn, ...own];
  const used = Object.values(words).flat();
  const key = (w: Record<string, string[]>, o: string[]) => JSON.stringify([Object.keys(w).sort().map((c) => [c, w[c]]), [...o].sort()]);
  const changed = key(words, out) !== key(saved, savedOut);
  const isDefault = key(words, out) === key(defaults, []);
  const norm = (t: string) => t.trim().toLowerCase().replace(/\s+/g, " ");
  const addWord = (c: string, t: string) => {
    const w = norm(t);
    if (w && !(words[c] ?? []).includes(w)) setWords({ ...words, [c]: [...(words[c] ?? []), w] });
  };
  const remove = (c: string, w: string) => setWords({ ...words, [c]: (words[c] ?? []).filter((x) => x !== w) });
  const addClass = (t: string) => {
    const name = norm(t);
    setErr(null);
    if (classes.includes(name)) { setErr(`There is already a class called ${name}.`); return; }
    setWords({ ...words, [name]: CLASS_WORDS[name] ?? [name] });
    setOut([...out, name]);  // a new class is not a vehicle until said so
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
  const classGroups: Groups = SUGGESTED_CLASSES.map(([g, cs]) => [g, cs.map(([c]) => c)]);
  return (
    <div className="samwords">
      <div className="row samwords-head">
        <h3>Words SAM 3.1 looks for</h3>
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
        <table className="samwords-table">
          <thead><tr><th>Class</th><th>Search words</th><th title="Counted as a vehicle">Counted</th><th /></tr></thead>
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
                      <AddPick groups={groupsFor(c)} have={used} disabled={busy} label={`Add a word for ${c}`} prompt="+ type a word"
                        maxLength={40} onAdd={(w) => addWord(c, w)} />
                    </div>
                  </td>
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
            label="Add a class" prompt="+ type a class" maxLength={24} onAdd={addClass} />
          <span className="spacer" />
          {!isDefault && <button className="linkish" disabled={busy || saving} onClick={() => save(null, [])}>Reset to defaults</button>}
          {changed && <button className="primary" disabled={busy || saving} onClick={() => save(words, out)}>{saving ? "Saving…" : "Save"}</button>}
        </div>
        <p className="muted small">SAM 3.1 has no fixed class list: pick a suggested word or type your own. Each word adds about half a second per image. Classes you add are reviewed and exported like the others, and are not used to train the standard detector.</p>
      </>}
      {err && <div className="msg err" role="alert">{err}</div>}
    </div>
  );
}

const COLS: [string, string, string][] = [["filename", "File", ""], ["capture_utc", "Local time", ""], ["view", "View", ""],
  ["vehicle_count", "Vehicles", "num"], ["", "Unresolved at night", "num"], ["processing_state", "Processing", ""], ["review_status", "Review", ""]];

function Results({ missionId }: { missionId: string }) {
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
        <label className="row muted">Vehicles ≥ <input type="number" min={0} style={{ width: 64 }} value={q.filter.min_vehicles ?? ""} onChange={(e) => setF("min_vehicles", e.target.value)} /></label>
        <span style={{ flex: 1 }} />
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
              <input value={names[v.id] ?? v.name} onChange={(e) => setNames({ ...names, [v.id]: e.target.value })} style={{ fontWeight: 600 }} />
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
        <input value={name} onChange={(e) => setName(e.target.value)} placeholder="New view name (optional)" />
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
        <input type="number" min={4} value={far} onChange={(e) => setFar(e.target.value)} style={{ width: 64 }} />
        <span className="muted">px wide, close from</span>
        <input type="number" min={5} value={close} onChange={(e) => setClose(e.target.value)} style={{ width: 64 }} />
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

function DeleteMission({ m }: { m: any }) {
  const [open, setOpen] = useState(false);
  const [name, setName] = useState("");
  const [err, setErr] = useState<string | null>(null);
  return (
    <div className="card">
      <div className="row" style={{ justifyContent: "space-between" }}>
        <span className="muted">Deleting a mission removes its database, crops and clips. Source images and exported files stay where they are. Administrators only.</span>
        <button className="danger" onClick={() => { setOpen(true); setName(""); setErr(null); }}>Delete mission…</button>
      </div>
      {open && (
        <div className="dialog-bg" onClick={() => setOpen(false)}>
          <div className="dialog" onClick={(e) => e.stopPropagation()}>
            <h2>Delete mission</h2>
            <p>This permanently removes the analysis results, corrections, reviews, datasets and evidence of <b className="bidi">{m.config.name}</b>. It cannot be undone except from a backup. The deletion is recorded in the audit log.</p>
            <p className="muted">Not touched: the source folder <span className="mono bidi">{m.config.source_dir}</span> and exports in <span className="mono bidi">{m.config.output_dir}</span>.</p>
            <label>Type the mission's name to confirm<input dir="auto" value={name} onChange={(e) => setName(e.target.value)} /></label>
            {err && <div className="msg err">{err}</div>}
            <div className="row" style={{ justifyContent: "flex-end" }}>
              <button onClick={() => setOpen(false)}>Cancel</button>
              <button className="danger" disabled={name.trim() !== m.config.name} onClick={async () => {
                try { await post(`/missions/${m.id}/delete`, { confirm_name: name }); go("/"); }
                catch (e) { setErr((e as ApiError).message); }
              }}>Delete permanently</button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
