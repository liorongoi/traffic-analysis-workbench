import { useCallback, useEffect, useState } from "react";
import { api, ApiError, Box, fmtDuration, Job, post } from "../api";
import { go } from "../router";
import { fmtT } from "./Events";

type Target = { category: string; subtype: string | null; colours: string[]; exclude_colours: string[] };
type Win = { days: string[]; start: string; end: string };
export type Spec = {
  schema: number;
  targets: Target[];
  exclude: { categories: string[]; subtypes: string[] };
  regions: { include: string[]; exclude: string[] };
  time: { windows: Win[]; date_from: string | null; date_to: string | null; conditions: string[] };
  min_count: number;
  uncertain: "list_separately" | "exclude";
  not_understood: string[];
  notes: string[];
};
type Request = {
  id: number; text: string; interpreter: string; spec: Spec; machine_spec: Spec; description: string[];
  confirmed: boolean; confirmed_by: string | null; confirmed_at: string | null; created_at: string;
  runs?: { id: number; state: string; started_at: string; finished_at: string | null; summary: any }[];
};
type Vocab = { categories: string[]; subtypes: string[]; colours: string[]; conditions: string[]; weekdays: string[] };
type Home = { requests: Request[]; region_names: string[]; vocabulary: Vocab; mode: string };
type Hit = {
  id: number; source_version_id: number; filename: string; capture_local: string | null; object_uid: string; event_t?: number | null;
  status: "match" | "possible"; reasons: string[]; target: number; cls: string; box: Box;
  colour: { state: string; values?: string[]; reason?: string | null; algo?: string }; decision: string;
};

const EXAMPLES = ["red pickup trucks", "white vans between 07:00 and 09:00", "buses at night", "trucks, not white, only certain ones"];
const DETECTOR = ["car", "bus", "truck", "motorcycle", "bicycle"];
const SUBTYPE_LABEL: Record<string, string> = { pickup: "pickup", van: "van", suv: "SUV", taxi: "taxi", minibus: "minibus", semi_trailer: "semi-trailer" };

export default function SearchTab({ missionId, runId }: { missionId: string; runId: number | null }) {
  const [home, setHome] = useState<Home | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const load = useCallback(() => api<Home>(`/missions/${missionId}/search`).then(setHome).catch((e: ApiError) => setErr(e.message)), [missionId]);
  useEffect(() => { load(); }, [load]);
  if (err) return <div className="msg err">{err}</div>;
  if (!home) return <div className="muted">Loading…</div>;
  if (home.mode !== "object_search")
    return <div className="msg warn">Object search runs in object search missions only. Create a new mission with the mode "object search" over the same source folder.</div>;
  if (runId) return <RunView missionId={missionId} runId={runId} vocab={home.vocabulary} />;
  return <Ask missionId={missionId} home={home} reload={load} />;
}

function Ask({ missionId, home, reload }: { missionId: string; home: Home; reload: () => void }) {
  const [text, setText] = useState("");
  const [req, setReq] = useState<Request | null>(null);
  const [spec, setSpec] = useState<Spec | null>(null);
  const [ignore, setIgnore] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [job, setJob] = useState<Job | null>(null);
  const [busy, setBusy] = useState(false);

  const ask = async () => {
    setErr(null);
    setBusy(true);
    try {
      const r = await post<Request>(`/missions/${missionId}/search/ask`, { text });
      setReq(r);
      setSpec(structuredClone(r.spec));
      setIgnore(false);
    } catch (e) { setErr((e as ApiError).message); }
    setBusy(false);
  };

  const confirmAndRun = async () => {
    if (!req || !spec) return;
    setErr(null);
    setBusy(true);
    try {
      await post(`/missions/${missionId}/search/${req.id}/confirm`, { spec, ignore_unknown: ignore });
      const j = await post<Job>(`/missions/${missionId}/search/${req.id}/run`);
      setJob(j);
    } catch (e) { setErr((e as ApiError).message); setBusy(false); }
  };

  // Follow the running search, then open its results.
  useEffect(() => {
    if (!job || !["running", "paused"].includes(job.state)) {
      if (job && job.result?.search_run) go(`/m/${missionId}/search/${job.result.search_run}`);
      return;
    }
    const t = setTimeout(async () => {
      const m = await api(`/missions/${missionId}`);
      setJob(m.job);
    }, 700);
    return () => clearTimeout(t);
  }, [job, missionId]);

  return (
    <div className="grid cols-2" style={{ alignItems: "start" }}>
      <div className="grid">
        <div className="card">
          <h2>What are you looking for?</h2>
          <p className="muted">Type in English. Nothing runs until you have checked how the request was understood.</p>
          <textarea rows={3} style={{ width: "100%" }} value={text} onChange={(e) => setText(e.target.value)} placeholder="e.g. red pickup trucks between 17:00 and 05:00"
            onKeyDown={(e) => { if (e.key === "Enter" && (e.ctrlKey || e.metaKey) && text.trim()) ask(); }} />
          <div className="row" style={{ marginTop: 8, justifyContent: "space-between" }}>
            <div className="row">{EXAMPLES.map((x) => <button key={x} className="chip" onClick={() => setText(x)}>{x}</button>)}</div>
            <button className="primary" disabled={!text.trim() || busy} onClick={ask}>Interpret</button>
          </div>
        </div>
        {req && spec && (
          <div className="card">
            <h2>How the request was understood</h2>
            <p className="mono muted" style={{ marginTop: 0 }}>“{req.text}” · interpreter {req.interpreter}</p>
            <ul style={{ marginTop: 0 }}>{req.description.map((d, i) => <li key={i}>{d}</li>)}</ul>
            {spec.notes.length > 0 && <div className="msg warn"><b>Please note</b><ul style={{ margin: "4px 0 0" }}>{spec.notes.map((n, i) => <li key={i}>{n}</li>)}</ul></div>}
            {spec.not_understood.length > 0 && (
              <div className="msg err">
                <b>Not understood</b>
                <ul style={{ margin: "4px 0" }}>{spec.not_understood.map((n, i) => <li key={i}>{n}</li>)}</ul>
                <label className="row"><input type="checkbox" checked={ignore} onChange={(e) => setIgnore(e.target.checked)} /> Ignore these words and run anyway</label>
              </div>
            )}
            <SpecEditor spec={spec} setSpec={setSpec} vocab={home.vocabulary} regionNames={home.region_names} />
            {err && <div className="msg err">{err}</div>}
            {job ? <JobLine job={job} mid={missionId} /> : (
              <div className="row" style={{ marginTop: 10 }}>
                <button className="primary" disabled={busy || spec.targets.length === 0 || (spec.not_understood.length > 0 && !ignore)} onClick={confirmAndRun}>Confirm and search</button>
                <button onClick={() => { setReq(null); setSpec(null); }}>Discard</button>
              </div>
            )}
          </div>
        )}
        {!req && err && <div className="msg err">{err}</div>}
      </div>
      <History requests={home.requests} missionId={missionId} reload={reload} />
    </div>
  );
}

function JobLine({ job, mid }: { job: Job; mid: string }) {
  const pct = job.total ? Math.round((100 * job.done) / job.total) : 0;
  return (
    <div style={{ marginTop: 10 }}>
      <div className="row" style={{ justifyContent: "space-between" }}>
        <b>{job.state === "running" ? "Searching…" : job.state}</b>
        <span className="muted">{job.done} of {job.total} images · {fmtDuration(job.elapsed_s)}</span>
      </div>
      <div className="progress" style={{ margin: "6px 0" }}><div style={{ width: `${pct}%` }} /></div>
      {["running", "paused"].includes(job.state) && <button className="danger" onClick={() => post(`/missions/${mid}/job/cancel`)}>Cancel search</button>}
      {job.error && <div className="msg err">{job.error}</div>}
    </div>
  );
}

function toggle(list: string[], v: string): string[] {
  return list.includes(v) ? list.filter((x) => x !== v) : [...list, v];
}

function Chips({ all, on, set, label }: { all: string[]; on: string[]; set: (v: string[]) => void; label?: (v: string) => string }) {
  return (
    <div className="row" style={{ gap: 4 }}>
      {all.map((v) => (
        <button key={v} type="button" className={`chip ${on.includes(v) ? "on" : ""}`} aria-pressed={on.includes(v)} onClick={() => set(toggle(on, v))}>
          {label ? label(v) : v}
        </button>
      ))}
    </div>
  );
}

export function SpecEditor({ spec, setSpec, vocab, regionNames }: { spec: Spec; setSpec: (s: Spec) => void; vocab: Vocab; regionNames: string[] }) {
  const upd = (fn: (s: Spec) => void) => { const c = structuredClone(spec); fn(c); setSpec(c); };
  return (
    <details open className="spec">
      <summary><b>Edit the search</b></summary>
      <h3 style={{ marginTop: 10 }}>Look for</h3>
      {spec.targets.map((t, i) => (
        <div key={i} className="card" style={{ padding: 10, marginBottom: 8 }}>
          <div className="row">
            <select value={t.category} onChange={(e) => upd((s) => { s.targets[i].category = e.target.value; })}>
              {vocab.categories.map((c) => <option key={c} value={c}>{c === "vehicle" ? "any vehicle" : c}</option>)}
            </select>
            <select value={t.subtype ?? ""} onChange={(e) => upd((s) => { s.targets[i].subtype = e.target.value || null; })}>
              <option value="">no subtype</option>
              {vocab.subtypes.map((c) => <option key={c} value={c}>{SUBTYPE_LABEL[c] ?? c}</option>)}
            </select>
            {spec.targets.length > 1 && <button className="danger" onClick={() => upd((s) => { s.targets.splice(i, 1); })}>Remove</button>}
          </div>
          {t.subtype && <p className="muted" style={{ margin: "6px 0" }}>The detector does not tell a {SUBTYPE_LABEL[t.subtype]} apart, so results are listed as possible.</p>}
          <div style={{ marginTop: 6 }}><span className="muted">Colour (any of)</span><Chips all={vocab.colours} on={t.colours} set={(v) => upd((s) => { s.targets[i].colours = v; s.targets[i].exclude_colours = s.targets[i].exclude_colours.filter((c) => !v.includes(c)); })} /></div>
          <div style={{ marginTop: 6 }}><span className="muted">Not these colours</span><Chips all={vocab.colours} on={t.exclude_colours} set={(v) => upd((s) => { s.targets[i].exclude_colours = v; s.targets[i].colours = s.targets[i].colours.filter((c) => !v.includes(c)); })} /></div>
        </div>
      ))}
      <button onClick={() => upd((s) => { s.targets.push({ category: "vehicle", subtype: null, colours: [], exclude_colours: [] }); })}>Add another kind</button>

      <h3 style={{ marginTop: 14 }}>Leave out</h3>
      <Chips all={DETECTOR} on={spec.exclude.categories} set={(v) => upd((s) => { s.exclude.categories = v; })} />
      <div style={{ marginTop: 6 }}><Chips all={vocab.subtypes} on={spec.exclude.subtypes} label={(v) => SUBTYPE_LABEL[v] ?? v} set={(v) => upd((s) => { s.exclude.subtypes = v; })} /></div>

      <h3 style={{ marginTop: 14 }}>Where</h3>
      {regionNames.length === 0 ? <p className="muted">No named regions in this mission. The views' own analysis regions apply.</p> : (
        <>
          <div><span className="muted">Only inside</span><Chips all={regionNames} on={spec.regions.include} set={(v) => upd((s) => { s.regions.include = v; s.regions.exclude = s.regions.exclude.filter((x) => !v.includes(x)); })} /></div>
          <div style={{ marginTop: 6 }}><span className="muted">Not inside</span><Chips all={regionNames} on={spec.regions.exclude} set={(v) => upd((s) => { s.regions.exclude = v; s.regions.include = s.regions.include.filter((x) => !v.includes(x)); })} /></div>
          <p className="muted">With no region chosen, each view's own analysis regions apply.</p>
        </>
      )}

      <h3 style={{ marginTop: 14 }}>When (local time)</h3>
      {spec.time.windows.map((w, i) => (
        <div key={i} className="row" style={{ marginBottom: 6 }}>
          <Chips all={vocab.weekdays} on={w.days} set={(v) => upd((s) => { s.time.windows[i].days = v.length ? vocab.weekdays.filter((d) => v.includes(d)) : s.time.windows[i].days; })} />
          <input type="time" value={w.start} onChange={(e) => upd((s) => { s.time.windows[i].start = e.target.value; })} aria-label="start" />
          <span>to</span>
          <input value={w.end} style={{ width: 70 }} onChange={(e) => upd((s) => { s.time.windows[i].end = e.target.value; })} aria-label="end (24:00 is midnight)" />
          <button className="danger" onClick={() => upd((s) => { s.time.windows.splice(i, 1); })}>Remove</button>
        </div>
      ))}
      <button onClick={() => upd((s) => { s.time.windows.push({ days: [...vocab.weekdays], start: "07:00", end: "09:00" }); })}>Add a time window</button>
      <span className="muted"> An end at or before the start runs past midnight. 24:00 means midnight.</span>
      <div className="row" style={{ marginTop: 8 }}>
        <label className="row">From <input type="date" value={spec.time.date_from ?? ""} onChange={(e) => upd((s) => { s.time.date_from = e.target.value || null; })} /></label>
        <label className="row">to <input type="date" value={spec.time.date_to ?? ""} onChange={(e) => upd((s) => { s.time.date_to = e.target.value || null; })} /></label>
      </div>
      <div style={{ marginTop: 8 }}><span className="muted">Image condition (measured from the image)</span><Chips all={vocab.conditions} on={spec.time.conditions} set={(v) => upd((s) => { s.time.conditions = v; })} /></div>

      <h3 style={{ marginTop: 14 }}>Results</h3>
      <label className="row">Only images with at least <input type="number" min={1} max={500} style={{ width: 70 }} value={spec.min_count} onChange={(e) => upd((s) => { s.min_count = Math.max(1, Math.min(500, Number(e.target.value) || 1)); })} /> matching objects</label>
      <div className="row" style={{ marginTop: 6 }}>
        <label className="row"><input type="radio" checked={spec.uncertain === "list_separately"} onChange={() => upd((s) => { s.uncertain = "list_separately"; })} /> List uncertain ones separately as possible</label>
        <label className="row"><input type="radio" checked={spec.uncertain === "exclude"} onChange={() => upd((s) => { s.uncertain = "exclude"; })} /> Only certain matches</label>
      </div>
    </details>
  );
}

function History({ requests, missionId }: { requests: Request[]; missionId: string; reload: () => void }) {
  return (
    <div className="card">
      <h2>Earlier searches</h2>
      {requests.length === 0 && <p className="muted">None yet.</p>}
      {requests.map((r) => (
        <div key={r.id} style={{ borderTop: "1px solid var(--line)", padding: "8px 0" }}>
          <div className="row" style={{ justifyContent: "space-between" }}>
            <b className="bidi">“{r.text}”</b>
            <span className="muted mono">{r.created_at.replace("T", " ").replace("Z", " UTC")}</span>
          </div>
          {!r.confirmed && <span className="muted">Not confirmed</span>}
          {(r.runs ?? []).map((x) => (
            <div key={x.id} className="row">
              <a href={`#/m/${missionId}/search/${x.id}`}>Search {x.id}</a>
              <span className="muted">{x.state}{x.summary?.hits_match != null ? ` · ${x.summary.hits_match} matches, ${x.summary.hits_possible} possible in ${x.summary.images_with_hits} images` : ""}</span>
            </div>
          ))}
        </div>
      ))}
    </div>
  );
}

const DECISION_LABEL: Record<string, string> = { correct: "Correct", not_a_match: "Not a match", undecided: "Not checked" };

function RunView({ missionId, runId, vocab }: { missionId: string; runId: number; vocab: Vocab }) {
  const [data, setData] = useState<any>(null);
  const [status, setStatus] = useState("");
  const [decision, setDecision] = useState("");
  const [offset, setOffset] = useState(0);
  const [msg, setMsg] = useState<{ kind: string; text: string } | null>(null);
  const LIMIT = 60;
  const load = useCallback(() => {
    const q = new URLSearchParams({ offset: String(offset), limit: String(LIMIT) });
    if (status) q.set("status", status);
    if (decision) q.set("decision", decision);
    return api(`/missions/${missionId}/search-runs/${runId}?${q}`).then(setData).catch((e: ApiError) => setMsg({ kind: "err", text: e.message }));
  }, [missionId, runId, status, decision, offset]);
  useEffect(() => { load(); }, [load]);

  const decide = async (h: Hit, d: string) => {
    await post(`/missions/${missionId}/search-hits/${h.id}/decision`, { decision: d });
    load();
  };
  const setColour = async (h: Hit, value: string) => {
    await api(`/missions/${missionId}/items/${h.source_version_id}/objects/${h.object_uid}/colour`, { method: "PUT", body: { value: value || null } });
    setMsg({ kind: "ok", text: "Colour saved. It is used by the next search; this result list stays as it was run." });
  };
  const doExport = async () => {
    setMsg(null);
    try {
      const r = await post(`/missions/${missionId}/search-runs/${runId}/export`, {});
      setMsg({ kind: "ok", text: `Exported: ${r.files.join(", ")}` });
    } catch (e) { setMsg({ kind: "err", text: (e as ApiError).message }); }
  };

  if (!data) return msg ? <div className={`msg ${msg.kind}`}>{msg.text}</div> : <div className="muted">Loading…</div>;
  const s = data.summary ?? {};
  const isVideo = s.events_total != null;
  const pages = Math.ceil(data.total / LIMIT);
  return (
    <div className="grid">
      <div className="row" style={{ justifyContent: "space-between" }}>
        <a href={`#/m/${missionId}/search`}>← New search</a>
        <button onClick={doExport}>Export hits</button>
      </div>
      {msg && <div className={`msg ${msg.kind}`}>{msg.text}</div>}
      <div className="grid cols-2" style={{ alignItems: "start" }}>
        <div className="card">
          <h2 className="bidi">“{data.text}”</h2>
          <ul style={{ marginTop: 0 }}>{data.description.map((d: string, i: number) => <li key={i}>{d}</li>)}</ul>
          <p className="muted" style={{ marginBottom: 0 }}>Search {data.id} · {data.state} · started by {data.started_by} at <span className="mono">{data.started_at}</span> · colour reader {data.attribute_algo}</p>
        </div>
        <div className="grid cols-4">
          <div className="card stat"><div className="n">{s.hits_match ?? 0}</div><div className="l">matches</div></div>
          <div className="card stat"><div className="n">{s.hits_possible ?? 0}</div><div className="l">possible (a fact could not be checked)</div></div>
          {isVideo ? <>
            <div className="card stat"><div className="n">{s.recordings_with_hits ?? 0}</div><div className="l">recordings with results</div></div>
            <div className="card stat"><div className="n">{s.events_searched ?? 0}</div><div className="l">events searched, of {s.events_total ?? 0}</div></div>
          </> : <>
            <div className="card stat"><div className="n">{s.images_with_hits ?? 0}</div><div className="l">images with results, of {s.images_searched ?? 0} searched</div></div>
            <div className="card stat"><div className="n">{s.unresolved_night_candidates_not_searched ?? 0}</div><div className="l">unresolved night lights not searched (type and colour unknown)</div></div>
          </>}
        </div>
      </div>
      {s.note && <div className="muted">{s.note}</div>}
      <details className="card">
        <summary>Why {isVideo ? "events" : "images or objects"} were left out</summary>
        <table className="t" style={{ marginTop: 8 }}><tbody>
          {(isVideo ? [["Events rejected, duplicate or merged", s.events_not_counted], ["Events outside the time or dates", s.events_outside_time],
            ["Events with no absolute time (time limits need it)", s.events_time_unknown], ["Events in another light condition", s.events_other_condition],
            ["Recordings with too few results", s.recordings_below_min_count], ["Possible results left out (only certain matches)", s.possible_left_out],
            ["Colour readings: observed / uncertain / not observable", s.colour ? `${s.colour.observed} / ${s.colour.uncertain} / ${s.colour.not_observable}` : "–"]] : [["Images outside the time or dates", s.images_outside_time], ["Images in another light condition", s.images_other_condition],
            ["Images with condition not measured", s.images_condition_unknown], ["Images whose view has none of the named regions", s.images_without_named_region],
            ["Images not processed yet", s.images_not_processed], ["Images with too few results", s.images_below_min_count],
            ["Possible results left out (only certain matches)", s.possible_left_out], ["Objects checked", s.objects_checked],
            ["Colour readings: observed / uncertain / not observable", s.colour ? `${s.colour.observed} / ${s.colour.uncertain} / ${s.colour.not_observable}` : "–"]])
            .map(([k, v]) => <tr key={k as string}><td>{k}</td><td className="num">{v ?? 0}</td></tr>)}
        </tbody></table>
      </details>
      <div className="row">
        <select value={status} onChange={(e) => { setOffset(0); setStatus(e.target.value); }}>
          <option value="">Matches and possible</option><option value="match">Matches only</option><option value="possible">Possible only</option>
        </select>
        <select value={decision} onChange={(e) => { setOffset(0); setDecision(e.target.value); }}>
          <option value="">Any verdict</option><option value="undecided">Not checked</option><option value="correct">Correct</option><option value="not_a_match">Not a match</option>
        </select>
        <span className="muted">{data.total} results</span>
      </div>
      <div className="hits">
        {data.hits.map((h: Hit) => (
          <div key={h.id} className={`card hit ${h.status}`}>
            {h.object_uid.startsWith("ev") ? (
              <a href={`#/m/${missionId}/events/${h.object_uid.slice(2)}`} title="Open the event in the player">
                <img src={`/api/missions/${missionId}/events/${h.object_uid.slice(2)}/thumb?h=130`} alt={`${h.cls} in ${h.filename}`} loading="lazy" />
              </a>
            ) : (
              <a href={`#/m/${missionId}/i/${h.source_version_id}`} title="Open the image">
                <img src={`/api/missions/${missionId}/crops/${h.source_version_id}?box=${h.box.map((v) => v.toFixed(1)).join(",")}`} alt={`${h.cls} in ${h.filename}`} loading="lazy" />
              </a>
            )}
            <div className="row" style={{ justifyContent: "space-between" }}>
              <span className={`badge ${h.status === "match" ? "approved" : "bulk_approved_unreviewed"}`}>{h.status === "match" ? "Match" : "Possible"}</span>
              <span>{h.cls}{h.colour.state !== "not_needed" ? ` · ${h.colour.state === "observed" ? h.colour.values?.[0] : h.colour.state === "uncertain" ? `${h.colour.values?.join(" or ")}?` : "colour not observable"}` : ""}</span>
            </div>
            {h.reasons.length > 0 && <ul className="reasons">{h.reasons.map((r, i) => <li key={i}>{r}</li>)}</ul>}
            <div className="mono muted" style={{ fontSize: 11.5 }}>{h.filename}{h.event_t != null ? ` at ${fmtT(h.event_t)}` : ""}<br />{h.capture_local?.replace("T", " ") ?? (h.event_t != null ? "absolute time unknown" : "")}</div>
            <div className="row" style={{ marginTop: 6, gap: 4 }}>
              <button className={h.decision === "correct" ? "primary" : ""} onClick={() => decide(h, h.decision === "correct" ? "undecided" : "correct")}>Correct</button>
              <button className={h.decision === "not_a_match" ? "primary" : ""} onClick={() => decide(h, h.decision === "not_a_match" ? "undecided" : "not_a_match")}>Not a match</button>
              <select aria-label="Correct the colour" defaultValue="" onChange={(e) => setColour(h, e.target.value)}>
                <option value="">Colour…</option>
                {vocab.colours.map((c) => <option key={c} value={c}>{c}</option>)}
                <option value="not_observable">not observable</option>
              </select>
            </div>
            <div className="muted" style={{ fontSize: 12 }}>{DECISION_LABEL[h.decision]}</div>
          </div>
        ))}
      </div>
      {pages > 1 && (
        <div className="row">
          <button disabled={offset === 0} onClick={() => setOffset(Math.max(0, offset - LIMIT))}>Previous</button>
          <span className="muted">Page {offset / LIMIT + 1} of {pages}</span>
          <button disabled={offset + LIMIT >= data.total} onClick={() => setOffset(offset + LIMIT)}>Next</button>
        </div>
      )}
    </div>
  );
}
