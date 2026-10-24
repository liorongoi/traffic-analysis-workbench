import { useCallback, useEffect, useRef, useState } from "react";
import { api, ApiError, Filter, fmtDuration, Job, post, REVIEW_LABEL, ResultRow } from "../api";
import { go } from "../router";
import SearchTab from "./Search";

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
  const tabs = [["overview", "Overview"], ...(m.config.mode === "object_search" ? [["search", "Search"]] : []),
    ["results", "Results"], ["views", "Camera views"]];
  return (
    <div className="page">
      <div className="row" style={{ justifyContent: "space-between" }}>
        <h1 className="bidi">{m.config.name}</h1>
        <span className="muted">{m.config.input_type}, {m.config.mode.replace(/_/g, " ")}, {m.config.iana_zone}</span>
      </div>
      <nav className="tabs">
        {tabs.map(([k, l]) => <a key={k} href={`#/m/${missionId}/${k}`} className={tab === k ? "on" : ""}>{l}</a>)}
      </nav>
      {tab === "overview" && <Overview m={m} reload={load} />}
      {tab === "results" && <Results missionId={missionId} />}
      {tab === "views" && <Views missionId={missionId} />}
      {tab === "search" && <SearchTab missionId={missionId} runId={extra ? Number(extra) : null} />}
    </div>
  );
}

function JobCard({ job, mid, reload }: { job: Job | null; mid: string; reload: () => void }) {
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
            <span className="muted">{job.done} of {job.total} images</span>
          </div>
          <div className="progress" style={{ margin: "8px 0" }}><div style={{ width: `${pct}%` }} /></div>
          <div className="muted">Elapsed {fmtDuration(job.elapsed_s)} · remaining {job.eta_s != null ? `about ${fmtDuration(job.eta_s)}` : "–"} ({job.eta_basis})</div>
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

function Overview({ m, reload }: { m: any; reload: () => void }) {
  const s = m.status;
  const pc = m.processing_counts as Record<string, number>;
  const toggle = async () => {
    await api(`/missions/${m.id}/monitoring`, { method: "PUT", body: { enabled: !m.config.monitoring_enabled } });
    reload();
  };
  const stats: [string, number][] = [["Files", s.source_files], ["Processed", pc.done ?? 0], ["Queued", (pc.queued ?? 0) + (pc.running ?? 0)],
    ["Failed", pc.failed ?? 0], ["Waiting to finish copying", s.pending_stability], ["Missing from source", s.missing], ["Names not readable", s.malformed], ["Damaged files", s.corrupt]];
  return (
    <div className="grid" style={{ gap: 14 }}>
      <div className="grid cols-4">
        {stats.map(([l, n]) => <div key={l} className="card stat"><div className="n">{n}</div><div className="l">{l}</div></div>)}
      </div>
      <div className="grid cols-2">
        <JobCard job={m.job} mid={m.id} reload={reload} />
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
        </div>
      )}
    </div>
  );
}
