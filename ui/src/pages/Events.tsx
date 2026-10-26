import { useCallback, useEffect, useRef, useState } from "react";
import { api, ApiError, post } from "../api";
import { go } from "../router";

// Video events: one row per continuous appearance of a vehicle. Times are seconds from the start of the recording,
// read from the frames' presentation timestamps. Absolute UTC is shown only when the file name gave the start.

export type VEvent = {
  id: number; key: string; source_version_id: number; file: string; first_t: number; last_t: number;
  machine_first_t: number; machine_last_t: number; cls: string; machine_cls: string; cls_share: number;
  flags: string[]; related: string[]; end_reason: string; status: string; duplicate_of: string | null;
  merged_into: number | null; absorbed: string[]; split_at: number[]; parts: [number, number][];
  first_utc: string | null; last_utc: string | null; revisions: number;
};
type Obs = { t: number; x1: number; y1: number; x2: number; y2: number; cls: string; score: number };

export const fmtT = (t: number | null | undefined) => {
  if (t == null) return "";
  const ms = Math.round(t * 1000);
  const h = Math.floor(ms / 3600000), m = Math.floor((ms % 3600000) / 60000), s = Math.floor((ms % 60000) / 1000);
  return `${String(h).padStart(2, "0")}:${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}.${String(ms % 1000).padStart(3, "0")}`;
};

const FLAG_LABEL: Record<string, string> = {
  appeared_inside_frame: "appeared inside the frame", ended_inside_frame: "lost inside the frame",
  possible_duplicate: "possible duplicate", class_uncertain: "class uncertain",
  cut_by_scene_change: "cut by a scene change", cut_by_end_of_recording: "cut by the end of the recording",
};
const STATUS_LABEL: Record<string, string> = {
  pending: "Not reviewed", confirmed: "Confirmed", rejected: "Rejected", duplicate: "Duplicate", merged: "Merged",
};
const CLASSES = ["car", "bus", "truck", "motorcycle", "bicycle"];

export default function EventsTab({ missionId, eventId }: { missionId: string; eventId: number | null }) {
  return eventId ? <EventDetail missionId={missionId} eventId={eventId} /> : <EventList missionId={missionId} />;
}

function EventList({ missionId }: { missionId: string }) {
  const [recs, setRecs] = useState<any[]>([]);
  const [evs, setEvs] = useState<VEvent[] | null>(null);
  const [rec, setRec] = useState("");
  const [status, setStatus] = useState("");
  const [flag, setFlag] = useState("");
  const [startTimes, setStartTimes] = useState<Record<number, string>>({});
  const [precisions, setPrecisions] = useState<Record<number, string>>({});
  const [timeRevision, setTimeRevision] = useState(0);
  const [msg, setMsg] = useState<{ kind: string; text: string } | null>(null);
  useEffect(() => { api(`/missions/${missionId}/recordings`).then((r) => setRecs(r.recordings)); }, [missionId]);
  useEffect(() => {
    const q = new URLSearchParams();
    if (rec) q.set("version_id", rec);
    if (status) q.set("status", status);
    if (flag) q.set("flag", flag);
    api(`/missions/${missionId}/events?${q}`).then((r) => setEvs(r.events));
  }, [missionId, rec, status, flag, timeRevision]);
  const saveStart = async (r: any) => {
    setMsg(null);
    try {
      await api(`/missions/${missionId}/recordings/${r.id}/start-time`, { method: "PUT", body: {
        action: "set", utc: startTimes[r.id] ?? r.start_utc, precision: precisions[r.id] ?? r.start_precision ?? "second",
        note: "Operator confirmed recording start in UTC",
      } });
      const updated = await api(`/missions/${missionId}/recordings`);
      setRecs(updated.recordings); setTimeRevision((x) => x + 1);
      setMsg({ kind: "ok", text: "UTC recording start saved. Event UTC times and exports now use this correction." });
    } catch (e) { setMsg({ kind: "err", text: (e as ApiError).message }); }
  };
  const clearStart = async (r: any) => {
    setMsg(null);
    try {
      await api(`/missions/${missionId}/recordings/${r.id}/start-time`, { method: "PUT", body: {
        action: "clear", note: "Operator cleared manual UTC start",
      } });
      const updated = await api(`/missions/${missionId}/recordings`);
      setRecs(updated.recordings); setTimeRevision((x) => x + 1);
      setMsg({ kind: "ok", text: "Manual UTC start cleared; the filename time is in use when available." });
    } catch (e) { setMsg({ kind: "err", text: (e as ApiError).message }); }
  };
  const doExport = async () => {
    setMsg(null);
    try {
      const r = await post(`/missions/${missionId}/events/export`, {});
      setMsg({ kind: "ok", text: `Exported: ${r.files.join(", ")}` });
    } catch (e) { setMsg({ kind: "err", text: (e as ApiError).message }); }
  };
  return (
    <div className="grid" style={{ gap: 10 }}>
      <div className="card">
        <h2>Recordings</h2>
        <table className="t"><thead><tr><th>File</th><th>Start (UTC)</th><th className="num">Length</th><th>Frame rate</th><th>Processing</th><th className="num">Events</th></tr></thead>
          <tbody>{recs.map((r) => (
            <tr key={r.id}>
              <td className="mono bidi">{r.rel_path}</td>
              <td>
                <div className="mono">{r.start_utc ?? <span className="muted" title="No confirmed UTC start">unknown</span>}</div>
                <div className="muted">{r.start_source === "manual" ? `operator confirmed by ${r.start_revision_author} · ${r.start_revision_created_at}` : r.start_source === "filename" ? "from filename" : "no UTC start"}</div>
                <details style={{ marginTop: 4 }}>
                  <summary>{r.start_source === "manual" ? "Edit UTC start" : "Set UTC start"}</summary>
                  <div className="grid" style={{ gap: 5, marginTop: 5 }}>
                    <input aria-label={`UTC start for ${r.rel_path}`} value={startTimes[r.id] ?? r.start_utc ?? ""}
                      onChange={(e) => setStartTimes({ ...startTimes, [r.id]: e.target.value })}
                      placeholder="2026-10-21T12:00:00Z" />
                    <select aria-label="UTC start precision" value={precisions[r.id] ?? r.start_precision ?? "second"}
                      onChange={(e) => setPrecisions({ ...precisions, [r.id]: e.target.value })}>
                      <option value="minute">Minute</option><option value="second">Second</option>
                    </select>
                    <button disabled={!(startTimes[r.id] ?? r.start_utc)} onClick={() => saveStart(r)}>Save confirmed UTC start</button>
                    {r.start_source === "manual" && <button onClick={() => clearStart(r)}>Clear manual correction</button>}
                    <span className="muted">Enter an ISO-8601 timestamp with a UTC offset (for example, ending in Z). This changes event UTC metadata, not video frames.</span>
                  </div>
                </details>
              </td>
              <td className="num mono">{fmtT(r.duration_s)}</td>
              <td>{r.nominal_fps ? `${Number(r.nominal_fps).toFixed(2)} fps` : ""}{r.variable_frame_rate ? " (variable)" : ""}</td>
              <td>{r.error ? <span className="badge failed" title={r.error}>cannot decode</span> : r.processing_state ?? ""}</td>
              <td className="num">{r.events}</td>
            </tr>))}</tbody></table>
      </div>
      <div className="card row">
        <select value={rec} onChange={(e) => setRec(e.target.value)} aria-label="Recording">
          <option value="">All recordings</option>{recs.map((r) => <option key={r.id} value={r.id}>{r.rel_path}</option>)}
        </select>
        <select value={status} onChange={(e) => setStatus(e.target.value)} aria-label="Review status">
          <option value="">Any review status</option>{Object.entries(STATUS_LABEL).map(([k, v]) => <option key={k} value={k}>{v}</option>)}
        </select>
        <select value={flag} onChange={(e) => setFlag(e.target.value)} aria-label="Flag">
          <option value="">Any flag</option>{Object.entries(FLAG_LABEL).map(([k, v]) => <option key={k} value={k}>{v}</option>)}
        </select>
        <span className="muted">{evs?.length ?? 0} events</span>
        <span style={{ flex: 1 }} />
        <button onClick={doExport}>Export events</button>
      </div>
      {msg && <div className={`msg ${msg.kind}`}>{msg.text}</div>}
      {evs && evs.length === 0 && <div className="card muted">No events match. Events appear after the recordings are processed.</div>}
      {evs && evs.length > 0 && (
        <div className="tablewrap">
          <table className="t">
            <thead><tr><th style={{ width: 90 }}></th><th>Event</th><th>Recording</th><th>Start</th><th>End</th><th>Start (UTC)</th><th>Class</th><th>Review</th><th>Flags</th></tr></thead>
            <tbody>{evs.map((e) => (
              <tr key={e.id} className="click" onClick={() => go(`/m/${missionId}/events/${e.id}`)}>
                <td><img loading="lazy" src={`/api/missions/${missionId}/events/${e.id}/thumb?h=48`} alt="" style={{ height: 40, display: "block", borderRadius: 3 }} /></td>
                <td className="mono">{e.key}</td>
                <td className="mono bidi">{e.file}</td>
                <td className="mono">{fmtT(e.first_t)}</td>
                <td className="mono">{fmtT(e.last_t)}</td>
                <td className="mono">{e.first_utc ?? <span className="muted">unknown</span>}</td>
                <td>{e.cls}{e.cls !== e.machine_cls ? <span className="muted"> (machine: {e.machine_cls})</span> : ""}</td>
                <td><span className={`badge ${e.status === "confirmed" ? "approved" : e.status === "rejected" ? "failed" : ""}`}>{STATUS_LABEL[e.status]}</span>{e.parts.length > 1 ? <span className="muted"> · {e.parts.length} parts</span> : ""}</td>
                <td>{e.flags.map((f) => <span key={f} className="badge flag" style={{ marginRight: 4 }}>{FLAG_LABEL[f] ?? f}</span>)}</td>
              </tr>))}</tbody>
          </table>
        </div>
      )}
    </div>
  );
}

function EventDetail({ missionId, eventId }: { missionId: string; eventId: number }) {
  const [ev, setEv] = useState<(VEvent & { observations: Obs[]; clips: any[] }) | null>(null);
  const [others, setOthers] = useState<VEvent[]>([]);
  const [err, setErr] = useState<string | null>(null);
  const [msg, setMsg] = useState<{ kind: string; text: string } | null>(null);
  const [now, setNow] = useState(0);
  const [before, setBefore] = useState("");
  const [after, setAfter] = useState("");
  const [method, setMethod] = useState("copy");
  const [busy, setBusy] = useState(false);
  const [fallback, setFallback] = useState(false);   // the browser cannot play this codec: show decoded frames
  const video = useRef<HTMLVideoElement>(null);
  const still = useRef<HTMLImageElement>(null);
  const canvas = useRef<HTMLCanvasElement>(null);

  const load = useCallback(async () => {
    try {
      const e = await api(`/missions/${missionId}/events/${eventId}`);
      setEv(e);
      const r = await api(`/missions/${missionId}/events?version_id=${e.source_version_id}`);
      setOthers(r.events.filter((o: VEvent) => o.id !== eventId));
    } catch (e) { setErr((e as ApiError).message); }
  }, [missionId, eventId]);
  useEffect(() => { load(); }, [load]);

  // Draw the tracked box nearest the playhead (within 0.3 s), so the operator sees which vehicle the event is.
  const draw = useCallback(() => {
    const c = canvas.current;
    const el = fallback ? still.current : video.current;
    if (!el || !c || !ev) return;
    const vw = fallback ? still.current!.naturalWidth : video.current!.videoWidth;
    const vh = fallback ? still.current!.naturalHeight : video.current!.videoHeight;
    const t = fallback ? now : video.current!.currentTime;
    if (!fallback) setNow(t);
    c.width = el.clientWidth;
    c.height = el.clientHeight;
    const g = c.getContext("2d");
    if (!g || !vw) return;
    g.clearRect(0, 0, c.width, c.height);
    let best: Obs | null = null;
    for (const o of ev.observations) if (!best || Math.abs(o.t - t) < Math.abs(best.t - t)) best = o;
    if (!best || Math.abs(best.t - t) > 0.3) return;
    const s = Math.min(c.width / vw, c.height / vh);
    const ox = (c.width - vw * s) / 2, oy = (c.height - vh * s) / 2;
    g.strokeStyle = "#ffd400";
    g.lineWidth = 2;
    g.strokeRect(ox + best.x1 * s, oy + best.y1 * s, (best.x2 - best.x1) * s, (best.y2 - best.y1) * s);
    g.fillStyle = "#ffd400";
    g.font = "12px sans-serif";
    g.fillText(`${ev.key} ${best.cls}`, ox + best.x1 * s, Math.max(12, oy + best.y1 * s - 4));
  }, [ev, fallback, now]);
  useEffect(() => {
    let raf = 0;
    const loop = () => { draw(); raf = requestAnimationFrame(loop); };
    raf = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(raf);
  }, [draw]);

  const seek = (t: number) => {
    if (fallback) setNow(Math.max(0, Math.round(t * 1000) / 1000));
    else if (video.current) video.current.currentTime = Math.max(0, t);
  };
  const step = (d: number) => seek((fallback ? now : video.current?.currentTime ?? 0) + d);
  const revise = async (action: string, payload: Record<string, unknown> = {}) => {
    setMsg(null);
    try { await post(`/missions/${missionId}/events/${eventId}/revise`, { action, ...payload }); await load(); }
    catch (e) { setMsg({ kind: "err", text: (e as ApiError).message }); }
  };
  const clip = async () => {
    setMsg(null);
    if (before === "" || after === "") { setMsg({ kind: "err", text: "Enter the seconds of context before and after the event." }); return; }
    setBusy(true);
    try {
      const r = await post(`/missions/${missionId}/events/${eventId}/clip`, { before_s: Number(before), after_s: Number(after), method });
      const note = r.method === "copy" && r.actual?.[0] != null && r.actual[0] < r.requested[0] - 0.001
        ? ` It starts at ${fmtT(r.actual[0])}, the keyframe before the requested ${fmtT(r.requested[0])}.` : "";
      setMsg({ kind: "ok", text: `Clip written: ${r.path}.${note}` });
      await load();
    } catch (e) { setMsg({ kind: "err", text: (e as ApiError).message }); }
    setBusy(false);
  };

  if (err) return <div className="msg err">{err}</div>;
  if (!ev) return <div className="muted">Loading…</div>;
  const idx = others.length;
  return (
    <div className="grid" style={{ gap: 10 }}>
      <div className="row">
        <a href={`#/m/${missionId}/events`}>← All events</a>
        <h2 style={{ margin: 0 }}>{ev.key}</h2>
        <span className="mono bidi muted">{ev.file}</span>
        <span className={`badge ${ev.status === "confirmed" ? "approved" : ev.status === "rejected" ? "failed" : ""}`}>{STATUS_LABEL[ev.status]}</span>
        {ev.duplicate_of && <span className="muted">duplicate of {ev.duplicate_of}</span>}
        {ev.merged_into && <a href={`#/m/${missionId}/events/${ev.merged_into}`}>merged into another event</a>}
        <span className="muted">{idx} other event(s) in this recording</span>
      </div>
      <div className="grid cols-2" style={{ gridTemplateColumns: "minmax(0, 2fr) minmax(280px, 1fr)" }}>
        <div className="card" style={{ padding: 8 }}>
          <div style={{ position: "relative", background: "#000" }}>
            {!fallback && <video ref={video} src={`/api/missions/${missionId}/videos/${ev.source_version_id}`} controls preload="metadata"
              style={{ width: "100%", display: "block", maxHeight: "62vh" }}
              onLoadedMetadata={() => seek(ev.first_t)} onError={() => { setFallback(true); setNow(ev.first_t); }} />}
            {fallback && <img ref={still} alt={`Frame at ${fmtT(now)}`} src={`/api/missions/${missionId}/videos/${ev.source_version_id}/frame?t=${now}&w=3840`}
              style={{ width: "100%", display: "block", maxHeight: "62vh", objectFit: "contain" }} />}
            <canvas ref={canvas} style={{ position: "absolute", left: 0, top: 0, pointerEvents: "none" }} />
          </div>
          <div className="row" style={{ marginTop: 6 }}>
            <span className="mono">{fmtT(now)}</span>
            <button onClick={() => seek(ev.first_t)}>Go to start</button>
            <button onClick={() => seek(ev.last_t)}>Go to end</button>
            <button onClick={() => step(fallback ? -0.2 : -0.04)} title="Step back">◀ {fallback ? "0.2 s" : "frame"}</button>
            <button onClick={() => step(fallback ? 0.2 : 0.04)} title="Step forward">{fallback ? "0.2 s" : "frame"} ▶</button>
          </div>
          {fallback && <>
            <input type="range" style={{ width: "100%" }} min={Math.max(0, ev.first_t - 10)} max={ev.last_t + 10} step={0.04} value={now}
              onChange={(e) => setNow(Number(e.target.value))} aria-label="Time in the recording" />
            <div className="msg warn">This browser cannot play the recording's video format, so frames are decoded on the workstation one at a time.</div>
          </>}
          <p className="muted" style={{ marginBottom: 0 }}>The yellow box is the tracked vehicle at the nearest analysed frame. Times are from the start of the recording.</p>
        </div>
        <div className="grid" style={{ gap: 10, alignContent: "start" }}>
          <div className="card">
            <table className="t"><tbody>
              <tr><td>Start</td><td className="mono">{fmtT(ev.first_t)}{ev.first_t !== ev.machine_first_t && <span className="muted"> (machine {fmtT(ev.machine_first_t)})</span>}</td></tr>
              <tr><td>End</td><td className="mono">{fmtT(ev.last_t)}{ev.last_t !== ev.machine_last_t && <span className="muted"> (machine {fmtT(ev.machine_last_t)})</span>}</td></tr>
              <tr><td>Start (UTC)</td><td className="mono">{ev.first_utc ?? "unknown: the file name has no start time"}</td></tr>
              <tr><td>Class</td><td>{ev.cls} <span className="muted">({Math.round(ev.cls_share * 100)}% of detections{ev.cls !== ev.machine_cls ? `, machine said ${ev.machine_cls}` : ""})</span></td></tr>
              <tr><td>Ended because</td><td>{ev.end_reason.replace(/_/g, " ")}</td></tr>
              <tr><td>Flags</td><td>{ev.flags.length ? ev.flags.map((f) => FLAG_LABEL[f] ?? f).join(", ") : "none"}{ev.related.length > 0 && ` (${ev.related.join(", ")})`}</td></tr>
              {ev.parts.length > 1 && <tr><td>Parts</td><td className="mono">{ev.parts.map(([a, b]) => `${fmtT(a)}–${fmtT(b)}`).join(", ")}</td></tr>}
              {ev.absorbed.length > 0 && <tr><td>Merged in</td><td>{ev.absorbed.join(", ")}</td></tr>}
              <tr><td>Operator changes</td><td>{ev.revisions}</td></tr>
            </tbody></table>
          </div>
          <div className="card grid" style={{ gap: 8 }}>
            <h2 style={{ margin: 0 }}>Review</h2>
            <div className="row">
              <button className="primary" onClick={() => revise("confirm")}>Confirm</button>
              <button className="danger" onClick={() => revise("reject")}>Reject</button>
              <button onClick={() => revise("reopen")}>Reopen</button>
            </div>
            <div className="row">
              <button onClick={() => revise("set_times", { first_t: now, last_t: ev.last_t })}>Start here</button>
              <button onClick={() => revise("set_times", { first_t: ev.first_t, last_t: now })}>End here</button>
              <button onClick={() => revise("split_at", { t: now })}>Split here</button>
            </div>
            <div className="row">
              <select aria-label="Set class" value="" onChange={(e) => e.target.value && revise("set_class", { cls: e.target.value })}>
                <option value="">Set class…</option>{CLASSES.map((c) => <option key={c}>{c}</option>)}
              </select>
              <select aria-label="Duplicate of" value="" onChange={(e) => e.target.value && revise("duplicate", { of: e.target.value })}>
                <option value="">Duplicate of…</option>{others.map((o) => <option key={o.id} value={o.key}>{o.key} ({fmtT(o.first_t)})</option>)}
              </select>
              <select aria-label="Merge into" value="" onChange={(e) => e.target.value && revise("merge_into", { target: Number(e.target.value) })}>
                <option value="">Merge into…</option>{others.filter((o) => o.status !== "merged").map((o) => <option key={o.id} value={o.id}>{o.key} ({fmtT(o.first_t)})</option>)}
              </select>
            </div>
            <p className="muted" style={{ margin: 0 }}>Times snap to the next frame. Every change is kept; the machine result is never changed.</p>
          </div>
          <div className="card grid" style={{ gap: 8 }}>
            <h2 style={{ margin: 0 }}>Export a clip</h2>
            <div className="row">
              <label className="row">Seconds before <input type="number" min={0} max={600} step={0.5} style={{ width: 70 }} value={before} onChange={(e) => setBefore(e.target.value)} /></label>
              <label className="row">after <input type="number" min={0} max={600} step={0.5} style={{ width: 70 }} value={after} onChange={(e) => setAfter(e.target.value)} /></label>
            </div>
            <div className="row">
              <select value={method} onChange={(e) => setMethod(e.target.value)} aria-label="Clip method">
                <option value="copy">Copy (fast, original quality, starts at a keyframe)</option>
                <option value="reencode">Re-encode (exact start, slower)</option>
              </select>
              <button className="primary" disabled={busy} onClick={clip}>{busy ? "Writing…" : "Export clip"}</button>
            </div>
            {ev.clips.length > 0 && (
              <ul style={{ margin: 0, paddingLeft: 18 }}>
                {ev.clips.map((c) => <li key={c.id} className="mono bidi" style={{ fontSize: 12 }}>{c.path} <span className="muted">({c.method}, {fmtT(c.actual_start_t)}–{fmtT(c.actual_end_t)})</span></li>)}
              </ul>
            )}
          </div>
          {msg && <div className={`msg ${msg.kind}`}>{msg.text}</div>}
        </div>
      </div>
    </div>
  );
}
