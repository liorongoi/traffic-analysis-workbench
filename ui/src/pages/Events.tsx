import { useCallback, useEffect, useRef, useState } from "react";
import { api, ApiError, post } from "../api";
import { go } from "../router";
import { Choice } from "../components/Pick";
import Player from "./Player";

// Video events: one row per continuous appearance of a vehicle. Times are seconds from the start of the recording,
// read from the frames' presentation timestamps. Absolute UTC is shown only when the file name gave the start.

export type VEvent = {
  id: number; key: string; source_version_id: number; file: string; first_t: number; last_t: number;
  machine_first_t: number; machine_last_t: number; cls: string; machine_cls: string; cls_share: number;
  flags: string[]; related: string[]; end_reason: string; status: string; duplicate_of: string | null;
  merged_into: number | null; absorbed: string[]; split_at: number[]; parts: [number, number][];
  first_utc: string | null; last_utc: string | null; revisions: number;
  continues?: number | null; continues_key?: string | null; attrs?: Attrs;
};
/** What a target did and looked like (phase 6a). Colours: null not read yet, [] not observable, 2+ uncertain. */
export type Attrs = {
  direction: string | null; dwell_s: number; stopped_s: number; speed: number | null; top_speed: number | null;
  unit: "km/h" | "px/s"; w: number | null; h: number | null; size_unit: "m" | "px";
  colour: string[] | null; upper: string[] | null; lower: string[] | null;
  path?: string[]; zones?: Record<string, string[]>;      // lines crossed (directions) and areas entered ("inside"), in order
};

export const ARROW: Record<string, string> = {
  right: "→", "down-right": "↘", down: "↓", "down-left": "↙", left: "←", "up-left": "↖", up: "↑", "up-right": "↗", stayed: "•",
};
const COLOURS = ["black", "white", "grey", "brown", "red", "orange", "yellow", "green", "blue"];
const SWATCH: Record<string, string> = {
  black: "#111", white: "#f4f4f4", grey: "#8a8a8a", brown: "#7a4b22", red: "#d33", orange: "#f08a24", yellow: "#f2d02a",
  green: "#2e9e4f", blue: "#2f6fd6",
};
/** Colour chips with their names (never colour alone). */
export function ColourChips({ v }: { v: string[] | null | undefined }) {
  if (v == null) return <span className="muted small">reading…</span>;
  if (!v.length) return <span className="muted small">not seen</span>;
  return <span className="chips">{v.map((c, i) => <span key={c} className="chip-colour">
    <i style={{ background: SWATCH[c] ?? "transparent" }} />{c}{i < v.length - 1 ? " or" : ""}</span>)}</span>;
}
export const fmtDur = (s: number) => s < 60 ? `${s.toFixed(s < 10 ? 1 : 0)} s` : `${Math.floor(s / 60)} min ${Math.round(s % 60)} s`;
const fmtSpeed = (a: Attrs) => a.speed == null ? "" : `${a.unit === "km/h" ? Math.round(a.speed) : Math.round(a.speed)} ${a.unit}`;

// Filters on what targets did and looked like; every one a pick from a list.
const DIRS = ["right", "down-right", "down", "down-left", "left", "up-left", "up", "up-right", "stayed"];
const IN_VIEW = [5, 10, 30, 60, 120, 300];
const STOPPED = [3, 10, 30, 60, 120, 300];
const KMH = [[0, 10], [10, 30], [30, 50], [50, 70], [70, 90], [90, 120], [120, 999]];
type Obs = { t: number; x1: number; y1: number; x2: number; y2: number; cls: string; score: number; found_by?: string };

export const fmtT = (t: number | null | undefined) => {
  if (t == null) return "";
  const ms = Math.round(t * 1000);
  const h = Math.floor(ms / 3600000), m = Math.floor((ms % 3600000) / 60000), s = Math.floor((ms % 60000) / 1000);
  return `${String(h).padStart(2, "0")}:${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}.${String(ms % 1000).padStart(3, "0")}`;
};

const REVIEW_TEXT: Record<string, string> = {
  full: "Approved: full labels", partial: "Some events reviewed", none: "Not reviewed",
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

const CLIP_SECONDS = [0, 1, 2, 3, 5, 10, 15, 20, 30, 45, 60, 90, 120, 180, 300, 600];
/** "2026-10-21T12:00:00Z" ⇄ the "2026-10-21T12:00:00" a date-and-time picker shows (the picker has no zone; it is UTC here). */
const toPicker = (iso: string) => {
  if (!iso) return "";
  const d = new Date(iso);
  return Number.isNaN(d.valueOf()) ? "" : d.toISOString().slice(0, 19);
};
const fromPicker = (v: string) => (v ? `${v.length === 16 ? `${v}:00` : v.slice(0, 19)}Z` : "");

export default function EventsTab({ missionId, extra }: { missionId: string; extra: string | null }) {
  // "#/m/<id>/events/<event id>" opens one event; "#/m/<id>/events/rec-<video id>" opens that video's tab.
  const eventId = extra && /^\d+$/.test(extra) ? Number(extra) : null;
  const recording = extra?.startsWith("rec-") ? extra.slice(4) : "";
  // "#/m/<id>/events/play-<video id>" plays that video with its boxes, to review and edit there.
  if (extra?.startsWith("play-")) return <Player missionId={missionId} versionId={Number(extra.slice(5))} />;
  return eventId ? <EventDetail missionId={missionId} eventId={eventId} /> : <EventList missionId={missionId} recording={recording} />;
}

type Counts = {
  classes: string[]; uncounted: string[]; bucket_s: number;
  total: { machine: Record<string, number>; counted: Record<string, number>; maybe_same: Record<string, number>; unreviewed: number; vehicles_counted: number };
  recordings: { source_version_id: number; file: string; duration_s: number | null; machine: Record<string, number>; counted: Record<string, number>;
    maybe_same: Record<string, number>; unreviewed: number; per_bucket: { start_s: number; counts: Record<string, number> }[] }[];
};

/** Unique targets per class: each tracked target once, however many frames it was seen in. The big number follows
 *  review (rejects, duplicates and merges left out); the machine count and unreviewed possible duplicates are beside it. */
function CountsCard({ missionId, versionId, refresh }: { missionId: string; versionId: string; refresh: number }) {
  const [c, setC] = useState<Counts | null>(null);
  const [hover, setHover] = useState<number | null>(null);
  useEffect(() => {
    const q = versionId.startsWith("g") ? `?group_id=${versionId.slice(1)}` : versionId ? `?version_id=${versionId}` : "";
    api(`/missions/${missionId}/counts${q}`).then(setC).catch(() => setC(null));
  }, [missionId, versionId, refresh]);
  if (!c || !c.classes.length) return null;
  const sum = (d: Record<string, number>) => Object.values(d).reduce((a, b) => a + b, 0);
  // Targets first seen in each minute, all recordings in view added up by minute from their start.
  const buckets = new Map<number, Record<string, number>>();
  for (const r of c.recordings) for (const b of r.per_bucket) {
    const cur = buckets.get(b.start_s) ?? {};
    for (const [k, v] of Object.entries(b.counts)) cur[k] = (cur[k] ?? 0) + v;
    buckets.set(b.start_s, cur);
  }
  const last = Math.max(0, ...[...buckets.keys()], ...c.recordings.map((r) => Math.floor((r.duration_s ?? 0) / c.bucket_s) * c.bucket_s));
  const bars = Array.from({ length: Math.floor(last / c.bucket_s) + 1 }, (_, i) => ({ s: i * c.bucket_s, counts: buckets.get(i * c.bucket_s) ?? {} }));
  const peak = Math.max(1, ...bars.map((b) => sum(b.counts)));
  const t = c.total;
  return (
    <div className="card counts viz">
      <div className="row" style={{ justifyContent: "space-between", alignItems: "baseline" }}>
        <h2>Unique targets{versionId ? " in this recording" : ""}</h2>
        <span className="muted small">{t.unreviewed ? `${t.unreviewed} event(s) not reviewed yet` : "all events reviewed"}</span>
      </div>
      <div className="count-tiles">
        {c.classes.map((k) => (
          <div key={k} className="count-tile">
            <div className="count-cls">{k}{c.uncounted.includes(k) ? <span className="muted"> · not a vehicle</span> : ""}</div>
            <div className="count-num">{t.counted[k] ?? 0}</div>
            <div className="muted small">machine {t.machine[k] ?? 0}{t.maybe_same[k] ? ` · ${t.maybe_same[k]} maybe the same` : ""}</div>
          </div>
        ))}
      </div>
      {bars.length >= 3 && <div className="count-chart" aria-label="New targets per minute" onMouseLeave={() => setHover(null)}>
        <div className="muted small">New targets per {c.bucket_s === 60 ? "minute" : `${c.bucket_s} s`} (peak {peak})</div>
        <div style={{ overflowX: "auto" }}><svg width={bars.length * 10} height="64" viewBox={`0 0 ${bars.length * 10} 64`} role="img">
          {bars.map((b, i) => {
            const h = (sum(b.counts) / peak) * 60;
            return (
              <g key={b.s} onMouseEnter={() => setHover(i)}>
                <rect x={i * 10} y={0} width={10} height={64} fill="transparent" />
                <rect x={i * 10 + 1} y={64 - Math.max(h, sum(b.counts) ? 2 : 0)} width={8} height={Math.max(h, sum(b.counts) ? 2 : 0)} rx={2}
                  fill={hover === i ? "var(--accent-hover)" : "var(--series-1)"} />
              </g>
            );
          })}
        </svg></div>
        <div className="count-tip small" aria-live="polite">
          {hover != null ? `${fmtT(bars[hover].s).slice(0, 8)} to ${fmtT(bars[hover].s + c.bucket_s).slice(0, 8)}: `
            + (sum(bars[hover].counts) ? Object.entries(bars[hover].counts).map(([k, v]) => `${v} ${k}`).join(", ") : "none") : "\u00a0"}
        </div>
      </div>}
      {!versionId && c.recordings.length > 1 && (
        <table className="t"><thead><tr><th>Recording</th>{c.classes.map((k) => <th key={k} className="num">{k}</th>)}</tr></thead>
          <tbody>{c.recordings.map((r) => (
            <tr key={r.source_version_id}><td className="mono bidi">{r.file}</td>
              {c.classes.map((k) => <td key={k} className="num">{r.counted[k] ?? 0}</td>)}</tr>
          ))}</tbody></table>
      )}
    </div>
  );
}

const FAMILY: Record<string, string> = { vehicle: "vehicles", person: "people" };
const CROWD = [5, 10, 20, 30, 50, 100];

/** The most targets in view at the same moment, per family, and the crowd alert (people in view at once). */
function InViewCard({ missionId, versionId, refresh }: { missionId: string; versionId: string; refresh: number }) {
  const [d, setD] = useState<any | null>(null);
  const [n, setN] = useState(0);
  useEffect(() => {
    const q = versionId.startsWith("g") ? `?group_id=${versionId.slice(1)}` : versionId ? `?version_id=${versionId}` : "";
    api(`/missions/${missionId}/in-view${q}`).then(setD).catch(() => setD(null));
  }, [missionId, versionId, refresh, n]);
  if (!d || !d.recordings.length) return null;
  const fams = new Map<string, { max: number; at_s: number; rec: any }>();
  for (const r of d.recordings) for (const [f, v] of Object.entries(r.families) as [string, any][])
    if (!fams.has(f) || v.max > fams.get(f)!.max) fams.set(f, { ...v, rec: r });
  const crowds = d.recordings.flatMap((r: any) => r.crowd.map((c: any) => ({ ...c, rec: r })));
  const setAlert = (v: string) => api(`/missions/${missionId}/crowd-alert`, { method: "PUT", body: { people: v ? Number(v) : null } }).then(() => setN((x) => x + 1));
  return (
    <div className="card counts in-view">
      <div className="row" style={{ justifyContent: "space-between", alignItems: "baseline" }}>
        <h2>Most in view at once</h2>
        <label className="row small">Crowd alert
          <select aria-label="Crowd alert" value={d.crowd_at ?? ""} onChange={(e) => setAlert(e.target.value)}>
            <option value="">Off</option>{CROWD.map((c) => <option key={c} value={c}>{c} or more people at once</option>)}
          </select></label>
      </div>
      <div className="count-tiles">
        {[...fams.entries()].map(([f, v]) => (
          <div key={f} className="count-tile">
            <div className="count-cls">{FAMILY[f] ?? f}</div>
            <div className="count-num">{v.max}</div>
            <div className="muted small">at <a href={`#/m/${missionId}/events/play-${v.rec.source_version_id}`}>{fmtT(v.at_s).slice(0, 8)}</a>
              {d.recordings.length > 1 ? <> in <span className="bidi">{v.rec.file.split("/").pop()}</span></> : null}</div>
          </div>))}
      </div>
      {d.crowd_at && (crowds.length ? <div className="msg warn"><strong>{crowds.length} crowd period(s)</strong> with {d.crowd_at} or more people in view:{" "}
        {crowds.slice(0, 12).map((c: any, i: number) => <span key={i}>{i ? ", " : ""}<a href={`#/m/${missionId}/events/play-${c.rec.source_version_id}`}>{fmtT(c.start_s).slice(0, 8)}–{fmtT(c.end_s).slice(0, 8)}</a> (up to {c.peak})</span>)}
        {crowds.length > 12 ? " …" : ""}</div>
        : <p className="muted small" style={{ margin: 0 }}>Never {d.crowd_at} or more people in view at once.</p>)}
    </div>
  );
}

/** Each counting line's crossings (in and out) and each area's entries and time inside, by class. */
function ZonesCard({ missionId, versionId, refresh }: { missionId: string; versionId: string; refresh: number }) {
  const [d, setD] = useState<any | null>(null);
  useEffect(() => {
    const q = versionId.startsWith("g") ? `?group_id=${versionId.slice(1)}` : versionId ? `?version_id=${versionId}` : "";
    api(`/missions/${missionId}/zones${q}`).then(setD).catch(() => setD(null));
  }, [missionId, versionId, refresh]);
  if (!d || !d.recordings.length) return null;
  // all videos in view added up, zone by zone (videos from one camera share its zones)
  const tot = new Map<string, { name: string; kind: string; by: Record<string, any> }>();
  for (const r of d.recordings) for (const z of r.zones) {
    const t = tot.get(z.uid) ?? { name: z.name, kind: z.kind, by: {} as Record<string, any> };
    for (const [c, v] of Object.entries(z.by_class) as [string, any][]) {
      const cur: Record<string, number> = t.by[c] ?? (z.kind === "line" ? { in: 0, out: 0 } : { entries: 0, inside_s: 0 });
      for (const k of Object.keys(cur)) cur[k] += v[k] ?? 0;
      t.by[c] = cur;
    }
    tot.set(z.uid, t);
  }
  const classes = [...new Set([...tot.values()].flatMap((t) => Object.keys(t.by)))];
  return (
    <div className="card">
      <h2>Lines and areas</h2>
      <table className="t"><thead><tr><th>Line or area</th>{classes.map((c) => <th key={c} className="num">{c}</th>)}</tr></thead>
        <tbody>{[...tot.entries()].map(([uid, t]) => (
          <tr key={uid}><td>{t.kind === "line" ? "Line" : "Area"}: {t.name}</td>
            {classes.map((c) => { const v = t.by[c]; return <td key={c} className="num mono">{!v ? "0" : t.kind === "line"
              ? `${v.in} in · ${v.out} out` : `${v.entries} · ${fmtDur(v.inside_s / Math.max(1, v.entries))} each`}</td>; })}
          </tr>))}</tbody></table>
      <p className="muted small" style={{ marginBottom: 0 }}>Draw lines and areas in the player ("Lines and areas"). Each target is
        counted once per crossing; the areas show how many entered and the average time inside.</p>
    </div>
  );
}

/** Where the targets of one video spent time (heatmap), or the paths they took, over the empty road. */
function MapCard({ missionId, versionId, ids }: { missionId: string; versionId: number; ids: number[] | null }) {
  const [kind, setKind] = useState("");
  const [m, setM] = useState<{ image: string; legend: any } | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const key = ids ? ids.join(",") : "all";
  useEffect(() => {
    if (!kind) { setM(null); return; }
    setErr(null);
    post(`/missions/${missionId}/recordings/${versionId}/map`, { kind, event_ids: ids }).then(setM).catch((e) => setErr((e as ApiError).message));
  }, [missionId, versionId, kind, key]);
  return (
    <div className="card grid" style={{ gap: 8 }}>
      <div className="row">
        <h2 style={{ margin: 0 }}>Map</h2>
        <select aria-label="Map" value={kind} onChange={(e) => setKind(e.target.value)}>
          <option value="">No map</option>
          <option value="heat">Where targets spent time (heatmap)</option>
          <option value="paths">The paths they took</option>
        </select>
        <span className="muted small">{ids ? `the ${ids.length} events the filters show` : "every counted event of this video"}, over the empty road</span>
      </div>
      {err && <div className="msg err">{err}</div>}
      {m && <>
        <img src={m.image} alt={kind === "heat" ? "Heatmap of where targets spent time" : "Paths of the targets"} style={{ width: "100%", maxWidth: 960, borderRadius: 6 }} />
        <div className="row small muted">
          {m.legend.kind === "heat" ? <><span className="heat-scale" aria-hidden="true" /> from little time to {m.legend.max_s.toFixed(1)} s at one spot · {m.legend.targets} targets</>
            : <>{Object.entries(m.legend.classes as Record<string, string>).map(([c, col]) => <span key={c} className="chip-colour"><i style={{ background: col }} />{c}</span>)}
              <span>· a dot where each was first seen · {m.legend.targets} targets</span></>}
        </div></>}
    </div>
  );
}

function EventList({ missionId, recording }: { missionId: string; recording: string }) {
  const [recs, setRecs] = useState<any[]>([]);
  const [groups, setGroups] = useState<{ id: number; name: string; members: number[] }[]>([]);
  const [evs, setEvs] = useState<VEvent[] | null>(null);
  const [rec, setRecState] = useState(recording);
  useEffect(() => { setRecState(recording); }, [recording]);
  // One tab per analysed video; the tab is part of the address, so it can be linked and survives a reload.
  const setRec = (v: string) => { setRecState(v); go(`/m/${missionId}/events${v ? `/rec-${v}` : ""}`); };
  const [status, setStatus] = useState("");
  const [flag, setFlag] = useState("");
  const [dir, setDir] = useState("");
  const [inView, setInView] = useState("");
  const [stopped, setStopped] = useState("");
  const [speed, setSpeed] = useState("");
  const [colour, setColour] = useState("");
  const [upper, setUpper] = useState("");
  const [lower, setLower] = useState("");
  const [reading, setReading] = useState(false);
  const [zoneNames, setZoneNames] = useState<Record<string, string>>({});
  const [zone, setZone] = useState("");
  const [from, setFrom] = useState("");
  const [to, setTo] = useState("");
  const [startTimes, setStartTimes] = useState<Record<number, string>>({});
  const [precisions, setPrecisions] = useState<Record<number, string>>({});
  const [timeRevision, setTimeRevision] = useState(0);
  const [msg, setMsg] = useState<{ kind: string; text: string } | null>(null);
  useEffect(() => { api(`/missions/${missionId}/recordings`).then((r) => { setRecs(r.recordings); setGroups(r.groups ?? []); }); }, [missionId]);
  useEffect(() => {
    const q = new URLSearchParams();
    if (rec.startsWith("g")) q.set("group_id", rec.slice(1));
    else if (rec) q.set("version_id", rec);
    if (status) q.set("status", status);
    if (flag) q.set("flag", flag);
    api(`/missions/${missionId}/events?${q}`).then((r) => { setEvs(r.events); setReading(!!r.reading_colours); setZoneNames(r.zones ?? {}); });
  }, [missionId, rec, status, flag, timeRevision]);
  // While colours are being read, the list fills in every few seconds.
  useEffect(() => {
    if (!reading) return;
    const t = setTimeout(() => setTimeRevision((x) => x + 1), 4000);
    return () => clearTimeout(t);
  }, [reading, evs]);
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
  const review = async (r: any, action: "approve" | "confirm_rest_and_approve" | "reopen") => {
    setMsg(null);
    try {
      await post(`/missions/${missionId}/recordings/${r.id}/review`, { action });
      const updated = await api(`/missions/${missionId}/recordings`);
      setRecs(updated.recordings); setTimeRevision((x) => x + 1);
      setMsg({ kind: "ok", text: action !== "reopen"
        ? "Recording approved for learning: its frames teach every reviewed target, and the empty road and rejected detections as background."
        : "Recording reopened: only its reviewed events teach, the rest of each frame is ignored." });
    } catch (e) { setMsg({ kind: "err", text: (e as ApiError).message }); }
  };
  const doExport = async () => {
    setMsg(null);
    try {
      const r = await post(`/missions/${missionId}/events/export`, {});
      setMsg({ kind: "ok", text: `Exported: ${r.files.join(", ")}` });
    } catch (e) { setMsg({ kind: "err", text: (e as ApiError).message }); }
  };
  const has = (v: string[] | null | undefined, c: string) => !!v && v.includes(c);
  const kmh = evs?.some((e) => e.attrs?.unit === "km/h");
  const pxSpeeds = (evs ?? []).filter((e) => e.attrs?.unit === "px/s" && e.attrs.speed != null).map((e) => e.attrs!.speed!).sort((a, b) => a - b);
  const third = (k: number) => pxSpeeds[Math.floor((pxSpeeds.length * k) / 3)] ?? 0;
  const inSpeed = (a: Attrs) => {
    if (!speed || a.speed == null) return !speed;
    if (speed.startsWith("k")) { const [lo, hi] = speed.slice(1).split("-").map(Number); return a.unit === "km/h" && a.speed >= lo && a.speed < hi; }
    if (a.unit !== "px/s") return false;
    return speed === "slow" ? a.speed < third(1) : speed === "medium" ? a.speed >= third(1) && a.speed < third(2) : a.speed >= third(2);
  };
  const inZone = (a: Attrs) => {
    if (!zone) return true;
    const [uid, how] = zone.split(":");
    const z = a.zones?.[uid];
    return !!z && (how === "any" || z.includes(how));
  };
  const onPath = (a: Attrs) => {
    if (!from && !to) return true;
    const p = a.path ?? [];
    if (from && !to) return p.includes(from);
    if (!from) return p.includes(to);
    const i = p.indexOf(from);
    return i >= 0 && p.slice(i + 1).includes(to);
  };
  const people = evs?.some((e) => e.attrs?.upper != null || e.cls === "person");
  const shown = (evs ?? []).filter((e) => { const a = e.attrs; if (!a) return true;
    return (!dir || a.direction === dir) && (!inView || a.dwell_s >= Number(inView)) && (!stopped || a.stopped_s >= Number(stopped))
      && inSpeed(a) && inZone(a) && onPath(a) && (!colour || has(a.colour, colour) || has(a.upper, colour) || has(a.lower, colour)) && (!upper || has(a.upper, upper)) && (!lower || has(a.lower, lower)); });
  const filtering = !!(dir || inView || stopped || speed || colour || upper || lower || zone || from || to);
  // One tab per analysed video; the parts of a merged recording share one tab.
  const inGroup = new Set(groups.flatMap((g) => g.members));
  const done = new Set(recs.filter((r) => r.processing_state === "done").map((r) => r.id));
  const groupTabs = groups.filter((g) => g.members.some((id) => done.has(id)));
  const soloTabs = recs.filter((r) => done.has(r.id) && !inGroup.has(r.id));
  return (
    <div className="grid" style={{ gap: 10 }}>
      {groupTabs.length + soloTabs.length > 1 && (
        <nav className="tabs videotabs" aria-label="Videos">
          <a href={`#/m/${missionId}/events`} className={rec === "" ? "on" : ""} onClick={(e) => { e.preventDefault(); setRec(""); }}>All videos</a>
          {groupTabs.map((g) => (
            <a key={`g${g.id}`} href={`#/m/${missionId}/events/rec-g${g.id}`} className={rec === `g${g.id}` ? "on" : ""}
              title={g.members.map((id) => recs.find((r) => r.id === id)?.rel_path).join(" + ")}
              onClick={(e) => { e.preventDefault(); setRec(`g${g.id}`); }}>{g.name}
              <span className="n">{g.members.reduce((a, id) => a + (recs.find((r) => r.id === id)?.events ?? 0), 0)}</span></a>))}
          {soloTabs.map((r) => (
            <a key={r.id} href={`#/m/${missionId}/events/rec-${r.id}`} className={String(r.id) === rec ? "on" : ""} title={r.rel_path}
              onClick={(e) => { e.preventDefault(); setRec(String(r.id)); }}>{r.rel_path.split("/").pop()}<span className="n">{r.events}</span></a>))}
        </nav>)}
      <CountsCard missionId={missionId} versionId={rec} refresh={timeRevision} />
      <InViewCard missionId={missionId} versionId={rec} refresh={timeRevision} />
      <ZonesCard missionId={missionId} versionId={rec} refresh={timeRevision} />
      <div className="card">
        <h2>Recordings</h2>
        <table className="t"><thead><tr><th>File</th><th>Start (UTC)</th><th className="num">Length</th><th>Frame rate</th><th>Processing</th><th className="num">Events</th><th>For learning</th></tr></thead>
          <tbody>{recs.filter((r) => !rec || (rec.startsWith("g") ? groups.find((g) => `g${g.id}` === rec)?.members.includes(r.id) : String(r.id) === rec)).map((r) => (
            <tr key={r.id}>
              <td className="mono bidi">{r.rel_path}
                {r.processing_state === "done" && <div><a className="play-link" href={`#/m/${missionId}/events/play-${r.id}`}>▶ Play and review</a></div>}</td>
              <td>
                <div className="mono">{r.start_utc ?? <span className="muted" title="No confirmed UTC start">unknown</span>}</div>
                <div className="muted">{r.start_source === "manual" ? `operator confirmed by ${r.start_revision_author} · ${r.start_revision_created_at}` : r.start_source === "filename" ? "from filename" : "no UTC start"}</div>
                <details style={{ marginTop: 4 }}>
                  <summary>{r.start_source === "manual" ? "Edit UTC start" : "Set UTC start"}</summary>
                  <div className="grid" style={{ gap: 5, marginTop: 5 }}>
                    <input type="datetime-local" step={1} aria-label={`UTC start for ${r.rel_path}`}
                      value={toPicker(startTimes[r.id] ?? r.start_utc ?? "")}
                      onChange={(e) => setStartTimes({ ...startTimes, [r.id]: fromPicker(e.target.value) })} />
                    <select aria-label="UTC start precision" value={precisions[r.id] ?? r.start_precision ?? "second"}
                      onChange={(e) => setPrecisions({ ...precisions, [r.id]: e.target.value })}>
                      <option value="minute">Minute</option><option value="second">Second</option>
                    </select>
                    <button disabled={!(startTimes[r.id] ?? r.start_utc)} onClick={() => saveStart(r)}>Save confirmed UTC start</button>
                    {r.start_source === "manual" && <button onClick={() => clearStart(r)}>Clear manual correction</button>}
                    <span className="muted">Pick the date and time in UTC. This changes event UTC metadata, not video frames.</span>
                  </div>
                </details>
              </td>
              <td className="num mono">{fmtT(r.duration_s)}</td>
              <td>{r.nominal_fps ? `${Number(r.nominal_fps).toFixed(2)} fps` : ""}{r.variable_frame_rate ? " (variable)" : ""}</td>
              <td>{r.error ? <span className="badge failed" title={r.error}>cannot decode</span> : r.processing_state ?? ""}</td>
              <td className="num">{r.events}</td>
              <td>
                <div>{REVIEW_TEXT[r.review] ?? ""}</div>
                {r.processing_state === "done" && (r.approved
                  ? <button onClick={() => review(r, "reopen")}>Reopen</button>
                  : <button onClick={() => review(r, r.pending ? "confirm_rest_and_approve" : "approve")}
                      title="Say that every target in this recording is an event and every event is right; events not decided yet are confirmed">
                      {r.pending ? `Confirm ${r.pending} left and approve` : "Approve recording"}</button>)}
              </td>
            </tr>))}</tbody></table>
        <p className="muted" style={{ marginBottom: 0 }}>Reviewed events teach the vehicle detector on the Learning tab. Approving a
          recording also says that nothing was missed, so the empty road and the rejected detections in its frames are learnt
          as background, as approving a still image does.</p>
      </div>
      <div className="card row">
        <select value={rec} onChange={(e) => setRec(e.target.value)} aria-label="Recording">
          <option value="">All recordings</option>
          {groups.map((g) => <option key={`g${g.id}`} value={`g${g.id}`}>{g.name} (merged)</option>)}
          {recs.map((r) => <option key={r.id} value={r.id}>{r.rel_path}</option>)}
        </select>
        <select value={status} onChange={(e) => setStatus(e.target.value)} aria-label="Review status">
          <option value="">Any review status</option>{Object.entries(STATUS_LABEL).map(([k, v]) => <option key={k} value={k}>{v}</option>)}
        </select>
        <select value={flag} onChange={(e) => setFlag(e.target.value)} aria-label="Flag">
          <option value="">Any flag</option>{Object.entries(FLAG_LABEL).map(([k, v]) => <option key={k} value={k}>{v}</option>)}
        </select>
        <span className="muted">{filtering ? `${shown.length} of ${evs?.length ?? 0}` : evs?.length ?? 0} events</span>
        <span style={{ flex: 1 }} />
        {rec && !rec.startsWith("g") && <button className="primary" onClick={() => go(`/m/${missionId}/events/play-${rec}`)}>▶ Play and review this video</button>}
        <button onClick={doExport}>Export events</button>
      </div>
      {evs && evs.length > 0 && <div className="card row attr-filters">
        <span className="muted small">What they did</span>
        <select aria-label="Direction" value={dir} onChange={(e) => setDir(e.target.value)}>
          <option value="">Any direction</option>{DIRS.map((d) => <option key={d} value={d}>{ARROW[d]} {d === "stayed" ? "stayed in place" : `went ${d}`}</option>)}
        </select>
        <select aria-label="Speed" value={speed} onChange={(e) => setSpeed(e.target.value)}>
          <option value="">Any speed</option>
          {kmh && KMH.map(([lo, hi]) => <option key={lo} value={`k${lo}-${hi}`}>{hi > 900 ? `over ${lo}` : `${lo} to ${hi}`} km/h</option>)}
          {pxSpeeds.length > 2 && <><option value="slow">Slowest third{kmh ? " (camera not measured)" : ""}</option>
            <option value="medium">Middle third</option><option value="fast">Fastest third</option></>}
        </select>
        <select aria-label="Time in view" value={inView} onChange={(e) => setInView(e.target.value)}>
          <option value="">Any time in view</option>{IN_VIEW.map((s) => <option key={s} value={s}>In view {fmtDur(s)} or more</option>)}
        </select>
        <select aria-label="Time stopped" value={stopped} onChange={(e) => setStopped(e.target.value)}>
          <option value="">Stopped or not</option>{STOPPED.map((s) => <option key={s} value={s}>Stopped {fmtDur(s)} or more</option>)}
        </select>
        <span className="muted small">How they looked</span>
        <select aria-label="Colour" value={colour} onChange={(e) => setColour(e.target.value)}>
          <option value="">Any colour</option>{COLOURS.map((c) => <option key={c} value={c}>{c}</option>)}
        </select>
        {people && <>
          <select aria-label="Upper body colour" value={upper} onChange={(e) => setUpper(e.target.value)}>
            <option value="">Any top (people)</option>{COLOURS.map((c) => <option key={c} value={c}>{c} top</option>)}
          </select>
          <select aria-label="Lower body colour" value={lower} onChange={(e) => setLower(e.target.value)}>
            <option value="">Any trousers or skirt</option>{COLOURS.map((c) => <option key={c} value={c}>{c} below</option>)}
          </select></>}
        {Object.keys(zoneNames).length > 0 && <>
          <span className="muted small">Where they went</span>
          <select aria-label="Line or area" value={zone} onChange={(e) => setZone(e.target.value)}>
            <option value="">Any line or area</option>
            {Object.entries(zoneNames).map(([uid, n]) => (evs ?? []).some((e) => e.attrs?.zones?.[uid]?.includes("inside"))
              ? <option key={uid} value={`${uid}:any`}>Was in {n}</option>
              : [<option key={uid} value={`${uid}:any`}>Crossed {n}</option>,
                 <option key={`${uid}i`} value={`${uid}:in`}>Crossed {n}, in</option>,
                 <option key={`${uid}o`} value={`${uid}:out`}>Crossed {n}, out</option>])}
          </select>
          <select aria-label="From" value={from} onChange={(e) => setFrom(e.target.value)}>
            <option value="">From anywhere</option>{Object.entries(zoneNames).map(([uid, n]) => <option key={uid} value={uid}>From {n}</option>)}
          </select>
          <select aria-label="To" value={to} onChange={(e) => setTo(e.target.value)}>
            <option value="">to anywhere</option>{Object.entries(zoneNames).map(([uid, n]) => <option key={uid} value={uid}>to {n}</option>)}
          </select></>}
        {filtering && <button onClick={() => { setDir(""); setSpeed(""); setInView(""); setStopped(""); setColour(""); setUpper(""); setLower(""); setZone(""); setFrom(""); setTo(""); }}>Clear</button>}
        {reading && <span className="muted small">Reading colours…</span>}
      </div>}
      {rec && !rec.startsWith("g") && evs && evs.length > 0 &&
        <MapCard missionId={missionId} versionId={Number(rec)} ids={filtering ? shown.map((e) => e.id) : null} />}
      {msg && <div className={`msg ${msg.kind}`}>{msg.text}</div>}
      {evs && evs.length === 0 && <div className="card muted">No events match. Events appear after the recordings are processed.</div>}
      {evs && evs.length > 0 && shown.length === 0 && <div className="card muted">No event matches these filters.</div>}
      {evs && shown.length > 0 && (
        <div className="tablewrap">
          <table className="t">
            <thead><tr><th style={{ width: 90 }}></th><th>Event</th><th>Recording</th><th>Start</th><th>End</th><th>Start (UTC)</th><th>Class</th><th>Went</th><th>Path</th><th className="num">Speed</th><th className="num">In view</th><th>Colour</th><th>Review</th><th>Flags</th></tr></thead>
            <tbody>{shown.map((e) => (
              <tr key={e.id} className="click" onClick={() => go(`/m/${missionId}/events/${e.id}`)}>
                <td><img loading="lazy" src={`/api/missions/${missionId}/events/${e.id}/thumb?h=48`} alt="" style={{ height: 40, display: "block", borderRadius: 3 }} /></td>
                <td className="mono">{e.key}</td>
                <td className="mono bidi">{e.file}</td>
                <td className="mono">{fmtT(e.first_t)}</td>
                <td className="mono">{fmtT(e.last_t)}</td>
                <td className="mono">{e.first_utc ?? <span className="muted">unknown</span>}</td>
                <td>{e.cls}{e.cls !== e.machine_cls ? <span className="muted"> (machine: {e.machine_cls})</span> : ""}</td>
                <td title={e.attrs?.direction ?? ""}>{e.attrs?.direction ? `${ARROW[e.attrs.direction]} ${e.attrs.direction}` : ""}</td>
                <td className="small">{(e.attrs?.path ?? []).map((u) => zoneNames[u] ?? "?").join(" → ")}</td>
                <td className="num mono">{e.attrs ? fmtSpeed(e.attrs) : ""}</td>
                <td className="num mono">{e.attrs ? fmtDur(e.attrs.dwell_s) : ""}{e.attrs && e.attrs.stopped_s >= 3 ? <div className="muted small">stopped {fmtDur(e.attrs.stopped_s)}</div> : null}</td>
                <td>{e.attrs && (e.attrs.upper !== null || e.attrs.lower !== null) && e.attrs.colour == null && (e.attrs.upper || e.attrs.lower)
                  ? <span className="small">top <ColourChips v={e.attrs.upper} /> · below <ColourChips v={e.attrs.lower} /></span>
                  : <ColourChips v={e.attrs?.colour} />}</td>
                <td><span className={`badge ${e.status === "confirmed" ? "approved" : e.status === "rejected" ? "failed" : ""}`}>{STATUS_LABEL[e.status]}</span>{e.parts.length > 1 ? <span className="muted"> · {e.parts.length} parts</span> : ""}</td>
                <td>{e.continues_key && <span className="badge part" style={{ marginRight: 4 }} title="The same target, carried on from the part before; counted once">continues {e.continues_key}</span>}
                  {e.flags.map((f) => <span key={f} className="badge flag" style={{ marginRight: 4 }}>{FLAG_LABEL[f] ?? f}</span>)}</td>
              </tr>))}</tbody>
          </table>
        </div>
      )}
    </div>
  );
}

const FOUND_LABEL: Record<string, string> = { standard: "standard detector", sam3: "SAM 3.1", both: "both" };

/** Which detector found the tracked boxes of an event, e.g. "standard detector 41 · both 7 · SAM 3.1 2 frames". */
function foundBy(obs: Obs[]): string {
  const n: Record<string, number> = {};
  for (const o of obs) n[o.found_by ?? "standard"] = (n[o.found_by ?? "standard"] ?? 0) + 1;
  return Object.entries(n).sort((a, b) => b[1] - a[1]).map(([k, v]) => `${FOUND_LABEL[k] ?? k} ${v}`).join(" · ") + " frames";
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
  const [classes, setClasses] = useState<string[]>(CLASSES);   // the vehicle classes and the mission's own
  const video = useRef<HTMLVideoElement>(null);
  const still = useRef<HTMLImageElement>(null);
  const canvas = useRef<HTMLCanvasElement>(null);

  const load = useCallback(async () => {
    try {
      const e = await api(`/missions/${missionId}/events/${eventId}`);
      setEv(e);
      const r = await api(`/missions/${missionId}/events?version_id=${e.source_version_id}`);
      setOthers(r.events.filter((o: VEvent) => o.id !== eventId));
      if (r.classes) setClasses(r.classes);
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
    if (before === "" || after === "") { setMsg({ kind: "err", text: "Choose the seconds of context before and after the event." }); return; }
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
              <tr><td>Found by</td><td>{foundBy(ev.observations)}</td></tr>
              {ev.attrs && <>
                <tr><td>Went</td><td>{ev.attrs.direction ? `${ARROW[ev.attrs.direction]} ${ev.attrs.direction === "stayed" ? "stayed in place" : ev.attrs.direction}` : ""}</td></tr>
                <tr><td>Speed</td><td>{ev.attrs.speed == null ? <span className="muted">too short to tell</span> : <>{fmtSpeed(ev.attrs)} <span className="muted">(top {Math.round(ev.attrs.top_speed ?? 0)})</span>
                  {ev.attrs.unit === "px/s" && <span className="muted small"> · measure the camera in the player for km/h</span>}</>}</td></tr>
                <tr><td>In view</td><td>{fmtDur(ev.attrs.dwell_s)}{ev.attrs.stopped_s >= 1 ? `, stopped ${fmtDur(ev.attrs.stopped_s)}` : ""}</td></tr>
                <tr><td>Size</td><td className="mono">{ev.attrs.w} × {ev.attrs.h} {ev.attrs.size_unit}</td></tr>
                {ev.attrs.colour == null && (ev.attrs.upper != null || ev.attrs.lower != null)
                  ? <tr><td>Clothes</td><td>top <ColourChips v={ev.attrs.upper} /> · below <ColourChips v={ev.attrs.lower} /></td></tr>
                  : <tr><td>Colour</td><td><ColourChips v={ev.attrs.colour} /></td></tr>}
              </>}
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
                <option value="">Set class…</option>{classes.map((c) => <option key={c}>{c}</option>)}
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
              <label className="row">Seconds before <Choice label="Seconds before" value={before === "" ? "" : Number(before)} onChange={(n) => setBefore(String(n))} options={CLIP_SECONDS} /></label>
              <label className="row">after <Choice label="Seconds after" value={after === "" ? "" : Number(after)} onChange={(n) => setAfter(String(n))} options={CLIP_SECONDS} /></label>
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
