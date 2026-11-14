import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api, ApiError, post } from "../api";
import { fmtT } from "./Events";
import { NamePick } from "../components/Pick";
import { Labels, ScoreKey, Tails, TAILS, boxColourSetting, dotColourSetting, hueOf, scoreAt, type TailLen } from "../components/TrackMarks";

// The player: a recording plays with every event's box drawn on it, and is reviewed and edited where it plays, as a
// still image is. Click a box (or a bar on the timeline) to pick its event; drag on the paused picture to redraw the
// picked event's box on this frame, or, after "Missed target", to add what the analysis did not find.

type PBox = [number, number, number, number, number];          // t, x1, y1, x2, y2
type PEvent = {
  id: number; key: string; first_t: number; last_t: number; cls: string; machine_cls: string; status: string;
  origin: "machine" | "human"; fixes: { t: number; box: number[] }[]; boxes: PBox[]; flags: string[];
  scores?: (number | null)[];                                   // the detector's score beside each box (null: yours)
};
// A target the operator marked (one SAM 3.1 does not know): followed both ways from their boxes, then confirmed.
type Marked = { id: number; cls: string; marks: { t: number; box: number[] }[]; steps: { t: number; box: number[]; score: number }[];
  start_t: number | null; end_t: number | null; state: "suggested" | "confirmed" };
const SURE = 0.55;
type Overlay = {
  recording: { id: number; rel_path: string; width: number | null; height: number | null; duration_s: number | null;
    nominal_fps: number | null; review: string; approved: boolean; pending: number };
  events: PEvent[]; classes: string[];
};

const STATUS_TEXT: Record<string, string> = {
  pending: "not reviewed", confirmed: "confirmed", rejected: "rejected", duplicate: "duplicate", merged: "merged",
};
const REVIEW_TEXT: Record<string, string> = {
  full: "Approved for learning", partial: "Some events reviewed", none: "Not reviewed",
};
// Line style and a letter say what a box is, not colour alone (UX-01): M machine, H added or redrawn by you.
const LOOK: Record<string, { stroke: string; dash: string }> = {
  pending: { stroke: "var(--machine)", dash: "" },
  confirmed: { stroke: "var(--human)", dash: "10 3 2 3" },
  human: { stroke: "var(--human)", dash: "8 4" },
  rejected: { stroke: "var(--muted)", dash: "2 4" },
};
const SPEEDS = [0.25, 0.5, 1, 2, 4];
const SHOW = { all: "Every box", review: "Boxes not reviewed yet", none: "No boxes" } as const;

/** The event's box at time t: its box on the nearest analysed frame (within a sixth of a second), else the two
 *  around t blended (an analysed gap of up to 2 s; any gap between the boxes you drew for a target you added). */
function boxAt(e: PEvent, t: number): number[] | null {
  const b = e.boxes;
  if (!b.length || t < e.first_t - 0.17 || t > e.last_t + 0.17) return null;
  let i = 0;
  while (i < b.length && b[i][0] < t) i++;
  const after = b[i], before = b[i - 1];
  const near = !before ? after : !after ? before : t - before[0] <= after[0] - t ? before : after;
  if (Math.abs(near[0] - t) <= 0.17) return near.slice(1);
  if (!before || !after || (after[0] - before[0] > 2 && e.origin !== "human")) return null;
  const w = (t - before[0]) / (after[0] - before[0]);
  return [1, 2, 3, 4].map((k) => before[k] + w * (after[k] - before[k]));
}

/** Timeline lanes: each event a bar from its first to its last second, packed so bars do not overlap. */
function lanes(evs: PEvent[]): PEvent[][] {
  const out: PEvent[][] = [];
  for (const e of [...evs].sort((a, b) => a.first_t - b.first_t)) {
    const lane = out.find((l) => l[l.length - 1].last_t + 0.3 < e.first_t);
    if (lane) lane.push(e); else out.push([e]);
  }
  return out;
}

// Tails and labels (components/TrackMarks): the box's look, and what pointing at a box says (all the small label
// leaves out).
const lookOfEv = (e: PEvent) => LOOK[e.status === "rejected" ? "rejected" : e.origin === "human" || e.fixes.length ? "human" : e.status] ?? LOOK.pending;
function tip(e: PEvent, t: number): string {
  const s = scoreAt(e, t);
  return `${e.key} · ${e.cls} · ${STATUS_TEXT[e.status] ?? e.status}${s != null ? ` · score ${s.toFixed(2)}` : ""}`;
}

export default function Player({ missionId, versionId }: { missionId: string; versionId: number }) {
  const [ov, setOv] = useState<Overlay | null>(null);
  // The events panel folds away so the video can take the whole width; remembered on this computer.
  const [panel, setPanel] = useState<boolean>(() => { try { return localStorage.getItem("twb.player-panel") !== "off"; } catch { return true; } });
  const [err, setErr] = useState<string | null>(null);
  const [msg, setMsg] = useState<{ kind: string; text: string } | null>(null);
  const [now, setNow] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [show, setShow] = useState<keyof typeof SHOW>("all");
  const [sel, setSel] = useState<number | null>(null);
  const [adding, setAdding] = useState(false);
  const [addCls, setAddCls] = useState("car");
  const [drag, setDrag] = useState<{ x0: number; y0: number; x1: number; y1: number } | null>(null);
  const [fallback, setFallback] = useState(false);
  const [help, setHelp] = useState(false);
  // Tails and labels: how long a tail, and what the small label over each target says. Kept for next time.
  const keep = <T,>(k: string, v: T) => { try { localStorage.setItem(`twb.player.${k}`, JSON.stringify(v)); } catch { /* not kept */ } };
  const kept = <T,>(k: string, d: T): T => { try { const v = localStorage.getItem(`twb.player.${k}`); return v == null ? d : JSON.parse(v); } catch { return d; } };
  const [tail, setTailRaw] = useState<TailLen>(() => kept("tail", "3" as TailLen));
  const [showId, setShowIdRaw] = useState<boolean>(() => kept("ids", true));
  const [showScore, setShowScoreRaw] = useState<boolean>(() => kept("scores", false));
  const setTail = (v: TailLen) => { setTailRaw(v); keep("tail", v); };
  // Colours of the boxes and the tail's dots: chosen in Settings, Video.
  const [dotColour] = useState<string>(dotColourSetting);
  const [boxColour] = useState<string>(boxColourSetting);
  // The box's look with the chosen colour (the line style keeps telling the review state).
  const lookFor = (e: PEvent) => {
    const l = lookOfEv(e);
    return boxColour === "state" ? l : { ...l, stroke: boxColour === "target" ? hueOf(e.id) : boxColour };
  };
  const setShowId = (v: boolean) => { setShowIdRaw(v); keep("ids", v); };
  const setShowScore = (v: boolean) => { setShowScoreRaw(v); keep("scores", v); };
  // Measuring the camera: two points on the road and the real distance between them give km/h and metres.
  type Cal = { own: boolean; points: number[]; meters: number; m_per_px: number; from_version_id: number } | null;
  const [cal, setCal] = useState<Cal>(null);
  const [distances, setDistances] = useState<number[]>([]);
  const [measuring, setMeasuring] = useState(false);
  const [line, setLine] = useState<number[] | null>(null);
  // Lines and areas of this camera (every video in the same folder): drawn here, counted on the events page.
  type Zone = { uid: string; kind: "line" | "area"; name: string; points: number[][] };
  const [zones, setZones] = useState<Zone[]>([]);
  const [zoneNames, setZoneNames] = useState<{ line: string[]; area: string[] }>({ line: [], area: [] });
  const [zoning, setZoning] = useState<"" | "line" | "area">("");
  const [zoneName, setZoneName] = useState("");
  const [zpts, setZpts] = useState<number[][]>([]);
  const video = useRef<HTMLVideoElement>(null);
  const svg = useRef<SVGSVGElement>(null);
  // Teaching a new target: draw it once, the app follows it both ways; correct, set where it is in view, confirm.
  const [marking, setMarking] = useState(false);
  const [markCls, setMarkCls] = useState("");
  const [marked, setMarked] = useState<Marked[]>([]);
  const [selMark, setSelMark] = useState<number | null>(null);
  const [following, setFollowing] = useState(false);
  const loadMarks = useCallback(() => api(`/missions/${missionId}/recordings/${versionId}/marks`)
    .then((r) => setMarked(r.targets)).catch(() => undefined), [missionId, versionId]);
  useEffect(() => { loadMarks(); }, [loadMarks]);

  const load = useCallback(async () => {
    try { setOv(await api(`/missions/${missionId}/recordings/${versionId}/overlay`)); setErr(null); }
    catch (e) { setErr((e as ApiError).message); }
  }, [missionId, versionId]);
  useEffect(() => { load(); }, [load]);
  useEffect(() => {
    api(`/missions/${missionId}/recordings/${versionId}/calibration`)
      .then((r) => { setCal(r.calibration); setDistances(r.distances_m); }).catch(() => undefined);
  }, [missionId, versionId]);
  // Where a drone or aircraft camera was (DJI subtitle file or KLV in the video), when the recording carries it.
  // Optional: read only when the operator asks for it; nothing else depends on it.
  const [tel, setTel] = useState<{ source: string | null; summary?: any; samples?: any[] } | null>(null);
  useEffect(() => { setTel(null); }, [missionId, versionId]);
  const askFlight = () => api(`/missions/${missionId}/recordings/${versionId}/telemetry`)
    .then((r) => setTel(r)).catch((e) => setMsg({ kind: "err", text: (e as ApiError).message }));
  const loadZones = useCallback(() => api(`/missions/${missionId}/recordings/${versionId}/zones`)
    .then((r) => { setZones(r.zones); setZoneNames({ line: r.line_names, area: r.area_names }); }).catch(() => undefined), [missionId, versionId]);
  useEffect(() => { loadZones(); }, [loadZones]);
  const startZone = (k: "" | "line" | "area") => {
    setZoning(k); setZpts([]); setAdding(false); setMeasuring(false); setSel(null); pause();
    if (k) setZoneName((k === "line" ? zoneNames.line : zoneNames.area).find((n) => !zones.some((z) => z.name === n)) ?? "");
  };
  const saveZone = async (pts: number[][]) => {
    await act(async () => {
      await post(`/missions/${missionId}/recordings/${versionId}/zones`, { kind: zoning, name: zoneName, points: pts });
      await loadZones(); setZoning(""); setZpts([]);
    }, `${zoning === "line" ? "Line" : "Area"} "${zoneName}" added. Its counts are on the events page, for every video from this camera.`);
  };
  const deleteZone = (z: Zone) => act(async () => {
    await api(`/missions/${missionId}/zones/${z.uid}`, { method: "DELETE" }); await loadZones();
  }, `"${z.name}" removed.`);
  const saveMeasure = async (meters: number) => {
    if (!line) return;
    await act(async () => {
      const r = await api(`/missions/${missionId}/recordings/${versionId}/calibration`, { method: "PUT", body: { points: line, meters } });
      setCal(r.calibration); setLine(null); setMeasuring(false);
    }, `Measured: ${meters} m. Speeds of this video, and of the other videos from the same folder without a measure of their own, are now in km/h.`);
  };
  const clearMeasure = () => act(async () => {
    const r = await api(`/missions/${missionId}/recordings/${versionId}/calibration`, { method: "DELETE" });
    setCal(r.calibration); setLine(null);
  }, "Measure removed: speeds of this video are in pixels per second again.");

  // The playhead follows the video every animation frame; in the frame-by-frame fallback a timer moves it.
  useEffect(() => {
    if (fallback) {
      if (!playing) return;
      const id = setInterval(() => setNow((t) => Math.min(t + 0.2 * speed, ov?.recording.duration_s ?? t)), 200);
      return () => clearInterval(id);
    }
    let raf = 0;
    const loop = () => { if (video.current) setNow(video.current.currentTime); raf = requestAnimationFrame(loop); };
    raf = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(raf);
  }, [fallback, playing, speed, ov?.recording.duration_s]);
  useEffect(() => { if (video.current) video.current.playbackRate = speed; }, [speed]);

  const W = ov?.recording.width ?? 1280, H = ov?.recording.height ?? 720;
  const dur = ov?.recording.duration_s ?? 0;
  const frame = 1 / (ov?.recording.nominal_fps || 25);
  const evs = ov?.events.filter((e) => e.status !== "merged") ?? [];
  const selEv = evs.find((e) => e.id === sel) ?? null;
  const classes = ov?.classes ?? ["car"];
  const toReview = evs.filter((e) => e.status === "pending").sort((a, b) => a.first_t - b.first_t);
  const myBoxHere = selEv?.fixes.find((f) => Math.abs(f.t - now) < frame * 0.6) ?? null;
  const tl = useMemo(() => lanes(evs), [ov]);

  const seek = (t: number) => {
    const c = Math.max(0, Math.min(dur || t, t));
    if (fallback) setNow(c); else if (video.current) { video.current.currentTime = c; setNow(c); }
  };
  const pause = () => { video.current?.pause(); setPlaying(false); };
  const toggle = () => {
    if (fallback) { setPlaying((p) => !p); return; }
    const v = video.current;
    if (!v) return;
    if (v.paused) { v.play(); setPlaying(true); } else { v.pause(); setPlaying(false); }
  };
  const pick = (e: PEvent | null, jump = false) => {
    setSel(e?.id ?? null); setAdding(false);
    if (e && jump) { pause(); seek(e.boxes.find((b) => b[0] >= e.first_t)?.[0] ?? e.first_t); }
  };
  const act = async (fn: () => Promise<unknown>, ok?: string) => {
    setMsg(null);
    try { await fn(); await load(); if (ok) setMsg({ kind: "ok", text: ok }); }
    catch (e) { setMsg({ kind: "err", text: (e as ApiError).message }); }
  };
  const revise = (id: number, action: string, payload: object = {}) =>
    act(() => post(`/missions/${missionId}/events/${id}/revise`, { action, ...payload }));
  // Confirm or reject, then go on to the next event not reviewed yet, so a review is one key per event.
  const decide = async (action: "confirm" | "reject") => {
    if (!selEv) return;
    const next = toReview.find((e) => e.first_t >= selEv.first_t && e.id !== selEv.id) ?? toReview.find((e) => e.id !== selEv.id);
    await revise(selEv.id, action);
    if (next) pick(next, true);
  };
  const setClass = (cls: string) => {
    if (selEv && selEv.cls !== cls) revise(selEv.id, "set_class", { cls });
    else setAddCls(cls);
  };
  const approve = () => act(() => post(`/missions/${missionId}/recordings/${versionId}/review`,
    { action: ov?.recording.approved ? "reopen" : ov?.recording.pending ? "confirm_rest_and_approve" : "approve" }),
    ov?.recording.approved ? "Reopened: only the events you reviewed teach now." :
      "Approved for learning: every target here teaches, and the empty road and rejected boxes teach as background.");

  // Screen point to video pixels, through the letterboxed picture.
  const toImage = (ev: React.PointerEvent) => {
    const r = svg.current!.getBoundingClientRect();
    const s = Math.min(r.width / W, r.height / H);
    const x = (ev.clientX - r.left - (r.width - W * s) / 2) / s, y = (ev.clientY - r.top - (r.height - H * s) / 2) / s;
    return { x: Math.max(0, Math.min(W, x)), y: Math.max(0, Math.min(H, y)) };
  };
  const curMark = marked.find((x) => x.id === selMark) ?? null;
  const drawing = adding || marking || measuring || !!zoning || selEv != null;
  const onDown = (ev: React.PointerEvent) => {
    if (!drawing || ev.button !== 0) return;
    pause();
    const p = toImage(ev);
    if (zoning === "area") {                             // an area: one click per corner; click the first to close it
      const first = zpts[0];
      if (first && zpts.length >= 3 && Math.hypot(first[0] - p.x, first[1] - p.y) < 12 * Math.max(W / (svg.current?.clientWidth || W), 1)) saveZone(zpts);
      else setZpts([...zpts, [p.x, p.y]]);
      return;
    }
    setDrag({ x0: p.x, y0: p.y, x1: p.x, y1: p.y });
    (ev.target as Element).setPointerCapture?.(ev.pointerId);
  };
  const onMove = (ev: React.PointerEvent) => { if (drag) { const p = toImage(ev); setDrag({ ...drag, x1: p.x, y1: p.y }); } };
  const onUp = async () => {
    if (!drag) return;
    const box = [Math.min(drag.x0, drag.x1), Math.min(drag.y0, drag.y1), Math.max(drag.x0, drag.x1), Math.max(drag.y0, drag.y1)];
    setDrag(null);
    if (zoning === "line") {
      if (Math.hypot(drag.x1 - drag.x0, drag.y1 - drag.y0) >= 10) saveZone([[drag.x0, drag.y0], [drag.x1, drag.y1]]);
      return;
    }
    if (measuring) {                                     // a line between two points, not a box
      if (Math.hypot(drag.x1 - drag.x0, drag.y1 - drag.y0) >= 10) setLine([drag.x0, drag.y0, drag.x1, drag.y1]);
      return;
    }
    if (box[2] - box[0] < 4 || box[3] - box[1] < 4) { if (!adding) setSel(null); return; }   // a click on the picture: let go
    const t = fallback ? now : video.current?.currentTime ?? now;
    if (marking) {
      if (!curMark && !markCls.trim()) { setMsg({ kind: "err", text: "Choose what the target is first." }); return; }
      setFollowing(true);
      await act(async () => {
        const r = curMark ? await post(`/missions/${missionId}/marks/${curMark.id}/correct`, { t, box })
          : await post(`/missions/${missionId}/recordings/${versionId}/marks`, { cls: markCls, t, box });
        await loadMarks(); setSelMark(r.id);
      }, curMark ? "Followed again from all your boxes. Check the start and the end, then confirm."
        : "Followed both ways. Play through it: where the box goes wrong, pause and draw it again. Then check the start and the end, and confirm.");
      setFollowing(false);
      return;
    }
    if (adding) {
      await act(async () => {
        const e = await post(`/missions/${missionId}/recordings/${versionId}/targets`, { t, box, cls: addCls });
        setSel(e.id); setAdding(false);
      }, `Added ${addCls}. Pause on another frame and draw it again to follow it; the frames between are filled in.`);
    } else if (selEv) {
      await revise(selEv.id, "set_box", { t, box });
    }
  };

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement).tagName;
      if (!ov || tag === "INPUT" || tag === "SELECT" || tag === "TEXTAREA" || e.ctrlKey || e.metaKey || e.altKey) return;
      const k = e.key.toLowerCase();
      const i = toReview.findIndex((x) => x.id === sel);
      if (k === " ") toggle();
      else if (k === "arrowright" || k === ".") { pause(); seek(now + (e.shiftKey ? 1 : frame)); }
      else if (k === "arrowleft" || k === ",") { pause(); seek(now - (e.shiftKey ? 1 : frame)); }
      else if (k === "j" && toReview.length) pick(toReview[(i + 1) % toReview.length], true);
      else if (k === "k" && toReview.length) pick(toReview[(i - 1 + toReview.length) % toReview.length], true);
      else if (k === "c" && selEv) decide("confirm");
      else if ((k === "x" || k === "delete" || k === "backspace") && selEv) decide("reject");
      else if (k === "b") { setAdding((a) => !a); setSel(null); setMeasuring(false); pause(); }
      else if (k === "t") { setMarking((v) => !v); setSelMark(null); setAdding(false); setSel(null); pause(); }
      else if (k === "m") { setMeasuring((v) => !v); setLine(null); setAdding(false); setSel(null); pause(); }
      else if (k === "a") approve();
      else if (k === "h") setShow((s) => (s === "none" ? "all" : "none"));
      else if (k === "s") setShowScore(!showScore);
      else if (k === "]") setPanel((p) => { try { localStorage.setItem("twb.player-panel", p ? "off" : "on"); } catch { /* this visit only */ } return !p; });
      else if (k === "l") startZone(zoning === "line" ? "" : "line");
      else if (k === "enter" && zoning === "area" && zpts.length >= 3) saveZone(zpts);
      else if (k === "escape") { setDrag(null); setAdding(false); setMarking(false); setSelMark(null); setMeasuring(false); setLine(null); setZoning(""); setZpts([]); setSel(null); setHelp(false); }
      else if (k === "?") setHelp((h) => !h);
      else if (/^[1-9]$/.test(k) && Number(k) <= classes.length) setClass(classes[Number(k) - 1]);
      else return;
      e.preventDefault();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  });

  if (err) return <div className="msg err">{err}</div>;
  if (!ov) return <div className="muted">Loading…</div>;
  const rec = ov.recording;
  const togglePanel = () => setPanel((p) => { try { localStorage.setItem("twb.player-panel", p ? "off" : "on"); } catch { /* this visit only */ } return !p; });
  // Lines and labels keep their size on screen whatever the video's resolution.
  const cw = svg.current?.clientWidth || W, ch = svg.current?.clientHeight || H;
  const px = Math.max(W / cw, H / ch);              // video pixels per screen pixel
  const sw = 2 * px;
  const shown = evs.filter((e) => show === "all" ? true : show === "review" ? e.status === "pending" : false);
  const fs = 12 * px;

  return (
    <div className="grid player" style={{ gap: 10 }}>
      <div className="row">
        <a href={`#/m/${missionId}/events/rec-${versionId}`}>← Events of this video</a>
        <h2 style={{ margin: 0 }} className="mono bidi">{rec.rel_path.split("/").pop()}</h2>
        <span className={`badge ${rec.approved ? "approved" : ""}`}>{REVIEW_TEXT[rec.review] ?? rec.review}</span>
        <span className="muted">{toReview.length ? `${toReview.length} of ${evs.length} events to review` : `all ${evs.length} events reviewed`}</span>
        <span style={{ flex: 1 }} />
        <button className="iconbtn" aria-pressed={panel} onClick={togglePanel} title="Show or hide the events panel, so the video can take the whole width">{panel ? "Hide events" : "Events"}</button>
        <button className={rec.approved ? "" : "primary"} onClick={approve}
          title="Say every target in this video is marked and every box is right; it then teaches the detector (A)">
          {rec.approved ? "Reopen" : rec.pending ? `Confirm ${rec.pending} left and approve for learning` : "Approve for learning"}</button>
      </div>
      <div className="grid" style={{ gridTemplateColumns: panel ? "minmax(0, 1fr) 320px" : "minmax(0, 1fr)", gap: 10 }}>
        <div className="card" style={{ padding: 8 }}>
          <div className="stage" style={{ position: "relative", background: "#000", aspectRatio: `${W} / ${H}`, maxHeight: "calc(100vh - 330px)", minHeight: 240 }}>
            {!fallback && <video ref={video} src={`/api/missions/${missionId}/videos/${versionId}`} preload="auto"
              style={{ width: "100%", height: "100%", display: "block", objectFit: "contain" }}
              onPlay={() => setPlaying(true)} onPause={() => setPlaying(false)} onEnded={() => setPlaying(false)}
              onError={() => setFallback(true)} />}
            {fallback && <img alt={`Frame at ${fmtT(now)}`} src={`/api/missions/${missionId}/videos/${versionId}/frame?t=${now.toFixed(2)}&w=1920`}
              style={{ width: "100%", height: "100%", display: "block", objectFit: "contain" }} />}
            <svg ref={svg} viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="xMidYMid meet"
              style={{ position: "absolute", inset: 0, width: "100%", height: "100%", cursor: drawing ? "crosshair" : "default" }}
              onPointerDown={onDown} onPointerMove={onMove} onPointerUp={onUp}>
              <Tails events={shown} now={now} len={tail} sel={sel} px={px} lookOf={lookFor} boxAt={boxAt}
                fps={ov.recording.nominal_fps || 25} colour={dotColour} />
              {shown.map((e) => {
                const b = boxAt(e, now);
                if (!b) return null;
                const look = lookFor(e);
                const on = e.id === sel;
                return (
                  <g key={e.id} onPointerDown={(ev) => { if (!adding && !measuring && !zoning && e.id !== sel) { ev.stopPropagation(); pick(e); pause(); } }}
                    style={{ cursor: "pointer" }}>
                    <rect x={b[0]} y={b[1]} width={b[2] - b[0]} height={b[3] - b[1]} fill={on ? "rgba(255,255,255,0.06)" : "transparent"}
                      stroke={look.stroke} strokeDasharray={look.dash ? look.dash.split(" ").map((v) => Number(v) * sw / 2).join(" ") : undefined}
                      strokeWidth={on ? sw * 2 : sw} />
                    <title>{tip(e, now)}</title>
                  </g>
                );
              })}
              {show !== "none" && zones.map((z) => {
                const pts = z.points;
                if (z.kind === "area") return <g key={z.uid} className="measure-line">
                  <polygon points={pts.map((q) => q.join(",")).join(" ")} fill="rgba(74,163,255,0.10)" stroke="var(--accent, #4aa3ff)" strokeWidth={sw} strokeDasharray={`${4 * sw} ${2 * sw}`} />
                  <text x={pts[0][0]} y={pts[0][1] + fs} fontSize={fs} fill="var(--accent, #4aa3ff)" paintOrder="stroke" stroke="#000" strokeWidth={3 * px}>{z.name}</text></g>;
                const [[x1, y1], [x2, y2]] = pts;
                const mx = (x1 + x2) / 2, my = (y1 + y2) / 2, len = Math.hypot(x2 - x1, y2 - y1) || 1;
                // the "in" side: to the right of the line as drawn from its first point to its second
                const nx = -(y2 - y1) / len, ny = (x2 - x1) / len, a = 18 * px;
                return <g key={z.uid} className="measure-line">
                  <line x1={x1} y1={y1} x2={x2} y2={y2} stroke="var(--accent, #4aa3ff)" strokeWidth={sw * 1.5} />
                  <line x1={mx} y1={my} x2={mx + nx * a} y2={my + ny * a} stroke="var(--accent, #4aa3ff)" strokeWidth={sw} markerEnd="url(#zarrow)" />
                  <text x={mx + nx * a * 1.4} y={my + ny * a * 1.4} fontSize={fs} fill="var(--accent, #4aa3ff)" paintOrder="stroke" stroke="#000" strokeWidth={3 * px}>{z.name} · in</text></g>;
              })}
              <defs><marker id="zarrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse">
                <path d="M 0 0 L 10 5 L 0 10 z" fill="var(--accent, #4aa3ff)" /></marker></defs>
              {zoning === "area" && zpts.length > 0 && <g className="measure-line">
                <polyline points={zpts.map((q) => q.join(",")).join(" ")} fill="none" stroke="var(--human)" strokeWidth={sw} />
                {zpts.map((q, i) => <circle key={i} cx={q[0]} cy={q[1]} r={(i === 0 ? 6 : 4) * px} fill="var(--human)" />)}</g>}
              {zoning === "line" && drag && <line x1={drag.x0} y1={drag.y0} x2={drag.x1} y2={drag.y1} stroke="var(--human)" strokeWidth={sw * 1.5} />}
              {(() => {
                const l = measuring ? (drag ? [drag.x0, drag.y0, drag.x1, drag.y1] : line) : cal ? cal.points : null;
                if (!l) return null;
                const lab = measuring ? (line && !drag ? "how far apart?" : "") : `${cal!.meters} m${cal!.own ? "" : " (same camera)"}`;
                return <g className="measure-line">
                  <line x1={l[0]} y1={l[1]} x2={l[2]} y2={l[3]} stroke="var(--warn, #f2b84b)" strokeWidth={sw} strokeDasharray={measuring ? undefined : `${3 * sw} ${2 * sw}`} />
                  {[0, 2].map((k) => <circle key={k} cx={l[k]} cy={l[k + 1]} r={4 * px} fill="var(--warn, #f2b84b)" />)}
                  {lab && <text x={(l[0] + l[2]) / 2} y={(l[1] + l[3]) / 2 - 6 * px} fontSize={fs} fill="var(--warn, #f2b84b)" paintOrder="stroke"
                    stroke="#000" strokeWidth={3 * px} textAnchor="middle">{lab}</text>}
                </g>;
              })()}
              {show !== "none" && marked.map((mk) => {
                const st = mk.steps.reduce<Marked["steps"][number] | null>((b, x) => (!b || Math.abs(x.t - now) < Math.abs(b.t - now) ? x : b), null);
                const inRange = mk.start_t != null && mk.end_t != null && now >= mk.start_t - 0.1 && now <= mk.end_t + 0.1;
                if (!st || Math.abs(st.t - now) > 0.15 || !inRange) return null;
                const b = st.box, on = mk.id === selMark;
                return <g key={`m${mk.id}`} onPointerDown={(ev) => { if (!marking) { ev.stopPropagation(); setSelMark(mk.id); setMarking(true); setSel(null); pause(); } }}
                  style={{ cursor: "pointer" }}>
                  <rect x={b[0]} y={b[1]} width={b[2] - b[0]} height={b[3] - b[1]} fill="transparent" stroke="var(--accent, #4aa3ff)"
                    strokeWidth={on ? sw * 2 : sw} strokeDasharray={st.score >= SURE ? undefined : `${2 * sw} ${2 * sw}`} />
                  <text x={b[0]} y={Math.max(fs, b[1] - 3)} fontSize={fs} fill="var(--accent, #4aa3ff)" paintOrder="stroke" stroke="#000" strokeWidth={3 * px}>
                    T{mk.id} {mk.cls}{st.score < SURE ? " · unsure" : ""}{mk.state === "confirmed" ? " · confirmed" : ""}</text></g>;
              })}
              {show !== "none" && <Labels events={shown} now={now} sel={sel} px={px} showId={showId} showScore={showScore}
                boxAt={boxAt} statusText={(e: PEvent) => e.status !== "pending" ? STATUS_TEXT[e.status] : ""} W={W} />}
              {drag && !measuring && !zoning && <rect x={Math.min(drag.x0, drag.x1)} y={Math.min(drag.y0, drag.y1)} width={Math.abs(drag.x1 - drag.x0)}
                height={Math.abs(drag.y1 - drag.y0)} fill="none" stroke="var(--human)" strokeWidth={sw} strokeDasharray={`${4 * sw} ${2 * sw}`} />}
            </svg>
          </div>
          <div className="row" style={{ marginTop: 6 }}>
            <button className="primary" onClick={toggle} style={{ minWidth: 76 }} title="Space">{playing ? "Pause" : "Play"}</button>
            <button onClick={() => { pause(); seek(now - frame); }} title="Previous frame (←)">◀ frame</button>
            <button onClick={() => { pause(); seek(now + frame); }} title="Next frame (→)">frame ▶</button>
            <span className="mono">{fmtT(now)} / {fmtT(dur)}</span>
            <select aria-label="Speed" value={speed} onChange={(e) => setSpeed(Number(e.target.value))}>
              {SPEEDS.map((s) => <option key={s} value={s}>{s}×</option>)}
            </select>
            <select aria-label="Boxes shown" value={show} onChange={(e) => setShow(e.target.value as keyof typeof SHOW)}>
              {Object.entries(SHOW).map(([k, v]) => <option key={k} value={k}>{v}</option>)}
            </select>
            <select aria-label="Tail" value={tail} onChange={(e) => setTail(e.target.value as TailLen)}
              title="A line behind each target showing where it has been; a ring where it stood still">
              {Object.entries(TAILS).map(([k, v]) => <option key={k} value={k}>{v}</option>)}
            </select>
            <a className="small" href="#/settings/video" title="Colours of the vehicle boxes and of the tail's dots">Colours…</a>
            <label className="check small" title="A small ID over each target, coloured by how sure the detector was">
              <input type="checkbox" checked={showId} onChange={(e) => setShowId(e.target.checked)} /> IDs</label>
            <label className="check small" title="The detector's score (0 to 1) beside each ID (S)">
              <input type="checkbox" checked={showScore} onChange={(e) => setShowScore(e.target.checked)} /> Score</label>
            {(showId || showScore) && <ScoreKey />}
            <span style={{ flex: 1 }} />
            <button className={adding ? "primary" : ""} onClick={() => { setAdding(!adding); setSel(null); pause(); }}
              title="Draw a target the analysis missed (B)">+ Missed target</button>
            {adding && <select aria-label="Class of the new target" value={addCls} onChange={(e) => setAddCls(e.target.value)}>
              {classes.map((c) => <option key={c}>{c}</option>)}</select>}
            <button className={marking ? "primary" : ""} onClick={() => { setMarking(!marking); setSelMark(null); setAdding(false); setSel(null); pause(); }}
              title="A kind of target the detectors do not find (a drone, a tractor): draw it once, the app follows it, and once you confirm it teaches the detector (T)">🎯 Teach a new target</button>
            {marking && !curMark && <NamePick label="What the target is" value={markCls} onChange={setMarkCls}
              suggestions={[...classes.filter((c) => !["car", "truck", "bus", "motorcycle", "bicycle"].includes(c)), "drone", "tractor", "scooter", "cart"]} />}
            <button className={measuring ? "primary" : ""} onClick={() => { setMeasuring(!measuring); setLine(null); setAdding(false); setSel(null); pause(); }}
              title="Measure the camera, for speeds in km/h and sizes in metres (M)">📏 {measuring ? "Measuring" : cal ? "Measured" : "Measure"}</button>
            {measuring && line && <select aria-label="Distance between the two points" value="" onChange={(e) => e.target.value && saveMeasure(Number(e.target.value))}>
              <option value="">The two points are…</option>
              <option value={4.5}>a car's length apart (4.5 m)</option>
              <option value={3.5}>a lane's width apart (3.5 m)</option>
              <option value={12}>a bus's length apart (12 m)</option>
              {distances.map((d) => <option key={d} value={d}>{d} m apart</option>)}
            </select>}
            {measuring && cal && <button onClick={clearMeasure}>Remove the measure</button>}
            <button onClick={() => { pause(); location.hash = `#/m/${missionId}/map/${versionId}`; }}
              title="Put this camera on a map, and see each counted target's path on it">🗺 On the map</button>
            <select aria-label="Lines and areas" value={zoning} onChange={(e) => startZone(e.target.value as "" | "line" | "area")}>
              <option value="">Lines and areas…</option>
              <option value="line">Draw a counting line (L)</option>
              <option value="area">Draw an area</option>
            </select>
            {zoning && <select aria-label="Name" value={zoneName} onChange={(e) => setZoneName(e.target.value)}>
              {(zoning === "line" ? zoneNames.line : zoneNames.area).map((n) => <option key={n} value={n}>{n}{zones.some((z) => z.name === n) ? " (another)" : ""}</option>)}
            </select>}
            {zoning === "area" && zpts.length >= 3 && <button className="primary" onClick={() => saveZone(zpts)}>Finish the area</button>}
            <button onClick={() => setHelp(!help)} title="Keys (?)">?</button>
          </div>
          <Timeline lanes={tl} dur={dur} now={now} sel={sel} onSeek={(t) => { pause(); seek(t); }} onPick={(e) => pick(e, true)} />
          {fallback && <div className="msg warn">This browser cannot play the video's format, so its frames are decoded on the workstation, five a second.</div>}
          {zoning === "line" && <div className="msg">Drag the line across the road. Targets crossing it are counted both ways; "in" is
            to the right of the line as you draw it (the arrow shows it). Every video from this camera uses it.</div>}
          {zoning === "area" && <div className="msg">Click each corner of the area; click the first corner again (or Enter) to finish.
            The time each target spends inside is counted. Every video from this camera uses it.</div>}
          {measuring && <div className="msg">{line ? "Pick the real distance between the two points." :
            "Pause where you can see the road. Drag a line between two points on the road whose distance you know (a car's length, a lane's width, road markings), at the depth where the targets pass."}
            {" "}One measure covers this video and the other videos from the same folder.</div>}
          {marking && <div className="msg">{following ? "Following the target through the video…" : curMark
            ? `Drawing now adds another box of T${curMark.id} (${curMark.cls}) on this frame and follows it again. Esc to let go.`
            : "Pause where the target is clear and drag a box around all of it. The app follows it forwards and backwards and suggests where it enters and leaves."}</div>}
          {adding && <div className="msg">Drag a box around the target the analysis missed. It is added as a confirmed {addCls}.</div>}
          {help && <div className="msg small">
            Space play or pause · ← → one frame (Shift: one second) · J / K next or previous event to review · C confirm ·
            X or Delete reject · 1–{classes.length} class ({classes.map((c, i) => `${i + 1} ${c}`).join(", ")}) ·
            B missed target · T teach a new target · M measure the camera · L counting line · drag on the paused picture to redraw the picked event's box · A approve for learning ·
            H hide boxes · S show scores · Esc let go</div>}
        </div>
        <div className="grid" style={{ gap: 10, alignContent: "start", display: panel ? undefined : "none" }}>
          <div className="card grid" style={{ gap: 8 }}>
            {selEv ? <>
              <div className="row" style={{ justifyContent: "space-between" }}>
                <strong className="mono">{selEv.key}</strong>
                <span className={`badge ${selEv.status === "confirmed" ? "approved" : selEv.status === "rejected" ? "failed" : ""}`}>{STATUS_TEXT[selEv.status]}</span>
              </div>
              <div className="muted small">{fmtT(selEv.first_t)} to {fmtT(selEv.last_t)} · {selEv.origin === "human" ? "added by you" : "found by the analysis"}
                {selEv.fixes.length ? ` · ${selEv.fixes.length} box(es) drawn by you` : ""}</div>
              <div className="row">
                <button className="primary" onClick={() => decide("confirm")} title="C">Confirm</button>
                <button className="danger" onClick={() => decide("reject")} title="X">Reject</button>
                <select aria-label="Class" value={selEv.cls} onChange={(e) => setClass(e.target.value)}>
                  {[...new Set([...classes, selEv.cls])].map((c) => <option key={c}>{c}</option>)}</select>
              </div>
              <div className="muted small">Pause where the box is wrong and drag a new one on the picture: it replaces the box on that frame.</div>
              {myBoxHere && <button onClick={() => revise(selEv.id, "clear_box", { t: myBoxHere.t })}>Take back my box on this frame</button>}
              <a href={`#/m/${missionId}/events/${selEv.id}`}>Times, split, merge and clips…</a>
            </> : <div className="muted">Click a box on the video, or a bar under it, to review that event. Press J to go to the
              next event not reviewed yet.</div>}
          </div>
          {msg && <div className={`msg ${msg.kind}`}>{msg.text}</div>}
          {curMark && <MarkCard mk={curMark} now={now} missionId={missionId}
            onSeek={(t) => { pause(); seek(t); }} onClose={() => { setSelMark(null); setMarking(false); }}
            onChange={async (fn, ok) => { await act(fn, ok); await loadMarks(); }} />}
          {marked.length > 0 && !curMark && <div className="card grid" style={{ gap: 4 }}>
            <strong className="small">Targets you are teaching</strong>
            {marked.map((mk) => <div key={mk.id} className="row small click" style={{ justifyContent: "space-between" }}
              onClick={() => { setSelMark(mk.id); setMarking(true); setSel(null); pause(); if (mk.start_t != null) seek(mk.start_t); }}>
              <span>T{mk.id} {mk.cls} · {mk.start_t != null ? `${fmtT(mk.start_t)} to ${fmtT(mk.end_t ?? mk.start_t)}` : "not found"}</span>
              <span className={`badge ${mk.state === "confirmed" ? "approved" : ""}`}>{mk.state === "confirmed" ? "teaches" : "to check"}</span></div>)}
          </div>}
          {tel === null ? <button onClick={askFlight} title="Read the drone's or aircraft's own flight record (a DJI .srt beside the video, or KLV inside it)">Flight data</button>
            : tel.source ? <Aircraft tel={tel as any} t={now} /> : <div className="muted small">This recording carries no flight data.</div>}
          {zones.length > 0 && <div className="card grid" style={{ gap: 4 }}>
            <strong className="small">Lines and areas of this camera</strong>
            {zones.map((z) => <div key={z.uid} className="row small" style={{ justifyContent: "space-between" }}>
              <span>{z.kind === "line" ? "Line" : "Area"}: {z.name}</span>
              <button className="danger" onClick={() => deleteZone(z)}>Remove</button></div>)}
          </div>}
          <div className="card" style={{ padding: 0, maxHeight: "48vh", overflow: "auto" }}>
            <table className="t"><tbody>
              {[...evs].sort((a, b) => a.first_t - b.first_t).map((e) => (
                <tr key={e.id} className={`click ${e.id === sel ? "sel" : ""}`} onClick={() => pick(e, true)}>
                  <td className="mono">{e.origin === "human" ? "H" : "M"} {e.key}</td>
                  <td>{e.cls}</td>
                  <td className="mono small">{fmtT(e.first_t).slice(3)}</td>
                  <td><span className={`badge ${e.status === "confirmed" ? "approved" : e.status === "rejected" ? "failed" : ""}`}>{STATUS_TEXT[e.status]}</span></td>
                </tr>))}
            </tbody></table>
          </div>
        </div>
      </div>
    </div>
  );
}

/** One marked target: where it is in view (start and end, set from the playhead), its boxes, confirm or remove. */
function MarkCard({ mk, now, missionId, onSeek, onClose, onChange }: {
  mk: Marked; now: number; missionId: string; onSeek: (t: number) => void; onClose: () => void;
  onChange: (fn: () => Promise<unknown>, ok?: string) => Promise<void>;
}) {
  const put = (body: object, ok: string) => onChange(() => api(`/missions/${missionId}/marks/${mk.id}`, { method: "PUT", body }), ok);
  const sure = mk.steps.filter((s) => s.score >= SURE && mk.start_t != null && s.t >= mk.start_t && s.t <= (mk.end_t ?? 0)).length;
  return (
    <div className="card grid" style={{ gap: 8 }}>
      <div className="row" style={{ justifyContent: "space-between" }}>
        <strong>T{mk.id} · {mk.cls}</strong>
        <span className={`badge ${mk.state === "confirmed" ? "approved" : ""}`}>{mk.state === "confirmed" ? "teaches the detector" : "to check"}</span>
      </div>
      {mk.start_t == null ? <div className="muted small">It could not be followed from your box. Draw it again on a clearer frame.</div> : <>
        <div className="row small">
          <button onClick={() => onSeek(mk.start_t!)}>Enters {fmtT(mk.start_t)}</button>
          <button onClick={() => put({ start_t: now }, `Starts at ${fmtT(now)} now.`)} disabled={now > (mk.end_t ?? now)}>Starts here</button>
        </div>
        <div className="row small">
          <button onClick={() => onSeek(mk.end_t!)}>Leaves {fmtT(mk.end_t!)}</button>
          <button onClick={() => put({ end_t: now }, `Ends at ${fmtT(now)} now.`)} disabled={now < (mk.start_t ?? now)}>Ends here</button>
        </div>
        <div className="muted small">{mk.marks.length} box{mk.marks.length === 1 ? "" : "es"} drawn by you · found surely on {sure} frames
          in the range. A dashed box is where the app is unsure: pause there and draw it again.</div>
        <div className="row">
          {mk.state === "confirmed"
            ? <button onClick={() => put({ state: "suggested" }, "It no longer teaches until you confirm it again.")}>Take back</button>
            : <button className="primary" onClick={() => put({ state: "confirmed" }, `Confirmed: up to 60 frames between ${fmtT(mk.start_t)} and ${fmtT(mk.end_t!)} teach the detector what a ${mk.cls} is.`)}>Confirm for training</button>}
          <button className="danger" onClick={() => onChange(() => api(`/missions/${missionId}/marks/${mk.id}`, { method: "DELETE" }), "Removed.").then(onClose)}>Remove</button>
          <button onClick={onClose}>Done</button>
        </div>
      </>}
    </div>
  );
}

function Timeline({ lanes: ls, dur, now, sel, onSeek, onPick }: {
  lanes: PEvent[][]; dur: number; now: number; sel: number | null; onSeek: (t: number) => void; onPick: (e: PEvent) => void;
}) {
  const ref = useRef<SVGSVGElement>(null);
  if (!dur) return null;
  const LH = 9, h = Math.max(1, ls.length) * LH + 8;
  const x = (t: number) => (t / dur) * 1000;
  const fill = (e: PEvent) => e.status === "rejected" ? "var(--muted)" : e.status === "pending" ? "var(--machine)" : "var(--human)";
  return (
    <svg ref={ref} className="timeline" viewBox={`0 0 1000 ${h}`} preserveAspectRatio="none" style={{ width: "100%", height: h, marginTop: 6, cursor: "pointer" }}
      role="img" aria-label="Events along the video" onClick={(ev) => {
        const r = ref.current!.getBoundingClientRect();
        onSeek(((ev.clientX - r.left) / r.width) * dur);
      }}>
      <rect x={0} y={0} width={1000} height={h} fill="var(--panel-2, rgba(127,127,127,0.08))" />
      {ls.map((lane, i) => lane.map((e) => (
        <rect key={e.id} x={x(e.first_t)} y={4 + i * LH} width={Math.max(2, x(e.last_t) - x(e.first_t))} height={LH - 3} rx={2}
          fill={fill(e)} opacity={e.id === sel ? 1 : 0.7} stroke={e.id === sel ? "var(--text, #fff)" : "none"} strokeWidth={1}
          onClick={(ev) => { ev.stopPropagation(); onPick(e); }}>
          <title>{`${e.key} ${e.cls}: ${fmtT(e.first_t)} to ${fmtT(e.last_t)}, ${STATUS_TEXT[e.status]}`}</title>
        </rect>)))}
      <line x1={x(now)} x2={x(now)} y1={0} y2={h} stroke="var(--accent, #4aa3ff)" strokeWidth={2} vectorEffect="non-scaling-stroke" />
    </svg>
  );
}

/** The aircraft's position, height and heading at the player's time, from its own metadata. */
function Aircraft({ tel, t }: { tel: { source: string; summary: any; samples: any[] }; t: number }) {
  const s = tel.samples;
  let a: any = null;
  if (s.length && t >= s[0].t - 0.5 && t <= s[s.length - 1].t + 0.5) {
    let i = 0;
    while (i + 1 < s.length && s[i + 1].t <= t) i++;
    const p = s[i], q = s[Math.min(i + 1, s.length - 1)];
    const w = q.t > p.t ? Math.min(1, Math.max(0, (t - p.t) / (q.t - p.t))) : 0;
    a = {} as any;
    for (const k of Object.keys(p)) a[k] = typeof p[k] === "number" && typeof q[k] === "number" && k !== "heading" ? p[k] + w * (q[k] - p[k]) : p[k];
  }
  const deg = (v: number, pos: string, neg: string) => `${Math.abs(v).toFixed(5)}°${v >= 0 ? pos : neg}`;
  const src = tel.source === "klv-0601" ? "KLV (MISB 0601)" : "DJI flight record";
  return (
    <div className="card grid" style={{ gap: 4 }}>
      <strong className="small">Aircraft <span className="muted">· {src}{tel.summary?.platform ? ` · ${tel.summary.platform}` : ""}</span></strong>
      {a && a.lat != null ? <>
        <div className="small mono">{deg(a.lat, "N", "S")} {deg(a.lon, "E", "W")}</div>
        <div className="small">
          {a.rel_alt != null && <>Height {Math.round(a.rel_alt)} m above take-off · </>}
          {a.alt != null && <>{Math.round(a.alt)} m above sea · </>}
          {a.heading != null && <>heading {Math.round(a.heading)}°</>}
          {a.hfov != null && <> · view {a.hfov.toFixed(1)}° wide</>}
        </div>
        {a.center_lat != null && <div className="small muted">Looking at {deg(a.center_lat, "N", "S")} {deg(a.center_lon, "E", "W")}</div>}
      </> : <div className="small muted">No position at this moment.</div>}
    </div>
  );
}
