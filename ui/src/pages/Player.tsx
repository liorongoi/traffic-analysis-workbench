import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api, ApiError, post } from "../api";
import { fmtT } from "./Events";

// The player: a recording plays with every event's box drawn on it, and is reviewed and edited where it plays, as a
// still image is. Click a box (or a bar on the timeline) to pick its event; drag on the paused picture to redraw the
// picked event's box on this frame, or, after "Missed target", to add what the analysis did not find.

type PBox = [number, number, number, number, number];          // t, x1, y1, x2, y2
type PEvent = {
  id: number; key: string; first_t: number; last_t: number; cls: string; machine_cls: string; status: string;
  origin: "machine" | "human"; fixes: { t: number; box: number[] }[]; boxes: PBox[]; flags: string[];
};
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

export default function Player({ missionId, versionId }: { missionId: string; versionId: number }) {
  const [ov, setOv] = useState<Overlay | null>(null);
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
  const video = useRef<HTMLVideoElement>(null);
  const svg = useRef<SVGSVGElement>(null);

  const load = useCallback(async () => {
    try { setOv(await api(`/missions/${missionId}/recordings/${versionId}/overlay`)); setErr(null); }
    catch (e) { setErr((e as ApiError).message); }
  }, [missionId, versionId]);
  useEffect(() => { load(); }, [load]);

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
  const drawing = adding || selEv != null;
  const onDown = (ev: React.PointerEvent) => {
    if (!drawing || ev.button !== 0) return;
    pause();
    const p = toImage(ev);
    setDrag({ x0: p.x, y0: p.y, x1: p.x, y1: p.y });
    (ev.target as Element).setPointerCapture?.(ev.pointerId);
  };
  const onMove = (ev: React.PointerEvent) => { if (drag) { const p = toImage(ev); setDrag({ ...drag, x1: p.x, y1: p.y }); } };
  const onUp = async () => {
    if (!drag) return;
    const box = [Math.min(drag.x0, drag.x1), Math.min(drag.y0, drag.y1), Math.max(drag.x0, drag.x1), Math.max(drag.y0, drag.y1)];
    setDrag(null);
    if (box[2] - box[0] < 4 || box[3] - box[1] < 4) { if (!adding) setSel(null); return; }   // a click on the picture: let go
    const t = fallback ? now : video.current?.currentTime ?? now;
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
      else if (k === "b") { setAdding((a) => !a); setSel(null); pause(); }
      else if (k === "a") approve();
      else if (k === "h") setShow((s) => (s === "none" ? "all" : "none"));
      else if (k === "escape") { setDrag(null); setAdding(false); setSel(null); setHelp(false); }
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
        <button className={rec.approved ? "" : "primary"} onClick={approve}
          title="Say every target in this video is marked and every box is right; it then teaches the detector (A)">
          {rec.approved ? "Reopen" : rec.pending ? `Confirm ${rec.pending} left and approve for learning` : "Approve for learning"}</button>
      </div>
      <div className="grid" style={{ gridTemplateColumns: "minmax(0, 1fr) 340px", gap: 10 }}>
        <div className="card" style={{ padding: 8 }}>
          <div className="stage" style={{ position: "relative", background: "#000", aspectRatio: `${W} / ${H}`, maxHeight: "56vh" }}>
            {!fallback && <video ref={video} src={`/api/missions/${missionId}/videos/${versionId}`} preload="auto"
              style={{ width: "100%", height: "100%", display: "block", objectFit: "contain" }}
              onPlay={() => setPlaying(true)} onPause={() => setPlaying(false)} onEnded={() => setPlaying(false)}
              onError={() => setFallback(true)} />}
            {fallback && <img alt={`Frame at ${fmtT(now)}`} src={`/api/missions/${missionId}/videos/${versionId}/frame?t=${now.toFixed(2)}&w=1920`}
              style={{ width: "100%", height: "100%", display: "block", objectFit: "contain" }} />}
            <svg ref={svg} viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="xMidYMid meet"
              style={{ position: "absolute", inset: 0, width: "100%", height: "100%", cursor: drawing ? "crosshair" : "default" }}
              onPointerDown={onDown} onPointerMove={onMove} onPointerUp={onUp}>
              {shown.map((e) => {
                const b = boxAt(e, now);
                if (!b) return null;
                const look = LOOK[e.status === "rejected" ? "rejected" : e.origin === "human" || e.fixes.length ? "human" : e.status] ?? LOOK.pending;
                const on = e.id === sel;
                return (
                  <g key={e.id} onPointerDown={(ev) => { if (!adding && e.id !== sel) { ev.stopPropagation(); pick(e); pause(); } }}
                    style={{ cursor: "pointer" }}>
                    <rect x={b[0]} y={b[1]} width={b[2] - b[0]} height={b[3] - b[1]} fill={on ? "rgba(255,255,255,0.06)" : "transparent"}
                      stroke={look.stroke} strokeDasharray={look.dash ? look.dash.split(" ").map((v) => Number(v) * sw / 2).join(" ") : undefined}
                      strokeWidth={on ? sw * 2 : sw} />
                    <text x={b[0]} y={Math.max(fs, b[1] - 3)} fontSize={fs} fill={look.stroke} paintOrder="stroke" stroke="#000" strokeWidth={3 * px}>
                      {e.origin === "human" ? "H" : "M"} {e.key} {e.cls}{e.status !== "pending" ? ` · ${STATUS_TEXT[e.status]}` : ""}</text>
                  </g>
                );
              })}
              {drag && <rect x={Math.min(drag.x0, drag.x1)} y={Math.min(drag.y0, drag.y1)} width={Math.abs(drag.x1 - drag.x0)}
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
            <span style={{ flex: 1 }} />
            <button className={adding ? "primary" : ""} onClick={() => { setAdding(!adding); setSel(null); pause(); }}
              title="Draw a target the analysis missed (B)">+ Missed target</button>
            {adding && <select aria-label="Class of the new target" value={addCls} onChange={(e) => setAddCls(e.target.value)}>
              {classes.map((c) => <option key={c}>{c}</option>)}</select>}
            <button onClick={() => setHelp(!help)} title="Keys (?)">?</button>
          </div>
          <Timeline lanes={tl} dur={dur} now={now} sel={sel} onSeek={(t) => { pause(); seek(t); }} onPick={(e) => pick(e, true)} />
          {fallback && <div className="msg warn">This browser cannot play the video's format, so its frames are decoded on the workstation, five a second.</div>}
          {adding && <div className="msg">Drag a box around the target the analysis missed. It is added as a confirmed {addCls}.</div>}
          {help && <div className="msg small">
            Space play or pause · ← → one frame (Shift: one second) · J / K next or previous event to review · C confirm ·
            X or Delete reject · 1–{classes.length} class ({classes.map((c, i) => `${i + 1} ${c}`).join(", ")}) ·
            B missed target · drag on the paused picture to redraw the picked event's box · A approve for learning ·
            H hide boxes · Esc let go</div>}
        </div>
        <div className="grid" style={{ gap: 10, alignContent: "start" }}>
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
